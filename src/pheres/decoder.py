# -*- coding: utf-8 -*-
"""
Partial recoding of the stdlib json module to support Jsonable decoding
"""
# This modules containes modification of the source code of python 3.8
# 'json' module, part of the standard library.
# The original source code is available at:
# https://github.com/python/cpython/tree/3.8/Lib/json
#
# The original source code is released under the
# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
# (see PYTHON-LICENSE file) and is the property of the Python Software Foundation:
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020 Python Software Foundation;
# All Rights Reserved
#
# The PSL authorize such derivative work (see PYTHON-LICENSE file)

# Standard library import
from abc import ABCMeta, ABC, abstractmethod
from dataclasses import dataclass, field, replace
import functools
from itertools import chain
import json
from json import JSONDecodeError, JSONDecoder
import types
import typing
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_origin,
    get_type_hints,
    overload,
)

# Local import
from .misc import JSONError
from .jsonable import (
    # Type Hints
    TypeHint,
    JSONValue,
    JSONArray,
    JSONObject,
    JSONType,
    _JSONArrayTypes,
    # Type utilities
    get_args,
    typeof,
    typecheck,
    normalize_json_type,
    # JSONable API
    _JSONableValue,
    _JSONableArray,
    _JSONableObject,
    SmartDecoder
)
from . import jsonable
from .utils import FlatKey

# import internals of stdlib 'json' module
# Those are not part of the public API
from json.decoder import WHITESPACE, WHITESPACE_STR, scanstring
from json.scanner import NUMBER_RE

__all__ = ["TypedJSONDecodeError", "TypedJSONDecoder", "deserialize"]

# Type Variable
U = TypeVar("U")

# Type Aliases
Pos = Union[int, FlatKey]

TypeOrig = Optional[TypeHint]
TypeArgs = Optional[TypeHint]

TypeTuple = Tuple[TypeHint, ...]
OrigTuple = Tuple[TypeOrig, ...]
ArgsTuple = Tuple[TypeArgs, ...]

TypeCache = Tuple[TypeTuple, OrigTuple, ArgsTuple]
TypeFilter = Callable[[TypeHint, TypeOrig, TypeArgs], bool]

# sentinel value
MISSING = object()


class TypedJSONDecodeError(JSONError, JSONDecodeError):
    """
    Raised when the decoded type is not the expected one
    """

    @overload
    def __init__(self, msg: str, doc: str, pos: int) -> None:
        ...

    @overload
    def __init__(self, msg: str, doc: JSONObject, pos: FlatKey) -> None:
        ...

    def __init__(self, msg, doc, pos):
        """
        Special case when the decoded document is an object
        """
        if not isinstance(doc, str):
            # quote the string keys only
            pos = ['"%s"' % p if isinstance(p, str) else str(p) for p in pos]
            keys = " -> ".join(("<base object>", *pos))
            errmsg = "%s: at %s" % (msg, keys)
            ValueError.__init__(self, errmsg)
            self.msg = msg
            self.doc = doc
            self.pos = pos
            self.lineno = None
            self.colno = None
        else:
            super().__init__(msg, doc, pos)


####################
# MODULE UTILITIES #
####################


def sync_filter(func, *iterables):
    """
    Filter multiple iterable at once, selecting values at index i
    such that func(iterables[0][i], iterables[1][i], ...) is True
    """
    return tuple(zip(*tuple(i for i in zip(*iterables) if func(*i)))) or ((),) * len(
        iterables
    )


def _make_pretty_type():
    tps = get_type_hints(jsonable)
    table = {
        # Standard Type Hints
        jsonable.JSONValue: "JSONValue",
        jsonable.JSONArray: "JSONArray",
        jsonable.JSONObject: "JSONObject",
        JSONType: "JSONType",
        # Resolved version
        tps["_jval"]: "JSONValue",
        tps["_jarr"]: "JSONArray",
        tps["_jobj"]: "JSONObject",
        tps["_jtyp"]: "JSONType",
    }

    def pretty_type(tp):
        return table.get(tp, tp)

    return pretty_type


pretty_type = _make_pretty_type()


####################
# DECODING CONTEXT #
####################


@dataclass(frozen=True)
class DecodeContext:
    """
    Internal class to keep tracks of types during typed decoding

    DecodeContext are immutable
    """

    doc: Union[str, JSONObject]
    pos: Pos
    types: TypeTuple
    origs: OrigTuple = None
    args: ArgsTuple = None
    parent_context: Optional["DecodeContext"] = None
    parent_key: Optional[Union[int, str]] = None

    @staticmethod
    def process_tp(tp):
        if get_origin(tp) is Union:
            return get_args(tp)
        return (tp,)

    def __post_init__(self, /):
        # cache types origins and arguments for performance
        if self.origs is None:
            object.__setattr__(
                self, "origs", tuple(get_origin(tp) for tp in self.types)
            )
        if self.args is None:
            object.__setattr__(self, "args", tuple(get_args(tp) for tp in self.types))
        if self.parent_context is not None and self.parent_key is None:
            raise ValueError("DecodeContext with a parent must be given a parent key")

    def err_msg(self, /, *, msg: str = None, value=MISSING) -> str:
        parts = []
        if msg:
            parts.append(msg)
        else:
            parts.append(f"Expected type {Union[self.types]}")
        if value is not MISSING:
            parts.append(f"got {value}")
        parts.append("at")
        return ", ".join(parts)

    # HELPER METHODS
    def filter_types(self, /, filter_func: TypeFilter) -> TypeCache:
        return sync_filter(filter_func, self.types, self.origs, self.args)

    def filtered(
        self, /, filter_func: TypeFilter, err_msg: str, *, err_pos=MISSING
    ) -> "DecodeContext":
        types, origs, args = sync_filter(filter_func, self.types, self.origs, self.args)
        if not types:
            raise TypedJSONDecodeError(
                msg=err_msg,
                doc=self.doc,
                pos=self.pos if err_pos is MISSING else err_pos,
            )
        return replace(self, types=types, origs=origs, args=args)

    def get_array_subtypes(self, /, index: int) -> TypeTuple:
        subtypes = []
        for tp, orig, arg in zip(self.types, self.origs, self.args):
            if isinstance(orig, type):
                if issubclass(orig, tuple):
                    subtypes.append(arg[index])
                    continue
                elif issubclass(orig, list):
                    subtypes.append(arg[0])
                    continue
            elif isinstance(tp, type) and issubclass(tp, _JSONableArray):
                if isinstance(tp._JTYPES, tuple):
                    subtypes.append(tp._JTYPES[index])
                else:
                    subtypes.append(type._JTYPES)
                continue
            raise JSONError(f"Unhandled Array type {tp}")
        return tuple(subtypes)

    def get_object_subtypes(self, /, key: str) -> TypeTuple:
        subtypes = []
        for tp, orig, arg in zip(self.types, self.origs, self.args):
            if isinstance(orig, type) and issubclass(orig, dict):
                subtypes.append(arg[1])
            elif isinstance(tp, type) and issubclass(tp, _JSONableObject):
                subtypes.append(tp._ALL_JATTRS[key].type_hint)
            else:
                raise JSONError(f"Unhandled Object type {tp}")
        return tuple(subtypes)

    # FILTERS AND FILTER FACTORIES
    @staticmethod
    def accept_array(tp: TypeHint, orig: TypeOrig, arg: TypeArgs) -> bool:
        return (
            isinstance(orig, type) and issubclass(orig, _JSONArrayTypes)
        ) or isinstance(tp, _JSONableArray)

    @staticmethod
    def accept_min_length(index: int) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, list):
                    return True
                elif issubclass(orig, tuple):
                    return len(args) > index
            elif isinstance(tp, type) and issubclass(tp, _JSONableArray):
                if isinstance(tp._JTYPES, tuple):
                    return len(tp._JTYPES) > index
                return True
            return False

        return accept

    @staticmethod
    def accept_array_value(index: int, value: object) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, tuple):
                    return typecheck(value, args[index])
                elif issubclass(orig, list):
                    return typecheck(value, args[0])
            elif isinstance(tp, type) and issubclass(tp, _JSONableArray):
                if issubclass(tp._JTYPES, tuple):
                    return typecheck(value, tp._JTYPES[index])
                return typecheck(value, tp._JTYPES[0])
            raise JSONError(f"Unhandled Array type {tp}")

        return accept

    @staticmethod
    def accept_length(length: int) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, list):
                    return True
                elif issubclass(orig, tuple):
                    return len(args) == length
            elif isinstance(tp, type) and issubclass(tp, _JSONableArray):
                if isinstance(tp._JTYPES, tuple):
                    return len(tp._JTYPES) == length
                return True
            raise JSONError(f"Unhandled Array type {tp}")

        return accept

    @staticmethod
    def accept_object(tp: TypeHint, orig: TypeOrig, arg: TypeArgs) -> bool:
        return (isinstance(orig, type) and issubclass(orig, dict)) or (
            isinstance(tp, type) and issubclass(tp, _JSONableObject)
        )

    @staticmethod
    def accept_key(key: str) -> TypeFilter:
        def accept(type_: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type) and issubclass(orig, dict):
                return True
            elif isinstance(type_, type) and issubclass(type_, _JSONableObject):
                return key in type_._ALL_JATTRS
            return False

        return accept

    @staticmethod
    def accept_object_value(key: str, value: object) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type) and issubclass(orig, dict):
                return typecheck(value, args[1])
            elif isinstance(tp, type) and issubclass(tp, _JSONableObject):
                return typecheck(value, tp._ALL_JATTRS[key].type_hint)
            raise JSONError(f"Unhandled Object type {tp}")

        return accept

    # CONTEXT SPECIALIZATION METHODS
    def array_context(self, /) -> "DecodeContext":
        return self.filtered(self.accept_array, self.err_msg(value="'Array'"))

    def index_context(self, /, index: int, pos: Pos) -> "DecodeContext":
        parent = self.filtered(
            self.accept_min_length(index),
            self.err_msg(msg=f"'Array' of length >={index+1}"),
        )
        return DecodeContext(
            doc=self.doc,
            pos=pos,
            types=parent.get_array_subtypes(index),
            parent_context=parent,
            parent_key=index,
        )

    def object_context(self, /) -> "DecodeContext":
        return self.filtered(self.accept_object, self.err_msg(value="'Object'"))

    def key_context(self, /, key: str, key_pos: Pos) -> "DecodeContext":
        parent = self.filtered(
            self.accept_key(key),
            self.err_msg(
                msg=f"Inferred type {Union[self.types]} has no key '{key}'",
            ),
            err_pos=key_pos,
        )
        return DecodeContext(
            doc=self.doc,
            pos=key_pos,
            types=parent.get_object_subtypes(key),
            parent_context=parent,
            parent_key=key,
        )

    # TYPECHECKING METHODS
    def typecheck_value(
        self, /, value: JSONValue, end_pos: U, start_pos: Pos
    ) -> Tuple[JSONValue, U, "DecodeContext"]:
        types, classes = [], []
        for tp in self.types:
            if isinstance(tp, type) and issubclass(tp, _JSONableValue):
                if typecheck(value, Union[tp._JTYPE]):
                    classes.append(tp)
            elif typecheck(value, tp):
                types.append(tp)
        if not types and not classes:
            raise TypedJSONDecodeError(
                msg=self.err_msg(value=value), doc=self.doc, pos=start_pos
            )
        if classes:
            if len(classes) > 1:
                raise TypedJSONDecodeError(
                    msg=self.err_msg(
                        msg=f"Multiple JSONable class found for value {value}"
                    ),
                    doc=self.doc,
                    pos=self.pos,
                )
            value = _JSONableValue.make(classes[0], value)
        parent = None
        if self.parent_context is not None:
            parent = self.parent_context
            key = self.parent_key
            if isinstance(key, int):
                filter_func = self.accept_array_value(key, value)
            elif isinstance(key, str):
                filter_func = self.accept_object_value(key, value)
            else:
                raise JSONError(f"Unhandled parent key {key}")
            parent = parent.filtered(
                filter_func, parent.err_msg(value=f"{value} of type {type(value)}")
            )
        return value, end_pos, parent

    def typecheck_array(
        self, /, array: JSONArray, end_pos: U
    ) -> Tuple[JSONArray, U, "DecodeContext"]:
        types, *_ = self.filter_types(self.accept_length(len(array)))
        if not types:
            raise TypedJSONDecodeError(
                msg=self.err_msg(
                    msg=f"Inferred type {Union[self.types]}",
                    value=f"{array}, an 'Array' of len {len(array)} which is too short",
                ),
                doc=self.doc,
                pos=self.pos,
            )
        classes = [
            tp
            for tp in types
            if isinstance(tp, type) and issubclass(tp, _JSONableArray)
        ]
        parent = self.parent_context
        if classes:
            if len(classes) > 1:
                raise TypedJSONDecodeError(
                    msg=self.err_msg(
                        msg=f"Multiple JSONable class found for array {array}"
                    ),
                    doc=self.doc,
                    pos=self.pos,
                )
            array = _JSONableArray.make(classes[0], array)
            if parent is not None:
                key = self.parent_key
                if isinstance(key, int):
                    filter_func = self.accept_array_value(key, array)
                elif isinstance(key, str):
                    filter_func = self.accept_object_value(key, array)
                else:
                    raise JSONError(f"Unhandled parent key {key}")
                parent = parent.filtered(
                    filter_func, parent.err_msg(value=f"{array} of type {type(array)}")
                )
        return array, end_pos, parent

    def typecheck_object(
        self, /, obj: JSONObject, end_pos: U
    ) -> Tuple[JSONObject, int, "DecodeContext"]:
        classes = [
            type_
            for type_ in self.types
            if isinstance(type_, type) and issubclass(type_, _JSONableObject)
        ]
        classes = [
            cls
            for i, cls in enumerate(classes)
            if all(not issubclass(cls, other) for other in classes[i + 1 :])
        ]
        parent = self.parent_context
        if classes:
            if len(classes) > 1:
                raise TypedJSONDecodeError(
                    msg=self.err_msg(
                        msg=f"Multiple JSONable class found for object {obj}"
                    ),
                    doc=self.doc,
                    pos=self.pos,
                )
            obj = _JSONableObject.make(classes[0], obj)
            if parent is not None:
                key = self.parent_key
                if isinstance(key, int):
                    filter_func = self.accept_array_value(key, obj)
                elif isinstance(key, str):
                    filter_func = self.accept_object_value(key, obj)
                else:
                    raise JSONError(f"Unhandled parent key {key}")
                parent = parent.filtered(
                    filter_func, parent.err_msg(value=f"{obj} of type {type(obj)}")
                )
        return obj, end_pos, parent


############################
# JSON INTERNALS OVERRIDES #
############################

# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
def make_string_scanner(
    context: JSONDecoder,
) -> Callable[[str, int, DecodeContext], Tuple[JSONType, int, DecodeContext]]:
    parse_object = context.parse_object
    parse_array = context.parse_array
    parse_string = context.parse_string
    match_number = NUMBER_RE.match
    strict = context.strict
    parse_float = context.parse_float
    parse_int = context.parse_int
    parse_constant = context.parse_constant
    object_hook = context.object_hook
    object_pairs_hook = context.object_pairs_hook
    memo = context.memo

    def _scan_once(
        string: str, idx: int, ctx: DecodeContext
    ) -> Tuple[JSONType, int, DecodeContext]:
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        if nextchar == '"':
            return ctx.typecheck_value(*parse_string(string, idx + 1, strict), idx)
        elif nextchar == "{":
            return parse_object(
                (string, idx + 1),
                strict,
                _scan_once,
                ctx.object_context(),
                object_hook,
                object_pairs_hook,
                memo,
            )
        elif nextchar == "[":
            return parse_array((string, idx + 1), _scan_once, ctx.array_context())
        elif nextchar == "n" and string[idx : idx + 4] == "null":
            return ctx.typecheck_value(None, idx + 4, idx)
        elif nextchar == "t" and string[idx : idx + 4] == "true":
            return ctx.typecheck_value(True, idx + 4, idx)
        elif nextchar == "f" and string[idx : idx + 5] == "false":
            return ctx.typecheck_value(False, idx + 5, idx)

        m = match_number(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or "") + (exp or ""))
            else:
                res = parse_int(integer)
            return ctx.typecheck_value(res, m.end(), idx)
        elif nextchar == "N" and string[idx : idx + 3] == "NaN":
            return ctx.typecheck_value(parse_constant("NaN"), idx + 3, idx)
        elif nextchar == "I" and string[idx : idx + 8] == "Infinity":
            return ctx.typecheck_value(parse_constant("Infinity"), idx + 8, idx)
        elif nextchar == "-" and string[idx : idx + 9] == "-Infinity":
            return ctx.typecheck_value(parse_constant("-Infinity"), idx + 9, idx)
        else:
            raise StopIteration(idx)

    def scan_once(
        string: str, idx: int, ctx: DecodeContext
    ) -> Tuple[JSONType, int, DecodeContext]:
        try:
            return _scan_once(string, idx, ctx)
        finally:
            memo.clear()

    return scan_once


# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
def JSONObjectParser(
    s_and_end: Tuple[str, int],
    strict: bool,
    scan_once: Callable[[str, int, DecodeContext], Tuple[object, int, DecodeContext]],
    ctx: DecodeContext,
    object_hook: Callable,
    object_pairs_hook: Callable,
    memo: Optional[dict] = None,
    _w: Callable = WHITESPACE.match,
    _ws: str = WHITESPACE_STR,
) -> Tuple[JSONObject, int, DecodeContext]:
    s, end = s_and_end
    pairs = []
    pairs_append = pairs.append
    # Backwards compatibility
    if memo is None:
        memo = {}
    memo_get = memo.setdefault
    # Use a slice to prevent IndexError from being raised, the following
    # check will raise a more specific ValueError if the string is empty
    nextchar = s[end : end + 1]
    # Normally we expect nextchar == '"'
    if nextchar != '"':
        if nextchar in _ws:
            end = _w(s, end).end()
            nextchar = s[end : end + 1]
        # Trivial empty object
        if nextchar == "}":
            if object_pairs_hook is not None:
                result = object_pairs_hook(pairs)
                return ctx.typecheck_object(result, end + 1)
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
            return ctx.typecheck_object(pairs, end + 1)
        elif nextchar != '"':
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes", s, end
            )
    end += 1
    while True:
        key_start = end - 1
        key, end = scanstring(s, end, strict)
        key = memo_get(key, key)
        # To skip some function call overhead we optimize the fast paths where
        # the JSON key separator is ": " or just ":".
        if s[end : end + 1] != ":":
            end = _w(s, end).end()
            if s[end : end + 1] != ":":
                raise JSONDecodeError("Expecting ':' delimiter", s, end)
        end += 1

        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
        except IndexError:
            pass
        try:
            value, end, ctx = scan_once(s, end, ctx=ctx.key_context(key, key_start))
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        pairs_append((key, value))
        try:
            nextchar = s[end]
            if nextchar in _ws:
                end = _w(s, end + 1).end()
                nextchar = s[end]
        except IndexError:
            nextchar = ""
        end += 1

        if nextchar == "}":
            break
        elif nextchar != ",":
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        end = _w(s, end).end()
        nextchar = s[end : end + 1]
        end += 1
        if nextchar != '"':
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes", s, end - 1
            )
    if object_pairs_hook is not None:
        result = object_pairs_hook(pairs)
        return ctx.typecheck_object(result, end)
    pairs = dict(pairs)
    if object_hook is not None:
        pairs = object_hook(pairs)
    return ctx.typecheck_object(pairs, end)


# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
def JSONArrayParser(
    s_and_end: Tuple[str, int],
    scan_once: Callable[[str, int, DecodeContext], Tuple[object, int, DecodeContext]],
    ctx: DecodeContext,
    _w: Callable = WHITESPACE.match,
    _ws: str = WHITESPACE_STR,
) -> Tuple[JSONArray, int, DecodeContext]:
    s, end = s_and_end
    values = []
    nextchar = s[end : end + 1]
    if nextchar in _ws:
        end = _w(s, end + 1).end()
        nextchar = s[end : end + 1]
    # Look-ahead for trivial empty array
    if nextchar == "]":
        return ctx.typecheck_array(values, end + 1)
    _append = values.append
    i = -1
    while True:
        try:
            i += 1
            value, end, ctx = scan_once(s, end, ctx=ctx.index_context(i, end))
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        _append(value)
        nextchar = s[end : end + 1]
        if nextchar in _ws:
            end = _w(s, end + 1).end()
            nextchar = s[end : end + 1]
        end += 1
        if nextchar == "]":
            break
        elif nextchar != ",":
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
        except IndexError:
            pass

    return ctx.typecheck_array(values, end)


###########################
# JSON OBJECT DESERILAZER #
###########################


def scan_json(
    obj: JSONType, pos: FlatKey, ctx: DecodeContext
) -> Tuple[JSONType, DecodeContext]:
    jtype = typeof(obj)

    if jtype is JSONValue:
        value, _, ctx = ctx.typecheck_value(obj, None, pos)
        return value, ctx
    elif jtype is JSONArray:
        ctx = ctx.array_context()
        arr = []
        for index, value in enumerate(obj):
            new_pos = (*pos, index)
            value, ctx = scan_json(value, new_pos, ctx.index_context(index, new_pos))
            arr.append(value)
        array, _, ctx = ctx.typecheck_array(arr, None)
        return array, ctx
    elif jtype is JSONObject:
        ctx = ctx.object_context()
        res = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_pos = (*pos, key)
                value, ctx = scan_json(value, new_pos, ctx.key_context(key, new_pos))
                res[key] = value
            res, _, ctx = ctx.typecheck_object(res, None)
        elif isinstance(obj, _JSONableObject):
            for jattr in obj._ALL_JATTRS:
                if jattr.json_only:
                    continue
                new_pos = (*pos, jattr.py_name)
                value = getattr(obj, jattr.py_name)
                value, ctx = scan_json(value, new_pos, ctx.key_context(key, new_pos))
                res[key] = value
            res, _, ctx = ctx.typecheck_object(res, None)
        else:
            raise JSONError(f"Unhandled JSONObject {obj}")
        return res, ctx
    else:
        raise JSONError(f"Unhandled JSON type {jtype}")


######################
# TYPED JSON DECODER #
######################


def _tp_cache(func):
    """Wrapper caching __class_getitem__ on type hints

    Provides a fallback if arguments are not hashable
    """
    cache = functools.lru_cache()(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return cache(*args, **kwargs)
        except TypeError as err:  # unhashable args
            print(err)  # TODO: remove
            pass
        return func(*args, **kwargs)

    return wrapper


def _exec_body(namespace, type_hint):
    """Internal helper to initialize parametrized TypedJSONDecoder"""
    namespace["type_hint"] = type_hint  # property(lambda self: type_hint)


class ParametrizedTypedJSONDecoderMeta(ABCMeta):
    """
    Metaclass for parametrized TypedJSONDecoder classes -- provides a nice repr()
    """

    def __repr__(self):
        tp = pretty_type(self.type_hint)
        if isinstance(tp, str):
            return f"TypedJSONDecoder[{tp}]"
        return f"TypedJSONDecoder[{tp!r}]"


class TypedJSONDecoder(ABC, SmartDecoder):
    """
    JSONDecoder subclass to typed JSON decoding

    The type to decode must be provided my indexing this class by
    a tye hint, like in the 'typing' module. The type hint must be
    valid in a JSON context.

    Jsonable subclasses are supported, as this is the whole point
    of that class

    Example:

    # type check that all values are int
    json.load(..., cls=JSONableDecoder[Dict[str, int]])
    """

    @property
    @abstractmethod
    def type_hint(self):
        """Type hint that this decoder decodes"""

    @classmethod
    @_tp_cache
    def _class_getitem_cache(cls, tp):
        """Parametrize the TypedJSONDecoder to decode the provided type hint

        Jsonable subclasses are supported
        """
        return types.new_class(
            "ParametrizedTypedJSONDecoder",
            (cls,),
            kwds={"metaclass": ParametrizedTypedJSONDecoderMeta},
            exec_body=functools.partial(_exec_body, type_hint=tp),
        )

    def __class_getitem__(cls, tp):
        """Parametrize the TypedJSONDecoder to decode the provided type hint

        Jsonable subclasses are supported
        """
        tp = normalize_json_type(tp)
        return cls._class_getitem_cache(tp)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace default decoder by our contextualized decoder
        self.parse_object = JSONObjectParser
        self.parse_array = JSONArrayParser
        self.scan_once = make_string_scanner(self)

    @functools.wraps(JSONDecoder.raw_decode)
    def raw_decode(self, s, idx=0):
        try:
            obj, end, _ = self.scan_once(
                s,
                idx,
                ctx=DecodeContext(
                    doc=s, pos=idx, types=DecodeContext.process_tp(self.type_hint)
                ),
            )
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        return obj, end

    @classmethod
    @functools.wraps(json.load)
    def load(cls, *args, **kwargs):
        if cls is TypedJSONDecoder:
            raise TypeError(f"You must parametrize {cls.__name__} before using it")
        return json.load(*args, cls=cls, **kwargs)

    @classmethod
    @functools.wraps(json.loads)
    def loads(cls, *args, **kwargs):
        if cls is TypedJSONDecoder:
            raise TypeError(f"You must parametrize {cls.__name__} before using it")
        return json.loads(*args, cls=cls, **kwargs)


def deserialize(obj: JSONObject, type_hint: TypeHint) -> JSONObject:
    """
    Deserializes a python object representing a JSON to a Type Hint

    This is the equivalent of TypedJSONDecoder for JSON object that were
    already loaded with json.loads()

    Arguments
        obj -- the object to deserialize
        type_hint -- the type to deserialize to, i.e. to type-check against

    Returns
        A JSONObject. It might not be equal to the original object, because
        JSONable serialization are converted to a proper class instance

    Raises
        TypedJSONDecodeError if obj cannot be deserialized to type_hint
    """
    obj, _ = scan_json(
        obj, tuple(), DecodeContext(obj, tuple(), DecodeContext.process_tp(type_hint))
    )
    return obj
