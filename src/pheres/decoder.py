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
    Union,
    get_origin,
    get_type_hints,
)

# Local import
from .misc import JSONError
from .jsonable import (
    TypeHint,
    JSONType,
    _JSONArrayTypes,
    get_args,
    normalize_json_type,
    typecheck,
    JSONable,
    _make_instance,
)
from . import jsonable
from .utils import FlatKey

# import internals of stdlib 'json' module
# Those are not part of the public API
from json.decoder import WHITESPACE, WHITESPACE_STR, scanstring
from json.scanner import NUMBER_RE

__all__ = ["TypedJSONDecodeError", "TypedJSONDecoder"]

# Type Aliases
TypeTuple = Tuple[TypeHint, ...]
OrigTuple = Tuple[Optional[TypeHint], ...]
ArgsTuple = Tuple[Optional[TypeHint], ...]
TypeFilter = Callable[[TypeHint, Optional[TypeHint], Optional[TypeHint]], bool]

# sentinel value
MISSING = object()


class TypedJSONDecodeError(JSONError, JSONDecodeError):
    """
    Raised when the decoded type is not the expected one
    """

    def __init__(
        self, msg: str, doc: Union[str, object], pos: Union[int, FlatKey]
    ) -> None:
        """
        Special case when the decoded document is an object
        """
        if not isinstance(doc, str):
            keys = " -> ".join(("<base object>", *pos))
            errmsg = "%s: at %s" % (msg, keys)
            ValueError.__init__(self, errmsg)
            self.msg = msg
            self.doc = doc
            self.pos = pos
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

    doc: object  # decoded JSON
    pos: Union[int, FlatKey]
    types: TypeTuple
    origs: OrigTuple = None
    args: ArgsTuple = None
    parent_context: Optional["DecodeContext"] = None
    parent_key: Optional[Union[int, str]] = None

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

    def err_msg(self, /, *, expected_msg: Optional[str]=None, value=MISSING) -> str:
        parts = []
        if expected_msg:
            parts.append(expected_msg)
        else:
            parts.append(f"Expected type {Union[self.types]}")
        if value is not MISSING:
            parts.append(f"got {value}")
        parts.append("at")
        return ", ".join(parts)

    # HELPER METHODS
    def filter_types(
        self,
        /,
        func: TypeFilter,
    ) -> Tuple[TypeTuple, OrigTuple, ArgsTuple]:
        return sync_filter(func, self.types, self.origs, self.args)

    @staticmethod
    def get_array_subtypes(index: int, types: TypeTuple, origs: OrigTuple, args: ArgsTuple) -> TypeTuple:
        subtypes = []
        for type_, orig, arg in zip(types, origs, args):
            if isinstance(orig, type):
                if issubclass(orig, tuple):
                    subtypes.append(arg[index])
                    continue
                elif issubclass(orig, list):
                    subtypes.append(arg[0])
                    continue
            raise JSONError(f"Unhandled Array type {type_}")
        return tuple(subtypes)

    @staticmethod
    def get_object_subtypes(key: str, types: TypeTuple, origs: OrigTuple, args: ArgsTuple) -> TypeTuple:
        subtypes = []
        for type_, orig, arg in zip(types, origs, args):
            if isinstance(orig, type) and issubclass(orig, dict):
                subtypes.append(arg[1])
            elif isinstance(type_, type) and issubclass(type_, JSONable):
                subtypes.append(type_._ALL_JATTRS[key].type_hint)
            else:
                raise JSONError(f"Unhandled Object type {type_}")
        return tuple(subtypes)

    # FILTER HELPER
    @staticmethod
    def accept_array(
        type_: TypeHint, orig: Optional[TypeHint], arg: Optional[TypeHint]
    ) -> bool:
        return isinstance(orig, type) and issubclass(orig, _JSONArrayTypes)

    @staticmethod
    def accept_min_length(
        index: int,
    ) -> TypeFilter:
        def accept(
            type_: TypeHint, orig: Optional[TypeHint], arg: Optional[TypeHint]
        ) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, list):
                    return True
                elif issubclass(orig, tuple) and len(arg) > index:
                    return True
            return False

        return accept

    @staticmethod
    def accept_array_value(
        index: int, value: object
    ) -> TypeFilter:
        def accept(
            type_: TypeHint, orig: Optional[TypeHint], arg: Optional[TypeHint]
        ) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, tuple):
                    return typecheck(value, arg[index])
                elif issubclass(orig, list):
                    return typecheck(value, arg[0])
                raise JSONError(f"Unhandled Array type {type_}")

        return accept
    
    @staticmethod
    def accept_length(length: int) -> TypeFilter:
        def accept(
            type_: TypeHint, orig: Optional[TypeHint], arg: Optional[TypeHint]
        ) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, list):
                    return True
                elif issubclass(orig, tuple):
                    return len(arg) == length
            raise JSONError(f"Unhandled Array type {type_}")
        return accept

    @staticmethod
    def accept_object(
        type_: TypeHint, orig: Optional[TypeHint], arg: Optional[TypeHint]
    ) -> bool:
        return (isinstance(orig, type) and issubclass(orig, dict)) or (
            isinstance(type_, type) and issubclass(type_, JSONable)
        )

    @staticmethod
    def accept_key(
        key: str,
    ) -> TypeFilter:
        def accept(
            type_: TypeHint, orig: Optional[TypeHint], arg: Optional[TypeHint]
        ) -> bool:
            if isinstance(orig, type) and issubclass(orig, dict):
                return True
            elif isinstance(type_, type) and issubclass(type_, JSONable):
                return key in type_._ALL_JATTRS
            return False

        return accept

    @staticmethod
    def accept_object_value(
        key: str, value: object
    ) -> TypeFilter:
        def accept(
            type_: TypeHint, orig: Optional[TypeHint], arg: Optional[TypeHint]
        ) -> bool:
            if isinstance(orig, type) and issubclass(orig, dict):
                return typecheck(value, arg[1])
            elif isinstance(type_, type) and issubclass(type_, JSONable):
                return typecheck(value, type_._ALL_JATTRS[key].type_hint)
            raise JSONError(f"Unhandled Object type {type_}")

        return accept

    # CONTEXT SPECIALIZATION METHODS
    def array_context(self, /) -> "DecodeContext":
        types, origs, args = self.filter_types(self.accept_array)
        if not types:
            raise TypedJSONDecodeError(
                msg=self.err_msg(value="'Array'"), doc=self.doc, pos=self.pos
            )
        return replace(self, types=types, origs=origs, args=args)

    def index_context(self, /, index: int, index_idx: int) -> "DecodeContext":
        types, origs, args = self.filter_types(self.accept_min_length(index))
        if not types:
            raise TypedJSONDecodeError(
                msg=self.err_msg(value=f"'Array' of length >={index+1}"),
                doc=self.doc,
                pos=self.pos,
            )
        parent = replace(self, types=types, origs=origs, args=args)
        return DecodeContext(
            doc=self.doc,
            pos=index_idx,
            types=self.get_array_subtypes(index, types, origs, args),
            parent_context=parent,
            parent_key=index,
        )

    def object_context(self, /) -> "DecodeContext":
        types, origs, args = self.filter_types(self.accept_object)
        if not types:
            raise TypedJSONDecodeError(
                msg=self.err_msg(value="'Object'"), doc=self.doc, pos=self.pos
            )
        return replace(self, types=types, origs=origs, args=args)

    def key_context(self, /, key: str, key_idx: int) -> "DecodeContext":
        types, origs, args = self.filter_types(self.accept_key(key))
        if not types:
            raise TypedJSONDecodeError(
                msg=self.err_msg(
                    expected_msg=f"Inferred type {Union[self.types]} has no key '{key}' (at char {key_idx})"
                ),
                doc=self.doc,
                pos=self.pos,
            )
        parent = replace(self, types=types, origs=origs, args=args)
        return DecodeContext(
            doc=self.doc,
            pos=key_idx,
            types=self.get_object_subtypes(key, types, origs, args),
            parent_context=parent,
            parent_key=key,
        )

    # TYPECHECKING METHODS
    def typecheck_value(
        self, /, value: object, end_idx: int, start_idx: int
    ) -> Tuple[object, int, "DecodeContext"]:
        if not any(typecheck(value, type_) for type_ in self.types):
            raise TypedJSONDecodeError(
                msg=self.err_msg(value=value), doc=self.doc, pos=start_idx
            )
        parent = None
        if self.parent_context is not None:
            parent = self.parent_context
            key = self.parent_key
            if isinstance(key, int):
                types, origs, args = parent.filter_types(
                    self.accept_array_value(key, value)
                )
            elif isinstance(key, str):
                types, args, origs = parent.filter_types(
                    self.accept_object_value(key, value)
                )
            else:
                raise JSONError(f"Unhandled parent key {key}")
            if not types:
                raise TypedJSONDecodeError(
                    msg=parent.err_msg(value=f"{value} of type {type(value)}"),
                    doc=parent.doc,
                    pos=parent.pos
                )
            parent = replace(parent, types=types, origs=origs, args=args)
        return value, end_idx, parent
    
    def typecheck_array(self, /, array: object, end_idx: int) -> Tuple[object, int, "DecodeContext"]:
        types, origs, args = self.filter_types(self.accept_length(len(array)))
        if not types:
            raise TypedJSONDecodeError(
                msg=self.err_msg(expected_msg=f"Inferred type {Union[self.types]}", value=f"{array}, which is too short"),
                doc=self.doc,
                pos=self.pos
            )
        return array, end_idx, self.parent_context
    
    def typecheck_object(self, /, obj: object, end_idx: int) -> Tuple[object, int, "DecodeContext"]:
        cls = [
            type_ for type_ in self.types
            if isinstance(type_, type) and issubclass(type_, JSONable)
        ]
        cls = [
            type_ for i, type_ in enumerate(cls)
            if all(not issubclass(type_, other) for other in cls[i + 1 :])
        ]
        if len(cls) > 1:
            raise TypedJSONDecodeError(
                msg=self.err_msg(expected_msg=f"Multiple JSONable class found"),
                doc=self.doc,
                pos=self.pos
            )
        if cls:
            return _make_instance(cls[0], obj), end_idx, self.parent_context
        return obj, end_idx, self.parent_context



############################
# JSON INTERNALS OVERRIDES #
############################


def py_make_scanner(
    context: JSONDecoder,
) -> Callable[[str, int, DecodeContext], Tuple[object, int, DecodeContext]]:
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
    ) -> Tuple[object, int, DecodeContext]:
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
    ) -> Tuple[object, int, DecodeContext]:
        try:
            return _scan_once(string, idx, ctx)
        finally:
            memo.clear()

    return scan_once


# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
def JSONObject(
    s_and_end: Tuple[str, int],
    strict: bool,
    scan_once: Callable[[str, int, DecodeContext], Tuple[object, int, DecodeContext]],
    ctx: DecodeContext,
    object_hook: Callable,
    object_pairs_hook: Callable,
    memo: Optional[dict] = None,
    _w: Callable = WHITESPACE.match,
    _ws: str = WHITESPACE_STR,
) -> Tuple[object, int, DecodeContext]:
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
def JSONArray(
    s_and_end: Tuple[str, int],
    scan_once: Callable[[str, int, DecodeContext], Tuple[object, int, DecodeContext]],
    ctx: DecodeContext,
    _w: Callable = WHITESPACE.match,
    _ws: str = WHITESPACE_STR,
) -> Tuple[object, int, DecodeContext]:
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


######################
# TYPED JSON DECODER #
######################

def _clean_tp(type_hint):
    # TODO: improve function
    # it should check that there are not ambiguities in the type_hint
    if get_origin(type_hint) is Union:
        return get_args(type_hint)
    return (type_hint,)


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


class TypedJSONDecoder(ABC, JSONDecoder):
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
            exec_body=functools.partial(_exec_body, type_hint=tp),
            kwds={"metaclass": ParametrizedTypedJSONDecoderMeta},
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
        self.parse_object = JSONObject
        self.parse_array = JSONArray
        self.scan_once = py_make_scanner(self)

    @functools.wraps(JSONDecoder.raw_decode)
    def raw_decode(self, s, idx=0):
        try:
            obj, end, _ = self.scan_once(
                s,
                idx,
                ctx=DecodeContext(
                    doc=s,
                    pos=idx,
                    types=_clean_tp(self.type_hint)
                )
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
