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
from __future__ import annotations

import re
import functools
import json
import types
import typing
from abc import ABC, ABCMeta, abstractmethod
from itertools import chain
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE, WHITESPACE_STR, scanstring
from json.scanner import NUMBER_RE
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
    get_args,
    get_origin,
    get_type_hints,
    overload,
)
from pathlib import Path

import attr

from pheres.datatypes import (
    MISSING,
    PHERES_ATTR,
    ArrayData,
    DictData,
    ObjectData,
    UsableDecoder,
    ValueData,
)
import pheres.jsontypes as jt
from pheres.exceptions import PheresInternalError, TypedJSONDecodeError
from pheres.misc import FlatKey
from pheres.typing import (
    JSONArray,
    JSONObject,
    JSONType,
    JSONValue,
    _JSONArrayTypes,
    _JSONObjectTypes,
    _normalize_factory,
    is_jarray_class,
    is_jdict_class,
    is_jobject_class,
    is_jobject_instance,
    is_jsonable_class,
    is_jvalue_class,
    typecheck,
    typeof,
)
from pheres.utils import (
    TypeHint,
    docstring,
    get_class_namespaces,
    get_eval_args,
    get_outer_namespaces,
    get_updated_class,
    sync_filter,
)

__all__ = ["TypedJSONDecoder", "deserialize"]

# Type Variable
U = TypeVar("U")

# Type Aliases
Pos = Union[int, FlatKey]  # pylint: disable=unsubscriptable-object

TypeOrig = Optional[TypeHint]  # pylint: disable=unsubscriptable-object
TypeArgs = Optional[TypeHint]  # pylint: disable=unsubscriptable-object

TypeTuple = Tuple[TypeHint, ...]
OrigTuple = Tuple[TypeOrig, ...]
ArgsTuple = Tuple[TypeArgs, ...]

TypeCache = Tuple[TypeTuple, OrigTuple, ArgsTuple]
TypeFilter = Callable[[TypeHint, TypeOrig, TypeArgs], bool]


##################
# MODULE HELPERS #
##################

def _make_value(cls: type, value):
    cls = get_updated_class(cls)
    return cls(value)


def _make_array(cls: type, array):
    cls = get_updated_class(cls)
    return cls(*array)


def _make_dict(cls: type, dct):
    cls = get_updated_class(cls)
    return cls(dct)


def _make_object(cls: type, obj):
    cls = get_updated_class(cls)
    data: ObjectData = getattr(cls, PHERES_ATTR)
    return cls(
        **{
            jattr.py_name: (
                obj[jattr.name] if jattr.name in obj else jattr.get_default()
            )
            for jattr in data.attrs.values()
            if not jattr.is_json_only
        }
    )

@attr.dataclass(frozen=True)
class DecodeContext:
    """
    Internal class to keep tracks of types during typed decoding

    DecodeContext are immutable
    """

    doc: Union[str, JSONObject]  # pylint: disable=unsubscriptable-object
    pos: Pos
    globalns: Dict
    localns: Dict
    types: TypeTuple
    get_eval_args: Callable[[TypeHint], Tuple[TypeHint]] = None
    origs: OrigTuple = None
    args: ArgsTuple = None
    parent: Optional["DecodeContext"] = None  # pylint: disable=unsubscriptable-object
    pkey: Optional[  # pylint: disable=unsubscriptable-object
        Union[int, str]  # pylint: disable=unsubscriptable-object
    ] = None

    @staticmethod
    def process_tp(tp, *, globalns=None, localns=None):
        if get_origin(tp) is Union:
            return get_args(tp)
        return (tp,)

    def __attrs_post_init__(self, /):
        # cache types origins and arguments for performance
        if self.get_eval_args is None:
            self.__dict__["get_eval_args"] = functools.partial(
                get_eval_args, globalns=self.globalns, localns=self.localns
            )
        if self.origs is None:
            self.__dict__["origs"] = tuple(get_origin(tp) for tp in self.types)
        if self.args is None:
            self.__dict__["args"] = tuple(self.get_eval_args(tp) for tp in self.types)
        if self.parent is not None and self.pkey is None:
            raise ValueError("DecodeContext with a parent must be given a parent key")

    def err_msg(self, /, *, msg: str = None, value=MISSING) -> str:
        parts = []
        if msg:
            parts.append(msg)
        else:
            parts.append(
                f"Expected type {Union[self.types]}"  # pylint: disable=unsubscriptable-object
            )
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
        return attr.evolve(self, types=types, origs=origs, args=args)

    def get_array_subtypes(self, /, index: int) -> TypeTuple:
        subtypes = []
        for tp, orig, arg in zip(self.types, self.origs, self.args):
            found = MISSING
            if isinstance(orig, type):
                if issubclass(orig, tuple):
                    found = arg[index]
                elif issubclass(orig, list):
                    found = arg[0]
            elif is_jarray_class(tp):
                data: ArrayData = getattr(tp, PHERES_ATTR)
                found = data.types[index if data.is_fixed else 0]
            if found is MISSING:
                raise PheresInternalError(f"Unhandled Array type {tp}")
            elif get_origin(found) is Union:
                subtypes.extend(self.get_eval_args(found))
            else:
                subtypes.append(found)
        return tuple(subtypes)

    def get_object_subtypes(self, /, key: str) -> TypeTuple:
        subtypes = []
        for tp, orig, arg in zip(self.types, self.origs, self.args):
            if isinstance(orig, type) and issubclass(orig, dict):
                tp = arg[1]
            elif is_jdict_class(tp):
                data: DictData = getattr(tp, PHERES_ATTR)
                tp = data.type
            elif is_jobject_class(tp):
                data: ObjectData = getattr(tp, PHERES_ATTR)
                tp = data.attrs[key].type
            else:
                raise PheresInternalError(f"Unhandled Object type {tp}")
            if get_origin(tp) is Union:
                subtypes.extend(self.get_eval_args(tp))
            else:
                subtypes.append(tp)
        return tuple(subtypes)

    # FILTERS AND FILTER FACTORIES
    @staticmethod
    def accept_array(tp: TypeHint, orig: TypeOrig, arg: TypeArgs) -> bool:
        return (
            isinstance(orig, type) and issubclass(orig, _JSONArrayTypes)
        ) or is_jarray_class(tp)

    @staticmethod
    def accept_min_length(index: int) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, list):
                    return True
                elif issubclass(orig, tuple):
                    return len(args) > index
            elif is_jarray_class(tp):
                data: ArrayData = getattr(tp, PHERES_ATTR)
                return not data.is_fixed or len(data.types) > index
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
            elif is_jarray_class(tp):
                data: ArrayData = getattr(tp, PHERES_ATTR)
                return typecheck(value, data.types[index if data.is_fixed else 0])
            raise PheresInternalError(f"Unhandled Array type {tp}")

        return accept

    @staticmethod
    def accept_length(length: int) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type):
                if issubclass(orig, list):
                    return True
                elif issubclass(orig, tuple):
                    return len(args) == length
            elif is_jarray_class(tp):
                data: ArrayData = getattr(tp, PHERES_ATTR)
                return not data.is_fixed or len(data.types) == length
            raise PheresInternalError(f"Unhandled Array type {tp}")

        return accept

    @staticmethod
    def accept_object(tp: TypeHint, orig: TypeOrig, arg: TypeArgs) -> bool:
        return (
            (isinstance(orig, type) and issubclass(orig, _JSONObjectTypes))
            or is_jdict_class(tp)
            or is_jobject_class(tp)
        )

    @staticmethod
    def accept_key(key: str) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type) and issubclass(orig, dict):
                return True
            elif is_jdict_class(tp):
                return True
            elif is_jobject_class(tp):
                data: ObjectData = getattr(tp, PHERES_ATTR)
                return key in data.attrs
            return False

        return accept

    @staticmethod
    def accept_object_value(key: str, value: object) -> TypeFilter:
        def accept(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
            if isinstance(orig, type) and issubclass(orig, dict):
                return typecheck(value, args[1])
            elif is_jdict_class(tp):
                data: DictData = getattr(tp, PHERES_ATTR)
                return typecheck(value, data.type)
            elif is_jobject_class(tp):
                data: ObjectData = getattr(tp, PHERES_ATTR)
                return typecheck(value, data.attrs[key].type)
            raise PheresInternalError(f"Unhandled Object type {tp}")

        return accept

    # CONTEXT SPECIALIZATION METHODS
    def array_context(self, /) -> "DecodeContext":
        return self.filtered(self.accept_array, self.err_msg(value="'Array'"))

    def index_context(self, /, index: int, pos: Pos) -> "DecodeContext":
        parent = self.filtered(
            self.accept_min_length(index),
            self.err_msg(value=f"'Array' of length >={index+1}"),
        )
        return DecodeContext(
            doc=self.doc,
            pos=pos,
            globalns=self.globalns,
            localns=self.localns,
            types=parent.get_array_subtypes(index),
            get_eval_args=self.get_eval_args,
            parent=parent,
            pkey=index,
        )

    def object_context(self, /) -> "DecodeContext":
        return self.filtered(self.accept_object, self.err_msg(value="'Object'"))

    def key_context(self, /, key: str, key_pos: Pos) -> "DecodeContext":
        parent = self.filtered(
            self.accept_key(key),
            self.err_msg(
                msg=f"Inferred type {Union[self.types]} has no key '{key}'",  # pylint: disable=unsubscriptable-object
            ),
            err_pos=key_pos,
        )
        return DecodeContext(
            doc=self.doc,
            pos=key_pos,
            globalns=self.globalns,
            localns=self.localns,
            types=parent.get_object_subtypes(key),
            get_eval_args=self.get_eval_args,
            parent=parent,
            pkey=key,
        )

    # TYPECHECKING METHODS
    def typecheck_value(
        self, /, value: JSONValue, end_pos: U, start_pos: Pos
    ) -> Tuple[JSONValue, U, "DecodeContext"]:
        types, classes = [], []
        for tp in self.types:
            if is_jvalue_class(tp):
                data: ValueData = getattr(tp, PHERES_ATTR)
                if typecheck(value, data.type):
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
            value = _make_value(classes[0], value)
        parent = None
        if self.parent is not None:
            parent = self.parent
            key = self.pkey
            if isinstance(key, int):
                filter_func = self.accept_array_value(key, value)
            elif isinstance(key, str):
                filter_func = self.accept_object_value(key, value)
            else:
                raise PheresInternalError(f"Unhandled parent key {key}")
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
                    msg=f"Inferred type {Union[self.types]}",  # pylint: disable=unsubscriptable-object
                    value=f"{array}, an 'Array' of len {len(array)} which is too short",
                ),
                doc=self.doc,
                pos=self.pos,
            )
        classes = [tp for tp in types if is_jarray_class(tp)]
        parent = self.parent
        if classes:
            if len(classes) > 1:
                raise TypedJSONDecodeError(
                    msg=self.err_msg(
                        msg=f"Multiple JSONable class found for array {array}"
                    ),
                    doc=self.doc,
                    pos=self.pos,
                )
            array = _make_array(classes[0], array)
            if parent is not None:
                key = self.pkey
                if isinstance(key, int):
                    filter_func = self.accept_array_value(key, array)
                elif isinstance(key, str):
                    filter_func = self.accept_object_value(key, array)
                else:
                    raise PheresInternalError(f"Unhandled parent key {key}")
                parent = parent.filtered(
                    filter_func, parent.err_msg(value=f"{array} of type {type(array)}")
                )
        return array, end_pos, parent

    def typecheck_object(
        self, /, obj: JSONObject, end_pos: U
    ) -> Tuple[JSONObject, int, "DecodeContext"]:
        classes = [
            tp for tp in self.types if is_jdict_class(tp) or is_jobject_class(tp)
        ]
        classes = [
            cls
            for i, cls in enumerate(classes)
            if all(not issubclass(cls, other) for other in classes[i + 1 :])
        ]
        parent = self.parent
        if classes:
            if len(classes) > 1:
                raise TypedJSONDecodeError(
                    msg=self.err_msg(
                        msg=f"Multiple JSONable class found for object {obj}"
                    ),
                    doc=self.doc,
                    pos=self.pos,
                )
            cls = classes[0]
            if is_jdict_class(cls):
                obj = _make_dict(cls, obj)
            elif is_jobject_class(cls):
                obj = _make_object(classes[0], obj)
            if parent is not None:
                key = self.pkey
                if isinstance(key, int):
                    filter_func = self.accept_array_value(key, obj)
                elif isinstance(key, str):
                    filter_func = self.accept_object_value(key, obj)
                else:
                    raise PheresInternalError(f"Unhandled parent key {key}")
                parent = parent.filtered(
                    filter_func, parent.err_msg(value=f"{obj} of type {type(obj)}")
                )
        return obj, end_pos, parent


############################
# JSON INTERNALS OVERRIDES #
############################

# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/scanner.py
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
            s, end = parse_string(string, idx + 1, strict)
            return ctx.typecheck_value(s, end, idx)
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

WHITESPACE = re.compile("([ \t\r]*\n)*([ \t\n\r]*)")

# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
def JSONObjectParser(
    s: str,
    end: int,
    lineno: int,
    colno: int,
    strict: bool,
    scan_once: Callable[[str, int, DecodeContext], Tuple[object, int, DecodeContext]],
    ctx: DecodeContext,
    object_hook: Callable,
    object_pairs_hook: Callable,
    memo: Optional[dict] = None,  # pylint: disable=unsubscriptable-object
    _w: Callable = WHITESPACE.match,
    _ws: str = WHITESPACE_STR,
    src: Optional[Path] = None # pylint: disable=unsubscriptable-object
) -> Tuple[JSONObject, int, DecodeContext]:
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
            m = _w(s, end)
            count = s.count('\n', end, m.end())
            lineno += count
            colno = (0 if count else colno) + len(m[2])
            end = m.end()
            nextchar = s[end : end + 1]
        # Trivial empty object
        if nextchar == "}":
            if object_pairs_hook is not None:
                result = object_pairs_hook(pairs)
                if src:
                    result = jt.Object(result, sources=jt.FileSource(src, lineno, colno))
                return ctx.typecheck_object(result, end + 1)
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
                if src:
                    pairs = jt.Object(result, sources=jt.FileSource(src, lineno, colno))
            return ctx.typecheck_object(pairs, end + 1)
        elif nextchar != '"':
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes", s, end
            )
    end += 1
    while True:
        key_start = end - 1
        key, _end = scanstring(s, end, strict)
        count = s.count("\n", end, _end)
        lineno += count
        colno = (0 if count else colno) + _end - s.rfind("\n", end, _end)
        key = memo_get(key, key)
        # To skip some function call overhead we optimize the fast paths where
        # the JSON key separator is ": " or just ":".
        if s[end : end + 1] != ":":
            m = _w(s, end)
            count = s.count('\n', end, m.end())
            lineno += count
            colno = (0 if count else colno) + len(m[2])
            end = m.end()
            if s[end : end + 1] != ":":
                raise JSONDecodeError("Expecting ':' delimiter", s, end)
        ## TODO: continue tracking lineno & colno from here
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
        # TODO: Handle jdict ?
        elif is_jobject_instance(obj):
            data: ObjectData = getattr(obj, PHERES_ATTR)
            for jattr in data.attrs:
                if jattr.is_json_only:
                    continue
                new_pos = (*pos, jattr.py_name)
                value = getattr(obj, jattr.py_name)
                value, ctx = scan_json(value, new_pos, ctx.key_context(key, new_pos))
                res[key] = value
            res, _, ctx = ctx.typecheck_object(res, None)
        else:
            raise PheresInternalError(f"Unhandled JSONObject {obj}")
        return res, ctx
    else:
        raise PheresInternalError(f"Unhandled JSON type {jtype}")


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


def _exec_body(namespace, type_hint, globalns, localns):
    """Internal helper to initialize parametrized TypedJSONDecoder"""
    namespace["type_hint"] = type_hint
    namespace["globalns"] = globalns
    namespace["localns"] = locals


class ParametrizedTypedJSONDecoderMeta(ABCMeta):
    """
    Metaclass for parametrized TypedJSONDecoder classes -- provides a nice repr()

    :meta private:
    """

    def __repr__(self):
        tp = self.type_hint
        if isinstance(tp, str):
            return f"TypedJSONDecoder[{tp}]"
        return f"TypedJSONDecoder[{tp!r}]"


class TypedJSONDecoder(ABC, UsableDecoder):
    """
    `json.JSONDecoder` subclass for typed JSON decoding

    The type to decode must be provided by indexing this class with the
    type as key (like in the 'typing' module). The type hint must be
    valid in a JSON context.

    jsonables are supported, as this is the whole point
    of that class

    Example:
        ::

            # type check that all values are str or int
            json.load(..., cls=JSONableDecoder[Dict[str, int]])
            # The class exposed a `load` method
            JSONableDecoder[Dict[str, int]].load(...)
    """

    @property
    @abstractmethod
    def type_hint(self):
        """Type hint that this decoder decodes

        :meta private:"""

    @classmethod
    @_tp_cache
    def _class_getitem_cache(cls, globalns, localns, tp):
        """Parametrize the TypedJSONDecoder to decode the provided type hint

        Jsonable subclasses are supported
        """
        return types.new_class(
            "ParametrizedTypedJSONDecoder",
            (cls,),
            kwds={"metaclass": ParametrizedTypedJSONDecoderMeta},
            exec_body=functools.partial(
                _exec_body, type_hint=tp, localns=localns, globalns=globalns
            ),
        )

    def __class_getitem__(cls, tp):
        """Parametrize the TypedJSONDecoder to decode the provided type hint

        Jsonable subclasses are supported

        Args:
            tp: type hint to parametrize a TypedJSONDecoder for

        Returns:
            A special subclass of TypedJSONDecoder for use with
            `json.load`
        """
        if is_jsonable_class(tp):
            globalns, localns = get_class_namespaces(get_updated_class(tp))
            tp = _normalize_factory(globalns, localns)(tp)
            globalns, localns = None, None
        else:
            globalns, localns = get_outer_namespaces()
            tp = _normalize_factory(globalns, localns)(tp)
        return cls._class_getitem_cache(globalns, localns, tp)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace default decoder by our contextualized decoder
        self.parse_object = JSONObjectParser
        self.parse_array = JSONArrayParser
        self.scan_once = make_string_scanner(self)

    @docstring(post="\n:meta private:")
    @functools.wraps(JSONDecoder.raw_decode)
    def raw_decode(self, s, idx=0):
        globalns, localns = self.globalns, self.localns  # pylint: disable=no-member
        if is_jsonable_class(self.type_hint):
            globalns, localns = get_class_namespaces(get_updated_class(self.type_hint))
        try:
            obj, end, _ = self.scan_once(
                s,
                idx,
                ctx=DecodeContext(
                    doc=s,
                    pos=idx,
                    globalns=globalns,
                    localns=localns,
                    types=DecodeContext.process_tp(
                        self.type_hint, globalns=globalns, localns=localns
                    ),
                ),
            )
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        return obj, end

    @classmethod
    @docstring(
        pre="""
            Thin wrapper around `json.load` that use this class as the ``cls`` argument.
            The `TypedJSONDecoder` must be parametrized.
            
            Wrapped function docstring:\n    """
    )
    @functools.wraps(json.load)
    def load(cls, *args, **kwargs):
        if cls is TypedJSONDecoder:
            raise TypeError(f"You must parametrize {cls.__name__} before using it")
        return json.load(*args, cls=cls, **kwargs)

    @classmethod
    @docstring(
        pre="""
        Thin wrapper around `json.loads` that use this class as the ``cls`` argument.
        The TypedJSONDecoder must be parametrized.
        
        Wrapped function docstring:\n    """
    )
    @functools.wraps(json.loads)
    def loads(cls, *args, **kwargs):
        if cls is TypedJSONDecoder:
            raise TypeError(f"You must parametrize {cls.__name__} before using it")
        return json.loads(*args, cls=cls, **kwargs)


def deserialize(obj: JSONObject, type_hint: TypeHint) -> JSONObject:
    """
    Deserializes a python object representing a JSON  to a given type

    This is the equivalent of `TypedJSONDecoder` for JSON object that were
    already loaded with `json.loads`.

    Args:
        obj: the object to deserialize
        type_hint: the type to deserialize to, i.e. to type-check against

    Returns:
        A `JSONObject`. It might not be equal to the original object, because
        serialized Jsonable are converted to a proper class instance

    Raises:
        `TypedJSONDecodeError`: ``obj`` cannot be deserialized to type_hint
    """
    if is_jsonable_class(type_hint):
        globalns, localns = get_class_namespaces(type_hint)
    else:
        globalns, localns = get_outer_namespaces()
    obj, _ = scan_json(
        obj,
        tuple(),
        DecodeContext(
            doc=obj,
            pos=tuple(),
            globalns=globalns,
            localns=localns,
            types=DecodeContext.process_tp(
                type_hint, globalns=globalns, localns=localns
            ),
        ),
    )
    return obj
