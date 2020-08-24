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
from dataclasses import dataclass
import functools
from itertools import chain
import json
from json import JSONDecodeError, JSONDecoder
import types
import typing
from typing import (
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
from . import jsonize
from .jsonize import (
    JSONType,
    get_args,
    normalize_json_tp,
    typecheck,
    JSONable,
    _make_instance,
)

# import internals of stdlib 'json' module
# Those are not part of the public API
from json.decoder import WHITESPACE, WHITESPACE_STR, scanstring
from json.scanner import NUMBER_RE

__all__ = ["TypedJSONDecodeError", "TypedJSONDecoder"]


class TypedJSONDecodeError(JSONError, JSONDecodeError):
    """
    Raised when the decoded type is not the expected one
    """


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
    tps = get_type_hints(jsonize)
    table = {
        # Standard Type Hints
        jsonize.JSONValue: "JSONValue",
        jsonize.JSONArray: "JSONArray",
        jsonize.JSONObject: "JSONObject",
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


@dataclass
class DecodeContext:
    string: str
    start: int
    type_hints: List[Type[Type]]  # type of the value to decode
    jsonable: Optional[type] = None  # Last Jsonable subclass seen
    key: Tuple[str, ...] = ()  # Key of the last Jsonable subclass currently decoding
    upper_context: "DecodeContext" = None

    cur_key = None

    def __post_init__(self):
        if not isinstance(self.type_hints, tuple):
            if isinstance(self.type_hints, Iterable):
                self.type_hints = tuple(self.type_hints)
            else:
                self.type_hints = (self.type_hints,)  # 1-tuple
        self.type_hints = tuple(
            chain.from_iterable(
                get_args(tp) if get_origin(tp) is Union else (tp,)
                for tp in self.type_hints
            )
        )
        self.tps = self.type_hints
        self.origs = tuple(get_origin(tp) for tp in self.type_hints)
        self.args = tuple(get_args(tp) for tp in self.type_hints)

    def prefix_msg(self, msg: str):
        """Prefix a message with context infos"""
        return (
            "".join(
                [
                    f"In '{self.jsonable.__name__}' object, " if self.jsonable else "",
                    f"at {self.key!s}, " if self.key else "",
                ]
            )
            + msg
        )

    def get_msg(self, value=Ellipsis, match=None):
        return self.prefix_msg(
            "Expected "
            + (match if match else f"type {pretty_type(Union[self.type_hints])}")
            + (f", got {value!s}" if value is not Ellipsis else "")
            + ", at"
        )

    def notify_object(self):
        self.tps, self.origs, self.args = sync_filter(
            lambda tp, orig, args: orig is dict
            or (isinstance(tp, type) and issubclass(tp, JSONable)),
            self.tps,
            self.origs,
            self.args,
        )
        if not self.tps:
            raise TypedJSONDecodeError(
                self.get_msg("'Object'"), self.string, self.start
            )

    def notify_array(self):
        self.tps, self.origs, self.args = sync_filter(
            lambda tp, orig, args: orig in (tuple, list),
            self.tps,
            self.origs,
            self.args,
        )
        if not self.tps:
            raise TypedJSONDecodeError(self.get_msg("'Array'"), self.string, self.start)

    def notify_value(self, value):
        if self.cur_key is None:
            raise JSONError(
                "[!! This is a bug !! Please report] Anomalous notification of DecodeContext"
            )
        old_tps = self.tps
        if isinstance(self.cur_key, int):
            self.tps, self.origs, self.args = sync_filter(
                lambda tp, orig, args: typecheck(
                    value, args[0 if orig is list else self.cur_key]
                ),
                self.tps,
                self.origs,
                self.args,
            )
        elif isinstance(self.cur_key, str):
            self.tps, self.origs, self.args = sync_filter(
                lambda tp, orig, args: typecheck(
                    value,
                    args[1]
                    if orig is dict
                    else next(
                        filter(
                            lambda jattr: jattr.name == self.cur_key,
                            tp._ALL_JATTRS.values(),
                        )
                    ).type_hint,
                ),
                self.tps,
                self.origs,
                self.args,
            )
        else:
            raise JSONError(
                "[!! This is a bug !! Please report] Anomalous notification of DecodeContext"
            )
        if not self.tps:
            self.key = (*self.key, self.cur_key)
            raise TypedJSONDecodeError(
                self.prefix_msg(
                    f"{type(value).__name__} does not match inferred type {pretty_type(Union[old_tps])}, at"
                ),
                self.string,
                self.start,
            )

    def __getitem__(self, key_and_start):
        key, start = key_and_start
        self.cur_key = key

        if isinstance(key, int):
            self.tps, self.origs, self.args = sync_filter(
                lambda tp, orig, args: orig is list
                or (orig is tuple and len(args) > key),
                self.tps,
                self.origs,
                self.args,
            )
            if not self.tps:
                raise TypedJSONDecodeError(
                    self.get_msg(f"'Array' of length >= {key+1}"),
                    self.string,
                    self.start,
                )
            return DecodeContext(
                string=self.string,
                start=start,
                type_hints=tuple(
                    args[0 if orig is list else key]
                    for orig, args in zip(self.origs, self.args)
                ),
                jsonable=self.jsonable,
                key=(*self.key, key),
                upper_context=self,
            )

        elif isinstance(key, str):
            old_tps = self.tps
            self.tps, self.origs, self.args = sync_filter(
                lambda tp, orig, args: (
                    orig is dict
                    or (
                        isinstance(tp, type)
                        and issubclass(tp, JSONable)
                        and any(jattr.name == key for jattr in tp._ALL_JATTRS.values())
                    )
                ),
                self.tps,
                self.origs,
                self.args,
            )
            if not self.tps:
                raise TypedJSONDecodeError(
                    self.prefix_msg(
                        f"inferred type {pretty_type(Union[old_tps])} has no key '{key}', at"
                    ),
                    self.string,
                    self.start,
                )
            return DecodeContext(
                string=self.string,
                start=start,
                type_hints=tuple(
                    args[1]
                    if orig is dict
                    else next(
                        filter(lambda jattr: jattr.name == key, tp._ALL_JATTRS.values())
                    ).type_hint
                    for tp, orig, args in zip(self.tps, self.origs, self.args)
                ),
                jsonable=self.jsonable,
                key=(*self.key, key),
                upper_context=self,
            )
        raise JSONError(
            "[!! This is a bug !! Please report] Anomalous indexing of DecodeContext"
        )

    def check_value(self, val, end, start):
        if not any(typecheck(val, tp) for tp in self.tps):
            raise TypedJSONDecodeError(self.get_msg(val), self.string, start)
        if self.upper_context:
            self.upper_context.notify_value(val)
        return val, end

    def check_object(self, obj, end):
        cls_list = list(
            filter(
                lambda tp: isinstance(tp, type) and issubclass(tp, JSONable), self.tps
            )
        )
        cls_list = [
            cls
            for i, cls in enumerate(cls_list)
            if all(not issubclass(cls, other) for other in cls_list[i + 1 :])
        ]
        if len(cls_list) > 1:
            raise JSONError(
                "[!! This is a bug !! Please report] Multiple Jsonable subclass available at deserialization"
            )
        if cls_list:
            return (
                _make_instance(cls_list[0], obj),
                end,
            )
        return obj, end

    def check_array(self, array, end):
        length = len(array)
        self.origs, self.args = sync_filter(
            lambda orig, args: orig is list or len(args) == length,
            self.origs,
            self.args,
        )
        if not self.origs:
            raise TypedJSONDecodeError(
                self.prefix_msg(
                    f"{array} is incomplete for inferred type {pretty_type(self.tps)}, at"
                ),
                self.string,
                self.start,
            )
        return array, end


############################
# JSON INTERNALS OVERRIDES #
############################

# Original source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/scanner.py
def py_make_scanner(context):
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

    def _scan_once(string, idx, ctx: DecodeContext):
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        if nextchar == '"':
            return ctx.check_value(*parse_string(string, idx + 1, strict), idx)
        elif nextchar == "{":
            ctx.notify_object()
            return parse_object(
                (string, idx + 1),
                strict,
                _scan_once,
                ctx,
                object_hook,
                object_pairs_hook,
                memo,
            )
        elif nextchar == "[":
            ctx.notify_array()
            return parse_array((string, idx + 1), _scan_once, ctx)
        elif nextchar == "n" and string[idx : idx + 4] == "null":
            return ctx.check_value(None, idx + 4, idx)
        elif nextchar == "t" and string[idx : idx + 4] == "true":
            return ctx.check_value(True, idx + 4, idx)
        elif nextchar == "f" and string[idx : idx + 5] == "false":
            return ctx.check_value(False, idx + 5, idx)

        m = match_number(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or "") + (exp or ""))
            else:
                res = parse_int(integer)
            return ctx.check_value(res, m.end(), idx)
        elif nextchar == "N" and string[idx : idx + 3] == "NaN":
            return ctx.check_value(parse_constant("NaN"), idx + 3, idx)
        elif nextchar == "I" and string[idx : idx + 8] == "Infinity":
            return ctx.check_value(parse_constant("Infinity"), idx + 8, idx)
        elif nextchar == "-" and string[idx : idx + 9] == "-Infinity":
            return ctx.check_value(parse_constant("-Infinity"), idx + 9, idx)
        else:
            raise StopIteration(idx)

    def scan_once(string, idx, ctx):
        try:
            return _scan_once(string, idx, ctx)
        finally:
            memo.clear()

    return scan_once


# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
def JSONObject(
    s_and_end,
    strict,
    scan_once,
    ctx,
    object_hook,
    object_pairs_hook,
    memo=None,
    _w=WHITESPACE.match,
    _ws=WHITESPACE_STR,
):
    # TODO: use DecodeContext
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
                return ctx.check_object(result, end + 1)
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
            return ctx.check_object(pairs, end + 1)
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
            value, end = scan_once(s, end, ctx=ctx[key, key_start])
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
        return ctx.check_object(result, end)
    pairs = dict(pairs)
    if object_hook is not None:
        pairs = object_hook(pairs)
    return ctx.check_object(pairs, end)


# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
def JSONArray(s_and_end, scan_once, ctx, _w=WHITESPACE.match, _ws=WHITESPACE_STR):
    # TODO: use decode context
    s, end = s_and_end
    values = []
    nextchar = s[end : end + 1]
    if nextchar in _ws:
        end = _w(s, end + 1).end()
        nextchar = s[end : end + 1]
    # Look-ahead for trivial empty array
    if nextchar == "]":
        return ctx.check_array(values, end + 1)
    _append = values.append
    i = -1
    while True:
        try:
            i += 1
            value, end = scan_once(s, end, ctx=ctx[i, end])
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

    return ctx.check_array(values, end)


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
        tp = normalize_json_tp(tp)
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
            obj, end = self.scan_once(
                s,
                idx,
                ctx=DecodeContext(
                    string=s,
                    start=idx,
                    type_hints=self.type_hint,
                    jsonable=self.type_hint
                    if isinstance(self.type_hint, type)
                    and issubclass(self.type_hint, JSONable)
                    else None,
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
