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
from dataclasses import dataclass
from functools import partial
from json import JSONDecodeError
import typing
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Type, Union

# import of stdlib 'json' module internals
# Those are not part of the public API
from json.decoder import WHITESPACE, WHITESPACE_STR, scanstring
from json.scanner import NUMBER_RE

# Local import
from ._misc import JsonError
from ._jsonize import Jsonable
from ._jtypes import JsonArray, JsonObject, _JsonValueTypes


class TypedJSONDecodeError(JSONDecodeError):
    """
    Raised when the decoded type is not the expected one
    """


def sync_filter(func, *iterables):
    """
    Filter multiple iterable at once, selecting values at index i
    such that func(iterables[0][i], iterables[1][i], ...) is True
    """
    return tuple(zip(*tuple(i for i in zip(*iterables) if func(*i)))) or ((),) * len(iterables)


def typecheck(tp, value) -> bool:
    """internal helper - Type check a value against a JSON type hint"""
    if tp in _JsonValueTypes:
        return isinstance(value, tp)
    elif (orig := typing.get_origin(tp)) is None and issubclass(tp, Jsonable):
        return isinstance(value, tp)
    elif orig is Literal:
        return value in typing.get_args(tp)
    elif orig is Union:
        return any(typecheck(arg, value) for arg in typing.get_args(tp))
    raise JsonError("[BUG] Unhandled typecheck")


@dataclass
class DecodeContext:
    string: str
    start: int
    type_hints: List[Type[Type]]  # type of the value to decode
    jsonable: Optional[type] = None  # Last Jsonable subclass seen
    key: Tuple[
        str, ...
    ] = ()  # Key of the last Jsonable subclass currently decoding
    upper_context: "DecodeContext" = None

    cur_key = None

    def __post_init__(self):
        if not isinstance(self.type_hints, tuple):
            if isinstance(self.type_hints, Iterable):
                self.type_hints = tuple(self.type_hints)
            elif typing.get_origin(self.type_hints) is Union:
                self.type_hints = typing.get_args(self.type_hints)
            else:
                self.type_hints = (self.type_hints,)  # 1-tuple
        self.tps = self.type_hints
        self.origs = tuple(typing.get_origin(tp) for tp in self.type_hints)
        self.args = tuple(typing.get_args(tp) for tp in self.type_hints)

    def prefix_msg(self, msg: str):
        """Prefix a message with context infos"""
        return (
            "".join(
                [
                    f"While deserializing a {self.jsonable.__name__} object, "
                    if self.jsonable
                    else "",
                    f"Under key {self.key!s}, " if self.key else "",
                ]
            )
            + msg
        )

    def get_msg(self, value=Ellipsis, match=None):
        return self.prefix_msg(
            f"Expected value matching {match or Union[self.type_hints]}"
            + (f", got {value!s}" if value is not Ellipsis else "")
            + ", at"
        )

    def notify_object(self):
        self.tps, self.origs, self.args = sync_filter(
            lambda tp, orig, args: orig is dict
            or (orig is None and issubclass(tp, Jsonable)),
            self.tps,
            self.origs,
            self.args,
        )
        if not self.tps:
            raise TypedJSONDecodeError(self.get_msg(), self.string, self.start)

    def notify_array(self):
        self.tps, self.origs, self.args = sync_filter(
            lambda tp, orig, args: orig in (tuple, list),
            self.tps,
            self.origs,
            self.args,
        )
        if not self.tps:
            raise TypedJSONDecodeError(self.get_msg(), self.string, self.start)

    def notify_value(self, value):
        if self.cur_key is None:
            raise JsonError(
                "[!! This is a bug !! Please report] Anomalous notification of DecodeContext"
            )
        prev_tps = self.tps
        if isinstance(self.cur_key, int):
            self.tps, self.origs, self.args = sync_filter(
                lambda tp, orig, args: typecheck(
                    args[0 if orig is list else self.cur_key], value
                ),
                self.tps,
                self.origs,
                self.args,
            )
        elif isinstance(self.cur_key, str):
            self.tps, self.origs, self.args = sync_filter(
                lambda tp, orig, args: typecheck(
                    args[1]
                    if orig is dict
                    else next(
                        filter(lambda jattr: jattr.name == self.cur_key, tp._ALL_JATTRS)
                    ).type_hint,
                    value,
                ),
                self.tps,
                self.origs,
                self.args,
            )
        else:
            raise JsonError(
                "[!! This is a bug !! Please report] Anomalous notification of DecodeContext"
            )
        if not self.tps:
            self.key = (*self.key, self.cur_key)
            raise TypedJSONDecodeError(
                self.get_msg(value, prev_tps), self.string, self.start
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
                    self.get_msg(f"an array of length >= {key+1}"),
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
                        orig is None
                        and issubclass(tp, Jsonable)
                        and any(jattr.name == key for jattr in tp._ALL_JATTRS)
                    )
                ),
                self.tps,
                self.origs,
                self.args,
            )
            if not self.tps:
                raise TypedJSONDecodeError(
                    self.prefix_msg(
                        f"Key '{key}' is invalid for type {Union[old_tps]} (previous keys may have narrowed the type), at"
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
                        filter(lambda jattr: jattr.name == key, tp._ALL_JATTRS)
                    ).type_hint
                    for tp, orig, args in zip(self.tps, self.origs, self.args)
                ),
                jsonable=self.jsonable,
                key=(*self.key, key),
                upper_context=self,
            )
        raise JsonError(
            "[!! This is a bug !! Please report] Anomalous indexing of DecodeContext"
        )

    def check_value(self, val, end, start):
        if not any(typecheck(tp, val) for tp in self.tps):
            raise TypedJSONDecodeError(self.get_msg(val), self.string, start)
        if self.upper_context:
            self.upper_context.notify_value(val)
        return val, end

    def check_object(self, obj, end):
        jsonable_cls = [
            tp
            for tp, orig in zip(self.tps, self.origs)
            if orig is None and issubclass(tp, Jsonable)
        ]
        if len(jsonable_cls) > 1:
            raise JsonError(
                "[!! This is a bug !! Please report] Multiple Jsonable subclass available at deserialization"
            )
        if jsonable_cls:
            cls = jsonable_cls[0]
            return (
                cls(
                    **{
                        jattr.py_name: obj.get(jattr.name, jattr.default)
                        for jattr in cls._ALL_JATTRS
                    }
                ),
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
                self.prefix_msg(f"the value {array} miss keys to match {self.tps}, at"),
                self.string,
                self.start,
            )
        return array, end


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
