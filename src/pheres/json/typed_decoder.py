"""
Typed JSON decoding
"""
from __future__ import annotations

import functools
import json
import re
import types
import typing
from abc import ABC, ABCMeta, abstractmethod
from itertools import chain
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE, WHITESPACE_STR, scanstring
from json.scanner import NUMBER_RE
from pathlib import Path
from typing import (
    Any,
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

import attr

from pheres.core import typecheck
from pheres.exceptions import PheresInternalError, TypedJSONDecodeError
from pheres.typed_json.aliases import JSONArray, JSONObject, JSONType, JSONValue
from pheres.utils import docstring, sync_filter

__all__ = [
    "TypedJSONDecoder",
    "deserialize",
]

# Type Variable
U = TypeVar("U")

# Type Aliases
Pos = Union[int, FlatKey]  # pylint: disable=unsubscriptable-object

TypeHint = Any
TypeOrig = Optional[TypeHint]  # pylint: disable=unsubscriptable-object
TypeArgs = Tuple[Optional[TypeHint], ...]  # pylint: disable=unsubscriptable-object

TypeTuple = Tuple[TypeHint, ...]
OrigTuple = Tuple[TypeOrig, ...]
ArgsTuple = Tuple[TypeArgs, ...]

TypeCache = Tuple[TypeTuple, OrigTuple, ArgsTuple]
TypeFilter = Callable[[TypeHint, TypeOrig, TypeArgs], bool]


def _unpack_union(tp) -> TypeTuple:
    """
    Unpacks the arguments of a Union type
    """
    if get_origin(tp) is Union:
        return get_args(tp)
    return (tp,)


def _filter_arrays(tp: TypeHint, orig: TypeOrig, arg: TypeArgs) -> bool:
    """
    Filter types that can typecheck arrays
    """
    return (
        isinstance(orig, type) and issubclass(orig, _JSONArrayTypes)
    ) or is_jarray_class(tp)


def _filter_array_index_factory(index: int) -> TypeFilter:
    """
    Factory to filtering types that typecheck an array with a specific index
    """

    def filter_array_index(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
        if isinstance(orig, type):
            if issubclass(orig, list):
                return True
            elif issubclass(orig, tuple):
                return len(args) > index
        elif is_jarray_class(tp):
            data: ArrayData = getattr(tp, PHERES_ATTR)
            return not data.is_fixed or len(data.types) > index
        return False

    return filter_array_index


def _filter_array_value_factory(index: int, value: object) -> TypeFilter:
    """
    Factory for filtering array types that typecheck a value at an index
    """

    def filter_array_value(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
        if isinstance(orig, type):
            if issubclass(orig, tuple):
                return typecheck(value, args[index])
            elif issubclass(orig, list):
                return typecheck(value, args[0])
        elif is_jarray_class(tp):
            data: ArrayData = getattr(tp, PHERES_ATTR)
            return typecheck(value, data.types[index if data.is_fixed else 0])
        raise PheresInternalError(f"Unhandled Array type {tp}")

    return filter_array_value


def _filter_array_length_factory(length: int) -> TypeFilter:
    """
    Factory for filtering array types for a given length
    """

    def filter_array_length(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
        if isinstance(orig, type):
            if issubclass(orig, list):
                return True
            elif issubclass(orig, tuple):
                return len(args) == length
        elif is_jarray_class(tp):
            data: ArrayData = getattr(tp, PHERES_ATTR)
            return not data.is_fixed or len(data.types) == length
        raise PheresInternalError(f"Unhandled Array type {tp}")

    return filter_array_length


def _filter_dict(tp: TypeHint, orig: TypeOrig, arg: TypeArgs) -> bool:
    """
    Filter types that typecheck a dict
    """
    return (
        (isinstance(orig, type) and issubclass(orig, _JSONObjectTypes))
        or is_jdict_class(tp)
        or is_jobject_class(tp)
    )


def _filter_dict_key_factory(key: str) -> TypeFilter:
    """
    Filter dict types that accept a key
    """

    def filter_dict_key(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
        if isinstance(orig, type) and issubclass(orig, dict):
            return True
        elif is_jdict_class(tp):
            return True
        elif is_jobject_class(tp):
            data: ObjectData = getattr(tp, PHERES_ATTR)
            return key in data.attrs
        return False

    return filter_dict_key


def _filter_dict_value_factory(key: str, value: object) -> TypeFilter:
    """
    Filter dict types that typecheck the value of a given key
    """

    def filter_dict_value(tp: TypeHint, orig: TypeOrig, args: TypeArgs) -> bool:
        if isinstance(orig, type) and issubclass(orig, dict):
            return typecheck(value, args[1])
        elif is_jdict_class(tp):
            data_d: DictData = getattr(tp, PHERES_ATTR)
            return typecheck(value, data_d.type)
        elif is_jobject_class(tp):
            data_o: ObjectData = getattr(tp, PHERES_ATTR)
            return typecheck(value, data_o.attrs[key].type)
        raise PheresInternalError(f"Unhandled Object type {tp}")

    return filter_dict_value


def _make_value(cls: type, value):
    """
    Builds a jsonable instance from a value-typed json
    """
    cls = _get_updated_class(cls)
    return cls(value)


def _make_tuple(cls: type, tpl: tuple):
    """
    Builds a jsonable instance from a tuple-typed json
    """
    cls = _get_updated_class(cls)
    return cls(*tpl)


def _make_list(cls: type, lst: list):
    """
    Builds a jsonable instance from a list-typed json
    """
    cls = _get_updated_class(cls)
    return cls(lst)


def _make_dict(cls: type, dct: dict):
    """
    Builds a jsonable instance from a dict-typed json
    """
    cls = _get_updated_class(cls)
    return cls(dct)


def _make_object(cls: type, obj):
    """
    Builds a jsonable instance from a typeddict-typed json
    """
    cls = _get_updated_class(cls)
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
class _DecodeContext:
    """
    Internal class to keep tracks of valid types during JSON decoding

    DecodeContext are immutable

    Attributes:
        doc: JSON source being decoded
        pos: position in the JSON source
        get_eval_args: get_args wrapper that resolves the type arguments
            in the calling namespace
        types: types that are still valids
        origs: cache of the origins of the still valid types
        args: cache of the arguments (as from typing.get_args) of the still valid types
        parent: parent _DecodeContext instance that created this one
        parent_key: key of the parent type that this _DecodeContext is linked to
    """

    doc: Union[str, JSONObject]  # pylint: disable=unsubscriptable-object
    pos: Pos
    get_eval_args: Callable[[TypeHint], Tuple[TypeHint]]
    types: TypeTuple
    origs: OrigTuple = ()
    args: ArgsTuple = ()
    parent: Optional["_DecodeContext"] = None  # pylint: disable=unsubscriptable-object
    parent_key: Optional[  # pylint: disable=unsubscriptable-object
        Union[int, str]  # pylint: disable=unsubscriptable-object
    ] = None

    def __attrs_post_init__(self, /):
        # cache types origins and arguments for performance
        if not self.origs:
            self.__dict__["origs"] = tuple(get_origin(tp) for tp in self.types)
        if not self.args:
            self.__dict__["args"] = tuple(self.get_eval_args(tp) for tp in self.types)
        if self.parent is not None and self.parent_key is None:
            raise ValueError("_DecodeContext with a parent must be given a parent key")

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
    def filter_types(self, /, type_filter: TypeFilter) -> TypeCache:
        return sync_filter(type_filter, self.types, self.origs, self.args)

    def filtered(
        self, /, type_filter: TypeFilter, err_msg: str, *, err_pos=MISSING
    ) -> "_DecodeContext":
        types, origs, args = sync_filter(type_filter, self.types, self.origs, self.args)
        if not types:
            raise TypedJSONDecodeError(
                msg=err_msg,
                doc=self.doc,
                pos=self.pos if err_pos is MISSING else err_pos,
            )
        return attr.evolve(self, types=types, origs=origs, args=args)

    def get_array_subtypes(self, /, index: int) -> TypeTuple:
        """
        Returns the subtypes at index ``index`` from its array types

        Types should already be filtered for array types
        """
        subtypes: List[TypeHint] = []
        for tp, orig, arg in zip(self.types, self.origs, self.args):
            found = MISSING  # type: ignore
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
        """
        Return the subtype at key ``key`` from its dict types

        Types should already be filtered for dict types
        """
        subtypes: List[TypeHint] = []
        for tp, orig, arg in zip(self.types, self.origs, self.args):
            if isinstance(orig, type) and issubclass(orig, dict):
                tp = arg[1]
            elif is_jdict_class(tp):
                data_d: DictData = getattr(tp, PHERES_ATTR)
                tp = data_d.type
            elif is_jobject_class(tp):
                data_o: ObjectData = getattr(tp, PHERES_ATTR)
                tp = data_o.attrs[key].type
            else:
                raise PheresInternalError(f"Unhandled Object type {tp}")
            if get_origin(tp) is Union:
                subtypes.extend(self.get_eval_args(tp))
            else:
                subtypes.append(tp)
        return tuple(subtypes)

    def filter_for_arrays(self, /) -> "_DecodeContext":
        """
        Return a copy of this _DecodeCOntext filtered for array types
        """
        return self.filtered(_filter_arrays, self.err_msg(value="'Array'"))

    def get_index_subcontext(self, /, index: int, pos: Pos) -> "_DecodeContext":
        """
        Returns the subcontext for decoding at index ``index``
        """
        parent = self.filtered(
            _filter_array_index_factory(index),
            self.err_msg(value=f"'Array' of length >={index+1}"),
        )
        return _DecodeContext(
            doc=self.doc,
            pos=pos,
            get_eval_args=self.get_eval_args,
            types=parent.get_array_subtypes(index),
            parent=parent,
            parent_key=index,
        )

    def filter_for_dict(self, /) -> "_DecodeContext":
        """
        Return a copy of this _DecodeContext filtered for dict types
        """
        return self.filtered(_filter_dict, self.err_msg(value="'Object'"))

    def get_key_subcontext(self, /, key: str, key_pos: Pos) -> "_DecodeContext":
        """
        Returns the subcontext for decoding at key ``key``
        """
        parent = self.filtered(
            _filter_dict_key_factory(key),
            self.err_msg(
                msg=f"Inferred type {Union[self.types]} has no key '{key}'",  # pylint: disable=unsubscriptable-object
            ),
            err_pos=key_pos,
        )
        return _DecodeContext(
            doc=self.doc,
            pos=key_pos,
            get_eval_args=self.get_eval_args,
            types=parent.get_object_subtypes(key),
            parent=parent,
            parent_key=key,
        )

    # TYPECHECKING METHODS
    def typecheck_value(
        self, /, value: JSONValue, end_pos: U, start_pos: Pos
    ) -> Tuple[JSONValue, U, "_DecodeContext"]:
        """
        Typechek a decoded value against the types of this context
        """
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
            key = self.parent_key
            if isinstance(key, int):
                type_filter = _filter_array_value_factory(key, value)
            elif isinstance(key, str):
                type_filter = _filter_dict_value_factory(key, value)
            else:
                raise PheresInternalError(f"Unhandled parent key {key}")
            parent = parent.filtered(
                type_filter, parent.err_msg(value=f"{value} of type {type(value)}")
            )
        return value, end_pos, parent

    def typecheck_array(
        self, /, array: JSONArray, end_pos: U
    ) -> Tuple[JSONArray, U, "_DecodeContext"]:
        """
        Typechek a decoded array against the types of this context
        """
        types, *_ = self.filter_types(_filter_array_length_factory(len(array)))
        if not types:
            raise TypedJSONDecodeError(
                msg=self.err_msg(
                    msg=f"Inferred type {Union[self.types]}",  # pylint: disable=unsubscriptable-object
                    value=f"{array}, an 'Array' of len {len(array)}, which is too short",
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
            array = _make_tuple(classes[0], array)
            if parent is not None:
                key = self.parent_key
                if isinstance(key, int):
                    type_filter = _filter_array_value_factory(key, array)
                elif isinstance(key, str):
                    type_filter = _filter_dict_value_factory(key, array)
                else:
                    raise PheresInternalError(f"Unhandled parent key {key}")
                parent = parent.filtered(
                    type_filter, parent.err_msg(value=f"{array} of type {type(array)}")
                )
        return array, end_pos, parent

    def typecheck_object(
        self, /, obj: JSONObject, end_pos: U
    ) -> Tuple[JSONObject, int, "_DecodeContext"]:
        """
        Typecheck a decoded dict against the types of this context
        """
        classes = [
            tp for tp in self.types if is_jdict_class(tp) or is_jobject_class(tp)
        ]
        classes = [
            cls
            for i, cls in enumerate(classes)
            if not issubclass(cls, classes[i + 1 :])
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
                key = self.parent_key
                if isinstance(key, int):
                    type_filter = _filter_array_value_factory(key, obj)
                elif isinstance(key, str):
                    type_filter = _filter_dict_value_factory(key, obj)
                else:
                    raise PheresInternalError(f"Unhandled parent key {key}")
                parent = parent.filtered(
                    type_filter, parent.err_msg(value=f"{obj} of type {type(obj)}")
                )
        return obj, end_pos, parent
