"""
Module containing internal classes used for storing internal data

Exposed classes offer a bit of introspection that can be useful
"""
from __future__ import annotations

import enum
import functools
import inspect
import json
from copy import deepcopy
from types import ModuleType
from typing import Any, Callable, ClassVar, Iterable, Literal, Union, get_origin

from attr import Factory, attrib
from attr import dataclass as attrs

from pheres._exceptions import (
    JsonAttrError,
    JsonAttrTypeError,
    JsonAttrValueError,
    PheresInternalError,
)
from pheres._utils import AnyClass, TypeHint, get_args, get_class_namespaces, post_init

PHERES_ATTR = "__pheres_data__"


class _MISSING:
    def __repr__(self):
        return "MISSING"


MISSING = _MISSING()
"""Sentinel object used when `None` cannot be used"""

__all__ = [
    "MISSING",
    "JsonAttr",
    "ValueData",
    "ArrayData",
    "DictData",
    "ObjectData",
    "DelayedData",
    "UsableDecoder",
]


class JsonableEnum(enum.Enum):
    """
    Enumeration of the different kind of jsonable

    The actual values are an implementation detail and should
    not be used. Refer to `Enum` for how to use enumeration in
    python.

    Attributes:
        VALUE: for jsonable values
        ARRAY: for jsonable arrays
        DICT: for jsonable dicts
        OBJECT: for jsonable objects
    """

    VALUE = "value"
    ARRAY = "array"
    DICT = "dict"
    OBJECT = "object"


@attrs(frozen=True)
class ValueData:
    """
    Opaque type storing pheres data for jsonable values

    Exposed for used with `issubclass`
    """

    type: TypeHint
    type_hint: TypeHint = attrib(init=False)

    def __attrs_post_init__(self):
        self.__dict__["type_hint"] = self.type


@attrs(frozen=True)
class ArrayData:
    """
    Opaque type storing pheres data for jsonable arrays

    Exposed for used with `issubclass`
    """

    types: tuple[TypeHint]
    is_fixed: bool = attrib(init=False, default=MISSING)
    type_hint: TypeHint = attrib(init=False, default=MISSING)

    def __attrs_post_init__(self):
        if len(self.types) == 2 and self.types[1] is Ellipsis:
            self.__dict__["is_fixed"] = False
            self.__dict__["type_hint"] = list[self.types[0]]
            self.__dict__["types"] = self.types[:1]
        else:
            self.__dict__["is_fixed"] = True
            self.__dict__["type_hint"] = tuple[self.types]


@attrs(frozen=True)
class DictData:
    """
    Opaque type storing pheres data for jsonable dicts

    Exposed for used with `issubclass`
    """

    type: TypeHint
    type_hint: TypeHint = attrib(init=False)

    def __attrs_post_init__(self):
        self.__dict__["type_hint"] = dict[str, self.type]


@post_init
@attrs
class JsonAttr:
    """
    Stores information for a json attribute

    Attributes:
        name: name of this attribute in JSON
        py_name: name of this attribute in python
        type: type of this attribute
        is_json_only: if this attribute should be only
          present in JSON
    """

    module: ModuleType
    cls_name: str
    name: str
    py_name: str
    type: TypeHint
    default: Any = attrib(default=MISSING)
    is_json_only: bool = attrib(default=MISSING)

    def __post_init__(self, *, cls=None):
        from pheres._typing import is_json, typecheck

        if cls is None:
            raise ValueError("Missing parent class")

        if self.default is not MISSING:
            value = self.default() if callable(self.default) else self.default
            if not is_json(value):
                raise JsonAttrValueError(value)
            if not typecheck(value, self.type):
                raise JsonAttrTypeError(self.type, value)

        if self.is_json_only is MISSING:
            globalns, localns = get_class_namespaces(cls)
            if (
                get_origin(self.type) is Literal
                and len(
                    (args := get_args(self.type, globalns=globalns, localns=localns))
                )
                == 1
            ):
                arg = args[0]
                default = self.default() if callable(self.default) else self.default
                if default is not MISSING and default != arg:
                    raise JsonAttrError(
                        f"Incoherent Literal and default for json-only attribute: {arg} != {default}"
                    )
                self.__dict__["is_json_only"] = True
                if default is MISSING:
                    self.__dict__["default"] = arg
            else:
                self.__dict__["is_json_only"] = False

    @functools.cached_property
    def is_required(self):
        """
        Returns True if this attribute is mandatory in JSON
        """
        return self.default is MISSING or self.is_json_only

    @property
    def cls(self):
        return getattr(self.module, self.cls_name)

    def get_default(self):
        """
        Returns the default value of this attribute

        The returned value is a copy and is safe to use and modify
        """
        if callable(self.default):
            return self.default()
        return deepcopy(self.default)


@attrs(frozen=True)
class ObjectData:
    """
    Opaque type storing pheres data for jsonable objects

    Exposed for used with `issubclass`
    """

    attrs: dict[str, JsonAttr]
    req_attrs: dict[str, JsonAttr] = attrib(init=False)

    def __attrs_post_init__(self):
        self.__dict__["req_attrs"] = {
            name: attr for name, attr in self.attrs.items() if attr.is_required
        }


@attrs(frozen=True)
class DelayedData:
    """
    Opaque type storing pheres data for jsonable that have been delayed

    Exposed for used with `issubclass`

    Attributes:
        kind: What type of jsonable the class will be once decorated
         (This has not happened yet !)
    """

    func: Callable[[type], type]
    kind: JsonableEnum


class UsableDecoder(json.JSONDecoder):
    """
    `json.JSONDecoder` subclass with wrapper methods ``load()`` and
    ``loads()`` using itself for the decoder class
    """

    @classmethod
    def load(cls, *args, **kwargs):
        """
        Thin wrapper around `json.load` that use this class as the default ``cls`` argument
        """
        return json.load(*args, cls=cls, **kwargs)

    @classmethod
    def loads(cls, *args, **kwargs):
        """
        Thin wrapper around `json.loads` that use this class as the default ``cls`` argument
        """
        return json.loads(*args, cls=cls, **kwargs)
