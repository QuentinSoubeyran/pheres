"""
module for internal data storing classes
"""
import enum
import functools
import inspect
from copy import deepcopy
from types import ModuleType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Tuple,
    Union,
    get_origin,
)

from attr import Factory, attrib
from attr import dataclass as attrs

from .exceptions import (
    JsonAttrError,
    JsonAttrTypeError,
    JsonAttrValueError,
    PheresInternalError,
)
from .utils import AnyClass, TypeHint, get_args, get_class_namespaces, post_init

PHERES_ATTR = "__pheres_data__"
MISSING = object()

__all__ = [
    "JsonableEnum",
    "JsonAttr",
    "ValueData",
    "ArrayData",
    "DictData",
    "ObjectData",
    "DelayedData",
]


class JsonableEnum(enum.Enum):
    VALUE = enum.auto()
    ARRAY = enum.auto()
    DICT = enum.auto()
    OBJECT = enum.auto()


@attrs(frozen=True)
class ValueData:
    """
    Stores pheres data for jsonable values
    """

    type: TypeHint
    type_hint: TypeHint = attrib(init=False)

    def __attrs_post_init__(self):
        self.__dict__["type_hint"] = self.type


@attrs(frozen=True)
class ArrayData:
    """
    Stores pheres data for jsonable arrays
    """

    types: Tuple[TypeHint]
    is_fixed: bool = attrib(init=False, default=MISSING)
    type_hint: TypeHint = attrib(init=False, default=MISSING)

    def __attrs_post_init__(self):
        if len(self.types) == 2 and self.types[1] is Ellipsis:
            self.__dict__["is_fixed"] = False
            self.__dict__["type_hint"] = List[self.types[0]]
            self.__dict__["types"] = self.types[:1]
        else:
            self.__dict__["is_fixed"] = True
            self.__dict__["type_hint"] = Tuple[self.types]


@attrs(frozen=True)
class DictData:
    """
    Stores pheres data for jsonable dicts
    """

    type: TypeHint
    type_hint: TypeHint = attrib(init=False)

    def __attrs_post_init__(self):
        self.__dict__["type_hint"] = Dict[str, self.type]


@post_init
@attrs
class JsonAttr:
    """
    Stores information for a json attribute
    """

    module: ModuleType
    cls_name: str
    name: str
    py_name: str
    type: TypeHint
    default: Any = attrib(default=MISSING)
    is_json_only: bool = attrib(default=MISSING)

    def __post_init__(self, *, cls=None):
        from .typing import is_json, typecheck

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
    def required(self):
        return self.default is MISSING or self.is_json_only

    @property
    def cls(self):
        return getattr(self.module, self.cls_name)

    def get_default(self):
        if callable(self.default):
            return self.default()
        return deepcopy(self.default)


@attrs(frozen=True)
class ObjectData:
    """
    Stores pheres data for jsonable objects
    """

    attrs: Dict[str, JsonAttr]
    req_attrs: Dict[str, JsonAttr] = attrib(init=False)

    def __attrs_post_init__(self):
        self.__dict__["req_attrs"] = {
            name: attr for name, attr in self.attrs.items() if not attr.required
        }


@attrs(frozen=True)
class DelayedData:
    """
    Stores the data for delayed jsonable objects
    """

    func: Callable[[type], type]
    kind: JsonableEnum
