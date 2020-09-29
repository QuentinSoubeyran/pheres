"""
module for internal data storing classes
"""
import enum
import inspect
from types import ModuleType
from typing import (
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from attr import attrib
from attr import dataclass as attrs

from .exceptions import PheresInternalError
from .utils import AnyClass, TypeHint

PHERES_ATTR = "__pheres_data__"

set_attr = object.__setattr__

__all__ = [
    "JsonableEnum",
    "JsonAttr",
    "ValueData",
    "ArrayData",
    "DictData",
    "ObjectData",
    "DelayedData",
    "DelayedJsonableHandler",
]


class JsonableEnum(enum.Enum):
    VALUE = enum.auto()
    ARRAY = enum.auto()
    DICT = enum.auto()
    OBJECT = enum.auto()


@attrs
class JsonAttr:
    """
    Stores information for a json attribute
    """

    name: str
    py_name: str
    type: TypeHint
    required: bool = attrib(init=False)


@attrs(frozen=True)
class ValueData:
    """
    Stores pheres data for jsonable values
    """

    type: TypeHint
    type_hint: TypeHint = attrib(init=False)

    def __attrs_post_init__(self):
        set_attr(self, "type_hint", self.type)


@attrs(frozen=True)
class ArrayData:
    """
    Stores pheres data for jsonable arrays
    """

    types: Tuple[TypeHint]
    fixed: bool = attrib(init=False)
    type_hint: TypeHint = attrib(init=False)

    def __attrs_post_init__(self):
        if len(self.types == 2) and self.types[1] is Ellipsis:
            set_attr(self, "fixed", False)
            set_attr(self, "type_hint", List[self.types[0]])
            set_attr(self, "types", self.types[:1])
        else:
            set_attr(self, "fixed", True)
            set_attr(self, "type_hint", Tuple[self.types])


@attrs(frozen=True)
class DictData:
    """
    Stores pheres data for jsonable dicts
    """

    type: TypeHint
    type_hint: TypeHint = attrib(init=False)

    def __attrs_post_init__(self):
        set_attr(self, "type_hint", Dict[str, self.type])


@attrs(frozen=True)
class ObjectData:
    """
    Stores pheres data for jsonable objects
    """

    attrs: Dict[str, JsonAttr]
    req_attrs: Dict[str, JsonAttr] = attrib(init=False)


@attrs(frozen=True)
class DelayedData:
    """
    Stores the data for delayed jsonable objects
    """

    func: Callable[[type], type]
    kind: JsonableEnum


class DelayedJsonableHandler:
    dependencies: ClassVar[  # pylint: disable=unsubscriptable-object
        Dict[ModuleType, Dict[str, FrozenSet[str]]]
    ] = {}

    @classmethod
    def _add(cls, /, jsonable: AnyClass, deps: Iterable[str]) -> None:
        module = inspect.getmodule(jsonable)
        deps = frozenset(deps)
        name = jsonable.__name__
        cls.dependencies.setdefault(module, {})[name] = deps

    @classmethod
    def _contains(cls, /, jsonable: AnyClass):
        module = inspect.getmodule(jsonable)
        return jsonable.__name__ in cls.dependencies.get(module, {})

    @classmethod
    def decorate(cls):
        # todo
        raise NotImplementedError("TODO")
