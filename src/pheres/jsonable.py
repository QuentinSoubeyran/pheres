"""
Module for the jsonable interface
"""
import dataclasses
import functools
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Set,
    Tuple,
    Union,
    get_origin,
)

import attr
from attr import dataclass as attrs

from .datatypes import (
    PHERES_ATTR,
    ArrayData,
    DelayedData,
    DelayedJsonableHandler,
    DictData,
    JsonableEnum,
    ObjectData,
    ValueData,
)
from .exceptions import JsonableError
from .typing import is_jsonable_class
from .utils import AnyClass, Subscriptable, TypeHint, type_repr

__all__ = [
    "jsonable",
]

_decorator_table = {
    JsonableEnum.VALUE: None,
    JsonableEnum.ARRAY: None,
    JsonableEnum.DICT: None,
    JsonableEnum.OBJECT: None,
}

_kwargs_table = {
    JsonableEnum.VALUE: ("type_hint",),
    JsonableEnum.ARRAY: ("type_hint",),
    JsonableEnum.DICT: ("type_hint",),
    JsonableEnum.OBJECT: ("auto_attrs",),
}


@attrs(frozen=True)
class jsonable:
    """
    Class decorator that makes a class JSONable.

    Can be used directly, or arguments that controls the JSONization
    process can be specified.
    It can also be indexed with type hint(s) to create a JSONable value
    or array

    Arguments:
        all_attrs [Optional, True] -- use all attributes for JSONized attribute
        after -- register this jsonable only after the listed foward ref are available to pheres

    Usage:
        # Default behavior
        @jsonable
        @jsonable()

        # Specify arguments
        @jsonable(all_attrs=False)

        # JSONable values
        @jsonable[int]              # single value
        @jsonable[int, ...]         # variable length array
        @jsonable[int, int]         # fixed length array
    """

    INIT_ARGS: ClassVar[Iterable[str]] = (  # pylint: disable=unsubscriptable-object
        "all_attrs",
        "after",
    )

    auto_attrs: bool = True
    after: Union[str, Iterable[str]] = ()  # pylint: disable=unsubscriptable-object
    type_hint: TypeHint = attr.ib(init=False, default=None)
    kind: JsonableEnum = attr.ib(init=False, default=JsonableEnum.OBJECT)

    @classmethod
    def _factory(
        cls,
        type_hint: TypeHint,
        kind: JsonableEnum = None,
        cls_arg: type = None,
        /,
        *args,
        **kwargs,
    ):
        decorator = cls(*args, **kwargs)._parametrize(type_hint, kind=kind)
        if cls_arg is not None:
            return decorator(cls_arg)
        return decorator

    def __new__(cls, cls_arg=None, /, *args, **kwargs):
        decorator = super().__new__(cls)
        if cls_arg is not None:
            # __init__ hasn't been called automatically
            cls.__init__(decorator, *args, **kwargs)
            # __init__ is skipped if the return value of __new__
            # is not an instance of the class, so this is safe
            return decorator(cls_arg)
        return decorator

    def __attrs_post_init__(self):
        if isinstance(self.after, str):
            after = frozenset((self.after,))
        elif isinstance(self.after, Iterable):
            for dep in self.after:
                if not isinstance(dep, str):
                    raise TypeError("@jsonable dependencies must be str")
            after = frozenset(self.after)
        else:
            raise TypeError(
                "@jsonable dependencies must be a str of an iterable of str"
            )
        self.__dict__["after"] = after

    def _parametrize(self, type_hint: TypeHint, *, kind: JsonableEnum = None):
        if self.type_hint is not None:
            raise TypeError("Cannot parametrize @jsonable twice")
        if kind is None:
            # Guess the kind from the provided type-hint
            if isinstance(type_hint, tuple):
                if len(type_hint) == 2 and type_hint[1] is Ellipsis:
                    type_hint = List[type_hint[0]]
                else:
                    type_hint = Tuple[type_hint]
                kind = JsonableEnum.ARRAY
            elif isinstance((orig := get_origin(type_hint)), type):
                if issubclass(orig, (list, tuple)):
                    kind = JsonableEnum.ARRAY
                elif issubclass(orig, dict):
                    kind = JsonableEnum.DICT
            else:
                kind = JsonableEnum.VALUE
        self.__dict__["kind"] = kind
        self.__dict__["type_hint"] = type_hint
        return self

    @classmethod
    def __class_getitem__(cls, /, type_hint):
        return functools.partial(cls._factory, type_hint, None)

    @Subscriptable
    def Value(tp):  # pylint: disable=no-self-argument
        tp = Union[tp]  # pylint: disable=unsubscriptable-object
        return functools.partial(
            jsonable._factory,
            tp,
            kind=JsonableEnum.VALUE,
        )

    @Subscriptable
    def Array(
        tp: Union[tuple, TypeHint]
    ):  # pylint: disable=no-self-argument, unsubscriptable-object
        if not isinstance(tp, tuple):
            tp = Tuple[tp]
        elif len(tp) == 2 and tp[1] is Ellipsis:
            tp = List[tp[0]]  # pylint: disable=unsubscriptable-object
        else:
            tp = Tuple[tp]
        return functools.partial(jsonable._factory, tp, kind=JsonableEnum.ARRAY)

    @Subscriptable
    def Dict(tp):  # pylint: disable=no-self-argument
        tp = Dict[str, Union[tp]]  # pylint: disable=unsubscriptable-object
        return functools.partial(jsonable._factory, tp, kind=JsonableEnum.DICT)

    def __repr__(self):
        return "%s%s%s(%s)" % (
            "" if self.__module__ == "builtins" else f"{self.__module__}.",
            self.__class__.__qualname__,
            "" if self.type_hint is None else f"[{type_repr(self.type_hint)}]",
            ", ".join(
                [f"{attr}={getattr(self, attr)!r}" for attr in jsonable.INIT_ARGS]
            ),
        )

    def __call__(self, cls: AnyClass) -> AnyClass:
        if not isinstance(cls, type):
            raise TypeError("Can only decorate classes")
        if attr.has(cls):
            raise JsonableError(
                "@jsonable must be the inner-most decorator when used with @attr.s"
            )
        if dataclasses.is_dataclass(cls):
            raise JsonableError(
                "@jsonable must be the inner-most decorator when used with @dataclass"
            )
        if is_jsonable_class(cls) or DelayedJsonableHandler._contains(cls):
            # dont' decorate or delay twice
            return cls
        decorator = _decorator_table[self.kind]
        kwargs = {getattr(self, kwarg) for kwarg in _kwargs_table[self.kind]}
        if self.after:
            setattr(
                cls,
                PHERES_ATTR,
                DelayedData(functools.partial(decorator, **kwargs), self.kind),
            )
            return cls
        else:
            return decorator(cls, **kwargs)
