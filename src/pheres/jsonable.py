"""
Module for the jsonable interface
"""
from enum import Enum, auto
from typing import Callable, ClassVar, Dict, Iterable, Set, Tuple, Union

from attr import attrib
from attr import dataclass as attrs

from .utils import AnyClass, TypeHint


class JsonableEnum(Enum):
    VALUE: auto()
    ARRAY: auto()
    OBJECT: auto()
    CLASS: auto()


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

    all_attrs: bool = True
    after: Union[str, Iterable[str]] = ()  # pylint: disable=unsubscriptable-object
    type_hint: TypeHint = attrib(init=False)
    kind: JsonableEnum = JsonableEnum.CLASS

    @classmethod
    def _factory(cls, type_hint, virtual=None, cls_arg=None, /, *args, **kwargs):
        decorator = cls(*args, **kwargs)._parametrize(type_hint, virtual)
        if cls_arg is not None:
            return decorator(cls_arg)
        return decorator

    def __new__(cls, cls_arg=None, /, *args, **kwargs):
        decorator = super().__new__(cls)
        if cls_arg is not None:
            # __init__ hasn't been called automatically
            cls.__init__(decorator, *args, **kwargs)
            # __init__ is skipped if the return value of __new__
            # is not an instance of the class
            return decorator(cls_arg)
        return decorator

    def __post_init__(self):
        object.__setattr__(self, "type_hint", None)
        object.__setattr__(self, "virtual_class", JSONableClass)
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
        object.__setattr__(self, "after", after)

    def _parametrize(self, type_hint, virtual=None):
        if self.type_hint is not None:
            raise TypeError("Cannot parametrize @jsonable twice")
        if virtual is None:
            if isinstance(type_hint, tuple):
                if len(type_hint) == 2 and type_hint[1] is Ellipsis:
                    type_hint = List[type_hint[0]]
                else:
                    type_hint = Tuple[type_hint]
                virtual = JSONableArray
            elif isinstance((orig := get_origin(type_hint)), type):
                if issubclass(orig, (list, tuple)):
                    virtual = JSONableArray
                elif issubclass(orig, dict):
                    virtual = JSONableObject
            else:
                virtual = JSONableValue
        object.__setattr__(self, "virtual_class", virtual)
        object.__setattr__(self, "type_hint", type_hint)
        return self

    @classmethod
    def __class_getitem__(cls, /, type_hint):
        return functools.partial(cls._factory, type_hint, None)

    @Subscriptable
    def Value(tp):  # pylint: disable=no-self-argument
        return functools.partial(
            jsonable._factory,
            Union[tp],  # pylint: disable=unsubscriptable-object
            JSONableValue,
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
        return functools.partial(jsonable._factory, tp, JSONableArray)

    @Subscriptable
    def Object(tp):  # pylint: disable=no-self-argument
        return functools.partial(jsonable._factory, Dict[str, tp], JSONableObject)

    def __repr__(self):
        return "%s%s%s(%s)" % (
            "" if self.__module__ == "builtins" else f"{self.__module__}.",
            self.__class__.__qualname__,
            "" if self.type_hint is None else f"[{_type_repr(self.type_hint)}]",
            ", ".join([f"{attr}={getattr(self, attr)!r}" for attr in ("all_attrs",)]),
        )

    def __call__(self, cls: AnyClass) -> AnyClass:
        if not isinstance(cls, type):
            raise TypeError("Can only decorate classes")
        # already decorated
        # Avoid call to issubclass to prevent ABCMeta from caching
        if cls in self.virtual_class.registry or cls in self._delayed:
            return cls
        _register_class_ref(cls)
        with on_success(self._register_delayed_classes), on_error(
            unregister_forward_ref, cls.__name__
        ):
            if self.virtual_class in (JSONableValue, JSONableArray, JSONableObject):
                decorate = functools.partial(
                    _decorate_jsonable_simple, self.virtual_class, cls, self.type_hint
                )
            elif self.virtual_class is JSONableClass:
                decorate = functools.partial(
                    _decorate_jsonable_class, cls, self.all_attrs
                )
            else:
                raise TypeError("Unknown virtual jsonable registry")
            if (
                self.after
                and self.after & register_forward_ref._table.keys() != self.after
            ):
                self._delayed[cls] = (self.after, decorate)
            else:
                decorate()
        return cls
