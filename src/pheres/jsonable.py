"""
Module for the jsonable interface
"""
import dataclasses
import functools
import inspect
import json
from enum import Enum, auto
from types import ModuleType
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Set,
    Tuple,
    Union,
    get_origin,
    get_type_hints,
)

import attr
from attr import dataclass as attrs

from .datatypes import (
    MISSING,
    PHERES_ATTR,
    ArrayData,
    DelayedData,
    DictData,
    JsonableEnum,
    JsonAttr,
    ObjectData,
    ValueData,
)
from .decoder import TypedJSONDecoder, deserialize
from .exceptions import JsonableError, JsonAttrError
from .typing import (
    _normalize_hint,
    is_jobject_class,
    is_jobject_instance,
    is_jsonable_class,
    is_jsonable_instance,
)
from .utils import (
    AnyClass,
    Subscriptable,
    TypeHint,
    TypeT,
    classproperty,
    get_args,
    get_class_namespaces,
    get_updated_class,
    type_repr,
)

__all__ = [
    "JsonMark",
    "Marked",
    "marked",
    "jsonable",
]


class _DelayedJsonableHandler:
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
    def decorate_delayed(cls):
        # todo
        raise NotImplementedError("TODO")


def _get_decoder(cls):
    return TypedJSONDecoder[cls]


def _from_json(cls: type, /, obj):
    """Deserialize to an instance of the class this method is called on.

    Tries to guess how to deserialize in the following order:
        - if ``obj`` supports ``read``, use load()
        - if ``obj`` is a string or bytes, use loads()
        - else, deserialize it as a python JSON object
    """
    if hasattr(obj, "read"):
        return json.load(obj, cls=cls.Decoder)  # pylint: disable=no-member
    elif isinstance(obj, (str, bytes)):
        return json.loads(obj, cls=cls.Decoder)  # pylint: disable=no-member
    else:
        return deserialize(obj, cls)


###################
# JSONABLE VALUES #
###################


def _value_to_json(self, with_defaults: bool = False):
    """
    Serialize the instance to a JSON python object
    """
    raise NotImplementedError(
        "jsonable value classes must implement the to_json() method"
    )


def _decorate_value(cls: AnyClass, *, type_hint: TypeHint) -> AnyClass:
    globalns, localns = get_class_namespaces(cls)
    type_hint = _normalize_hint(globalns, localns, type_hint)
    data = ValueData(type_hint)
    setattr(cls, PHERES_ATTR, data)
    for name, member in (
        ("Decoder", classproperty(_get_decoder)),
        ("from_json", classmethod(_from_json)),
        ("to_json", _value_to_json),
    ):
        if not name in cls.__dict__:
            setattr(cls, name, member)
    return cls


def _make_value(cls: AnyClass, value):
    cls = get_updated_class(cls)
    return cls(value)


###################
# JSONABLE ARRAYS #
###################


def _array_to_json(self, with_defaults: bool = False):
    """
    Serialize the instance to a JSON python object
    """
    raise NotImplementedError(
        "jsonable array classes must implement the to_json() method"
    )


def _decorate_array(cls: AnyClass, *, type_hint: TypeHint) -> AnyClass:
    globalns, localns = get_class_namespaces(cls)
    types = get_args(type_hint, globalns=globalns, localns=localns)
    data = ArrayData(types)
    setattr(cls, PHERES_ATTR, data)
    for name, member in (
        ("Decoder", classproperty(_get_decoder)),
        ("from_json", classmethod(_from_json)),
        ("to_json", _array_to_json),
    ):
        if not name in cls.__dict__:
            setattr(cls, name, member)
    return cls


def _make_array(cls, array):
    cls = get_updated_class(cls)
    return cls(*array)


#################
# JSONABLE DICT #
#################


def _dict_to_json(self, with_defaults: bool = False):
    """
    Serialize the instance to a JSON python object
    """
    raise NotImplementedError(
        "jsonable dict classes must implement the to_json() method"
    )


def _decorate_dict(cls: AnyClass, *, type_hint: TypeHint) -> AnyClass:
    globalns, localns = get_class_namespaces(cls)
    type_ = get_args(type_hint, globalns=globalns, localns=localns)[1]
    data = DictData(type_)
    setattr(cls, PHERES_ATTR, data)
    for name, member in (
        ("Decoder", classproperty(_get_decoder)),
        ("from_json", classmethod(_from_json)),
        ("to_json", _dict_to_json),
    ):
        if not name in cls.__dict__:
            setattr(cls, name, member)
    return cls


def _make_dict(cls, dct):
    cls = get_updated_class(cls)
    return cls(dct)


####################
# JSONABLE OBJECTS #
####################


@attrs
class JsonMark:
    """
    Annotation for JSONized arguments type that provides more control
    on the JSONized attribute behavior. All arguments are optional.

    Arguments
        key -- Set the name of the key in JSON for that attribute.
            Defaults to: the name of the attribute in python

        json_only -- Make the attribute only present in JSON. The attribute
            must have a default value or be a Literal of a single value. The
            attribute is removed from the class' annotations at runtime
            Defaults to:
             * True for Literals of a single value
             * False for all other types
    """

    key: str = None
    json_only: bool = MISSING


Marked = Annotated[TypeT, JsonMark()]


def marked(tp: TypeHint, /, **kwargs) -> TypeHint:
    """
    Shortcut for Annotated[T, JSONAttr(**kwargs)]

    See JSONAttr for a list of supported keyword arguments. Not compatible
    with type checkers due to being runtime
    """
    return Annotated[tp, JsonMark(**kwargs)]


def _get_jattrs(cls: type, auto_attrs: bool = True) -> Dict[str, JsonAttr]:
    attr_names = set(cls.__annotations__.keys())
    for parent in cls.__mro__[1:]:
        if is_jobject_class(parent):
            attr_names |= {
                jattr.py_name for jattr in getattr(parent, PHERES_ATTR).attrs.values()
            }
        elif _DelayedJsonableHandler._contains(parent):
            raise JsonableError(
                f"Cannot register jsonable object {cls.__name__} before its parent class {parent.__name__}"
            )
    # Gather jsonized attributes
    jattrs = {}
    globalns, localns = get_class_namespaces(cls)
    _get_args = functools.partial(get_args, localns=localns, globalns=globalns)
    module = inspect.getmodule(cls)
    for py_name, tp in get_type_hints(
        cls, localns={cls.__name__: cls}, include_extras=True
    ).items():
        if py_name not in attr_names:
            continue
        name = py_name
        is_json_only = MISSING
        # Retrieve name, type hint and annotation
        if get_origin(tp) is Annotated:
            tp, *args = _get_args(tp)
            if any(isinstance((found := arg), JsonMark) for arg in args):
                name = found.key or name  # pylint: disable=undefined-variable
                is_json_only = found.json_only  # pylint: disable=undefined-variable
            elif not auto_attrs:
                continue
        elif not auto_attrs:
            continue
        # Check for name conflict
        if name in jattrs:
            raise JsonAttrError(cls, py_name, "{attr} name is already used")
        # Get default value
        default = getattr(cls, py_name, MISSING)
        # Handle dataclass Field
        if isinstance(default, dataclasses.Field):
            if default.default is not dataclasses.MISSING:
                default = default.default
            elif default.default_factory is not dataclasses.MISSING:
                default = default.default_factory
            else:
                default = MISSING
        # Handle attr.ib
        if isinstance(default, attr.Attribute):
            if default.default is not attr.NOTHING:
                if isinstance(default.default, attr.Factory):
                    default = default.default.factory
                else:
                    default = default.default
            else:
                default = MISSING
        # Create JsonAttr
        jattrs[name] = JsonAttr(
            module=module,
            cls_name=cls.__name__,
            name=name,
            py_name=py_name,
            type=_normalize_hint(globalns, localns, tp),
            default=default,
            is_json_only=is_json_only,
        )
    return jattrs


def _object_to_json(self, *, with_defaults: bool = False):
    """
    Serialize the instance to a JSON python object
    """
    obj = {}
    data: ObjectData = getattr(type(self), PHERES_ATTR)
    for jattr in data.req_attrs.values():
        default = jattr.get_default()
        if jattr.is_json_only:
            obj[jattr.name] = default
            continue
        value = getattr(self, jattr.py_name)
        if value == default and not with_defaults:
            continue
        if is_jsonable_instance(value):
            value = value.to_json(with_defaults=with_defaults)
        obj[jattr.py_name] = value
    return obj


def _decorate_object(cls: AnyClass, *, auto_attrs: bool) -> AnyClass:
    attrs = _get_jattrs(cls, auto_attrs=auto_attrs)
    data = ObjectData(attrs)
    setattr(cls, PHERES_ATTR, data)
    for name, member in (
        ("Decoder", TypedJSONDecoder[cls]),
        ("from_json", classmethod(_from_json)),
        ("to_json", _object_to_json),
    ):
        if name not in cls.__dict__:
            setattr(cls, name, member)
    return cls


def _make_object(cls, obj):
    cls = get_updated_class(cls)
    data: ObjectData = getattr(cls, PHERES_ATTR)
    return cls(
        **{
            jattr.py_name: (
                obj[jattr.name] if jattr.name in obj else jattr.get_default()
            )
            for jattr in data.attrs.values()
            if not jattr.json_only
        }
    )


######################
# JSONABLE DECORATOR #
######################

_decorator_table = {
    JsonableEnum.VALUE: _decorate_value,
    JsonableEnum.ARRAY: _decorate_array,
    JsonableEnum.DICT: _decorate_dict,
    JsonableEnum.OBJECT: _decorate_object,
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


#################
# SERIALIZATION #
#################


class JSONableEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that supports JSONable classes
    """

    def default(self, obj: object):
        if is_jsonable_instance(obj):
            return obj.to_json()
        return super().default(obj)


@functools.wraps(json.dump)
def dump(*args, cls=JSONableEncoder, **kwargs):
    return json.dump(*args, cls=cls, **kwargs)


@functools.wraps(json.dumps)
def dumps(*args, cls=JSONableEncoder, **kwargs):
    return json.dumps(*args, cls=cls, **kwargs)
