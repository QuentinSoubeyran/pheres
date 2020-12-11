"""
Module for the jsonable interface
"""
from __future__ import annotations

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
    Type,
    Union,
    get_origin,
    get_type_hints,
)
from warnings import warn

import attr
from attr import dataclass as attrs

from pheres._datatypes import (
    MISSING,
    PHERES_ATTR,
    ArrayData,
    DelayedData,
    DictData,
    JsonableEnum,
    JsonAttr,
    ObjectData,
    UsableDecoder,
    ValueData,
)
from pheres._decoder import TypedJSONDecoder, deserialize
from pheres._exceptions import JsonableError, JsonAttrError, PheresInternalError
from pheres._typing import (
    JSONType,
    _normalize_factory,
    is_jobject_class,
    is_jobject_instance,
    is_jsonable_class,
    is_jsonable_instance,
)
from pheres._utils import (
    AnyClass,
    Subscriptable,
    TypeHint,
    TypeT,
    Virtual,
    classproperty,
    get_args,
    get_class_namespaces,
    get_updated_class,
    type_repr,
)

__all__ = [
    "JsonableDummy",
    "JsonMark",
    "Marked",
    "marked",
    "jsonable",
    "dump",
    "dumps",
]


class JsonableDummy:
    """
    Dummy class providing dummy members for all members
    added by `@jsonable <pheres._jsonable.jsonable>`. Allows type checkers and linters to detect
    said attributes
    """

    Decoder: ClassVar[Type[UsableDecoder]] = UsableDecoder

    @classmethod
    def from_json(cls: AnyClass, /, obj: Any) -> AnyClass:
        """
        Converts a JSON file, string or object to an instance of that class
        """
        raise NotImplementedError

    def to_json(self: AnyClass) -> JSONType:
        """Converts an instance of that class to a JSON object"""
        raise NotImplementedError


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


def _to_json_factory(internal: str):
    def to_json(self, with_defaults: bool = False) -> JSONType:
        """
        Serializes this instance to JSON
        """
        return getattr(self, internal)

    return to_json


###################
# JSONABLE VALUES #
###################


def _decorate_value(cls: AnyClass, *, type_hint: TypeHint, internal: str) -> AnyClass:
    type_hint = _normalize_factory(*get_class_namespaces(cls))(type_hint)
    data = ValueData(type_hint)
    setattr(cls, PHERES_ATTR, data)
    for name, member in (
        ("Decoder", classproperty(_get_decoder)),
        ("from_json", classmethod(_from_json)),
    ):
        if not name in cls.__dict__:
            setattr(cls, name, member)
    if internal:
        if "to_json" in cls.__dict__:
            raise JsonableError("Cannot override already-defined to_json() method")
        setattr(cls, "to_json", _to_json_factory(internal))
    elif not hasattr(cls, "to_json"):
        raise JsonableError(
            "jsonable value must implement to_json() or provide the 'internal' arg"
        )
    return cls


###################
# JSONABLE ARRAYS #
###################


def _decorate_array(cls: AnyClass, *, type_hint: TypeHint, internal: str) -> AnyClass:
    globalns, localns = get_class_namespaces(cls)
    types = get_args(type_hint, globalns=globalns, localns=localns)
    data = ArrayData(types)
    setattr(cls, PHERES_ATTR, data)
    for name, member in (
        ("Decoder", classproperty(_get_decoder)),
        ("from_json", classmethod(_from_json)),
    ):
        if not name in cls.__dict__:
            setattr(cls, name, member)
    if internal:
        if "to_json" in cls.__dict__:
            raise JsonableError("Cannot override already-defined to_json() method")
        setattr(cls, "to_json", _to_json_factory(internal))
    elif not hasattr(cls, "to_json"):
        raise JsonableError(
            "jsonable array must implement to_json() or provide the 'internal' arg"
        )
    return cls


#################
# JSONABLE DICT #
#################


def _decorate_dict(cls: AnyClass, *, type_hint: TypeHint, internal: str) -> AnyClass:
    globalns, localns = get_class_namespaces(cls)
    type_ = get_args(type_hint, globalns=globalns, localns=localns)[1]
    data = DictData(type_)
    setattr(cls, PHERES_ATTR, data)
    for name, member in (
        ("Decoder", classproperty(_get_decoder)),
        ("from_json", classmethod(_from_json)),
    ):
        if not name in cls.__dict__:
            setattr(cls, name, member)
    if internal:
        if "to_json" in cls.__dict__:
            raise JsonableError("Cannot override already-defined to_json() method")
        setattr(cls, "to_json", _to_json_factory(internal))
    elif not hasattr(cls, "to_json"):
        raise JsonableError(
            "jsonable value must implement to_json() or provide the 'internal' arg"
        )
    return cls


####################
# JSONABLE OBJECTS #
####################


@attrs
class JsonMark:
    """
    Annotation for jsonized arguments that provides more control
    on the jsonized attribute behavior. All arguments are optional.

    Arguments:
        key: Set the name of the key in JSON for that attribute.
            Defaults to the name of the attribute in python
        json_only: Make the attribute only present in JSON. The attribute
            must have a default value or be a `typing.Literal` of a single value. The
            attribute is removed from the class' annotations at runtime
            Defaults to `True` for Literals of a single value; `False` for all other types
    """

    key: str = None
    json_only: bool = MISSING


Marked = Annotated[TypeT, JsonMark()]
"""Simple type alias to quickly mark an attribute as jsonized

``Marked[T]`` is equivalent to ``Annotated[T, JsonMark()]``
"""


def marked(tp: TypeHint, /, **kwargs) -> TypeHint:
    """
    Shortcut for ``Annotated[T, JsonMark(**kwargs)]``

    See `JsonMark` for a list of supported keyword arguments.
    `marked` may not be compatible with type checkers due to being
    a runtime definition

    Args:
        tp: Type hint to mark
        **kwargs: additional info to pass to `JsonMark`

    See also:
        `JsonMark`
    """
    return Annotated[tp, JsonMark(**kwargs)]


def _get_jattrs(cls: type, only_marked: bool = False) -> Dict[str, JsonAttr]:
    attr_names = set(cls.__annotations__.keys())
    for parent in cls.__mro__[1:]:
        if is_jobject_class(parent):
            attr_names |= {
                jattr.py_name for jattr in getattr(parent, PHERES_ATTR).attrs.values()
            }
        elif _jsonable.is_delayed(parent):
            raise JsonableError(
                f"Cannot decorate jsonable object {cls.__name__} before its parent class {parent.__name__}"
            )
    # Gather jsonized attributes
    jattrs = {}
    globalns, localns = get_class_namespaces(cls)
    _get_args = functools.partial(get_args, localns=localns, globalns=globalns)
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
            elif only_marked:
                continue
        elif only_marked:
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
            module=inspect.getmodule(cls),
            cls_name=cls.__name__,
            name=name,
            py_name=py_name,
            type=_normalize_factory(globalns, localns)(tp),
            default=default,
            is_json_only=is_json_only,
            cls=cls,
        )
    return jattrs


def _cleanup_object_class(cls):
    data: ObjectData = getattr(cls, PHERES_ATTR)
    annotations = cls.__dict__.get("__annotations__", {})
    for jattr in data.attrs.values():
        if jattr.is_json_only:
            if hasattr(cls, jattr.py_name):
                delattr(cls, jattr.py_name)
            if jattr.py_name in annotations:
                del annotations[jattr.py_name]


def _object_to_json(self, *, with_defaults: bool = False):
    """
    Serialize the instance to a JSON python object
    """
    obj = {}
    data: ObjectData = getattr(type(self), PHERES_ATTR)
    for jattr in data.attrs.values():
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


def _decorate_object(cls: AnyClass, *, only_marked: bool) -> AnyClass:
    attrs = _get_jattrs(cls, only_marked=only_marked)
    data = ObjectData(attrs)
    setattr(cls, PHERES_ATTR, data)
    _cleanup_object_class(cls)
    for name, member in (
        ("Decoder", classproperty(_get_decoder)),
        ("from_json", classmethod(_from_json)),
        ("to_json", _object_to_json),
    ):
        if name not in cls.__dict__:
            setattr(cls, name, member)
    return cls


######################
# JSONABLE DECORATOR #
######################

_decorator_table = {
    JsonableEnum.VALUE: _decorate_value,
    JsonableEnum.ARRAY: _decorate_array,
    JsonableEnum.DICT: _decorate_dict,
    JsonableEnum.OBJECT: _decorate_object,
}


@attrs(frozen=True)
class _jsonable:
    """
    Internal implementation of the jsonable decorator
    """

    _DELAYED: ClassVar[  # pylint: disable=unsubscriptable-object
        Dict[ModuleType, Dict[str, FrozenSet[str]]]
    ] = {}

    _KWARGS_IN: ClassVar[  # pylint: disable=unsubscriptable-object
        dict[JsonableEnum, set[str]]
    ] = {
        JsonableEnum.VALUE: {"after", "internal"},
        JsonableEnum.ARRAY: {"after", "internal"},
        JsonableEnum.DICT: {"after", "internal"},
        JsonableEnum.OBJECT: {"after", "only_marked"},
    }

    _KWARGS_OUT: ClassVar[  # pylint: disable=unsubscriptable-object
        dict[JsonableEnum, set[str]]
    ] = {
        JsonableEnum.VALUE: {"type_hint", "internal"},
        JsonableEnum.ARRAY: {"type_hint", "internal"},
        JsonableEnum.DICT: {"type_hint", "internal"},
        JsonableEnum.OBJECT: {"only_marked"},
    }

    # Internal attributes
    type_hint: TypeHint
    kind: JsonableEnum
    # Decorator arguments
    after: Union[str, Iterable[str]] = ()  # pylint: disable=unsubscriptable-object
    internal: str = None
    only_marked: bool = False

    @staticmethod
    def _clean_tp_kind(
        type_hint: TypeHint, kind: JsonableEnum
    ) -> Tuple[TypeHint, JsonableEnum]:
        """
        Cleanup the type hint is any and infer the jsonable kind if not provided
        """
        if type_hint is None:
            if not (kind is None or kind is JsonableEnum.OBJECT):
                raise PheresInternalError(
                    "%s kind is incompatible with type-hint %s" % (kind, type_hint)
                )
            return type_hint, JsonableEnum.OBJECT
        if kind is None:
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
        return type_hint, kind

    @classmethod
    def _clean_args(cls, kind: JsonableEnum, **kwargs) -> Dict[str, Any]:
        """
        Checks and clean the arguments of the decorator
        """
        used_args = cls._KWARGS_IN[kind]
        if kwargs.keys() - used_args:
            raise JsonableError(
                f"Unused args {', '.join(kwargs.keys() - used_args)} for jsonable {kind._value_}"
            )
        if "after" in kwargs:
            after = kwargs[after]
            if isinstance(after, str):
                after = frozenset((after,))
            elif isinstance(after, Iterable):
                for dep in after:
                    if not isinstance(dep, str):
                        raise TypeError("@jsonable dependencies must be str")
                after = frozenset(after)
            else:
                raise TypeError(
                    "@jsonable dependencies must be a str of an iterable of str"
                )
            kwargs["after"] = after
        return kwargs

    @classmethod
    def _make(
        cls,
        type_hint: TypeHint = None,
        kind: JsonableEnum = None,
        cls_arg: type = None,
        /,
        **kwargs,
    ):
        """
        Internal method to make a parametrized version of this decorator

        Arguments:
            cls: class the method was called on
            type_hint: type hint to parametrize the decorator with
            kind: type of jsonable object to produce
            cls_arg: class to decorate
            *args: positional arguments passed to the decorator by the user
            **kwargs: keyword arguments passed to the decorator by the user
        """
        type_hint, kind = cls._clean_tp_kind(type_hint, kind)
        kwargs = cls._clean_args(kind, **kwargs)
        decorator = cls(type_hint=type_hint, kind=kind, **kwargs)
        if cls_arg is not None:
            return decorator(cls_arg)
        return decorator

    def __repr__(self):
        return "%s%s%s(%s)" % (
            "" if self.__module__ == "builtins" else f"{self.__module__}.",
            self.__class__.__qualname__,
            "" if self.type_hint is None else f"[{type_repr(self.type_hint)}]",
            ", ".join(
                [
                    f"{attr}={getattr(self, attr)!r}"
                    for attr in self._KWARGS_IN[self.kind]
                ]
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
        if is_jsonable_class(cls) or self.is_delayed(cls):
            # dont' decorate or delay twice
            return cls
        decorator = _decorator_table[self.kind]
        kwargs = {kwarg: getattr(self, kwarg) for kwarg in self._KWARGS_OUT[self.kind]}
        if self.after:
            setattr(
                cls,
                PHERES_ATTR,
                DelayedData(functools.partial(decorator, **kwargs), self.kind),
            )
            self.delay(cls, self.after)
        else:
            cls = decorator(cls, **kwargs)
        self.decorate_delayed()
        return cls

    @classmethod
    def delay(cls, /, jsonable: AnyClass, deps: Iterable[str]) -> None:
        module = inspect.getmodule(jsonable)
        deps = frozenset(deps)
        name = jsonable.__name__
        cls._DELAYED.setdefault(module, {})[name] = deps

    @classmethod
    def is_delayed(cls, /, jsonable: AnyClass):
        module = inspect.getmodule(jsonable)
        return jsonable.__name__ in cls._DELAYED.get(module, {})

    @classmethod
    def decorate_delayed(cls):
        decorated = {}
        for module, cls_deps_map in cls._DELAYED.items():
            for cls_name, dependencies in cls_deps_map.items():
                if all(hasattr(module, deps) for deps in dependencies):
                    kls = getattr(module, cls_name)
                    data: DelayedData = getattr(kls, PHERES_ATTR)
                    setattr(module, cls_name, data.func(kls))
                    decorated.setdefault(module, []).append(cls_name)
        for module, cls_list in decorated.items():
            for cls_name in cls_list:
                del cls._DELAYED[module][cls_name]
                if not cls._DELAYED[module]:
                    del cls._DELAYED[module]


class jsonable:
    r"""
    Class decorator to make a class jsonable

    .. _jsonable-class:

    This decorator adds members to the decorated class to serialize or
    deserialize instance from JSON format. Class decorated this way are
    called ``jsonable`` classess. The members added by this decorator
    are described in `JsonableDummy`. Some may not be added by certain
    usage of this decorator, see below.

    Jsonable classes are **typed**. This means the values found in JSON
    must have the correct types when deserializing. Types are specified
    in python code by using PEP 526 type annotations. No typecheck occurs
    on serialization, as python's type annotations are not binding.
    `pheres` trusts that the code follows the annotations it exposes, but
    does not enforce it. Ill-implemented code may well crash when
    deserializing instances that did not abide to their annotations.

    ``@jsonable`` can be parametrized by type-hints, much like types from
    the `typing` module (e.g. ``@jsonable[int]``). This will produce
    various types of jsonable classes. Refer to the *Usage* section below
    for details.

    There are four types of :ref:`jsonable class <jsonable-class>`\ es:
    `jsonable value <jsonable-value>`, `jsonable array <jsonable-array>`,
    `jsonable dict <jsonable-dict>` and `jsonable object <jsonable-object>`.
    These jsonable class have different JSON representation and are
    documented below.

    This class decorator is fully compatible with the `dataclasses`
    module and the `attr.s` decorator, though it must be used as the
    inner-most decorator (that is, it must be executed first on the class).
    It will raise an error if this is not the case.

    Arguments:
        after (Union[str, Iterable[str]]): Modify the decorated class
            only after the listed member become available in the **class'
            module**. This allows for circular definitions. When the class
            is modified, the class object is retrieved again from the
            module it was defined in. This makes ``after`` compatibles
            with decorators that replace the class object with another one,
            such as `attr.s` in certain circumstances.

            Checks for dependencies are performed after each use of the
            ``@jsonable`` decorator. If your class is the last decorated
            one in the file, call `jsonable.decorate_delayed` to modify
            it before usage.
        internal (str): For jsonable value, array of dict. Name of the
            attribute where the JSON value is stored, allowing Pheres
            to make add the ``to_json`` method to the decorated class
        only_marked (bool): For jsonable object only. If `True`, all
            annotated attributes are used. If `False`, only *marked*
            attributes are used. See *jsonable objects* below. Attributes
            lacking a PEP 526 annotation are always ignored.

    Raises:
        TypeError: an argument has a wrong type
        JsonableError: the decorator is not the innermost with respect to
            `dataclasses.dataclass` or `attr.s`

    .. _jsonable-value:

    Jsonable values:
        Jsonable values are represented by a single JSON value (null, a
        number, a string). To make a jsonable value by decorating a class
        with `@jsonable <pheres._jsonable.jsonable>`, the class must:

        * have an ``__init__`` method accepting the call ``Class(value)``,
          where value is the JSON value for an instance of the class
          (this is used for deserialization)
        * implement the ``to_json(self)`` method, that returns the value
          to represents the instance with in JSON (e.g. `None`, an `int`
          or a `str`). This is because `pheres` cannot know what the class
          does with the value, like it does with jsonable objects.

        See the *Usage* section below for how to make jsonable values

    .. _jsonable-array:

    Jsonable arrays:
        Jsonable arrays are represented by a JSON array. There are two kind
        of arrays, fixed-length arrays and arbitrary length arrays. To make
        a jsonable array by decorating a class with
        `@jsonable <pheres._jsonable.jsonable>`, the class must:

        * have an ``__init__`` method that accept the call ``Class(*array)``,
          where ``array`` is the python list for the JSON array. This means:

          * For fixed-length array, a signature ``__init__(self, v1, v2, ...)``
            with a fixed number of arguments is accepted

          * For arbitrary length array, the number of arguments is not known
            in advance, so the signature must be similar to ``__init__(self,
            *array)`` (this signature is also valid for fixed-length arrays
            as it does accept the call above).

        * implement the ``to_json(self)`` method, that returns the python
          `list` that represents the instance in JSON. This is because
          `pheres` cannot know what the class does with the values like
          it does with jsonable objects.

        See the *Usage* section below for how to make jsonable arrays

    .. _jsonable-dict:

    Jsonable dicts:
        Jsonable dicts are represented by a JSON Object with arbitrary
        key-value pairs. For JSON object with known key-value pairs,
        see *jsonable objects* below. To make a jsonable dict by
        decorating a class with `@jsonable <pheres._jsonable.jsonable>`,
        the class must:

        * have an ``__init__`` method that accept the call ``Class(**object)``,
          where ``object`` is the python dict representing the instance
          in JSON. The only signature that supports that in python is of
          the sort ``__init__(self, **object)`` (the actual name of the
          ``object`` parameter can vary). Optional arguments may be added,
          but all key-value pairs found in the deserialized JSON will be
          passed as keyword arguments. This is used for deserialization.

        * implement the ``to_json(self)`` method, that returns a python
          `dict` that represents the instance in JSON. This is because
          `pheres` cannot know what the class does with the key-value
          pairs like it does with jsonable objects.

        See the *Usage* section below for how to make jsonable dicts

    .. _jsonable-object:

    Jsonable objects:
        Jsonable objects are represented by a JSON Object with known
        key-value pairs. The key-value paires are obtained from the
        class by inspecting PEP 526 annotations. Attributes must be
        annotated to be considered by `pheres`, then:

        * If ``only_marked`` is `False` (the default), all annotated
          attributes are used
        * If ``only_marked`` is `False`, attributes must be marked for
          use (see `Marked` and `marked`)

        Irrespective of the value of ``only_marked``, `Marked` and `marked`
        can always be used for refined control (see their documentation).

        If an attribute has a value in the class body, it is taken to be the
        default value of this attribute. The attribute becomes optional in
        the JSON representation. Defaults are always deep-copied when
        instantiating the class from a JSON representation, so `list` and
        `dict` are safe to use.

        To make a jsonable object by decorating a class with ``@jsonable``,
        the class must:

        * have an ``__init__`` method that accept the call
          ``Class(attr1=val1, attr2=val2, ...)`` where attributes values
          are passed as keyword arguments. This is automatically the case
          if you used `dataclasses.dataclass` or `attr.s` and is
          the intended usage for this functuonality.

        In particular, jsonable object *do not* need to implement a
        ``to_json(self)`` method, as `pheres` knows exactly what attributes
        to use. This is in contrast with other jsonable classes.

        See the *Usage* section below for how to make jsonable objects

    Usages:
        There are several usage of this decorator.

        Type parametrization:
            To produce jsonable *values*, *arrays* or *dict*, the decorator
            must be **parametrized** by a type-hint. Any valid type-hint in
            JSON context can be used. The following syntax may be used::

                @jsonable[T]
                @jsonable.Value[T]
                @jsonable.Array[T]
                @jsonable.Dict[T]

            The first forms encompasses the following three: If the
            passed type ``T`` is a JSON value, it will produce a jsonable
            value after decorating the class. If ``T`` is `typing.List`
            or `typing.Tuple` (or `list` or `tuple` since python 3.9), it
            will produce a jsonable array. Finally, if ``T`` is
            `typing.Dict` (or `dict`), it will produce a jsonable dict.
            If you wish to produce a jsonable object, do not parametrize
            the decorator.

            The type must be fully parametrized, to provide the type of
            the JSON representation down to the last value. `typing.Union` can
            be used. This means that::

                @jsonable[List] # not valid
                @jsonable[List[int]] # valid
                @jsonable[List[Union[int, str]]] # valid

            For jsonable arrays, `typing.List` will produce an arbitrary-length
            array, while `typing.Tuple` will produce a fixed-length array, unless
            an ellipsis is used (e.g. ``jsonable[Tuple[int, Ellipsis]]``).
            The literal ellipsis `...` may be used, but this is avoided
            in this documentation to prevent confusion with the meaning
            that more arguments can be provided.

            ``@jsonable.Value[T, ...]`` is equivalent to
            ``jsonable[Union[T, ...]]`` (The ``...`` here is not python
            ellipsis but indicates that more than one types may be passed).

            ``@jsonable.Array[T, ...]`` is equivalent to
            ``jsonable[Tuple[T, ...]]`` (where the ``...`` indicates
            that more than one argument may be passed). This produces
            a fixed-length array -- use `Ellispis <...>` in second position
            to procude an arbitrary-length array.

            ``@jsonable.Dict[T, ...]`` is equivalent to
            ``@jsonable[dict[str, Union[T, ...]]]`` and produces a
            jsonable dict. To produce a jsonable object, do not parametrize
            the decorator.

        Arguments:
            If specified, arguments must be provided *after* the type
            parametrization. ``after`` can be used for all jsonable
            classes, but ``only_marked`` only has an effect on jsonable
            objects.

    Notes:
        ``@jsonable`` only add members that are not explicitely defined by the
        decorated class (inherited implementation are ignored). This means you
        can overwrite the default behavior if necessary.

        When parametrized by a type or passed arguments, this decorator returns
        an object that may be re-used. This means that the following code is
        valid::

            from pheres import jsonable, Marked

            my_decorator = jsonable(only_marked=True)

            @my_decorator
            class MyClass:
                python: int
                json: Marked[int]
    """

    # This is not really a class, prevent subclassing and instanciation
    def __init__(self, *args, **kwargs):
        raise TypeError("@jsonable is not a class")

    def __init_subclass__(self, *args, **kwargs):
        raise TypeError("@jsonable is not a class")

    # Shortcut syntaxes
    @Subscriptable
    def Value(tp):  # pylint: disable=no-self-argument
        tp = Union[tp]  # pylint: disable=unsubscriptable-object
        return functools.partial(
            _jsonable._make,
            tp,
            JsonableEnum.VALUE,
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
        return functools.partial(_jsonable._make, tp, JsonableEnum.ARRAY)

    @Subscriptable
    def Dict(tp):  # pylint: disable=no-self-argument
        tp = Dict[str, Union[tp]]  # pylint: disable=unsubscriptable-object
        return functools.partial(_jsonable._make, tp, JsonableEnum.DICT)

    @classmethod
    def __class_getitem__(cls, /, type_hint):
        """
        Makes this decorator parametrizable

        Arguments:
            type_hint: type to parametrize this decorator with

        Returns:
            A parametrized version of this decorator
        """
        return functools.partial(_jsonable._make, type_hint, None)

    def __new__(cls, cls_arg=None, /, **kwargs):
        """
        Creates a new instance of the class

        This is overridden to allow the class to be used as a decorator without
        empty parens, that is::

            @jsonable
            class A:
                pass

        is valid and has the same meaning as::

            @jsonable()
            class A:
                pass
        """
        return _jsonable._make(None, None, cls_arg, **kwargs)

    @staticmethod
    def decorate_delayed():
        """
        Modifies all delayed jsonables whose dependencies are now met

        You only need to call this if you do not use `@jsonable <pheres._jsonable.jsonable>`
        after the dependencies are made available
        """
        return _jsonable.decorate_delayed()


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
