# -*- coding: utf-8 -*-
"""
module for simple (de)serialization (from)to JSON in python

Part of the JsonPlus package
"""
# stdlib import
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import functools
import json
import types
import typing
from typing import Iterable, List, Literal, Optional, Type, Union

# Local import
from . import _jtypes
from . import _misc
from ._jtypes import JsonType, JsonObject
from ._module_data import *

__all__ = [
    # Errors
    "JsonableError",
    "JAttrError",
    # Jsonable API
    ## Jsonable Mixin
    "jsonize",
    "Jsonable",
    ## Serialization
    "JSONableEncoder",
    "dump",
    "dumps",
    ## Deserialization
    "JSONableDecoder",
    "TypedJSONDecodeError",
]

# Type aliases
TypeHint = type(List)
JsonLiteral = Union[bool, int, str]

# Sentinel used in this module
MISSING = object()


class JsonableError(_misc.JsonError):
    """
    Base exception for problem with the Jsonable mixin
    """


class JAttrError(JsonableError):
    """
    Exception for when an attribute is improperly jsonized
    """


# @dataclass(frozen=True)
# class JsonAttr:
#     f"""
#     Annotation to use to mark attributes as Jsonized.

#     Only does something in subclasses of {__name__}.Jsonable
#     JsonAttr must be indexed, do *not* create instances

#     Usage:
#         attribute : JsonAttr[TypeHint, name=None, value=None]

#     See JsonAttr.__class_getitem__ docstring for details
#     """
#     name: str = None
#     default: JsonType = Ellipsis
#     values: List[JsonLiteral] = None


class JsonAttr:
    """
    Internal marker class. More use in python 3.9 when API is extended
    """


# TODO: replace values by the use of Literal
def jsonize(
    type_hint: TypeHint,
    name: str = None,
    default: JsonType = MISSING,
    #values: Union[JsonLiteral, Iterable[JsonLiteral]] = None,
) -> TypeHint:
    f"""
    Return the type hint that marks an attribute as jsonized in a {__name__}.Jsonable subclass.

    Arguments
        type_hint - the type hint of the attributes. Must be valid for JSON (this includes sublass of Jsonable)
        name      - name of the attribute in JSON. Otherwise, the name is the same as in python
        default   - default value of the jsonized attribute
    
    Returns
        A special type hint that marks the attributes as jsonized
    """
    type_hint = _jtypes.normalize_json_tp(type_hint)  # raises on problem
    if default is MISSING:
        default = Ellipsis  # Hack to get a simple repr()
    elif not _jtypes.is_json(default):
        raise JAttrError(
            "Default value for jsonized attribute must be a valid JSON value"
        )
    kwargs = {"name": name, "default": default}
    return Union[
        JsonAttr, Literal[repr(kwargs)], type_hint
    ]  # Ugly hack for python 3.8; waiting on python 3.9's typing.Annotated


@dataclass(frozen=True)
class _JsonisedAttribute:
    """
    Internal class for a json key-value pair that should be in the JSON serialization of a Jsonable instance

    Attributes
        name      -- name of the attribute in JSON
        py_name   -- name of the attribute in Python
        type_hint -- type hint of the attribute in Python
        default   -- default value for the json attribute if not provided in JSON
        value     -- exact value of the attribute in JSON (if the value is forced)
    """

    name: str
    py_name: str
    type_hint: TypeHint
    default: object = MISSING
    literal_values: List[Union[bool, int, str]] = field(
        default=None, init=False, hash=True
    )

    def __post_init__(self) -> None:
        orig = typing.get_origin(self.type_hint)
        args = typing.get_args(self.type_hint)
        if orig is Literal:
            if self.default is not MISSING:
                raise JsonableError(
                    f"Literal jsonized attribute {self.py_name} cannot have a default value"
                )
            object.__setattr__(self, "literal_values", args)
            if not all(
                isinstance((errored_val := v), (bool, int, str))
                for v in self.literal_values
            ):
                raise JsonableError(
                    f"Literal jsonized attribute {self.py_name} must have a value of type bool, int or str,"  # pylint: disable=undefined-variable
                    f"got {type(errored_val)}"
                )
        if orig is tuple and len(args) > 1 and args[1] is Ellipsis:
            object.__setattr__(self, "type_hint", List[args[0]])

    def overlaps(self, /, other: "_JsonKey") -> bool:
        """
        Check if the exists a json key-value accepted by both _JsonKey

        Arguments
            other : _JsonKey to check conflicts with
        
        Return
            True in case of conflict, False otherwise
        """
        if self.name != other.name:
            return False
        if self.literal_values and other.literal_values:
            return bool(set(self.literal_values) & set(other.literal_values))
        ltype = self.type_hint
        rtype = other.type_hint
        if self.literal_values:
            ltype = Union[tuple(type(lit) for lit in self.literal_values)]
        if other.literal_values:
            rtype = Union[tuple(type(lit) for lit in other.literal_values)]
        return _jtypes.have_common_value(ltype, rtype)


def _get_jsonised_attr(cls: type):
    """Internal helper to find jsonized attribute on a class"""
    jsonized_attrs = []
    for attr_name, tp in typing.get_type_hints(cls).items():
        if (
            typing.get_origin(tp) is Union
            and (args := typing.get_args(tp))[0] is JsonAttr
        ):
            tp = Union[args[2:]]
            data = eval(
                typing.get_args(args[1])[0]
            )  # Retrieved the kwargs dict that was in Literal (see JsonAttr.__class_getitem__)
            if data["default"] is Ellipsis:
                default = getattr(cls, attr_name, MISSING)
                if default is not MISSING and not _jtypes.is_json(default):
                    raise JAttrError(
                        f"Default value for jsonized attribute must be valid JSON, got {default}"
                    )
                data["default"] = default
            jsonized_attrs.append(
                _JsonisedAttribute(
                    name=data["name"] or attr_name,
                    py_name=attr_name,
                    type_hint=tp,
                    default=data["default"],
                )
            )
    for i, jattr in enumerate(jsonized_attrs):
        for other in jsonized_attrs[i + 1 :]:
            if jattr.name == other.name:
                raise JAttrError(
                    f"Jsonized attributes {jattr.py_name} and {other.py_name} have the same json name"
                )
    return jsonized_attrs


def _is_jattr_subset(
    jattrs: List[_JsonisedAttribute], valids: List[_JsonisedAttribute]
):
    """
    Internal helper to test for conflicts between Jsonable subclasses

    Test if all jsonized attributes in 'jattrs' may be valid under 'valids', that is,
    if there exists a JSON object that would be valid for both sets of jsonized attributes
    """
    # This is more stringent than really necessary, as a valid attribute can
    # match multiple times
    return all(
        any(attr.overlaps(valid_attr) for valid_attr in valids) for attr in jattrs
    )


class Jsonable:
    f"""
    Mixin Class to make subclass (de)serializable (from)to JSON.
    
    Interacts well with dataclasses of NamedTuples
    This can be used e.g. to write config files as python classes and have them automatically loaded
    from JSON.

    Usage:

    Use the type hint generated by {__name__}.JsonAttr to mark an attribute as part of the JSON

    class Foo(Jsonable):
        value : Jsonize[int] # the attribute 'value' will be serialized
        names : Jsonize[List[int]] # attribute can be of any valid JsonType type
        id_   : JsonAttr[int, "id"] # change the attribute name in JSON. may not be supported by type checkers

    foo = Foo(value=5, names=["andrew", "bob"], id_=87456621)
    foo.asjson() == {{"value": 5, "names": ["andrew", "bob"], "id": 87456621}}
    """

    # All Jsonable subclasses
    _REGISTERED_CLASSES: List[Type["Jsonable"]] = []

    # Defined on subclasses by __init_suclass__
    _REQ_JATTRS: List[_JsonisedAttribute] = []
    _ALL_JATTRS: List[_JsonisedAttribute] = []
    Decoder = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        all_jattrs = _get_jsonised_attr(cls)
        req_jattrs = [jattr for jattr in all_jattrs if jattr.default is MISSING]
        # Check for conflict with previously registered classes
        for other_cls in Jsonable._REGISTERED_CLASSES:
            if (
                not issubclass(cls, other_cls)
                and _is_jattr_subset(req_jattrs, other_cls._ALL_JATTRS)
                and _is_jattr_subset(other_cls._REQ_JATTRS, all_jattrs)
            ):
                raise JsonableError(
                    f"Jsonable '{cls.__name__}' overlaps with '{other_cls.__name__}' without inheriting from it"
                )
        cls._REQ_JATTRS = req_jattrs
        cls._ALL_JATTRS = all_jattrs
        cls.Decoder = JSONableDecoder[cls]
        Jsonable._REGISTERED_CLASSES.append(cls)

    def to_json(self: "Jsonable"):
        """Serializes the instance as a JSON object"""
        return {
            jattr.name: value.to_json() if isinstance(value, Jsonable) else value
            for jattr in self._ALL_JATTRS
            if (value := getattr(self, jattr.py_name)) != jattr.default
        }

    @classmethod
    def from_json(cls: "Jsonable", /, obj: JsonObject):
        """Deserialize the provided JSON as a instance of this (sub)class"""
        # TODO:
        # - deserialize files, string and loaded object
        # - typecheck on deserialization
        # - track lineno and colno
        # -
        raise NotImplementedError


#################
# SERIALIZATION #
#################


class JSONableEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that supports Jsonable instance
    """

    def default(self, obj: object):
        if isinstance(obj, Jsonable):
            return obj.to_json()
        return super().default(obj)


@functools.wraps(json.dump)
def dump(*args, cls=JSONableEncoder, **kwargs):
    return json.dump(*args, cls=cls, **kwargs)


@functools.wraps(json.dumps)
def dumps(*args, cls=JSONableEncoder, **kwargs):
    return json.dumps(*args, cls=cls, **kwargs)


###################
# DESERIALIZATION #
###################


def _tp_cache(func):
    """Wrapper caching __class_getitem__ on type hints

    Provides a fallback if arguments are not hashable    
    """
    cache = functools.lru_cache()(func)

    @functools.wraps(func)
    def wrapper(type_hint):
        try:
            return cache(type_hint)
        except TypeError as err:  # unhashable args
            print(err)
            pass
        return func(type_hint)

    return wrapper


def _exec_body(namespace, type_hint):
    """Internal helper to initialize parametrized JSONableDecoder"""
    namespace["type_hint"] = property(lambda self: type_hint)


from ._decoder import (
    TypedJSONDecodeError,
    py_make_scanner,
    JSONArray,
    JSONObject,
    DecodeContext,
)


class JSONableDecoder(ABC, json.JSONDecoder):
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

    @staticmethod
    @_tp_cache
    def _class_getitem_cache(tp):
        """Parametrize the TypedJSONDecoder to decode the provided type hint

        Jsonable subclasses are supported
        """
        return types.new_class(
            "ParametrizedJSONableDecoder",
            (JSONableDecoder,),
            exec_body=functools.partial(_exec_body, type_hint=tp),
        )

    def __class_getitem__(cls, tp):
        """Parametrize the TypedJSONDecoder to decode the provided type hint

        Jsonable subclasses are supported
        """
        tp = _jtypes.normalize_json_tp(tp)
        return cls._class_getitem_cache(tp)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace default decoder by our contextualized decoder
        self.parse_object = JSONObject
        self.parse_array = JSONArray
        self.scan_once = py_make_scanner(self)

    @functools.wraps(json.JSONDecoder.raw_decode)
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
                    if typing.get_origin(self.type_hint) is None
                    and issubclass(self.type_hint, Jsonable)
                    else None,
                ),
            )
        except StopIteration as err:
            raise json.JSONDecodeError("Expecting value", s, err.value) from None
        return obj, end


# @dataclass(frozen=True)
# class _JsonPair:
#     """
#     Internal class that represents a key-value pair actually read from a JSON

#     The __eq__ is overload to support matching with _JsonEntry instances
#     """

#     key: str
#     value: JsonType

#     def match_entry(self, entry: _JsonEntry) -> bool:
#         return self.key == entry.name and (
#             entry.literal_value is _MISSING or self.value == entry.literal_value
#         )


# def jsonable_hook(obj: dict) -> Union[Jsonable, dict]:
#     """
#     Object hook for the json.load() and json.loads() methods to deserialize Jsonable subclasses
#     """
#     pairs = [_JsonPair(k, v) for k, v in obj.items()]
#     matched_cls = [
#         cls
#         for cls in Jsonable._REGISTERED_CLASSES
#         if (
#             all(
#                 any(e == p for p in pairs) for e in cls._REQ_ENTRIES
#             )  # All required keys are defined
#             and all(any(p == e for e in cls._ALL_ENTRIES) for p in pairs)
#         )  # All pairs are allowed
#     ]
#     matched_cls = [
#         cls
#         for i, cls in enumerate(matched_cls)
#         if all(not issubclass(cls, next_cls) for next_cls in matched_cls[i + 1 :])
#     ]
#     if len(matched_cls) > 1:
#         raise RuntimeError(
#             f"Multiple Jsonable subclass matches the JSON {obj}. This should never happen and is a bug in {MODULE}"
#         )
#     elif len(matched_cls) == 1:
#         kwargs = {}
#         return matched_cls[0](**kwargs)
#     else:
#         return obj


# @wraps(_json.load)
# def load(*args, object_hook=jsonable_hook, **kwargs):
#     return _json.load(*args, object_hook=object_hook, **kwargs)


# @wraps(_json.load)
# def loads(*args, object_hook=jsonable_hook, **kwargs):
#     return _json.loads(*args, object_hook=object_hook, **kwargs)


# from json import *


# ## TODO:
# # - Typechecking on Jsonized attributes
# # - Deserialization of a particular object (how ?)
# # - test
