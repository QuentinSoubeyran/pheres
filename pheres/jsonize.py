# -*- coding: utf-8 -*-
"""
module for simple (de)serialization (from)to JSON in python

Part of the Pheres package
"""
# stdlib import
from abc import ABC
from dataclasses import dataclass, field
import functools
import json
import types
import typing
from typing import Any, Iterable, List, Literal, Optional, Type, Union

# local imports
from . import jtyping
from .jtyping import JsonType, JsonObject
from .misc import JsonError

__all__ = [
    # Errors
    "JSONableError",
    "JAttrError",
    # Jsonable API
    "JSONable",
    #"jsonize",
    #"Jsonable",
    ## Serialization
    "JSONableEncoder",
    "dump",
    "dumps",
    ## Deserialization
    #"jsonable_hook",
    "load",
    "loads"
]

# Type aliases
TypeHint = type(List)
JsonLiteral = Union[bool, int, str]

# Sentinel used in this module
MISSING = object()


class JSONableError(JsonError):
    """
    Base exception for problem with the Jsonable mixin
    """


class JAttrError(JSONableError):
    """
    Exception for when an attribute is improperly jsonized
    """


class JSONable(ABC):
    """Asbstract class to represent objects that can be serialized and deserialized to JSON

    """
    _REGISTER = [] # List of registered classes

    def __init_subclass__(cls, **kwargs):
        # TODO
        raise NotImplementedError

    def to_json(self):
        # TODO
        raise NotImplementedError


# Local import that depends on JSONable being define
from .decoder import TypedJSONDecoder

#################
# JSONABLE  API #
#################


@dataclass(frozen=True)
class JAttr:
    name: str = None


# Activate in 3.9
# T = TypeVar("T", JsonType)
# JsonAttr = Annotated[T, JAttr()]


class JsonAttr:
    """
    Internal marker class. More use in python 3.9 when API is extended
    """


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
    # #literal_values: List[Union[bool, int, str]] = field(
    # #    default=None, init=False, hash=True
    # #)

    # def __post_init__(self) -> None:
    #     orig = typing.get_origin(self.type_hint)
    #     args = typing.get_args(self.type_hint)
    #     if orig is Literal:
    #         if self.default is not MISSING:
    #             raise JSONableError(
    #                 f"Literal jsonized attribute {self.py_name} cannot have a default value"
    #             )
    #         object.__setattr__(self, "literal_values", args)
    #         if not all(
    #             isinstance((errored_val := v), (bool, int, str))
    #             for v in self.literal_values
    #         ):
    #             raise JSONableError(
    #                 f"Literal jsonized attribute {self.py_name} must have a value of type bool, int or str,"  # pylint: disable=undefined-variable
    #                 f"got {type(errored_val)}"
    #             )
    #     if orig is tuple and len(args) > 1 and args[1] is Ellipsis:
    #         object.__setattr__(self, "type_hint", List[args[0]])

    def overlaps(self, /, other: "_JsonKey") -> bool:
        """
        Check if the exists a json key-value accepted by both _JsonKey

        Arguments
            other : _JsonKey to check conflicts with
        
        Return
            True in case of conflict, False otherwise
        """
        return self.name == other.name and jtyping.have_common_value(
            self.type_hint, other.type_hint
        )

def _get_jattrs(cls: type, all_attrs=bool) -> List[_JsonisedAttribute]:
    """Internal helper to find the attributes to jsonize on a class"""
    # TODO check if @dataclass has already been applied
    # TODO handle default list/dict in values, or callables to compute the value

    for attr_name, tp in typing.get_type_hints(cls, localns={cls.__name__: cls}).items():
        # TODO: handle dataclass fields !!
        pass
    raise NotImplementedError
    return []

def _is_jattr_subset(
    subset: List[_JsonisedAttribute], superset: List[_JsonisedAttribute]
):
    """
    Internal helper to test for conflicts between JSONable subclasses

    Test if there exist a value satisfying all of 'subset' that is
    valid in 'superset' (i.e. such that all its key match something in 'superset')
    """
    # TODO: may not work properly due to order
    # e.g.
    # a <-> x,y
    # b <-> x
    # Will match a against x, remove x and then find no overlap
    # because b doesn't match against y
    # May be very difficult to do properly
    # Quick fix: more stringent test, but assure the condition requires
    return all(
        any(subattr.overlaps(superattr) for superattr in superset)
        for subattr in subset
    )
    # Unreachable code, to debug
    superset = set(superset)
    for jattr in subset:
        if any(jattr.overlaps((match := superattr)) for superattr in superset):
            # Matched super jattr cannot match others
            superset.remove(match) # pylint: disable=undefined-variable
        else:
            return False
    return True
    

def _process_class(cls: type, /, *, all_attrs: bool) -> type:
    """Internal helper to make a class JSONable"""
    all_jattrs = _get_jattrs(cls, all_attrs)
    req_jattrs = [jattr for jattr in all_jattrs if jattr.default is MISSING]
    # Check for conflict with previously registered classes
    for other_cls in JSONable._REGISTER:
        if (
            not issubclass(cls, other_cls)
            and _is_jattr_subset(req_jattrs, other_cls._ALL_JATTRS)
            and _is_jattr_subset(other_cls._REQ_JATTRS, all_jattrs)
        ):
            raise JSONableError(
                f"Jsonable '{cls.__name__}' overlaps with '{other_cls.__name__}' without inheriting from it"
            )
    cls._REQ_JATTRS = req_jattrs
    cls._ALL_JATTRS = all_jattrs
    cls.Decoder = TypedJSONDecoder[cls]
    Jsonable._REGISTERED_CLASSES.append(cls)
    raise NotImplementedError


def jsonize(cls:type=None, /, *, all_attrs:bool=True) -> type:
    """Decorator to make a class JSONable

    By default, all type-hinted attributes are used. Fully compatible with dataclasses
    """
    if cls is not None:
        return _process_class(cls, all_attrs=all_attrs)
    else:
        return functools.partial(_process_class, all_attrs=all_attrs)

# TODO:
# Function to handle class
# -- find jsonized attr
# -- decorator: by default, take all annotated attributes
# -- full compatibility, both ways, with @dataclass
# -- Only 3.9 allows rename, with Annotated[]



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

# TODO
# Deserialization: make it infer types ! (additional to TypedJSONDecoder)

def jsonable_hook():
    # TODO
    raise NotImplementedError

@functools.wraps(json.load)
def load(*args, object_hook=jsonable_hook, **kwargs):
    return json.load(*args, object_hook=object_hook, **kwargs)


@functools.wraps(json.loads)
def loads(*args, object_hook=jsonable_hook, **kwargs):
    return json.loads(*args, object_hook=object_hook, **kwargs)

raise NotImplementedError("This module is under construction")

#################################################################################
#################################################################################
# OLD CODE BELOW -- NEEDS UPDATE
#################################################################################
#################################################################################


# TODO: replace values by the use of Literal
# def jsonize(
#     type_hint: TypeHint,
#     name: str = None,
#     default: JsonType = MISSING,
#     # values: Union[JsonLiteral, Iterable[JsonLiteral]] = None,
# ) -> TypeHint:
#     f"""
#     Return the type hint that marks an attribute as jsonized in a {__name__}.Jsonable subclass.

#     Arguments
#         type_hint - the type hint of the attributes. Must be valid for JSON (this includes sublass of Jsonable)
#         name      - name of the attribute in JSON. Otherwise, the name is the same as in python
#         default   - default value of the jsonized attribute
    
#     Returns
#         A special type hint that marks the attributes as jsonized
#     """
#     type_hint = jtyping.normalize_json_tp(type_hint)  # raises on problem
#     if default is MISSING:
#         default = Ellipsis  # Hack to get a simple repr()
#     elif not jtyping.is_json(default):
#         raise JAttrError(
#             "Default value for jsonized attribute must be a valid JSON value"
#         )
#     kwargs = {"name": name, "default": default}
#     return Union[
#         JsonAttr, Literal[repr(kwargs)], type_hint
#     ]  # Ugly hack for python 3.8; waiting on python 3.9's typing.Annotated


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
                if default is not MISSING and not jtyping.is_json(default):
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
                raise JSONableError(
                    f"Jsonable '{cls.__name__}' overlaps with '{other_cls.__name__}' without inheriting from it"
                )
        cls._REQ_JATTRS = req_jattrs
        cls._ALL_JATTRS = all_jattrs
        cls.Decoder = TypedJSONDecoder[cls]
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


###################
# DESERIALIZATION #
###################

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





# from json import *


# ## TODO:
# # - Typechecking on Jsonized attributes
# # - Deserialization of a particular object (how ?)
# # - test
