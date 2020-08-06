# -*- coding: utf-8 -*-
"""
module for simple (de)serialization (from)to JSON in python

Part of the Pheres package
"""
# stdlib import
from abc import ABC
from copy import deepcopy
import dataclasses
from dataclasses import dataclass
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

#################
# JSONABLE  API #
#################

class JSONable(ABC):
    """Asbstract class to represent objects that can be serialized and deserialized to JSON

    """
    _REGISTER = []  # List of registered classes
    _REQ_JATTRS = []  # Defined on subclasses/decorated classes
    _ALL_JATTRS = []  # Defined on subclasses/decorated classes
    Decoder = json.JSONDecoder # Defined on subclasses/decorated classes

    def __init_subclass__(cls, all_attrs: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        _process_class(cls, all_attrs=all_attrs)

    def to_json(self):
        return {
            jattr.name: value.to_json() if isinstance(value, JSONable) else value
            for jattr in self._ALL_JATTRS
            if (value := getattr(self, jattr.py_name)) != jattr.get_default()
        }
    
    @classmethod
    def from_json_file(cls, /, fp):
        """Deserialize ``fp`` (a ``.read()``-supporting file-like object containing
        a JSON document) to 
        """
        return json.load(fp, cls=cls.Decoder)
    
    @classmethod
    def from_json_str(cls, /, s):
        """Deserialize ``s`` (a ``str``, ``bytes`` or ``bytearray`` instance
        containing a JSON document) to an instance of the class this method is called on.
        """
        return json.loads(s, cls=cls.Decoder)

    @classmethod
    def from_json(cls, /, obj):
        """Deserialize a Python object to an instance of the class this method is called on.

        Implemented by serializing obj to a JSON string, and then deserializing
        """
        return json.loads(dump(obj), cls=cls.Decoder)

# Local import that depends on JSONable being define
from .decoder import TypedJSONDecoder

@dataclass(frozen=True)
class JAttr:
    name: str = None


# Activate in 3.9
# T = TypeVar("T", JsonType)
# JsonAttr = Annotated[T, JAttr()]


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

    def __post_init__(self, /) -> None:
        if callable(self.default) and not jtyping.is_json((value := self.default())):
            raise JAttrError(f"A callable default must produce a valid JSON value, got {value}")

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
    
    def get_default(self, /) -> JsonObject:
        """Return the default value for that attribute"""
        if callable(self.default):
            return
        return deepcopy(self.default)

def _get_jattrs(cls: type, all_attrs=bool) -> List[_JsonisedAttribute]:
    """Internal helper to find the attributes to jsonize on a class"""
    jattrs = []
    # TODO: handle 3.9 Annotated type hint
    for py_name, tp in typing.get_type_hints(cls, localns={cls.__name__: cls}).items():
        if all_attrs:
            default = getattr(cls, py_name, MISSING)
            if isinstance(default, dataclasses.Field):
                if default.default is not dataclasses.MISSING:
                    default = default.default
                elif default.default_factory is not dataclasses.MISSING:
                    default = default.default_factory
                else:
                    default = MISSING
            # TODO: handle name in python 3.9
            jattrs.append(
                _JsonisedAttribute(
                    name=py_name, # TODO: change in 3.9
                    py_name=py_name,
                    type_hint=jtyping.normalize_json_tp(tp),
                    default=default
                )
            )
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if field.default is dataclasses.MISSING and field.default_factory is not dataclasses.MISSING:
                # TODO: handle name in python 3.9
                jattrs.append(
                    _JsonisedAttribute(
                        name=field.name, # TODO : change in python 3.9
                        py_name=field.name,
                        type_hint=field.type,
                        default=field.default_factory
                    )
                )
    return jattrs

def _validates_under(
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
            and _validates_under(req_jattrs, other_cls._ALL_JATTRS)
            and _validates_under(other_cls._REQ_JATTRS, all_jattrs)
        ):
            raise JSONableError(
                f"Jsonable '{cls.__name__}' overlaps with '{other_cls.__name__}' without inheriting from it"
            )
    cls._REQ_JATTRS = req_jattrs
    cls._ALL_JATTRS = all_jattrs
    cls.Decoder = TypedJSONDecoder[cls]
    cls.to_json = JSONable.to_json
    cls.from_json = JSONable.from_json
    JSONable._REGISTER.append(cls)
    JSONable.register(cls)
    return cls


def jsonize(cls:type=None, /, *, all_attrs:bool=True) -> type:
    """Decorator to make a class JSONable

    By default, all type-hinted attributes are used. Fully compatible with dataclasses
    """
    if cls is not None:
        return _process_class(cls, all_attrs=all_attrs)
    else:
        return functools.partial(_process_class, all_attrs=all_attrs)



#################
# SERIALIZATION #
#################

class JSONableEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that supports JSONable classes
    """

    def default(self, obj: object):
        if isinstance(obj, JSONable):
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

raise NotImplementedError("This module is under construction")

# TODO
# Deserialization: make it infer types ! (additional to TypedJSONDecoder)

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

# def jsonable_hook(obj: dict) -> Union[Any, dict]:
#     """
#     Object hook for the json.load() and json.loads() methods to deserialize JSONable classes
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

def jsonable_hook(obj):
    raise NotImplementedError

@functools.wraps(json.load)
def load(*args, object_hook=jsonable_hook, **kwargs):
    return json.load(*args, object_hook=object_hook, **kwargs)


@functools.wraps(json.loads)
def loads(*args, object_hook=jsonable_hook, **kwargs):
    return json.loads(*args, object_hook=object_hook, **kwargs)