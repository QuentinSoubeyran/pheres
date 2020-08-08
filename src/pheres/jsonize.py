# -*- coding: utf-8 -*-
"""
Simple (de)serialization (from)to JSON in python

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
from typing import Any, Dict, Iterable, List, Literal, Optional, Type, Tuple, Union

# local imports
from . import jtyping
from .jtyping import JSONType, JSONObject
from .misc import JSONError, find_injection

__all__ = [
    # Errors
    "JSONableError",
    "JAttrError",
    # Jsonable API
    "JSONable",
    "jsonize",
    ## Serialization
    "JSONableEncoder",
    "dump",
    "dumps",
    ## Deserialization
    "jsonable_hook",
    "load",
    "loads",
]

# Type aliases
TypeHint = type(List)
JsonLiteral = Union[bool, int, str]

# Sentinel used in this module
MISSING = object()


class JSONableError(JSONError):
    """
    Base exception for problem with the JSONable ABC or the jsonize decorator
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
    _REQ_JATTRS = {}  # Defined on subclasses/decorated classes
    _ALL_JATTRS = {}  # Defined on subclasses/decorated classes
    Decoder = json.JSONDecoder  # Defined on subclasses/decorated classes

    def __init_subclass__(cls, all_attrs: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        _process_class(cls, all_attrs=all_attrs)

    def to_json(self):
        return {
            jattr.name: value.to_json() if isinstance(value, JSONable) else value
            for jattr in self._ALL_JATTRS.values()
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
            raise JAttrError(
                f"A callable default must produce a valid JSON value, got {value}"
            )

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

    def get_default(self, /) -> JSONObject:
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
                    name=py_name,  # TODO: change in 3.9
                    py_name=py_name,
                    type_hint=jtyping.normalize_json_tp(tp),
                    default=default,
                )
            )
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is not dataclasses.MISSING
            ):
                # TODO: handle name in python 3.9
                jattrs.append(
                    _JsonisedAttribute(
                        name=field.name,  # TODO : change in python 3.9
                        py_name=field.name,
                        type_hint=field.type,
                        default=field.default_factory,
                    )
                )
    return jattrs


def _is_jattr_subset(
    subset: Dict[str, _JsonisedAttribute], superset: Dict[str, _JsonisedAttribute]
):
    """
    Internal helper to test for conflicts between JSONable subclasses

    Test if there exist a value satisfying all of 'subset' that is
    valid in 'superset' (i.e. such that all its key match something in 'superset')
    """
    # Quick fix: more stringent test, but assure the condition requires
    # return all(
    #     any(subattr.overlaps(superattr) for superattr in superset) for subattr in subset
    # )
    return all(
        jattr.name in superset and jattr.overlaps(superset[jattr.name])
        for jattr in subset
    )


def _process_class(cls: type, /, *, all_attrs: bool) -> type:
    """Internal helper to make a class JSONable"""
    from .decoder import TypedJSONDecoder # avoid circular deps

    all_jattrs = {jattr.name: jattr for jattr in _get_jattrs(cls, all_attrs)}
    req_jattrs = {jattr.name: jattr for jattr in all_jattrs.values() if jattr.default is MISSING}
    # Check for conflict with previously registered classes
    for other_cls in JSONable._REGISTER:
        if (
            not issubclass(cls, other_cls)
            and _is_jattr_subset(req_jattrs, other_cls._ALL_JATTRS)
            and _is_jattr_subset(other_cls._REQ_JATTRS, all_jattrs)
            # and find_injection(
            #     req_jattrs, other_cls._ALL_JATTRS, jtyping.have_common_value
            # )
            # and find_injection(
            #     other_cls._REQ_JATTRS, all_jattrs, jtyping.have_common_value
            # )
        ):
            raise JSONableError(
                f"Jsonable '{cls.__name__}' overlaps with '{other_cls.__name__}' without inheriting from it"
            )
    cls._REQ_JATTRS = req_jattrs
    cls._ALL_JATTRS = all_jattrs
    cls.to_json = JSONable.to_json
    cls.from_json_file = JSONable.from_json_file
    cls.from_json_str = JSONable.from_json_str
    cls.from_json = JSONable.from_json
    JSONable.register(cls)
    JSONable._REGISTER.append(cls)
    cls.Decoder = TypedJSONDecoder[cls] # last because the class must already be JSONable
    return cls


def jsonize(cls: type = None, /, *, all_attrs: bool = True) -> type:
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


def jsonable_hook(obj: dict) -> Union[Any, dict]:
    """
    Object hook for the json.load() and json.loads() methods to deserialize JSONable classes
    """
    valid_cls = []
    for cls in JSONable._REGISTER:
        # req_jattrs = set(cls._REQ_JATTRS)
        if (
            # find_injection(
            #     A=obj.items(),
            #     B=cls._ALL_JATTRS,
            #     match_func=lambda kv, jattr: kv[0] == jattr.name
            #     and jtyping.typecheck(kv[1], jattr.type_hint),
            #     validator_func=lambda match: req_jattrs <= match.values(),
            # )
            # is not None
            all( # all required arguments are there
                key in obj and jtyping.typecheck(obj[key], jattr.type_hint)
                for key, jattr in cls._REQ_JATTRS.items()
            )
            and all(  # all keys are valid - don't test req, already did
                key in cls._ALL_JATTRS and jtyping.typecheck(obj[key], cls._ALL_JATTRS[key].type_hint)
                for key in obj.keys() - cls._REQ_JATTRS.items()
            )
        ):
            valid_cls.append(cls)
    # find less-specific class in case of inheritance
    valid_cls = [
        cls
        for i, cls in enumerate(valid_cls)
        if all(not issubclass(cls, next_cls) for next_cls in valid_cls[i + 1 :])
    ]
    if len(valid_cls) > 1:
        raise JSONError(
            f"[!! This is a bug !! Please report] Multiple valid JSONable class found while deserializing {obj}"
        )
    elif len(valid_cls) == 1:
        cls = valid_cls[0]
        return cls(
            **{
                jattr.py_name: obj[jattr.name]
                if jattr.name in obj
                else jattr.get_default()
                for jattr in cls._ALL_JATTRS.values()
            }
        )
    else:
        return obj


@functools.wraps(json.load)
def load(*args, object_hook=jsonable_hook, **kwargs):
    return json.load(*args, object_hook=object_hook, **kwargs)


@functools.wraps(json.loads)
def loads(*args, object_hook=jsonable_hook, **kwargs):
    return json.loads(*args, object_hook=object_hook, **kwargs)
