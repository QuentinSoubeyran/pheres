# -*- coding: utf-8 -*-
"""
Module for introspecting types in JSON

Part of the JsonPlus package
"""
# stdlib import
from itertools import repeat
from functools import lru_cache
from typing import Dict, List, Literal, Tuple, Type, Union, get_origin, get_args

# Local imports
from ._misc import AutoFormatMixin, JsonError

__all__ = [
    # Types
    "JsonNumber",
    "JsonValue",
    "JsonArray",
    "JsonObject",
    "JsonType",
    # Type utilities
    "JsonTypeError",
    "typeof",
    "is_json",
    "is_number",
    "is_value",
    "is_array",
    "is_object",
]

# Type hint aliases for JSON
JsonNumber = Union[int, float]
JsonValue = Union[None, bool, JsonNumber, str]
JsonArray = List["JsonType"]
JsonObject = Union[Dict[str, "JsonType"], "Jsonable"]
JsonType = Union[JsonValue, JsonArray, JsonObject]

# Type hint aliases for this module
TypeHint = Type[List]

# Constant
_JsonValueTypes = (type(None), bool, int, float, str)


class JsonTypeError(AutoFormatMixin, JsonError):
    """
    Raised when an object is not a valid JSON value, or on problems with types in JSON

    Attributes:
        obj -- the object with invalid type
        message -- explanation of the error
    """

    def __init__(self, obj, message="{obj} is not a valid JSON object"):
        super().__init__(message)
        self.obj = obj


class TypeHintError(AutoFormatMixin, JsonError):
    """
    Raised when a type hint is not valid for JSON values

    Attributes:
        type_hint -- invalid type hint
        message   -- explanation of the error
    """

    def __init__(
        self, type_hint, message="{type_hint} is not a valid type hint in JSON context"
    ):
        super().__init__(message)
        self.type_hint = type_hint


def typeof(obj: JsonType) -> Type[JsonType]:
    """
    Return the type alias of the type of the passed Json object.
    
    For nested types such as list or dict, the test is shallow
    and only checks the container
    The returned value is a type hints, equality testing must be done with `is`:
    
    typeof({}) == JsonObject # undefined
    typeof({}) is JsonObject # True

    Arguments
        obj -- object to get the type of
    
    Returns
        JsonValue, JsonArray or JsonObject based on the type of the passed object
    
    Raises
        JsonTypeError is the passed object is not a valid JSON
    """
    from ._jsonize import Jsonable  # delayed import to avoid circular deps

    if obj is None or isinstance(obj, _JsonValueTypes):
        return JsonValue
    elif isinstance(obj, list):
        return JsonArray
    elif isinstance(obj, (dict, Jsonable)):
        return JsonObject
    raise JsonTypeError(obj)


def is_json(obj: object) -> bool:
    """Check if a python object is valid JSON"""
    from ._jsonize import Jsonable

    type_ = type(obj)
    if type_ in _JsonValueTypes:
        return True
    elif issubclass(type_, Jsonable):
        return True
    elif type_ is list:
        return all(is_json(val) for val in obj)
    elif type_ is dict:
        return all(type(key) is str and is_json(val) for key, val in obj.items())
    return False

def is_number(obj: JsonType) -> bool:
    """
    Test if the JSON object is a JSON number

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json number, False otherwise
    
    Raises
        JsonTypeError if the object is not a valid JSON
    """
    return isinstance(obj, (int, float))


def is_value(obj: JsonType) -> bool:
    """
    Test if the JSON object is a JSON value

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json value, False otherwise
    
    Raises
        JsonTypeError if the object is not a valid JSON
    """
    return typeof(obj) is JsonValue


def is_array(obj: JsonType) -> bool:
    """
    Test if the JSON object is a JSON Array

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json array, False otherwise
    
    Raises
        JsonTypeError if the object is not a valid JSON
    """
    return typeof(obj) is JsonArray


def is_object(obj: JsonType) -> bool:
    """
    Test if the JSON object is a JSON Object

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json Object, False otherwise
    
    Raises
        JsonTypeError if the object is not a valid JSON
    """
    return typeof(obj) is JsonObject


@lru_cache
def normalize_json_tp(type_hint) -> TypeHint:
    """Normalize a type hint for JSON values

    Arguments
        type_hint -- JSON type hint to normalize
    
    Returns
        normalized representation of type_hint

    Raises
        TypeHintError when type_hint is invalid in JSON
    """
    from ._jsonize import Jsonable  # delayed import to prevent circular deps

    if type_hint in _JsonValueTypes:
        return type_hint
    elif (
        (orig := get_origin(type_hint)) is None
        and isinstance(type_hint, type)
        and issubclass(type_hint, Jsonable)
    ):
        return type_hint
    elif orig is Literal:
        if any(not isinstance(lit, (bool, int, str)) for lit in get_args(type_hint)):
            raise TypeHintError(type_hint)
        return type_hint
    elif orig is Union:
        norm_tp = tuple(normalize_json_tp(tp) for tp in get_args(type_hint))
        lits = sum(
            (get_args(tp) for tp in norm_tp if get_origin(tp) is Literal), tuple()
        )
        if lits:
            return Union[
                (
                    Literal[lits],
                    *tuple(tp for tp in norm_tp if not get_origin(tp) is Literal),
                )
            ]
        return Union[norm_tp]
    elif orig in (list, tuple):
        args = get_args(type_hint)
        if orig is list or (len(args) > 1 and args[1] is Ellipsis):
            return List[normalize_json_tp(args[0])]
        return Tuple[tuple(normalize_json_tp(tp) for tp in args)]
    elif orig is dict:
        args = get_args(type_hint)
        if args[0] is str:
            return Dict[str, normalize_json_tp(args[1])]
    raise TypeHintError(type_hint)


def is_json_type_hint(type_hint: TypeHint) -> bool:
    """Check that the type_hint is valid for JSON

    Supports Jsonable subclasses. Implemented by calling
    normalize_json_tp() and catching TypeHintError

    Arguments
        type_hint : type hint to test
    
    Return
        True if the type hint is valid for JSON, false otherwise
    """
    try:
        normalize_json_tp(type_hint)
        return True
    except TypeHintError:
        return False


@lru_cache
def have_common_value(ltp: TypeHint, rtp: TypeHint) -> bool:
    """Check if there is a json value valid for both type hints"""
    from ._jsonize import Jsonable

    lorig = get_origin(ltp)
    rorig = get_origin(rtp)
    largs = get_args(ltp)
    rargs = get_args(rtp)

    if ltp in _JsonValueTypes and rtp in _JsonValueTypes:
        return ltp == rtp or (ltp in (int, float) and rtp in (int, float))
    elif (
        isinstance(ltp, type)
        and isinstance(rtp, type)
        and issubclass(ltp, Jsonable)
        and issubclass(rtp, Jsonable)
    ):
        return ltp is rtp
    elif lorig is rorig is Union:
        return any(have_common_value(lsub, rsub) for lsub in largs for rsub in rargs)
    elif lorig in (list, tuple) and rorig in (list, tuple):
        lvariadic = lorig is list or len(largs) < 2 or largs[1] == Ellipsis
        rvariadic = rorig is list or len(rargs) < 2 or rargs[1] == Ellipsis
        if lvariadic and rvariadic:
            return have_common_value(largs[0], rargs[0])
        else:
            if lvariadic == rvariadic and len(largs) != len(rargs):
                return False
            return all(
                have_common_value(lsub, rsub)
                for lsub, rsub in zip(
                    repeat(largs[0]) if lvariadic else largs,
                    repeat(rargs[0]) if rvariadic else rargs,
                )
            )
    elif lorig is rorig is dict:
        return have_common_value(largs[1], rargs[1])
    return False
