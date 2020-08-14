# -*- coding: utf-8 -*-
"""
Module for introspecting types in JSON

Part of the Pheres package
"""
# stdlib import
from itertools import repeat
from functools import lru_cache
from threading import RLock
from typing import (
    AbstractSet,
    Any,
    Dict,
    List,
    Literal,
    Tuple,
    Type,
    Union,
    get_origin,
    get_args,
)

# Local imports
from .misc import AutoFormatMixin, JSONError, split

__all__ = [
    # Types
    "JSONValue",
    "JSONArray",
    "JSONObject",
    "JSONType",
    # Error
    "JSONTypeError",
    "TypeHintError",
    "CycleError",
    # Type utilities
    "typeof",
    "is_json",
    "is_number",
    "is_value",
    "is_array",
    "is_object",
    "typecheck",
    # Type Hint utilities
    "normalize_json_tp",
    "is_json_type_hint",
    "have_common_value",
]

# Type hint aliases for JSON
JSONValue = Union[None, bool, int, float, str]
JSONArray = Union[Tuple["JSONType", ...], List["JSONType"]]
JSONObject = Union[Dict[str, "JSONType"], "JSONable"]
JSONType = Union[JSONValue, JSONArray, JSONObject]
JSONLiteral = Union[bool, int, str]

# Type hint aliases for this module
TypeHint = Type[List]

# Constant
_JSONLiteralTypes = (bool, int, str)
_JSONValueTypes = (type(None), bool, int, float, str)
_JSONArrayTypes = (tuple, list)
_JSONObjectTypes = (dict,)

##############
# EXCEPTIONS #
##############


class JSONTypeError(AutoFormatMixin, JSONError):
    """
    Raised when an object is not a valid JSON value, or on problems with types in JSON

    Attributes:
        obj -- the object with invalid type
        message -- explanation of the error
    """

    def __init__(self, obj, message="{obj} is not a valid JSON object"):
        super().__init__(message)
        self.obj = obj


class TypeHintError(AutoFormatMixin, JSONError):
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


class CycleError(JSONError):
    """
    Raised when a value has cycles in it

    Attributes:
        obj -- cyclic object
        cycle -- detected cycle
        message -- explanation of the error
    """

    def __init__(self, obj, cycle, message="object has circular reference"):
        super().__init__(message)
        self.obj = obj


##################
# TYPE UTILITIES #
##################


def typeof(obj: JSONType) -> Type[JSONType]:
    """
    Return the type alias of the type of the passed JSON object.
    
    For nested types such as list or dict, the test is shallow
    and only checks the container type.
    The returned value is a type hints, equality testing must be done with `is`:
    
    typeof({}) == JSONObject # undefined
    typeof({}) is JSONObject # True

    Arguments
        obj -- object to get the type of
    
    Returns
        JSONValue, JSONArray or JSONObject based on the type of the passed object
    
    Raises
        JSONTypeError if the passed object is not a valid JSON
    """
    from .jsonize import JSONable

    if obj is None or isinstance(obj, _JSONValueTypes):
        return JSONValue
    elif isinstance(obj, (list, tuple)):
        return JSONArray
    elif isinstance(obj, (dict, JSONable)):
        return JSONObject
    raise JSONTypeError(obj)


def _is_json(obj: Any, rec_guard: Tuple[Any]) -> bool:
    """internal helper to check if object is valid JSON
    
    Has a guard to prevent infinite recursion """
    from .jsonize import JSONable

    if obj in rec_guard:
        raise CycleError(obj, rec_guard[rec_guard.index(obj) :])
    type_ = type(obj)
    if type_ in _JSONValueTypes:
        return True
    elif issubclass(type_, JSONable):
        return True
    elif type_ in _JSONArrayTypes:
        rec_guard = (*rec_guard, obj)
        return all(_is_json(elem, rec_guard) for elem in obj)
    elif type_ in _JSONObjectTypes:
        rec_guard = (*rec_guard, obj)
        return all(
            type(key) is str and _is_json(val, rec_guard) for key, val in obj.items()
        )
    return False


def is_json(obj: Any) -> bool:
    """Check if a python object is valid JSON
    
    Raises CycleError if the value has circular references
    Only tuples and lists are accepted for JSON arrays
    Dictionary *must* have string as keys
    """
    return _is_json(obj, ())


def is_number(obj: JSONType) -> bool:
    """
    Test if the JSON object is a JSON number

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json number, False otherwise
    
    Raises
        JSONTypeError if the object is not a valid JSON
    """
    return type(obj) in (int, float)


def is_value(obj: JSONType) -> bool:
    """
    Test if the JSON object is a JSON value

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json value, False otherwise
    """
    try:
        return typeof(obj) is JSONValue
    except JSONTypeError:
        return False


def is_array(obj: JSONType) -> bool:
    """
    Test if the JSON object is a JSON Array

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json array, False otherwise
    
    Raises
        JSONTypeError if the object is not a valid JSON
    """
    try:
        return typeof(obj) is JSONArray
    except JSONTypeError:
        return False


def is_object(obj: JSONType) -> bool:
    """
    Test if the JSON object is a JSON Object

    Argument
        obj -- object to test the type of
    
    Return
        True if the object is a json Object, False otherwise
    
    Raises
        JSONTypeError if the object is not a valid JSON
    """
    try:
        return typeof(obj) is JSONObject
    except JSONTypeError:
        return False


def typecheck(value: Any, tp: TypeHint) -> bool:
    """Test that a JSON value matches a JSON type hint
    
    Arguments
        tp -- type hint to check against. Must be normalized
        value -- value to test
    
    Return
        True if the value is valid for the type hint, False otherwise
    
    Raises
        TypeHintError if the type hint could not be handled
    """
    from .jsonize import JSONable

    if tp in _JSONValueTypes:
        return isinstance(value, tp)
    elif isinstance(tp, type) and issubclass(tp, JSONable):
        return isinstance(value, tp)
    elif (orig := get_origin(tp)) is Literal:
        return value in get_args(tp)
    elif orig is Union:
        return any(typecheck(arg, value) for arg in get_args(tp))
    elif orig in _JSONArrayTypes:
        if orig is tuple:
            args = get_args(tp)
            return (
                isinstance(value, _JSONArrayTypes)
                and len(value) == len(args)
                and all(typecheck(val, arg) for val, arg in zip(value, args))
            )
        elif orig is list:
            tp = get_args(tp)[0]
            return all(typecheck(val, tp) for val in value)
        raise JSONError(f"[BUG] Unhandled typecheck {tp}")
    elif orig in _JSONObjectTypes:
        if orig is dict:
            tp = get_args(tp)[1]
            return all(
                isinstance(key, str) and typecheck(val, tp)
                for key, val in value.items()
            )
        raise JSONError(f"[BUG] Unhandled typecheck {tp}")
    raise TypeHintError(tp, message="Unhandled type hint {type_hint} during type_check")


########################
# TYPE-HINTS UTILITIES #
########################


def _make_normalize():
    lock = RLock()
    guard = frozenset()

    @lru_cache
    def normalize_json_tp(tp):
        """Normalize a type hint for JSON values

        Arguments
            type_hint -- JSON type hint to normalize
        
        Returns
            normalized representation of type_hint

        Raises
            TypeHintError when type_hint is invalid in JSON
        """
        nonlocal lock, guard
        from .jsonize import JSONable  # Avoid circular deps

        with lock:
            if tp in guard:
                return tp
            old_guard = guard
            try:
                guard = guard | {tp}
                if tp in _JSONValueTypes or (
                    isinstance(tp, type) and issubclass(tp, JSONable)
                ):
                    return tp
                elif (orig := get_origin(tp)) is Literal:
                    if all(isinstance(lit, _JSONLiteralTypes) for lit in get_args(tp)):
                        return tp
                elif orig is Union:
                    lits, others = split(
                        lambda tp: get_origin(tp) is Literal,
                        (normalize_json_tp(tp) for tp in get_args(tp)),
                    )
                    if lits:
                        return Union[(Literal[sum(map(get_args, lits), ())], *others)]
                    return Union[others]
                elif orig in _JSONArrayTypes:
                    args = get_args(tp)
                    if orig is list or (len(args) > 1 and args[1] is Ellipsis):
                        return List[normalize_json_tp(args[0])]
                    return Tuple[tuple(normalize_json_tp(arg) for arg in args)]
                elif orig in _JSONObjectTypes:
                    args = get_args(tp)
                    if args[0] is str:
                        return Dict[str, normalize_json_tp(args[1])]
                raise TypeHintError(tp)  # handles all case that didn't return
            finally:
                guard = old_guard

    return normalize_json_tp


normalize_json_tp = _make_normalize()
del _make_normalize


def is_json_type_hint(type_hint: TypeHint) -> bool:
    """Check that the type_hint is valid for JSON

    Supports JSONable subclasses. Implemented by calling
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


# TODO: make recursive type proof !
@lru_cache
def have_common_value(ltp: TypeHint, rtp: TypeHint) -> bool:
    """Check if two JSON tye hints have common values
    
    Type hints must be normalized
    """
    from .jsonize import JSONable  # Avoid circular deps

    lorig = get_origin(ltp)
    rorig = get_origin(rtp)
    largs = get_args(ltp)
    rargs = get_args(rtp)

    if lorig is Literal or rorig is Literal:
        if lorig is Literal and rorig is Literal:
            return bool(set(largs) & set(rargs))
        values = largs if lorig is Literal else rargs
        type_ = ltp if lorig is Literal else rtp
        return any(typecheck(v, type_) for v in values)
    if ltp in _JSONValueTypes and rtp in _JSONValueTypes:
        return ltp == rtp or (ltp in (int, float) and rtp in (int, float))
    elif (
        isinstance(ltp, type)
        and isinstance(rtp, type)
        and issubclass(ltp, JSONable)
        and issubclass(rtp, JSONable)
    ):
        return issubclass(ltp, rtp) or issubclass(rtp, ltp)
    elif lorig is rorig is Union:
        return any(have_common_value(lsub, rsub) for lsub in largs for rsub in rargs)
    elif lorig in _JSONArrayTypes and rorig in _JSONArrayTypes:
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
