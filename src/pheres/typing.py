"""
JSON typing utilities

Part of the Pheres package
"""

import functools
import itertools
import typing
from abc import ABC
from threading import RLock
from typing import Any, Dict, List, Literal, Tuple, Type, TypeVar, Union, get_origin

from .datatypes import PHERES_ATTR, ArrayData, DictData, ObjectData, ValueData
from .exceptions import CycleError, JSONValueError, PheresInternalError, TypeHintError
from .utils import (
    AnyClass,
    TypeHint,
    Virtual,
    get_args,
    get_outer_namespaces,
    on_success,
    split,
)

# Constants
MISSING = object()  # sentinel

_JSONLiteralTypes = (bool, int, str)
_JSONValueTypes = (type(None), bool, int, float, str)
_JSONArrayTypes = (tuple, list)
_JSONObjectTypes = (dict,)

##########################
# VIRTUAL JSONABLE CLASS #
##########################


class JsonableValue(ABC, Virtual):
    __slots__ = ("__weakref__",)

    @classmethod
    def __subclasshook__(cls, /, subclass: AnyClass) -> bool:
        return is_jsonable_value(subclass)


class JsonableArray(ABC, Virtual):
    __slots__ = ("__weakref__",)

    @classmethod
    def __subclasshook__(cls, /, subclass: AnyClass) -> bool:
        return is_jsonable_array(subclass)


class JsonableDict(ABC, Virtual):
    __slots__ = ("__weakref__",)

    @classmethod
    def __subclasshook__(cls, /, subclass: AnyClass) -> bool:
        return is_jsonable_object(subclass)


class JsonableObject(ABC, Virtual):
    __slots__ = ("__weakref__",)

    @classmethod
    def __subclasshook__(cls, /, subclass: AnyClass) -> bool:
        return is_jsonable_class(subclass)


###################
# JSON TYPE HINTS #
###################

JSONValue = Union[  # pylint: disable=unsubscriptable-object
    None, bool, int, float, str, JsonableValue
]
JSONArray = Union[  # pylint: disable=unsubscriptable-object
    List["JSONType"], JsonableArray
]
JSONObject = Union[  # pylint: disable=unsubscriptable-object
    Dict[str, "JSONType"], JsonableDict, JsonableObject
]
JSONType = Union[  # pylint: disable=unsubscriptable-object
    JSONValue, JSONArray, JSONObject
]


########################
# TYPE-HINTS UTILITIES #
########################


@functools.lru_cache
def normalize_hint(tp: TypeHint, *, globalns=None, localns=None):
    """Normalize a JSON type hint

    Arguments
        tp -- JSON type hint to normalize

    Returns
        normalized representation of tp

    Raises
        TypeHintError when tp or an inner type is not a valid JSON type
    """
    if globalns is None and localns is None:
        globalns, localns = get_outer_namespaces()

    with normalize_hint._lock:
        if tp in normalize_hint._guard:
            return tp
        old_guard = normalize_hint._guard
        try:
            normalize_hint._guard = normalize_hint._guard | {tp}
            if isinstance(tp, type):
                if tp in (JsonableValue, JsonableArray, JsonableDict, JsonableObject):
                    return tp
                elif is_jsonable_class(tp):
                    return tp
                elif tp in _JSONValueTypes:
                    return tp
            elif (orig := get_origin(tp)) is Literal:
                if all(
                    isinstance(lit, _JSONLiteralTypes)
                    for lit in get_args(tp, globalns=globalns, localns=localns)
                ):
                    return tp
            elif orig is Union:
                others, lits = split(
                    lambda tp: get_origin(tp) is Literal,
                    (
                        normalize_hint(tp, globalns=globalns, localns=localns)
                        for tp in get_args(tp, globalns=globalns, localns=localns)
                    ),
                )
                if lits:
                    lits = sum(map(get_args, lits), ())
                    return Union[  # pylint: disable=unsubscriptable-object
                        (
                            Literal[lits],  # pylint: disable=unsubscriptable-object
                            *others,
                        )
                    ]
                return Union[others]  # pylint: disable=unsubscriptable-object
            elif isinstance(orig, type) and issubclass(orig, _JSONArrayTypes):
                args = get_args(tp, globalns=globalns, localns=localns)
                if orig is list or (len(args) > 1 and args[1] is Ellipsis):
                    return List[
                        normalize_hint(args[0], globalns=globalns, localns=localns)
                    ]
                return Tuple[
                    tuple(
                        normalize_hint(arg, globalns=globalns, localns=localns)
                        for arg in args
                    )
                ]
            elif isinstance(orig, type) and issubclass(orig, _JSONObjectTypes):
                args = get_args(tp, globalns=globalns, localns=localns)
                if args[0] is str:
                    return Dict[
                        str, normalize_hint(args[1], globalns=globalns, localns=localns)
                    ]
            raise TypeHintError(tp)  # handles all case that didn't return
        finally:
            normalize_hint._guard = old_guard


normalize_hint._lock = RLock()
normalize_hint._guard = frozenset()


def is_json_type_hint(type_hint: TypeHint, *, globalns=None, localns=None) -> bool:
    """Check that the type_hint is valid for JSON

    Supports JSONable subclasses. Implemented by calling
    normalize_json_type() and catching TypeHintError

    Arguments
        type_hint : type hint to test

    Return
        True if the type hint is valid for JSON, false otherwise
    """
    if globalns is None and localns is None:
        globalns, localns = get_outer_namespaces()
    try:
        normalize_hint(type_hint, globalns=globalns, localns=localns)
        return True
    except TypeHintError:
        return False


@functools.lru_cache
def find_collision(ltp: TypeHint, rtp: TypeHint) -> bool:
    """
    Finds a JSON common for the two type hints, or return MISSING

    The type hints must be in normalized form

    Arguments
        ltp -- left type hint
        rtp -- right type hint

    Returns
        value, such that typecheck(value, ltp) and typecheck(value, rtp)
        or MISSING if no such value exists
    """

    # early unpacking
    new_ltp, new_rtp = ltp, rtp
    while is_jsonable_class(new_ltp):
        tp = getattr(new_ltp, PHERES_ATTR).type
        if tp is not new_ltp:
            new_ltp = tp
        else:
            break
    while is_jsonable_class(new_rtp):
        tp = getattr(new_rtp, PHERES_ATTR).type
        if tp is not new_rtp:
            new_rtp = tp
        else:
            break
    if new_ltp is not ltp or new_rtp is not rtp:
        return find_collision(new_ltp, new_rtp)

    lorig = get_origin(ltp)
    rorig = get_origin(rtp)
    largs = get_args(ltp)
    rargs = get_args(rtp)

    # Literals
    if lorig is Literal or rorig is Literal:
        if lorig is Literal and rorig is Literal:
            intersect = set(largs) & set(rargs)
            return intersect.pop() if intersect else MISSING
        values = largs if lorig is Literal else rargs
        type_ = ltp if lorig is Literal else rtp
        for v in values:
            if typecheck(v, type_):
                return v
    # Unions
    if lorig is Union:
        for arg in largs:
            if (collision := find_collision(arg, rtp)) is not MISSING:
                return collision
    if rorig is Union:
        for arg in rargs:
            if (collision := find_collision(ltp, arg)) is not MISSING:
                return collision
    # Values
    if ltp in _JSONValueTypes and rtp in _JSONValueTypes:
        if ltp == rtp:
            return ltp()
    # Arrays
    elif lorig in _JSONArrayTypes and rorig in _JSONArrayTypes:
        lvariadic = lorig is list or (len(largs) == 2 and largs[1] == Ellipsis)
        rvariadic = rorig is list or (len(rargs) == 2 and rargs[1] == Ellipsis)
        if lvariadic and rvariadic:
            return []
        else:
            if lvariadic == rvariadic and len(largs) != len(rargs):
                return MISSING
            array = []
            for lsub, rsub in zip(
                itertools.repeat(largs[0]) if lvariadic else largs,
                itertools.repeat(rargs[0]) if rvariadic else rargs,
            ):
                collision = find_collision(lsub, rsub)
                if collision is MISSING:
                    return MISSING
                array.append(collision)
            return array
    # Objects
    elif lorig is dict or rorig is dict:
        if rorig is dict:
            ltp, lorig, largs, rtp, rorig, rargs = rtp, rorig, rargs, ltp, lorig, largs
        if is_jobject_class(rtp):
            data: ObjectData = getattr(rtp, PHERES_ATTR)
            obj = {}
            type_ = largs[1]
            for jattr in data.req_jattrs:
                collision = find_collision(type_, jattr.type)
                if collision is MISSING:
                    return MISSING
                obj[jattr.py_name] = collision
            return obj
        return {} if rorig is dict else MISSING
    elif is_jobject_class(ltp) and is_jobject_class(rtp):
        ldata, rdata = getattr(ltp, PHERES_ATTR), getattr(rtp, PHERES_ATTR)
        # TODO
        raise NotImplementedError
    return MISSING


########################
# TYPED JSON UTILITIES #
########################


def is_number(obj: Any) -> bool:
    """
    Test if the JSON object is a JSON number

    Argument
        obj -- object to test the type of

    Return
        True if the object is a json number, False otherwise
    """
    return type(obj) in (int, float)


def is_value(obj: Any) -> bool:
    """
    Test if the JSON object is a JSON value

    Argument
        obj -- object to test the type of

    Return
        True if the object is a json value, False otherwise
    """
    try:
        return typeof(obj) is JSONValue
    except JSONValueError:
        return False


def is_array(obj: Any) -> bool:
    """
    Test if the JSON object is a JSON Array

    Argument
        obj -- object to test the type of

    Return
        True if the object is a json array, False otherwise
    """
    try:
        return typeof(obj) is JSONArray
    except JSONValueError:
        return False


def is_object(obj: Any) -> bool:
    """
    Test if the JSON object is a JSON Object

    Argument
        obj -- object to test the type of

    Return
        True if the object is a json Object, False otherwise
    """
    try:
        return typeof(obj) is JSONObject
    except JSONValueError:
        return False


def is_jsonable_value(obj: Any) -> bool:
    """
    Return True if obj is a jsonable value
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return isinstance(getattr(cls, PHERES_ATTR, None), ValueData)


def is_jvalue_class(cls: Any) -> bool:
    """Return True if cls is a jsonable value class"""
    return isinstance(cls, type) and isinstance(
        getattr(cls, PHERES_ATTR, None), ValueData
    )


def is_jvalue_instance(obj: Any) -> bool:
    """Return True if obj a jsonable value instance"""
    return isinstance(getattr(type(obj), PHERES_ATTR, None), ValueData)


def is_jsonable_array(obj: Any) -> bool:
    """
    Return True if obj is a jsonable array
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return isinstance(getattr(cls, PHERES_ATTR, None), ArrayData)


def is_jarray_class(cls: Any) -> bool:
    """Return True if cls if a jsonable array class"""
    return isinstance(cls, type) and isinstance(
        getattr(cls, PHERES_ATTR, None), ArrayData
    )


def is_jarray_instance(obj: Any) -> bool:
    """Return True if obj is a jsonable array instance"""
    return isinstance(getattr(type(obj), PHERES_ATTR, None), ArrayData)


def is_jsonable_dict(obj: Any) -> bool:
    """
    Return True if obj is a jsonable dict
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return isinstance(getattr(cls, PHERES_ATTR, None), DictData)


def is_jdict_class(cls: Any) -> bool:
    """Return True if cls is a jsonable dict class"""
    return isinstance(cls, type) and isinstance(
        getattr(cls, PHERES_ATTR, None), DictData
    )


def is_jdict_instance(obj: Any) -> bool:
    """Return True if obj is a jsonable dict instance"""
    return isinstance(getattr(type(obj), PHERES_ATTR, None), DictData)


def is_jsonable_object(obj: Any) -> bool:
    """
    Return True if obj is a jsonable object
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return isinstance(getattr(cls, PHERES_ATTR, None), ObjectData)


def is_jobject_class(cls: Any) -> bool:
    """Return True if cls is a jsonable object class"""
    return isinstance(cls, type) and isinstance(
        getattr(cls, PHERES_ATTR, None), ObjectData
    )


def is_jobject_instance(obj: Any) -> bool:
    """Return True if obj is a jsonable object instance"""
    return isinstance(getattr(type(obj), PHERES_ATTR, None), ObjectData)


def is_jsonable(obj: Any) -> bool:
    """
    Return True if obj is a jsonable value or an instance of a jsonable
    value
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, PHERES_ATTR)


def is_jsonable_class(cls: Any) -> bool:
    """Return True if cls is a jsonable class"""
    return isinstance(cls, type) and hasattr(cls, PHERES_ATTR)


def is_jsonable_instance(obj: Any) -> bool:
    """Return True if obj is a jsonable instance"""
    return hasattr(type(obj), PHERES_ATTR)


def _is_json(obj: Any, rec_guard: Tuple[Any]) -> bool:
    """internal helper to check if object is valid JSON

    Has a guard to prevent infinite recursion"""

    if obj in rec_guard:
        raise CycleError(obj, rec_guard[rec_guard.index(obj) :])
    if isinstance(obj, _JSONValueTypes) or is_jsonable_instance(obj):
        return True
    elif isinstance(obj, _JSONArrayTypes):
        rec_guard = (*rec_guard, obj)
        return all(_is_json(elem, rec_guard) for elem in obj)
    elif isinstance(obj, _JSONObjectTypes):
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


def typeof(obj: JSONType) -> TypeHint:
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
        JSONValueError if the passed object is not a valid JSON
    """

    if obj is None or isinstance(obj, _JSONValueTypes) or is_jvalue_instance(obj):
        return JSONValue
    elif isinstance(obj, _JSONArrayTypes) or is_jarray_instance(obj):
        return JSONArray
    elif (
        isinstance(obj, _JSONObjectTypes)
        or is_jdict_instance(obj)
        or is_jobject_instance(obj)
    ):
        return JSONObject
    raise JSONValueError(obj)


def typecheck(value: JSONType, tp: TypeHint) -> bool:
    """Test that a JSON value matches a JSON type hint

    The type hint must be normalized (see normalize_hint()). This
    function may fail or return a wrong result if it is not

    Arguments:
        value -- value to test
        tp -- type hint to check against. Must be normalized

    Returns
        True if the value is valid for the type hint, False otherwise

    Raises
        TypeHintError if the type hint could not be handled
    """
    # Jsonables
    if isinstance(tp, type):
        if tp is JsonableValue:
            return is_jvalue_instance(value)
        elif tp is JsonableArray:
            return is_jarray_instance(value)
        elif tp is JsonableDict:
            return is_jdict_instance(value)
        elif tp is JsonableObject:
            return is_jobject_instance(value)
        elif is_jsonable_class(tp):
            return isinstance(value, tp)
    # Literal
    elif (orig := get_origin(tp)) is Literal:
        return value in get_args(tp)
    # Union
    elif orig is Union:
        return any(typecheck(value, arg) for arg in get_args(tp))
    # Values
    elif tp in _JSONValueTypes:
        return isinstance(value, tp)
    # Arrays
    elif orig in _JSONArrayTypes:
        if not isinstance(value, _JSONArrayTypes):
            return False
        if orig is tuple:
            args = get_args(tp)
            return len(value) == len(args) and all(
                typecheck(val, arg) for val, arg in zip(value, args)
            )
        elif orig is list:
            tp = get_args(tp)[0]
            return all(typecheck(val, tp) for val in value)
        raise PheresInternalError(
            f"Unhandled array type {tp} in typecheck(). This is a bug"
        )
    # Objects
    elif orig in _JSONObjectTypes:
        if orig is dict:
            if not isinstance(value, dict):
                return False
            tp = get_args(tp)[1]
            return all(
                isinstance(key, str) and typecheck(val, tp)
                for key, val in value.items()
            )
        raise PheresInternalError(
            f"Unhandled object type {tp} in typecheck(). This is a bug"
        )
    raise TypeHintError(tp)
