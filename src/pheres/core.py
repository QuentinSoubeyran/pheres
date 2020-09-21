# -*- coding: utf-8 -*-
"""
Typing of JSON and simple (de)serialization (from)to JSON in python

Part of the Pheres package
"""
# stdlib import
from abc import ABCMeta, ABC, abstractmethod
from copy import deepcopy
import dataclasses
from dataclasses import dataclass, field, replace
from enum import Enum
import functools
from itertools import repeat
import json
from threading import RLock
import types
import typing
from typing import (
    AbstractSet,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    MutableSet,
    Set,
    Type,
    TypeVar,
    Tuple,
    Union,
    get_origin,
    overload,
)
from weakref import WeakSet

# local imports
from .misc import (
    # Exception
    JSONError,
    # Classes
    AutoFormatMixin,
    Subscriptable,
    SmartDecoder,
    # Functions
    classproperty,
    on_success,
    on_error,
    split,
)

__all__ = [
    # Types
    "JSONValue",
    "JSONArray",
    "JSONObject",
    "JSONType",
    "TypeHint",
    # Errors
    "JSONTypeError",
    "TypeHintError",
    "CycleError",
    "JSONableError",
    "JAttrError",
    # Type utilities
    "typeof",
    "is_json",
    "is_number",
    "is_value",
    "is_array",
    "is_object",
    "typecheck",
    # Type Hint utilities
    "normalize_json_type",
    "is_json_type_hint",
    "have_common_value",
    "register_forward_ref",
    # Jsonable API
    "JSONable",
    "JSONAttr",
    "JAttr",
    "jattr",
    "jsonable",
    ## Serialization
    "JSONableEncoder",
    "dump",
    "dumps",
    ## Deserialization
    "SmartDecoder",
    "jsonable_hook",
    "load",
    "loads",
]

# Constant
_JSONLiteralTypes = (bool, int, str)
_JSONValueTypes = (type(None), bool, int, float, str)
_JSONArrayTypes = (tuple, list)
_JSONObjectTypes = (dict,)

# Type hint aliases for JSONget_args
JSONValue = Union[  # pylint: disable=unsubscriptable-object
    None, bool, int, float, str, "_VirtualValue"
]
JSONArray = Union[  # pylint: disable=unsubscriptable-object
    Tuple["JSONType", ...], List["JSONType"], "_VirtualArray"
]
JSONObject = Union[  # pylint: disable=unsubscriptable-object
    Dict[str, "JSONType"], "_VirtualObject", "_VirtualClass"
]
JSONLiteral = Union[bool, int, str]  # pylint: disable=unsubscriptable-object
JSONType = Union[  # pylint: disable=unsubscriptable-object
    JSONValue, JSONArray, JSONObject
]

# Type aliases for this module
_TypeHint_Types = (type, type(Union), type(Type), type(List))
TypeHint = Union[_TypeHint_Types]  # pylint: disable=unsubscriptable-object

# Annotations to force resolutions of the types hints
# Used in decoder.py to get the resolved types
_jval: JSONValue
_jarr: JSONArray
_jobj: JSONObject
_jtyp: JSONType

# Sentinel used in this module
MISSING = object()

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


class ForwardRefConflictError(AutoFormatMixin, JSONError):
    """
    Raised when registering a forward reference name that already exists

    Attributes
        name -- name of the foward reference
        type_ -- registered type
        new_type -- type attempted to be registered under 'name'
        message -- explanation of the error
    """

    def __init__(
        self,
        name,
        type_,
        new_type,
        message="Cannot register '{new_type}', name '{name}' is already used by '{type_}'",
    ):
        super().__init__(message)
        self.name = name
        self.type_ = type_
        self.new_type = type_


class JSONableError(JSONError):
    """
    Base exception for problem with the JSONable ABC or the jsonize decorator
    """


class JAttrError(JSONableError):
    """
    Exception for when an attribute is improperly jsonized
    """


####################
# MODULE UTILITIES #
####################


def register_forward_ref(name: str, type_hint: TypeHint):
    """
    Register a type under a name for use in FowardRef in JSONable classes

    Arguments
        name -- string name to use for the type
        type_ -- type matching the name

    Raises
        ValueError if 'name' is already used
    """
    with register_forward_ref._lock, on_success(jsonable._register_delayed_classes):
        if (type_ := register_forward_ref._table.get(name, MISSING)) is not MISSING:
            raise ForwardRefConflictError(name=name, type_=type_, new_type=type_hint)
        register_forward_ref._table[name] = normalize_json_type(type_hint)


register_forward_ref._lock = RLock()
register_forward_ref._table = {
    "JSONValue": JSONValue,
    "JSONArray": JSONArray,
    "JSONObject": JSONObject,
    "JSONType": JSONType,
}


def unregister_forward_ref(name: str):
    """
    Unregister a type from pheres forward references

    Arguments
        name -- the string name of the type to unregister
    """
    with register_forward_ref._lock:
        if name in register_forward_ref._table and name not in ("JSONType", "JSONable"):
            del register_forward_ref._table[name]


def _resolve_refs(tp):
    if isinstance(tp, typing.ForwardRef):
        with register_forward_ref._lock:
            return register_forward_ref._table.get(tp.__forward_arg__, tp)
    return tp


# Proactively resolve ForwardRef
@functools.wraps(typing.get_args)
def get_args(type_hint):
    return tuple(_resolve_refs(tp) for tp in typing.get_args(type_hint))


##################
# TYPE UTILITIES #
##################

# Adapted version of typing._type_repr
def _type_repr(tp):
    """Return the repr() of objects, special casing types and tuples"""
    if isinstance(tp, tuple):
        return ", ".join(map(_type_repr, tp))
    if isinstance(tp, type):
        if tp.__module__ == "builtins":
            return tp.__qualname__
        return f"{tp.__module__}.{tp.__qualname__}"
    if tp is Ellipsis:
        return "..."
    if isinstance(tp, types.FunctionType):
        return tp.__name__
    return repr(tp)


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
        JSONTypeError if the passed object is not a valid JSON
    """

    if (
        obj is None
        or isinstance(obj, _JSONValueTypes)
        or isinstance(obj, _VirtualValue)
    ):
        return JSONValue
    elif isinstance(obj, (*_JSONArrayTypes, _VirtualArray)):
        return JSONArray
    elif isinstance(obj, (*_JSONObjectTypes, _VirtualObject, _VirtualClass)):
        return JSONObject
    raise JSONTypeError(obj)


def _is_json(obj: Any, rec_guard: Tuple[Any]) -> bool:
    """internal helper to check if object is valid JSON

    Has a guard to prevent infinite recursion"""

    if obj in rec_guard:
        raise CycleError(obj, rec_guard[rec_guard.index(obj) :])
    type_ = type(obj)
    if type_ in _JSONValueTypes:
        return True
    elif type_ in JSONable.registry:
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


def typecheck(value: JSONType, tp: TypeHint) -> bool:
    """Test that a JSON value matches a JSON type hint

    Arguments
        tp -- type hint to check against. Must be normalized
        value -- value to test

    Return
        True if the value is valid for the type hint, False otherwise

    Raises
        TypeHintError if the type hint could not be handled
    """

    # Values
    if tp in _JSONValueTypes:
        return isinstance(value, tp)
    # JSONable
    elif isinstance(tp, type) and tp in JSONable.registry:
        return isinstance(value, tp)
    # Literal
    elif (orig := get_origin(tp)) is Literal:
        return value in get_args(tp)
    # Union
    elif orig is Union:
        return any(typecheck(value, arg) for arg in get_args(tp))
    # Arrays
    elif orig in _JSONArrayTypes:
        if not isinstance(value, _JSONArrayTypes):
            return False
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
        raise JSONError(f"[BUG] Unhandled typecheck {tp}")
    raise TypeHintError(tp, message="Unhandled type hint {type_hint} during type_check")


########################
# TYPE-HINTS UTILITIES #
########################


@functools.lru_cache
def normalize_json_type(tp: TypeHint):
    """Normalize a JSON type hint

    Arguments
        tp -- JSON type hint to normalize

    Returns
        normalized representation of tp

    Raises
        TypeHintError when tp or an inner type is not a valid JSON type
    """

    # if isinstance(tp, typing.ForwardRef):
    #    pass

    with normalize_json_type._lock:
        if tp in normalize_json_type._guard:
            return tp
        old_guard = normalize_json_type._guard
        try:
            normalize_json_type._guard = normalize_json_type._guard | {tp}
            # Avoid calls to issubclass to prevent ABCMeta
            # from caching the result
            # Allows _Registry.discard() to work
            # when @jsonable recovers from an error
            if tp in (JSONValue, JSONArray, JSONObject, JSONType):
                return tp
            if tp in _JSONValueTypes:
                return tp
            if isinstance(tp, type):
                if (
                    tp
                    in JSONable.registry  # pylint: disable=unsupported-membership-test
                    or tp in jsonable._delayed
                ):
                    return tp
                elif tp in _VirtualClsEnum.value_set:
                    return tp
            elif (orig := get_origin(tp)) is Literal:
                if all(isinstance(lit, _JSONLiteralTypes) for lit in get_args(tp)):
                    return tp
            elif orig is Union:
                others, lits = split(
                    lambda tp: get_origin(tp) is Literal,
                    (normalize_json_type(tp) for tp in get_args(tp)),
                )
                if lits:
                    return Union[  # pylint: disable=unsubscriptable-object
                        (
                            Literal[  # pylint: disable=unsubscriptable-object
                                sum(map(get_args, lits), ())
                            ],
                            *others,
                        )
                    ]
                return Union[others]  # pylint: disable=unsubscriptable-object
            elif isinstance(orig, type) and issubclass(orig, _JSONArrayTypes):
                args = get_args(tp)
                if orig is list or (len(args) > 1 and args[1] is Ellipsis):
                    return List[normalize_json_type(args[0])]
                return Tuple[tuple(normalize_json_type(arg) for arg in args)]
            elif isinstance(orig, type) and issubclass(orig, _JSONObjectTypes):
                args = get_args(tp)
                if args[0] is str:
                    return Dict[str, normalize_json_type(args[1])]
            raise TypeHintError(tp)  # handles all case that didn't return
        finally:
            normalize_json_type._guard = old_guard


normalize_json_type._lock = RLock()
normalize_json_type._guard = frozenset()


def is_json_type_hint(type_hint: TypeHint) -> bool:
    """Check that the type_hint is valid for JSON

    Supports JSONable subclasses. Implemented by calling
    normalize_json_type() and catching TypeHintError

    Arguments
        type_hint : type hint to test

    Return
        True if the type hint is valid for JSON, false otherwise
    """
    try:
        normalize_json_type(type_hint)
        return True
    except TypeHintError:
        return False


# TODO: make recursive type proof !
@functools.lru_cache
def have_common_value(ltp: TypeHint, rtp: TypeHint) -> bool:
    """Check if two JSON tye hints have common values

    Type hints must be normalized
    """

    # early unpacking
    new_ltp, new_rtp = ltp, rtp
    while new_ltp in (
        _VirtualValue.registry | _VirtualArray.registry | _VirtualObject.registry
    ):
        new_ltp = new_ltp.type
    while new_rtp in (
        _VirtualValue.registry | _VirtualArray.registry | _VirtualObject.registry
    ):
        new_rtp = new_rtp.type
    if new_ltp is not ltp or new_rtp is not rtp:
        return have_common_value(new_ltp, new_rtp)

    lorig = get_origin(ltp)
    rorig = get_origin(rtp)
    largs = get_args(ltp)
    rargs = get_args(rtp)

    # Literals
    if lorig is Literal or rorig is Literal:
        if lorig is Literal and rorig is Literal:
            return bool(set(largs) & set(rargs))
        values = largs if lorig is Literal else rargs
        type_ = ltp if lorig is Literal else rtp
        return any(typecheck(v, type_) for v in values)
    # Unions
    if lorig is Union:
        return any(have_common_value(larg, rtp) for larg in largs)
    if rorig is Union:
        return any(have_common_value(ltp, rarg) for rarg in rargs)
    # Values
    if ltp in _JSONValueTypes and rtp in _JSONValueTypes:
        return ltp == rtp or (ltp in (int, float) and rtp in (int, float))
    # Arrays
    elif lorig in _JSONArrayTypes and rorig in _JSONArrayTypes:
        lvariadic = lorig is list or (len(largs) == 2 and largs[1] == Ellipsis)
        rvariadic = rorig is list or (len(rargs) == 2 and rargs[1] == Ellipsis)
        if lvariadic and rvariadic:
            # return have_common_value(largs[0], rargs[0])
            return True  # empty array is valid for both
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
    # Objects
    elif lorig is dict:
        if rtp in _VirtualClass.registry:
            # Required Jsonized attributes are sufficient for conflicts
            return all(
                have_common_value(largs[1], jattr.type_hint)
                for jattr in rtp._REQ_JATTRS.values()
            )
        return rorig is dict
    elif rorig is dict:
        if ltp in _VirtualClass.registry:
            return all(
                have_common_value(jattr.type_hint, rargs[1])
                for jattr in ltp._REQ_JATTRS.values()
            )
        return lorig is dict
    elif ltp in _VirtualClass.registry and rtp in _VirtualClass.registry:
        # even if called during conflict checking in @jsonable
        # the method has been added to the class already
        ltp._pheres_conflict_with(rtp)
    # Default - no conflict have been found
    return False


################
# JSONABLE API #
################

AnyClass = TypeVar("AnyClass", bound=type)
T = TypeVar("T", *_TypeHint_Types)


class _Registry(ABC):
    """Class to have Abstract Base Class with accessible and mutable registry"""

    __slots__ = ()

    registry: ClassVar[MutableSet[AnyClass]]

    def __init_subclass__(
        cls, /, *args, _registry: MutableSet[AnyClass] = None, **kwargs
    ) -> None:
        if _registry is None:
            raise TypeError("Cannot subclass special pheres classes")
        super().__init_subclass__(*args, **kwargs)
        assert issubclass(type(_registry), MutableSet)
        cls.registry = _registry

    @classmethod
    def __subclasshook__(cls, /, subclass: AnyClass):
        if subclass in cls.registry:
            return True
        return False

    @classmethod
    def register(cls, /, subclass: AnyClass) -> AnyClass:
        """Register the class as a subclass of this ABC.

        WARNING: cannot be used as a decorator
        Can be used as a context manager expression to un-register on exceptions
        """
        cls.registry.add(subclass)
        return on_error(cls.registry.discard, subclass)

    @classmethod
    def discard(cls, /, subclass: AnyClass) -> None:
        """Unregister a class from an ABC

        Must be called before any call to isinstance() or issubclass()
        on the subclass due to ABCMeta caching the result"""
        cls.registry.discard(subclass)


class _VirtualJSONableBase(ABC):
    """
    Internal class - provides common code for all JSONable virtual classes
    """

    @staticmethod
    @abstractmethod
    def make(cls, obj):
        """Create an instance of cls from the JSON object obj"""

    @classproperty
    @abstractmethod
    def type(cls, /):  # pylint: disable=no-self-argument
        """Return a type-hint for the JSON representation of the class"""

    @classmethod
    def from_json(cls, /, obj):
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
            from .decoder import deserialize

            return deserialize(obj, cls)


class _VirtualValue(_VirtualJSONableBase, _Registry, _registry=WeakSet()):
    """
    Abstract Base Class for JSONable values

    Obtained with e.g. @jsonable[int]
    """

    __slots__ = ()

    registry: ClassVar[MutableSet[AnyClass]]

    # virtual subclasses attributes
    # added by the @jsonable decorator
    _JTYPE: Union[(*_JSONValueTypes, Type[Union[_JSONValueTypes]])]

    @staticmethod
    def process_type(
        tp: TypeHint,
    ) -> Union[(*_JSONValueTypes, Type[Union[_JSONValueTypes]])]:
        tp = type(None) if tp is None else tp
        types = get_args(tp) if get_origin(tp) is Union else (tp,)
        if not all(
            isinstance((invalid := arg), type) and issubclass(arg, _JSONValueTypes)
            for arg in types
        ):
            raise TypeHintError(
                invalid,  # pylint: disable=undefined-variable
                message="{type_hint} is not a valid type hint for a Jsonable value",
            )
        return tp

    @staticmethod
    def make(cls, value):
        """Makes an instance of cls from a JSON value"""
        return cls(value)

    @classproperty
    def type(cls, /):  # pylint: disable=no-self-argument
        return cls._JTYPE

    def to_json(self):
        raise NotImplementedError(
            "You must implement the 'to_json' method of jsonable value"
        )

    @classmethod
    def conflict_with(cls, /, other):
        """Test if this JSONable could have common serialization with another class"""
        if not (isinstance(other, type) and other in JSONable.registry):
            raise TypeError("Can only check conflicts with JSONable classes")
        if other not in _VirtualValue.registry:
            return False
        return have_common_value(cls._JTYPE, other._JTYPE)


class _VirtualArray(_VirtualJSONableBase, _Registry, _registry=WeakSet()):
    """
    Abstract Base Class for JSONable arrays

    Obtained with e.g. @jsonable[int, ...]
    """

    __slots__ = ()

    registry: ClassVar[MutableSet[AnyClass]]

    # virtual subclasses attributes
    # added by the @jsonable decorator
    _JTYPE: Union[TypeHint, Tuple[TypeHint, ...]]

    @staticmethod
    def process_type(types) -> Union[TypeHint, Tuple[TypeHint, ...]]:
        if not isinstance(types, tuple):
            # Assume list or tuple, should be filtered by @jsonable
            types = get_args(types)
        if len(types) == 2 and types[1] is Ellipsis:
            # Variable length array
            return normalize_json_type(types[0])
        else:
            # fixed size array
            return tuple(map(normalize_json_type, types))

    @staticmethod
    def make(cls, array):
        """Internal method - makes an instance of cls from a JSON array"""
        return cls(*array)

    @classproperty
    def type(cls, /):  # pylint: disable=no-self-argument
        if isinstance(cls._JTYPE, tuple):
            return List[cls._JTYPE]
        else:
            return Tuple[cls._JTYPE]

    def to_json(self):
        raise NotImplementedError(
            "You must implement the 'to_json' method of jsonable array"
        )

    @classmethod
    def conflict_with(cls, /, other):
        """Test if this JSONable could have common serialization with another class"""
        if not (isinstance(other, type) and other in JSONable.registry):
            raise TypeError("Can only check conflicts with JSONable classes")
        if other not in _VirtualArray.registry:
            return False
        self_type = (
            Tuple[cls._JTYPE] if isinstance(cls._JTYPE, tuple) else List[cls._JTYPE]
        )
        other_type = (
            Tuple[other._JTYPE]
            if isinstance(other._JTYPE, tuple)
            else List[other._JTYPE]
        )
        return have_common_value(self_type, other_type)


class _VirtualObject(_VirtualJSONableBase, _Registry, _registry=WeakSet()):
    """
    Abstract Base Class for jsonable object
    """

    __slots__ = ()

    registry: ClassVar[MutableSet[AnyClass]]
    # virtual subclasses attributes
    # added by the @jsonable decorator
    _JTYPE: TypeHint

    @staticmethod
    def process_type(tp: TypeHint) -> TypeHint:
        return normalize_json_type(tp)

    @staticmethod
    def make(cls, obj):
        return cls(obj)

    @classproperty
    def type(cls, /):  # pylint: disable=no-self-argument
        return Dict[str, cls._JTYPE]

    def to_json(self) -> JSONObject:
        raise NotImplementedError(
            "You must implement the 'to_json' method of jsonable object"
        )

    @classmethod
    def conflict_with(cls, /, other):
        """Test if this jsonable could have common serialization with another class"""
        if not (isinstance(other, type) and other in JSONable.registry):
            raise TypeError("Can only check conflicts with JSONable classes")
        if other not in _VirtualObject.registry:
            return False
        return have_common_value(cls._JTYPE, other._JTYPE)


class _VirtualClass(_VirtualJSONableBase, _Registry, _registry=WeakSet()):
    """Abstract Base Class for JSONable objects"""

    __slots__ = ()

    registry: ClassVar[MutableSet[AnyClass]]

    # attributes defined by the @jsonable decorator
    Decoder: json.JSONDecoder
    _REQ_JATTRS = {}
    _ALL_JATTRS = {}

    @staticmethod
    def make(cls, obj):
        """
        Internal method - makes a instance of cls from the JSON object obj
        """
        return cls(
            **{
                jattr.py_name: (
                    obj[jattr.name] if jattr.name in obj else jattr.get_default()
                )
                for jattr in cls._ALL_JATTRS.values()
                if not jattr.json_only
            }
        )

    @classproperty
    def type(cls, /):  # pylint: disable=no-self-argument
        return cls

    def to_json(self, /, *, with_defaults=False):
        obj = {}
        for jattr in self._ALL_JATTRS.values():
            value = jattr.to_json(self, with_default=with_defaults)
            if value is not MISSING:
                obj[jattr.name] = (
                    value.to_json(with_defaults=with_defaults)
                    if isinstance(value, _VirtualClass)
                    else value
                )
        return obj

    @classmethod
    def _pheres_conflict_with(cls, /, other):
        return (
            _is_jattr_subset(cls._REQ_JATTRS, other._ALL_JATTRS)
            and _is_jattr_subset(other._REQ_JATTRS, cls._ALL_JATTRS)
            and not issubclass(cls, other)
        )

    @classmethod
    def conflict_with(cls, /, other):
        """Test if this JSONable class could have common serialization with another class"""
        if not (isinstance(other, type) and other, JSONable.registry):
            raise TypeError("Can only check conflicts with JSONable classes")
        if other not in _VirtualClass.registry:
            return False
        return cls._pheres_conflict_with(other)


class _VirtualClsEnum(Enum):
    Value = _VirtualValue
    Array = _VirtualArray
    Object = _VirtualObject
    Class = _VirtualClass


_VirtualClsEnum.value_set = frozenset(item.value for item in _VirtualClsEnum)


class JSONable(ABC):
    """Asbtract Base Class with stub methods for JSONable classes

    Can be used with isinstance() and issubclass() to test if a class is JSONable
    also useful for linters and type-checkers, as inheriting from that class
    defines stubs for the attributes and methods added by @jsonable
    """

    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, /, subclass) -> bool:
        return subclass in cls.registry

    # attributes defined by the @jsonable decorator
    Decoder: ClassVar[SmartDecoder] = SmartDecoder

    @classproperty
    @classmethod
    def registry(cls) -> Set[type]:
        return (
            _VirtualValue.registry
            | _VirtualArray.registry
            | _VirtualObject.registry
            | _VirtualClass.registry
        )

    @classmethod
    def from_json(cls: AnyClass, /, obj: Any) -> AnyClass:
        """
        Converts a JSON file, string or object to an instance of that class
        """
        raise NotImplementedError

    def to_json(self: AnyClass) -> JSONObject:
        """Converts an instance of that class to a JSON object"""
        raise NotImplementedError

    @classmethod
    def conflict_with(cls, /, other: "JSONable") -> bool:
        """Test if this JSONable could have common serialization with another class"""
        raise NotImplementedError


register_forward_ref._table["JSONable"] = JSONable


@dataclass(frozen=True)
class JSONAttr:
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


JAttr = Annotated[T, JSONAttr()]


def jattr(tp: TypeHint, /, **kwargs) -> TypeHint:
    """
    Shortcut for Annotated[T, JSONAttr(**kwargs)]

    See JSONAttr for a list of supported keyword arguments. Not compatible
    with type checkers due to being runtime
    """
    return Annotated[tp, JSONAttr(**kwargs)]


@dataclass(frozen=True)
class _JsonisedAttribute:
    """
    Internal class to represent a JSONized attribute

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
    json_only: bool = field(default=MISSING, init=True)

    def __post_init__(self, /) -> None:
        if callable(self.default) and not is_json((value := self.default())):
            raise JAttrError(
                f"A callable default must produce a valid JSON value, got {value}"
            )
        if self.json_only is MISSING:
            if (
                get_origin(self.type_hint) is Literal
                and len(get_args(self.type_hint)) == 1
            ):
                arg = get_args(self.type_hint)[0]
                default = self.default() if callable(self.default) else self.default
                if default is not MISSING and default != arg:
                    raise JAttrError(
                        f"Incoherent Literal and default for json-only attribute: {arg} != {default}"
                    )
                object.__setattr__(self, "json_only", True)
                if default is MISSING:
                    object.__setattr__(self, "default", arg)
            else:
                object.__setattr__(self, "json_only", False)
        if self.json_only and self.default is MISSING:
            raise JAttrError("json-only JSONized Attributes must have a default value")

    def overlaps(self, /, other: "_JsonKey") -> bool:
        """
        Check if the exists a json key-value accepted by both _JsonKey

        Arguments
            other : _JsonKey to check conflicts with

        Return
            True in case of conflict, False otherwise
        """
        return self.name == other.name and have_common_value(
            self.type_hint, other.type_hint
        )

    def get_default(self, /) -> JSONObject:
        """Return the default value for that attribute"""
        if callable(self.default):
            return self.default()
        return deepcopy(self.default)

    def to_json(
        self, /, instance: _VirtualClass, *, with_default: bool = False
    ) -> JSONObject:
        if self.json_only:
            return self.get_default()
        value = getattr(instance, self.py_name)
        if value == self.get_default() and not with_default:
            return MISSING
        return value


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
        for jattr in subset.values()
    )


def _register_class_ref(cls: AnyClass):
    with register_forward_ref._lock:
        if (
            type_ := register_forward_ref._table.get(cls.__name__, MISSING)
        ) is not MISSING:
            raise JSONableError(
                f"Cannot register new JSONable class '{cls.__name__}':"
                f"forwad reference '{cls.__name__}' points to '{type_}'"
            )
        register_forward_ref._table[cls.__name__] = cls


def _add_jsonable_members(virtual_class: _VirtualClsEnum, cls: AnyClass) -> None:
    from .decoder import TypedJSONDecoder

    for (name, obj) in (
        ("Decoder", TypedJSONDecoder[cls]),
        ("from_json", classmethod(virtual_class.from_json.__func__)),
        ("to_json", virtual_class.to_json),
        ("conflict_with", classmethod(virtual_class.conflict_with.__func__)),
        ("type", classproperty(virtual_class.__dict__["type"].fget)),
    ):
        # test if class define the method already
        # ignore base classes, so no hasattr()
        if name not in cls.__dict__:
            setattr(cls, name, obj)


def _get_standard_jattrs(
    cls: AnyClass, all_attrs: bool = True
) -> Dict[str, _JsonisedAttribute]:
    # Only consider attributes from _JSONableObject parent classes
    allowed_attributes = set(cls.__annotations__.keys())
    # Do not test self
    for parent in cls.__mro__[1:]:
        if parent in  _VirtualClass.registry:
            allowed_attributes |= {jattr.py_name for jattr in parent._ALL_JATTRS}
        elif parent in jsonable._delayed:
            raise JSONableError("Cannot register JSONable class before its JSONable parent classes")
    # Gather jsonized attributes
    jattrs = {}
    for py_name, tp in typing.get_type_hints(
        cls, localns={cls.__name__: cls}, include_extras=True
    ).items():
        if py_name not in allowed_attributes:
            continue
        name = py_name
        json_only = MISSING
        # Retrieve name, type hint and annotation
        if typing.get_origin(tp) is Annotated:
            tp, *args = typing.get_args(tp)
            if any(isinstance((found := arg), JSONAttr) for arg in args):
                name = found.key or name  # pylint: disable=undefined-variable
                json_only = found.json_only  # pylint: disable=undefined-variable
            elif not all_attrs:
                continue
        elif not all_attrs:
            continue
        # Check for name conflict
        if name in jattrs:
            raise JAttrError(
                f"Duplicated attribute name {name} in JSONable class {cls.__name__}"
            )
        # Get default value, handle dataclass fields
        default = getattr(cls, py_name, MISSING)
        if isinstance(default, dataclasses.Field):
            if default.default is not dataclasses.MISSING:
                default = default.default
            elif default.default_factory is not dataclasses.MISSING:
                default = default.default_factory
            else:
                default = MISSING
        # Create JSONized attribute
        jattrs[name] = _JsonisedAttribute(
            name=name,
            py_name=py_name,
            type_hint=normalize_json_type(tp),
            default=default,
            json_only=json_only,
        )
    return jattrs


def _add_dataclass_jattrs(
    cls: AnyClass, jattrs: Dict[str, _JsonisedAttribute], all_attrs: bool = True
) -> Dict[str, _JsonisedAttribute]:
    # TODO: May need filter parent class fields for jsonable class only
    for field in dataclasses.fields(cls):
        if (
            field.default is not dataclasses.MISSING
            or field.default_factory is dataclasses.MISSING
        ):
            continue
        tp = field.type
        name = field.name
        json_only = MISSING
        if typing.get_origin(tp) is Annotated:
            tp, *args = typing.get_args(tp)
            if any(isinstance((found := arg), JSONAttr) for arg in args):
                name = found.key or name  # pylint: disable=undefined-variable
                json_only = found.json_only  # pylint: disable=undefined-variable
            elif not all_attrs:
                continue
        elif not all_attrs:
            continue
        if name in jattrs:
            raise JAttrError(
                f"Duplicated attribute name {name} in JSONable class {cls.__name__}"
            )
        jattrs[name] = _JsonisedAttribute(
            name=name,
            py_name=field.name,
            type_hint=normalize_json_type(tp),
            default=field.default_factory,
            json_only=json_only,
        )
    return jattrs


def _get_jattr_dicts(
    cls: AnyClass, all_attrs: bool = True
) -> Tuple[Dict[str, _JsonisedAttribute], Dict[str, _JsonisedAttribute]]:
    all_jattrs = _get_standard_jattrs(cls, all_attrs)
    if dataclasses.is_dataclass(cls):
        all_jattrs = _add_dataclass_jattrs(cls, all_jattrs, all_attrs)
    req_jattrs = {
        name: jattr
        for name, jattr in all_jattrs.items()
        if jattr.default is MISSING or jattr.json_only
    }
    return req_jattrs, all_jattrs


def _check_object_conflicts(cls: AnyClass) -> None:
    for other in _VirtualClass.registry:
        if other._pheres_conflict_with(cls):
            raise JSONableError(
                f"JSONable class '{cls.__name__}' conflicts with "
                f"'{other.__name__}' without inheriting from it"
            )


def _clean_object_class(cls: AnyClass):
    """Remove json_only attributes from the class annotations and dict"""
    cls_annotations = cls.__dict__.get("__annotations__", {})
    for jattr in cls._REQ_JATTRS.values():
        if jattr.json_only:
            if dataclasses.is_dataclass(cls):
                raise JSONableError(
                    "json-only attributes requires that @jsonable is applied before @dataclass"
                )
            if hasattr(cls, jattr.py_name):
                delattr(cls, jattr.py_name)
            if jattr.py_name in cls_annotations:
                del cls_annotations[jattr.py_name]


def _decorate_jsonable_simple(
    virtual_class: _VirtualClsEnum, cls: AnyClass, type_hint: TypeHint
) -> None:
    with virtual_class.register(cls):
        cls._JTYPE = virtual_class.process_type(type_hint)
        _add_jsonable_members(virtual_class, cls)


def _decorate_jsonable_class(cls: AnyClass, all_attrs: bool = True) -> None:
    with _VirtualClass.register(cls):
        req_jattrs, all_jattrs = _get_jattr_dicts(cls, all_attrs)
        cls._REQ_JATTRS = req_jattrs
        cls._ALL_JATTRS = all_jattrs
        # Required for conflict checks in recursive scenarios
        setattr(
            cls,
            "_pheres_conflict_with",
            classmethod(_VirtualClass._pheres_conflict_with.__func__),
        )
        _check_object_conflicts(cls)
        _add_jsonable_members(_VirtualClass, cls)
        _clean_object_class(cls)


@dataclass(frozen=True)
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

    _delayed: ClassVar[  # pylint: disable=unsubscriptable-object
        Dict[type, Tuple[Set, Callable]]
    ] = {}

    all_attrs: bool = True
    after: Union[str, Iterable[str]] = ()  # pylint: disable=unsubscriptable-object
    virtual_class: _VirtualClsEnum = field(
        default=_VirtualClass, init=False, repr=False
    )
    type_hint: TypeHint = field(default=None, init=False, repr=False)

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
        if virtual is not None:
            object.__setattr__(self, "virtual_class", virtual)
        else:
            if isinstance(type_hint, tuple):
                object.__setattr__(self, "virtual_class", _VirtualArray)
            elif isinstance((orig := get_origin(type_hint)), type):
                args = get_args(type_hint)
                if issubclass(orig, (tuple, list)):
                    type_hint = (
                        args[0] if len(args) > 1 and args[1] is Ellipsis else args
                    )
                    object.__setattr__(self, "virtual_class", _VirtualArray)
                elif issubclass(orig, dict):
                    if args[0] is not str:
                        raise TypeHintError(type_hint)
                    type_hint = args[1]
                    object.__setattr__(self, "virtual_class", _VirtualObject)
            else:
                object.__setattr__(self, "virtual_class", _VirtualValue)
        object.__setattr__(self, "type_hint", type_hint)
        return self

    @classmethod
    def __class_getitem__(cls, /, type_hint):
        return functools.partial(cls._factory, type_hint, None)

    @Subscriptable
    def Value(tp):  # pylint: disable=no-self-argument
        return functools.partial(jsonable._factory, tp, _VirtualValue)

    @Subscriptable
    def Array(tp):  # pylint: disable=no-self-argument
        return functools.partial(jsonable._factory, tp, _VirtualArray)

    @Subscriptable
    def Object(tp):  # pylint: disable=no-self-argument
        return functools.partial(jsonable._factory, tp, _VirtualObject)

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
            if self.virtual_class in (_VirtualValue, _VirtualArray, _VirtualObject):
                decorate = functools.partial(
                    _decorate_jsonable_simple, self.virtual_class, cls, self.type_hint
                )
            elif self.virtual_class is _VirtualClass:
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

    @classmethod
    def _register_delayed_classes(cls, /) -> None:
        """Register the delayed classes that can be registered"""
        registered = []
        for delayed_cls, (after, decorate) in cls._delayed.items():
            if (after & register_forward_ref._table.keys()) == after:
                decorate()
                registered.append(delayed_cls)
        for registered_cls in registered:
            del cls._delayed[registered_cls]


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


def jsonable_hook(obj: dict) -> JSONObject:
    """
    Object hook for the json.load() and json.loads() methods to deserialize JSONable classes
    """
    classes = []
    for cls in _VirtualClass.registry:
        if all(  # all required arguments are there
            key in obj and typecheck(obj[key], jattr.type_hint)
            for key, jattr in cls._REQ_JATTRS.items()
        ) and all(  # all keys are valid - don't test req, already did
            key in cls._ALL_JATTRS
            and typecheck(obj[key], cls._ALL_JATTRS[key].type_hint)
            for key in obj.keys() - cls._REQ_JATTRS.items()
        ):
            classes.append(cls)
    # find less-specific class in case of inheritance
    classes = [
        cls
        for i, cls in enumerate(classes)
        if all(not issubclass(cls, next_cls) for next_cls in classes[i + 1 :])
    ]
    if len(classes) > 1:
        raise JSONError(
            f"[!! This is a bug !! Please report] Multiple valid JSONable class found while deserializing {obj}"
        )
    elif len(classes) == 1:
        return _VirtualClass.make(classes[0], obj)
    else:
        return obj


@functools.wraps(json.load)
def load(*args, object_hook=jsonable_hook, **kwargs):
    return json.load(*args, object_hook=object_hook, **kwargs)


@functools.wraps(json.loads)
def loads(*args, object_hook=jsonable_hook, **kwargs):
    return json.loads(*args, object_hook=object_hook, **kwargs)
