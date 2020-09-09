# -*- coding: utf-8 -*-
"""
Typing of JSON and simple (de)serialization (from)to JSON in python

Part of the Pheres package
"""
# stdlib import
from abc import ABC
from copy import deepcopy
import dataclasses
from dataclasses import dataclass, field
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
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Tuple,
    Union,
    get_origin,
)

# local imports
from .misc import AutoFormatMixin, JSONError, split

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
    "jsonable",
    "JSONAttr",
    "JAttr",
    "jattr",
    ## Serialization
    "JSONableEncoder",
    "dump",
    "dumps",
    ## Deserialization
    "jsonable_hook",
    "load",
    "loads",
]

# Type hint aliases for JSONget_args
JSONValue = Union[None, bool, int, float, str]
JSONArray = Union[Tuple["JSONType", ...], List["JSONType"]]
JSONObject = Union[Dict[str, "JSONType"], "JSONable"]
JSONLiteral = Union[bool, int, str]
JSONType = Union[JSONValue, JSONArray, JSONObject]

# Type aliases for this module
TypeHint = Union[type, type(Union), type(Type), type(List)]

# Constant
_JSONLiteralTypes = (bool, int, str)
_JSONValueTypes = (type(None), bool, int, float, str)
_JSONArrayTypes = (tuple, list)
_JSONObjectTypes = (dict,)

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

# Temporary function until it is created
def _resolve_refs(tp):
    raise RuntimeError("Function hasn't been initialized")


def register_forward_ref(name: str, type_: TypeHint):
    raise RuntimeError("Function hasn't been initialized")


# The functions will be created later as they needs JSONable
# to be defined
def _make_refs_funcs():
    lock = RLock()
    refs_table = {"JSONType": JSONType, "JSONable": JSONable}

    def _resolve_refs(tp):
        if isinstance(tp, typing.ForwardRef):
            with lock:
                return refs_table.get(tp.__forward_arg__, tp)
        return tp

    def register_forward_ref(name: str, type_hint: TypeHint):
        """
        Register a type under a name for use in FowardRef in JSONable classes

        Arguments
            name -- string name to use for the type
            type_ -- type matching the name

        Raises
            ValueError if 'name' is already used
        """
        with lock:
            if name in refs_table:
                raise ForwardRefConflictError(
                    name=name, type_=refs_table[name], new_type=type_hint
                )
            refs_table[name] = normalize_json_type(type_hint)

    return (_resolve_refs, register_forward_ref)


# Proactively resolve ForwardRef
@functools.wraps(typing.get_args)
def get_args(type_hint):
    return tuple(_resolve_refs(tp) for tp in typing.get_args(type_hint))


##################
# TYPE UTILITIES #
##################


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

    if obj is None or isinstance(obj, _JSONValueTypes):
        return JSONValue
    elif isinstance(obj, (list, tuple)):
        return JSONArray
    elif isinstance(obj, (dict, JSONable)):
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

    if tp in _JSONValueTypes:
        return isinstance(value, tp)
    elif isinstance(tp, type) and issubclass(tp, JSONable):
        return isinstance(value, tp)
    elif (orig := get_origin(tp)) is Literal:
        return value in get_args(tp)
    elif orig is Union:
        return any(typecheck(value, arg) for arg in get_args(tp))
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


def _make_normalize():
    lock = RLock()
    guard = frozenset()

    @functools.lru_cache
    def normalize_json_type(tp: TypeHint):
        """Normalize a type hint for JSON values

        Arguments
            type_hint -- JSON type hint to normalize

        Returns
            normalized representation of type_hint

        Raises
            TypeHintError when type_hint is invalid in JSON
        """
        nonlocal lock, guard

        if isinstance(tp, typing.ForwardRef):
            pass

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
                    others, lits = split(
                        lambda tp: get_origin(tp) is Literal,
                        (normalize_json_type(tp) for tp in get_args(tp)),
                    )
                    if lits:
                        return Union[(Literal[sum(map(get_args, lits), ())], *others)]
                    return Union[others]
                elif orig in _JSONArrayTypes:
                    args = get_args(tp)
                    if orig is list or (len(args) > 1 and args[1] is Ellipsis):
                        return List[normalize_json_type(args[0])]
                    return Tuple[tuple(normalize_json_type(arg) for arg in args)]
                elif orig in _JSONObjectTypes:
                    args = get_args(tp)
                    if args[0] is str:
                        return Dict[str, normalize_json_type(args[1])]
                raise TypeHintError(tp)  # handles all case that didn't return
            finally:
                guard = old_guard

    return normalize_json_type


normalize_json_type = _make_normalize()
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
        # TODO: improve that check
        return issubclass(ltp, rtp) or issubclass(rtp, ltp)
    elif lorig is rorig is Union:
        return any(have_common_value(lsub, rsub) for lsub in largs for rsub in rargs)
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
    elif lorig is rorig is dict:
        # return have_common_value(largs[1], rargs[1])
        return True  # empty dict is valid for both
    return False


#################
# JSONABLE  API #
#################


class JSONable(ABC):
    """Abstract class to represent objects that can be serialized and deserialized to JSON"""

    _REGISTRY = []  # List of registered classes
    _TEMP_REGISTRY = []  # JSONable subclasses under registration
    _REQ_JATTRS = {}  # Defined on subclasses/decorated classes
    _ALL_JATTRS = {}  # Defined on subclasses/decorated classes
    Decoder = json.JSONDecoder  # Defined on subclasses/decorated classes

    @classmethod
    def __subclasshook__(cls, /, subclass):
        if subclass in cls._TEMP_REGISTRY:
            return True
        return NotImplemented

    def __init_subclass__(cls, all_attrs: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if JSONable in cls.__bases__:
            _process_class(cls, all_attrs=all_attrs)

    def to_json(self, /, *, with_defaults=False):
        obj = {}
        for jattr in self._ALL_JATTRS.values():
            value = jattr.write_json(self, with_default=with_defaults)
            if value is not MISSING:
                obj[jattr.name] = (
                    value.to_json() if isinstance(value, JSONable) else value
                )
        return obj

    @classmethod
    def from_json(cls, /, obj):
        """Deserialize to an instance of the class this method is called on.

        Tries to guess how to deserialize in the following order:
         - if ``obj`` supports ``read``, use load()
         - if ``obj`` is a string or bytes, use loads()
         - else, serialize it to JSON and deserialize with loads()
         NOTE: object are serialized then deserialized because its easier to get exact error this way
        """
        if hasattr(obj, "read"):
            return json.load(obj, cls=cls.Decoder)
        elif isinstance(obj, (str, bytes)):
            return json.loads(obj, cls=cls.Decoder)
        else:
            return json.loads(dumps(obj), cls=cls.Decoder)


# Delayed utils, JSONable needed to be define
_resolve_refs, register_forward_ref = _make_refs_funcs()


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


T = TypeVar("T", JSONType, JSONable, covariant=True)
JAttr = Annotated[T, JSONAttr()]


def jattr(T, /, **kwargs) -> TypeHint:
    """
    Shortcut for Annotated[T, JSONAttr(**kwargs)]

    See JSONAttr for a list of supported keyword arguments. Not compatible
    with type checkers due to being runtime
    """
    return Annotated[T, JSONAttr(**kwargs)]


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

    def __post_init__(self, /, json_only=MISSING) -> None:
        if callable(self.default) and not is_json((value := self.default())):
            raise JAttrError(
                f"A callable default must produce a valid JSON value, got {value}"
            )
        if json_only is MISSING:
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

    def write_json(
        self, /, instance: JSONable, *, with_default: bool = False
    ) -> JSONObject:
        if self.json_only:
            return self.get_default()
        value = getattr(instance, self.py_name)
        if value == self.get_default() and not with_default:
            return MISSING
        return value


def _get_jattrs(cls: type, all_attrs: bool) -> List[_JsonisedAttribute]:
    """Internal helper to find the attributes to jsonize on a class"""
    jattrs = []
    names = set()
    py_names = set()
    for py_name, tp in typing.get_type_hints(
        cls, localns={cls.__name__: cls}, include_extras=True
    ).items():
        name = py_name
        json_only = MISSING
        # Retrieve name, type hint and annotation
        if typing.get_origin(tp) is Annotated:
            tp, *args = typing.get_args(tp)
            if any(isinstance((found := arg), JSONAttr) for arg in args):
                name = found.key or name
                json_only = found.json_only
            elif not all_attrs:
                continue
        elif not all_attrs:
            continue
        # Check for name conflict
        if name in names:
            raise JAttrError(
                f"Duplicated attribute name {name} in JSONable class {cls.__name__}"
            )
        names.add(name)
        py_names.add(py_name)
        # Get default value
        default = getattr(cls, py_name, MISSING)
        if isinstance(default, dataclasses.Field):
            if default.default is not dataclasses.MISSING:
                default = default.default
            elif default.default_factory is not dataclasses.MISSING:
                default = default.default_factory
            else:
                default = MISSING
        # Create JSONized attribute
        jattrs.append(
            _JsonisedAttribute(
                name=name,
                py_name=py_name,
                type_hint=normalize_json_type(tp),
                default=default,
                json_only=json_only,
            )
        )
    # Handle dataclasses fields that were removed
    # from the class' annotations
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is not dataclasses.MISSING
            ):
                if field.name in py_names:
                    continue  # skip already handled attributes
                tp = field.type
                name = field.name
                json_only = MISSING
                if typing.get_origin(tp) is Annotated:
                    tp, *args = typing.get_args(tp)
                    if any(isinstance((found := arg), JSONAttr) for arg in args):
                        name = found.key or name
                        json_only = found.json_only
                    elif not all_attrs:
                        continue
                elif not all_attrs:
                    continue
                if name in names:
                    raise JAttrError(
                        f"Duplicated attribute name {name} in JSONable class {cls.__name__}"
                    )
                jattrs.append(
                    _JsonisedAttribute(
                        name=name,
                        py_name=field.name,
                        type_hint=normalize_json_type(tp),
                        default=field.default_factory,
                        json_only=json_only,
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
        for jattr in subset.values()
    )


def _process_class(cls: type, /, *, all_attrs: bool) -> type:
    """Internal helper to make a class JSONable"""
    from .decoder import TypedJSONDecoder  # avoid circular deps

    # If class has already been processed, skip it
    if cls in JSONable._REGISTRY:
        return cls
    # Temporary register
    JSONable._TEMP_REGISTRY.append(cls)
    try:
        try:
            register_forward_ref(cls.__name__, cls)
        except ForwardRefConflictError as err:
            raise JSONableError(
                f"Cannot register the new JSONable class under name '{err.name}': it is already used by '{err.type_}'"
            ) from None
        try:
            # Compute jsonized attributes of the class
            all_jattrs = {jattr.name: jattr for jattr in _get_jattrs(cls, all_attrs)}
            req_jattrs = {
                jattr.name: jattr
                for jattr in all_jattrs.values()
                if jattr.default is MISSING or jattr.json_only
            }
            # Check for conflict with previously registered classes
            for other_cls in JSONable._REGISTRY:
                if (
                    not issubclass(cls, other_cls)
                    and _is_jattr_subset(req_jattrs, other_cls._ALL_JATTRS)
                    and _is_jattr_subset(other_cls._REQ_JATTRS, all_jattrs)
                ):
                    raise JSONableError(
                        f"JSONable class '{cls.__name__}' overlaps with '{other_cls.__name__}' without inheriting from it"
                    )
            # Modify the class in-place
            JSONable.register(cls)
            JSONable._REGISTRY.append(cls)
            cls._REQ_JATTRS = req_jattrs
            cls._ALL_JATTRS = all_jattrs
            setattr(cls, "to_json", JSONable.to_json)
            setattr(cls, "from_json", classmethod(JSONable.from_json.__func__))
            cls.Decoder = TypedJSONDecoder[cls]
            # Remove json_only attribute from the class's annotations and dict
            cls_annotations = cls.__dict__.get("__annotations__", {})
            for jattr in req_jattrs.values():
                if jattr.json_only:
                    if dataclasses.is_dataclass(cls):
                        raise JSONableError(
                            "Using json-only attributes requires that @dataclass is applied after @jsonable"
                        )
                    if hasattr(cls, jattr.py_name):
                        delattr(cls, jattr.py_name)
                    if jattr.py_name in cls_annotations:
                        del cls_annotations[jattr.py_name]
            return cls
        except Exception:
            # On error, undo the foward_ref registration
            table = register_forward_ref.__closure__[1].cell_contents
            if cls.__name__ in table:
                del table[cls.__name__]
            raise
    finally:
        JSONable._TEMP_REGISTRY.remove(cls)


def _make_instance(cls, obj):
    return cls(
        **{
            jattr.py_name: (
                obj[jattr.name] if jattr.name in obj else jattr.get_default()
            )
            for jattr in cls._ALL_JATTRS.values()
            if not jattr.json_only
        }
    )


def jsonable(cls: type = None, /, *, all_attrs: bool = True) -> type:
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
    for cls in JSONable._REGISTRY:
        # req_jattrs = set(cls._REQ_JATTRS)
        if all(  # all required arguments are there
            key in obj and typecheck(obj[key], jattr.type_hint)
            for key, jattr in cls._REQ_JATTRS.items()
        ) and all(  # all keys are valid - don't test req, already did
            key in cls._ALL_JATTRS
            and typecheck(obj[key], cls._ALL_JATTRS[key].type_hint)
            for key in obj.keys() - cls._REQ_JATTRS.items()
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
        return _make_instance(valid_cls[0], obj)
    else:
        return obj


@functools.wraps(json.load)
def load(*args, object_hook=jsonable_hook, **kwargs):
    return json.load(*args, object_hook=object_hook, **kwargs)


@functools.wraps(json.loads)
def loads(*args, object_hook=jsonable_hook, **kwargs):
    return json.loads(*args, object_hook=object_hook, **kwargs)
