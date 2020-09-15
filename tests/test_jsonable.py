# Builtins
from dataclasses import dataclass, field
import json
from typing import Tuple, List, Union, Literal, Dict
import pytest

#thrid party import
from pheres import jsonable, JSONable
import pheres as ph



# fix tests being run mutiple times
from pheres.core import _JSONableObject
_JSONableObject.registry.clear()
for key in list(ph.register_forward_ref._table.keys()):
    if key not in ("JSONType", "JSONable"):
        del ph.register_forward_ref._table[key]


@dataclass
@jsonable
class BaseTypes(JSONable):
    null: None
    boolean: bool
    integer: int
    float_: float
    string: str


def test_base_types():
    obj = BaseTypes(None, True, 1, 1.0, "string")
    assert obj.to_json() == {
        "null": None,
        "boolean": True,
        "integer": 1,
        "float_": 1.0,
        "string": "string",
    }

    assert ph.dumps(obj) == (
        r'{"null": null, "boolean": true, "integer": 1, "float_": 1.0, "string": "string"}'
    )
    assert obj == BaseTypes.Decoder.loads(ph.dumps(obj))
    assert obj == BaseTypes.from_json(ph.dumps(obj))
    assert obj == BaseTypes.from_json(json.loads(ph.dumps(obj)))
    assert isinstance(ph.loads(ph.dumps(obj)), BaseTypes)


@dataclass
@jsonable
class ParamTypes(JSONable):
    literal: Literal[0, 1]
    array_fixed: Tuple[None, bool, int, float, str]
    array: List[int]
    obj: Dict[str, str]


def test_param_types():
    obj = ParamTypes(0, [None, True, 1, 1.0, "string"], [1, 2, 3], {"key": "value"})
    assert obj.to_json() == {
        "literal": 0,
        "array_fixed": [None, True, 1, 1.0, "string"],
        "array": [1, 2, 3],
        "obj": {"key": "value"},
    }
    assert ph.dumps(obj) == (
        r'{"literal": 0, "array_fixed": [null, true, 1, 1.0, "string"], "array": [1, 2, 3], "obj": {"key": "value"}}'
    )
    assert obj == ParamTypes.Decoder.loads(ph.dumps(obj))
    assert obj == ParamTypes.from_json(ph.dumps(obj))
    assert obj == ParamTypes.from_json(json.loads(ph.dumps(obj)))
    assert isinstance(ph.loads(ph.dumps(obj)), ParamTypes)


@dataclass
@jsonable
class DefaultBaseTypes(JSONable):
    type_: Literal["dbt"]
    null_d: None = None
    boolean_d: bool = False
    integer_d: int = 0
    float_d: float = 0.0
    string_d: str = ""


def test_default_bases():
    obj = DefaultBaseTypes()
    assert obj.to_json() == {"type_": "dbt"}
    assert obj.to_json(with_defaults=True) == {
        "type_": "dbt",
        "null_d": None,
        "boolean_d": False,
        "integer_d": 0,
        "float_d": 0.0,
        "string_d": "",
    }
    assert ph.dumps(obj) == r'{"type_": "dbt"}'
    assert obj == DefaultBaseTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultBaseTypes.from_json(ph.dumps(obj))
    assert obj == DefaultBaseTypes.from_json(json.loads(ph.dumps(obj)))
    assert isinstance(ph.loads(ph.dumps(obj)), DefaultBaseTypes)

    obj = DefaultBaseTypes(None, True, 1, 1.0, "string")
    assert obj.to_json() == {
        "type_": "dbt",
        "boolean_d": True,
        "integer_d": 1,
        "float_d": 1.0,
        "string_d": "string",
    }
    assert obj.to_json(with_defaults=True) == {
        "type_": "dbt",
        "null_d": None,
        "boolean_d": True,
        "integer_d": 1,
        "float_d": 1.0,
        "string_d": "string",
    }
    assert ph.dumps(obj) == (
        r'{"type_": "dbt", "boolean_d": true, "integer_d": 1, "float_d": 1.0, "string_d": "string"}'
    )
    assert obj == DefaultBaseTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultBaseTypes.from_json(ph.dumps(obj))
    assert obj == DefaultBaseTypes.from_json(json.loads(ph.dumps(obj)))
    assert isinstance(ph.loads(ph.dumps(obj)), DefaultBaseTypes)


# Test without dataclass
@jsonable
class _DefaultParamTypes(JSONable):
    type_: Literal["_dbt"]
    _literal_d: Literal[0, 1] = 0
    _array_fixed_d: Tuple[None, bool, int, float, str] = lambda: [
        None,
        False,
        0,
        0.0,
        "",
    ]
    _array_d: List[int] = lambda: []
    _obj_d: Dict[str, str] = lambda: {}

    def __init__(self, lit, arr_f, arr, obj):
        self._literal_d = lit
        self._array_fixed_d = arr_f
        self._array_d = arr
        self._obj_d = obj


@dataclass
@jsonable
class DefaultParamTypes(JSONable):
    type_: Literal["dpt"]
    literal_d: Literal[0, 1] = 0
    array_fixed_d: Tuple[None, bool, int, float, str] = field(
        default_factory=lambda: [None, False, 0, 0.0, ""]
    )
    array_d: List[int] = field(default_factory=lambda: [])
    obj_d: Dict[str, str] = field(default_factory=lambda: {})


def test_default_param():
    obj = DefaultParamTypes()
    assert obj.to_json() == {"type_": "dpt"}
    assert obj.to_json(with_defaults=True) == {
        "type_": "dpt",
        "literal_d": 0,
        "array_fixed_d": [None, False, 0, 0.0, ""],
        "array_d": [],
        "obj_d": {},
    }
    assert ph.dumps(obj) == r'{"type_": "dpt"}'
    assert obj == DefaultParamTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultParamTypes.from_json(ph.dumps(obj))
    assert obj == DefaultParamTypes.from_json(json.loads(ph.dumps(obj)))
    assert isinstance(ph.loads(ph.dumps(obj)), DefaultParamTypes)

    obj = DefaultParamTypes(
        1, [None, True, 1, 1.0, "string"], [1, 2, 3], {"key": "value"}
    )
    assert obj.to_json() == {
        "type_": "dpt",
        "literal_d": 1,
        "array_fixed_d": [None, True, 1, 1.0, "string"],
        "array_d": [1, 2, 3],
        "obj_d": {"key": "value"},
    }
    assert obj.to_json(with_defaults=True) == {
        "type_": "dpt",
        "literal_d": 1,
        "array_fixed_d": [None, True, 1, 1.0, "string"],
        "array_d": [1, 2, 3],
        "obj_d": {"key": "value"},
    }
    assert ph.dumps(obj) == (
        r'{"type_": "dpt", "literal_d": 1, "array_fixed_d": [null, true, 1, 1.0, "string"], "array_d": [1, 2, 3], "obj_d": {"key": "value"}}'
    )
    assert obj == DefaultParamTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultParamTypes.from_json(ph.dumps(obj))
    assert obj == DefaultParamTypes.from_json(json.loads(ph.dumps(obj)))
    assert isinstance(ph.loads(ph.dumps(obj)), DefaultParamTypes)


@dataclass
@jsonable
class JSONableTypes(JSONable):
    base_types: BaseTypes
    param_types: ParamTypes


def test_jsonables():
    base = BaseTypes(None, True, 1, 1.0, "string")
    param = ParamTypes(1, [None, True, 1, 1.0, "string"], [1, 2, 3], {"key": "value"})
    obj = JSONableTypes(base, param)
    assert obj.to_json() == {
        "base_types": {
            "null": None,
            "boolean": True,
            "integer": 1,
            "float_": 1.0,
            "string": "string",
        },
        "param_types": {
            "literal": 1,
            "array_fixed": [None, True, 1, 1.0, "string"],
            "array": [1, 2, 3],
            "obj": {"key": "value"},
        },
    }
    assert ph.dumps(obj) == (
        "{"
        r'"base_types": {"null": null, "boolean": true, "integer": 1, "float_": 1.0, "string": "string"}, '
        r'"param_types": {"literal": 1, "array_fixed": [null, true, 1, 1.0, "string"], "array": [1, 2, 3], "obj": {"key": "value"}}'
        "}"
    )
    assert obj == JSONableTypes.Decoder.loads(ph.dumps(obj))
    assert obj == JSONableTypes.from_json(ph.dumps(obj))
    assert obj == JSONableTypes.from_json(json.loads(ph.dumps(obj)))


@dataclass
@jsonable
class DefaultJSONableTypes(JSONable):
    base_types_d: DefaultBaseTypes = field(default_factory=DefaultBaseTypes)
    param_types_d: DefaultParamTypes = field(default_factory=DefaultParamTypes)

def test_default_jsonables():
    obj = DefaultJSONableTypes()
    assert obj.to_json() == {}
    assert obj.to_json(with_defaults=True) == {
        "base_types_d": {
            "type_": "dbt",
            "null_d": None,
            "boolean_d": False,
            "integer_d": 0,
            "float_d": 0.0,
            "string_d": "",
        },
        "param_types_d": {
            "type_": "dpt",
            "literal_d": 0,
            "array_fixed_d": [None, False, 0, 0.0, ""],
            "array_d": [],
            "obj_d": {},
        }
    }
    assert ph.dumps(obj) == r'{}'
    assert obj == DefaultJSONableTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultJSONableTypes.from_json(ph.dumps(obj))
    assert obj == DefaultJSONableTypes.from_json(json.loads(ph.dumps(obj)))

    base = DefaultBaseTypes(None, True, 1, 1.0, "string")
    param = DefaultParamTypes(1, [None, True, 1, 1.0, "string"], [1, 2, 3], {"key": "value"})
    obj = DefaultJSONableTypes(base, param)
    print(ph.dumps(obj, indent=2))
    assert obj.to_json(with_defaults=True) == {
        "base_types_d": {
            "type_": "dbt",
            "null_d": None,
            "boolean_d": True,
            "integer_d": 1,
            "float_d": 1.0,
            "string_d": "string",
        },
        "param_types_d": {
            "type_": "dpt",
            "literal_d": 1,
            "array_fixed_d": [None, True, 1, 1.0, "string"],
            "array_d": [1, 2, 3],
            "obj_d": {"key": "value"},
        },
    }
    assert ph.dumps(obj) == (
        "{"
        r'"base_types_d": {"type_": "dbt", "boolean_d": true, "integer_d": 1, "float_d": 1.0, "string_d": "string"}, '
        r'"param_types_d": {"type_": "dpt", "literal_d": 1, "array_fixed_d": [null, true, 1, 1.0, "string"], "array_d": [1, 2, 3], "obj_d": {"key": "value"}}'
        "}"
    )
    assert obj == DefaultJSONableTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultJSONableTypes.from_json(ph.dumps(obj))
    assert obj == DefaultJSONableTypes.from_json(json.loads(ph.dumps(obj)))


@jsonable[int]
class JsonableInt(JSONable):
    def __init__(self, value):
        self.value = value
    
    def to_json(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, JsonableInt):
            return self.value == other.value
        return NotImplemented

def test_jsonable_value():
    obj = JsonableInt(1)
    assert obj.to_json() == 1
    assert ph.dumps(obj) == '1'
    assert obj == JsonableInt.Decoder.loads(ph.dumps(obj))
    assert obj == JsonableInt.from_json(ph.dumps(obj))
    assert obj == JsonableInt.from_json(json.loads(ph.dumps(obj)))
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableInt.from_json(None)
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableInt.from_json(1.0)
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableInt.from_json(r'null')
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableInt.from_json(r'"string"')

@jsonable[int, int, int]
class JsonableArrayFixed(JSONable):
    def __init__(self, *array):
        self.array = list(array)
    
    def to_json(self):
        return self.array
    
    def __eq__(self, other):
        if isinstance(other, JsonableInt):
            return self.array == other.array
        return NotImplemented

def test_jsonable_array():
    obj = JsonableArrayFixed(1, 2, 3)
    assert obj.to_json() == [1, 2, 3]
    assert ph.dumps(obj) == r'[1, 2, 3]'
    assert obj == JsonableArrayFixed.Decoder.loads(ph.dumps(obj))
    assert obj == JsonableArrayFixed.from_json(ph.dumps(obj))
    assert obj == JsonableArrayFixed.from_json(json.loads(ph.dumps(obj)))

    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableArrayFixed.from_json([1, 2])
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableArrayFixed.from_json([1, 2, 3, 4])
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableArrayFixed.from_json(r"[1, 2]")
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableArrayFixed.from_json(r"[1,2,3,4]")
    with pytest.raises(ph.TypedJSONDecodeError):
        JsonableArrayFixed.from_json(r'[1, 2, "string"]')