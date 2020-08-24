from typing import *
from dataclasses import *
import pheres as ph

# fix tests being run mutiple times
ph.JSONable._REGISTER.clear()

@dataclass
class BaseTypes(ph.JSONable):
    null: None
    boolean: bool
    integer: int
    float_: float
    string: str

@dataclass
@ph.jsonable
class ParamTypes(ph.JSONable):
    literal: Literal[0, 1]
    array_fixed: Tuple[None, bool, int, float, str]
    array: List[int]
    obj: Dict[str, str]

@dataclass
@ph.jsonable
class DefaultBaseTypes(ph.JSONable):
    type_: Literal["dbt"]
    null_d: None = None
    boolean_d: bool = False
    integer_d: int = 0
    float_d: float = 0.0
    string_d: str = ""

# Test without dataclass
@ph.jsonable
class _DefaultParamTypes(ph.JSONable):
    type_: Literal["_dbt"]
    _literal_d: Literal[0, 1] = 0
    _array_fixed_d: Tuple[None, bool, int, float, str] = lambda : [None, False, 0, 0.0, ""]
    _array_d: List[int] = lambda : []
    _obj_d: Dict[str, str] = lambda: {}
    
    def __init__(self, lit, arr_f, arr, obj):
        self._literal_d = lit
        self._array_fixed_d = arr_f
        self._array_d = arr
        self._obj_d = obj

@dataclass
@ph.jsonable
class DefaultParamTypes(ph.JSONable):
    type_: Literal["dpt"]
    literal_d: Literal[0, 1] = 0
    array_fixed_d: Tuple[None, bool, int, float, str] = field(default_factory=lambda: [None, False, 0, 0.0, ""])
    array_d: List[int] = field(default_factory=lambda: [])
    obj_d : Dict[str, str] = field(default_factory=lambda:{})

@dataclass
@ph.jsonable
class JSONableTypes(ph.JSONable):
    base_types: BaseTypes
    param_types: ParamTypes

@dataclass
@ph.jsonable
class DefaultJSONableTypes(ph.JSONable):
    type_: Literal["dj"]
    base_types_d: DefaultBaseTypes
    param_types_d: DefaultParamTypes


def test_base_types():
    obj = BaseTypes(None, True, 100, 1e2, "maybe")
    assert obj.to_json() == {
        "null": None,
        "boolean": True,
        "integer": 100,
        "float_": 1e2,
        "string": "maybe",
    }

    assert ph.dumps(obj) == """{"null": null, "boolean": true, "integer": 100, "float_": 100.0, "string": "maybe"}"""
    assert obj == BaseTypes.Decoder.loads(ph.dumps(obj))
    assert obj == BaseTypes.from_json(ph.dumps(obj))
    assert isinstance(ph.loads(ph.dumps(obj)), BaseTypes)

def test_param_types():
    obj = ParamTypes(0, [None, True, 1, 1.0, "maybe"], [1], {"key": "value"})
    assert obj.to_json() == {"literal": 0, "array_fixed": [None, True, 1, 1.0, "maybe"], "array": [1], "obj": {"key": "value"}}
    assert ph.dumps(obj) == r'{"literal": 0, "array_fixed": [null, true, 1, 1.0, "maybe"], "array": [1], "obj": {"key": "value"}}'
    assert obj == ParamTypes.Decoder.loads(ph.dumps(obj))
    assert obj == ParamTypes.from_json(ph.dumps(obj))
    assert isinstance(ph.loads(ph.dumps(obj)), ParamTypes)


def test_default_bases():
    obj = DefaultBaseTypes()
    assert obj.to_json() == {"type_": "dbt"}
    assert obj.to_json(with_defaults=True) == {"type_": "dbt", "null_d": None, "boolean_d": False, "integer_d": 0, "float_d": 0.0, "string_d": ""}
    assert ph.dumps(obj) == r'{"type_": "dbt"}'
    assert obj == DefaultBaseTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultBaseTypes.from_json(ph.dumps(obj))
    assert isinstance(ph.loads(ph.dumps(obj)), DefaultBaseTypes)

    obj = DefaultBaseTypes(None, True, 1, 1.0, "string")
    assert obj.to_json() == {"type_": "dbt", "boolean_d": True, "integer_d": 1, "float_d": 1.0, "string_d": "string"}
    assert obj.to_json(with_defaults=True) == {"type_": "dbt", "null_d": None, "boolean_d": True, "integer_d": 1, "float_d": 1.0, "string_d": "string"}
    assert ph.dumps(obj) == r'{"type_": "dbt", "boolean_d": true, "integer_d": 1, "float_d": 1.0, "string_d": "string"}'
    assert obj == DefaultBaseTypes.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultBaseTypes.from_json(ph.dumps(obj))
    assert isinstance(ph.loads(ph.dumps(obj)), DefaultBaseTypes)

def test_default_param():
    obj = DefaultParamTypes()
    assert obj.to_json() == {"type_": "dpt"}