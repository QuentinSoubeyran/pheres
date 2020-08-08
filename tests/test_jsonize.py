from typing import *
from dataclasses import *
import pheres as ph

# fix tests being run mutiple times
# ph.JSONable._REGISTER.clear()


def test_base_types():
    @ph.jsonize
    @dataclass
    class BaseTypes:
        type_: Literal["base_types"]
        null: None
        boolean: bool
        integer: int
        float_: float
        name: str
        array: List[int]
        obj: Dict[str, str]

    obj = BaseTypes(
        "base_types", None, True, 100, 1e2, "bob", [1, 2, 3], {"key": "value"}
    )
    assert obj.to_json() == {
        "type_": "base_types",
        "null": None,
        "boolean": True,
        "integer": 100,
        "float_": 1e2,
        "name": "bob",
        "array": [1, 2, 3],
        "obj": {"key": "value"},
    }

    assert ph.dumps(obj) == """{"type_": "base_types", "null": null, "boolean": true, "integer": 100, "float_": 100.0, "name": "bob", "array": [1, 2, 3], "obj": {"key": "value"}}"""
    assert obj == BaseTypes.Decoder.loads(ph.dumps(obj))
    assert obj == BaseTypes.from_json(ph.dumps(obj))
    assert isinstance(ph.loads(ph.dumps(obj)), BaseTypes)


def test_defaults():
    @dataclass
    @ph.jsonize
    class DefaultBaseTypes:
        # test default values
        type_: Literal["default_base_types"] = "default_base_types"
        null: None = None
        boolean: bool = True
        integer: int = 100
        float_: float = 1e2
        name: str = ""
        array: List[int] = field(default_factory=lambda : [1, 2, 3])
        obj: Dict[str, str] = field(default_factory=lambda: {"key": "value"})
    
    obj = DefaultBaseTypes()
    assert obj.to_json() == {
        "type_": "default_base_types"
    }
    assert ph.dumps(obj) == '{"type_": "default_base_types"}'
    assert obj == obj.Decoder.loads(ph.dumps(obj))
    assert obj == DefaultBaseTypes.from_json(ph.dumps(obj))
    assert isinstance(ph.loads(ph.dumps(obj)), DefaultBaseTypes)
