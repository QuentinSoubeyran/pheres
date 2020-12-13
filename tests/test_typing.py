from pprint import pprint

import pytest
from hypothesis import given

from pheres import (
    JSONValueError,
    PheresError,
    is_json,
    is_json_type,
    normalize_hint,
    typecheck,
    typeof,
)
from pheres.types import *  # pylint: disable=unused-wildcard-import

from .strategies import (
    json_arrays,
    json_objects,
    json_types,
    json_value,
    jsons,
    typed_jsons,
)


@given(json_types())
def test_is_json(tp) -> None:
    assert is_json_type(tp)


@given(typed_jsons())
def test_typecheck_pairs(tp_ex) -> None:
    tp, ex = tp_ex
    assert typecheck(ex, tp)


@given(json_value)
def test_typecheck_values(value: JSONValue) -> None:
    assert typecheck(value, JSONValue)


@given(json_arrays())
def test_typecheck_arrays(array: JSONArray) -> None:
    assert typecheck(array, JSONArray)


@given(json_objects())
def test_typecheck_objects(obj: JSONObject) -> None:
    assert typecheck(obj, JSONObject)


@given(jsons())
def test_typecheck_any(obj: JSONType) -> None:
    assert typecheck(obj, JSONType)


@given(json_value)
def test_typeof_values(value: JSONValue) -> None:
    assert typeof(value) is JSONValue


@given(json_arrays())
def test_typeof_arrays(array: JSONArray) -> None:
    assert typeof(array) is JSONArray


@given(json_objects())
def test_typeof_objects(obj: JSONObject) -> None:
    assert typeof(obj) is JSONObject


@given(jsons())
def test_typeof_any(obj: JSONType) -> None:
    assert is_json(obj)


test_cases = [
    (None, JSONValue),
    (True, JSONValue),
    (0, JSONValue),
    (0.0, JSONValue),
    ("a string", JSONValue),
    (1j, JSONValueError),
    ((None, True, 0, 0.1, "a string"), JSONArray),
    ([None, True, 0, 0.1, "a string"], JSONArray),
    ({1, 2, 3}, JSONValueError),
    ({"key": 0.0}, JSONObject),
]


def test_typeof():
    from pheres import typeof

    for value, jtype in test_cases:
        print(f"Testing typeof({value}) == {jtype}")
        if isinstance(jtype, type) and issubclass(jtype, PheresError):
            with pytest.raises(jtype):
                typeof(value)
        else:
            assert typeof(value) is jtype


def test_is_json():
    from pheres import CycleError, is_json

    for value, jtype in test_cases:
        print(
            f"Testing is_json({value}) == {not (isinstance(jtype, type) and issubclass(jtype, PheresError))}"
        )
        assert is_json(value) == (
            not (isinstance(jtype, type) and issubclass(jtype, PheresError))
        )

    assert not is_json({0: 0.0})
    d = {}
    d["key"] = d
    with pytest.raises(CycleError):
        is_json(d)
