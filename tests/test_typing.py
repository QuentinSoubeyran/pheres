from pprint import pprint
from string import printable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pheres import JSONValueError, PheresError, is_json, typecheck, typeof
from pheres.types import *

json_value = (
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(printable)
)

def json_any(size):
    return st.recursive(
        json_value,
        lambda children: st.lists(children, max_size=10) | st.dictionaries(st.text(printable), children, max_size=10),
        max_leaves=size
    )
json_array = st.lists(json_any(15), max_size=15)
json_object = st.dictionaries(st.text(printable), json_any(15), max_size=15)

@given(json_value)
def test_json_values(value):
    assert typeof(value) is JSONValue
    assert typecheck(value, JSONValue)


@given(json_array)
def test_json_arrays(array):
    assert typeof(array) is JSONArray
    assert typecheck(array, JSONArray)


@given(json_object)
def test_json_objects(obj):
    assert typeof(obj) is JSONObject
    assert typecheck(obj, JSONObject)


@given(json_any(25))
def test_json_any(obj):
    assert is_json(obj)
    assert typecheck(obj, JSONType)


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
