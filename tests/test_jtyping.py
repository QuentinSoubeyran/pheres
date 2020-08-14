from typing import *
import pytest

from pheres import JSONError, JSONValue, JSONArray, JSONObject, JSONTypeError

test_cases = [
    (None, JSONValue),
    (True, JSONValue),
    (0, JSONValue),
    (0.0, JSONValue),
    ("a string", JSONValue),
    (1j, JSONTypeError),
    ((None, True, 0, 0.1, "a string"), JSONArray),
    ([None, True, 0, 0.1, "a string"], JSONArray),
    ({1, 2, 3}, JSONTypeError),
    ({"key": 0.0}, JSONObject),
]

def test_typeof():
    from pheres import typeof

    for value, jtype in test_cases:
        print(f"Testing typeof({value}) == {jtype}")
        if isinstance(jtype, type) and issubclass(jtype, JSONError):
            with pytest.raises(jtype):
                typeof(value)
        else:
            assert typeof(value) is jtype

def test_is_json():
    from pheres import is_json, CycleError
    
    for value, jtype in test_cases:
        print(f"Testing is_json({value}) == {not (isinstance(jtype, type) and issubclass(jtype, JSONError))}")
        assert is_json(value) == (not (isinstance(jtype, type) and issubclass(jtype, JSONError)))

    assert not is_json({0: 0.0})
    d = {}
    d["key"] = d
    with pytest.raises(CycleError):
        is_json(d)

