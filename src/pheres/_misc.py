from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List, Tuple, Type, Union

from pheres._datatypes import MISSING
from pheres._exceptions import JSONKeyError, JSONTypeError, PheresInternalError
from pheres._typing import JSONArray, JSONObject, JSONType, JSONValue, typeof

# Types used in his module
FlatKey = Tuple[Union[int, str], ...]  # pylint: disable=unsubscriptable-object
"""Typehint for keys of flattened JSON"""
FlatJSON = Dict[FlatKey, JSONValue]
"""Type hint of a flattened JSON"""


def _flatten(flat_json: FlatJSON, keys: FlatKey, obj: JSONType) -> FlatJSON:
    """
    Helper function to flatten a json object
    """
    jtype = typeof(obj)
    if jtype is JSONValue:
        flat_json[keys] = obj
    elif jtype is JSONArray:
        for index, value in enumerate(obj):
            _flatten(flat_json, (*keys, index), value)
    elif jtype is JSONObject:
        for key, value in obj.items():
            _flatten(flat_json, (*keys, key), value)
    else:
        raise PheresInternalError(f"Unhandled json type {jtype} in flatten()")
    return flat_json


def flatten(obj: JSONObject) -> FlatJSON:
    """
    Flattens a JSON object to a dict with a single level of mapping. Key become tuples that store the
    path from the top-level object down to the value. Object keys are stored as str, Array index as int

    Arguments:
        obj: json object to flatten

    Returns:
        A dict mapping tuples of index and key to the final value
    """
    return _flatten({}, tuple(), obj)


def _expand(flat_json: FlatJSON, array_as_dict: bool, sort: bool, pre_keys: FlatKey):
    """
    Helper function for expand
    """
    # Group value by the upper-most key
    groups = {}
    for keys, value in flat_json.items():
        if keys[1:]:
            groups.setdefault(keys[0], {})[keys[1:]] = value
        elif keys[0] in groups:
            raise ValueError(
                f"Flat key {pre_keys + keys} has mixed json type: JSONValue and other"
            )
        else:
            groups[keys[0]] = value
    # Check that the type of the key is consistent
    if all(isinstance(key, str) for key in groups):
        type_ = dict
    elif all(isinstance(key, int) for key in groups):
        type_ = list
    else:
        raise ValueError(
            f"Flat key {pre_keys} has mixed json type: JSONArray and JSONObject"
        )
    # Expand sub-values
    for key, value in groups.items():
        if isinstance(value, dict):
            groups[key] = _expand(value, array_as_dict, sort, (*pre_keys, key))
    # Convert & sort
    if array_as_dict or type_ is dict:
        if sort:
            return dict(sorted(groups.items()))
        return groups
    pairs = sorted(groups.items)
    for i, (index, value) in enumerate(pairs):
        if i != index:
            raise ValueError(
                f"Flat key {pre_keys} has invalid index: expected {i}, got {index}"
            )
    return [value for _, value in pairs]


def expand(
    flat_json: FlatJSON, *, array_as_dict: bool = False, sort: bool = True
) -> JSONObject:
    """
    Expand a flat JSON back into a JSON object. This is the inverse operation of flatten().
    If there are duplicated values under the same key, a random one is kept

    Arguments:
        flat_json: flat json to expand
        array_as_dict: whether to represent arrays as dict[int, JSONValue] instead of list
        sort: whether to sort JSONObject by keys

    Returns:
        A JSON object that is the expanded representation of the flat_json

    Raises:
        ValueError: the flat_json is invalid. See the error for details
    """
    return _expand(flat_json, array_as_dict, sort, tuple())


def compact(obj: JSONObject, *, sep=".") -> JSONObject:
    """
    Removes unnecessary levels in the JSON tree
    
    Returns a new JSONObject where keys with only one element are merged
    with their parent key. Keys are converted to `str` objects. The returned
    value contains only dicts, even if the original JSON contained arrays.

    Arguments:
        obj: json object object to compact
        sep: separator to use for merging keys

    Returns:
        A compact, dict-only, representation of json_obj
    """
    ret = {}
    for k, v in obj.items():
        if typeof(v) is JSONArray:
            v = {str(i): elem for i, elem in enumerate(v)}
        if typeof(v) is JSONObject:
            v = compact(v, sep=sep)
            if len(v) == 1:
                kp, v = next(iter(v.items()))
                ret[f"{k!s}{sep}{kp!s}"] = v
                continue
        ret[k] = v
    return ret


def get(
    obj: Union[JSONArray, JSONObject],  # pylint: disable=unsubscriptable-object
    key: Union[int, str, FlatKey],  # pylint: disable=unsubscriptable-object
    default=MISSING,
) -> JSONType:
    """Retrieve a value on a JSON array or object. Return ``default``
    if it was provided and ``key`` is missing

    Arguments:
        obj: JSON array or object to retrieve the value from
        key: key to index. Supports iterable of successive keys
        default: optional value to return if ``key`` is missing

    Raises:
        JSONKeyError: ``key`` is missing and ``default`` was not specified
    """
    if isinstance(key, Iterable) and not isinstance(key, str):
        key = tuple(key)
    else:
        key = (key,)
    try:
        for k in key[:-1]:
            obj = obj[k]
        return obj[key[-1]]
    except (IndexError, KeyError):
        if default is not MISSING:
            return default
        raise JSONKeyError(obj, key) from None


def has(
    obj: Union[JSONArray, JSONObject],  # pylint: disable=unsubscriptable-object
    key: Union[int, str, FlatKey],  # pylint: disable=unsubscriptable-object
):
    """Test if a JSON has the provided key

    Implemented by calling ``get(obj, key)`` and catching JSONKeyError

    See also:
        `pheres._misc.get`
    """
    try:
        get(obj, key)
        return True
    except JSONKeyError:
        return False


def set(
    obj: Union[JSONArray, JSONObject],  # pylint: disable=unsubscriptable-object
    key: Union[int, str, FlatKey],  # pylint: disable=unsubscriptable-object
    value: JSONObject,
):
    """Sets the value of the key in the JSON

    Possibly creates the full path at once. Setting a value in a array
    past its length or adding an element at the end is supported

    Arguments:
        obj: JSON to set the value ine
        key: path from the root JSON ``obj`` to the value to set.
            Supports iterables of keys. Key of type `int` will create new
            `list` if necessary, and `str` will create new `dict`
        value: the value to set under the key ``key``

    Raises:
        IndexError: The key is not valid
    """
    if isinstance(key, Iterable) and not isinstance(key, str):
        key = tuple(key)
    else:
        key = (key,)
    k = key[0]
    for next_key in key[1:]:
        if isinstance(next_key, int):
            next_obj = []
        elif isinstance(next_key, str):
            next_obj = {}
        else:
            raise JSONTypeError(
                type=Union[int, str],
                value=next_key,
                msg=f"JSON key must have type {{type}}, got {type(next_key)}",
            )
        if isinstance(obj, list) and k == len(obj):
            obj.append(next_obj)
        else:
            obj[k] = next_obj
        obj = next_obj
        k = next_key
    if isinstance(obj, list) and k == len(obj):
        obj.append(value)
    else:
        obj[k] = value
