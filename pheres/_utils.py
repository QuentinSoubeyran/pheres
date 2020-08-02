# -*- coding: utf-8 -*-
"""
module with utilities for using and transforming JSON

Part of the Pheres package
"""
# stdlib import
from typing import Dict, List, Tuple, Union

# Local import
from ._jtypes import JsonError, JsonType, JsonValue, JsonArray, JsonObject, typeof

__all__ = [
    # Types
    "FlatKey",
    "FlatJson",
    # Processing utilities
    "flatten",
    "expand",
    "compact",
]

# Types used in his module
FlatKey = Tuple[Union[int, str], ...]
FlatJson = Dict[FlatKey, JsonValue]


def _flatten(flat_json: FlatJson, keys: FlatKey, obj: JsonType) -> FlatJson:
    """
    Helper function to flatten a json object
    """
    jtype = typeof(obj)
    if jtype is JsonValue:
        flat_json[keys] = obj
    elif jtype is JsonArray:
        for index, value in enumerate(obj):
            _flatten(flat_json, (*keys, index), value)
    elif jtype is JsonObject:
        for key, value in obj.itesm():
            _flatten(flat_json, (*keys, key), value)
    else:
        raise JsonError(
            f"[!! This is a bug !! Please report] Unhandled json type {jtype} in flatten()"
        )
    return flat_json


def flatten(obj: JsonObject) -> FlatJson:
    """
    Flattens a JSON object to a dict with a single level of mapping. Key become tuples that store the
    path from the top-level object down to the value. Object keys are stored as str, Array index as int

    Arguments
        obj -- json object to flatten
    
    Returns
        A dict mapping tuples of index and key to the final value
    """
    return _flatten({}, tuple(), obj)


def _expand(flat_json: FlatJson, array_as_dict: bool, sort: bool, pre_keys: FlatKey):
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
                f"Flat key {pre_keys + keys} has mixed json type: JsonValue and other"
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
            f"Flat key {pre_keys} has mixed json type: JsonArray and JsonObject"
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
    flat_json: FlatJson, *, array_as_dict: bool = False, sort: bool = True
) -> JsonObject:
    """
    Expand a flat JSON back into a Json object. This is the inverse operation of flatten().
    If there are duplicated values under the same key, a random one is kept

    Arguments
        flat_json -- flat json to expand
        array_as_dict -- represent arrays as dict[int, JsonValue] instead of list
        sort -- sort JsonObject by keys
    
    Returns
        A Json object that is the expanded representation of the flat_json
    
    Raises
        ValueError -- the flat_json is invalid
    """
    return _expand(flat_json, array_as_dict, sort, tuple())


def compact(json_obj: JsonObject, *, sep="/") -> JsonObject:
    """
    Returns a new dict-only json that is a copy of `json_obj` where keys with only one element are
    merged with their parent key

    Arguments
        obj -- json object object to compact
        sep -- separator to use for merging keys
    
    Returns
        A compact, dict-only, representation of json_obj
    """
    ret = {}
    for k, v in json_obj.items():
        if typeof(v) is JsonArray:
            v = {str(i): elem for i, elem in enumerate(v)}
        if typeof(v) is JsonObject:
            v = compact(v, sep=sep)
            if len(v) == 1:
                kp, v = next(iter(v.items()))
                ret[f"{k!s}{sep}{kp!s}"] = v
                continue
        ret[k] = v
    return ret
