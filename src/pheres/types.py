"""
Module for quick import of type-hints

Contains:
    `typing.Dict`
    `typing.List`
    `typing.Union`
    `JSONValue`
    `JSONArray`
    `JSONObject`
    `JSONType`
    `JsonableValue`
    `JsonableArray`
    `JsonableDict`
    `JsonableObject`
"""
from __future__ import annotations

from typing import Dict, List, Union

from pheres._typing import (
    JsonableArray,
    JsonableDict,
    JsonableObject,
    JsonableValue,
    JSONArray,
    JSONObject,
    JSONType,
    JSONValue,
)