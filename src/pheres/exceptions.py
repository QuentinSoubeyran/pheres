"""
Pheres module for all exceptions
"""
import functools
import json
from typing import Any, Tuple

import graphlib
from attr import attrib
from attr import dataclass as attrs

from .utils import TypeHint, autoformat

exception = functools.partial(attrs, auto_exc=True)

__all__ = [
    "PheresInternalError",
    "PheresError",
    "TypeHintError",
    "JSONTypeError",
    "JSONValueError",
    "CycleError",
    "JsonableError",
    "JsonAttrError",
    "JsonAttrValueError",
    "JsonAttrTypeError",
    "DecodeError",
    "TypedJSONDecodeError",
]


@exception
class PheresInternalError(BaseException):
    """
    Raised when pheres encounters an internal error

    Attributes
        msg -- explanation of the error
    """


class PheresError(Exception):
    """
    Base exception in pheres
    """


####################
# TYPE HINT ERRORS #
####################


@autoformat
@exception
class TypeHintError(PheresError, TypeError):
    """
    Raised for invalid JSON type hints

    Attributes:
        type -- invalid type hint
        msg -- explanation of the error
    """

    type: TypeHint
    msg: str = "{type} is not a valid JSON type hint"


#################
# TYPING ERRORS #
#################


@autoformat
@exception
class JSONTypeError(PheresError, TypeError):
    """
    Raised when a value doesn't have the excepted type

    Attributes:
        type -- expected type
        value -- invalid value
        msg -- explanation of the error
    """

    type: TypeHint
    value: Any
    msg: str = "{value} doesn't have type {type}"


@autoformat
@exception
class JSONValueError(PheresError, ValueError):
    """
    Raised when an invalid json value is encountered

    Attributes
        value -- invalid value
        msg -- explanation of the error
    """

    value: Any
    msg: str = "invalid JSON {value}"


@autoformat
@exception
class CycleError(PheresError, graphlib.CycleError):
    """
    Raised when a value has cycles in it

    Attributes:
        obj -- object with a cycle
        cycle -- detected cycle
        msg -- explanation of the error
    """

    obj: Any
    cycle: Tuple[Any]
    msg: str = "{obj} contains the cycle {cycle}"


###################
# JSONABLE ERRORS #
###################


@exception
class JsonableError(PheresError):
    """
    Raised on problems with @jsonable when no better sub-exception
    exists

    Attributes
        msg -- explanation of the error
    """

    msg: str


@autoformat
@exception
class JsonAttrError(JsonableError):
    """
    Raised on problems with @jsonable due to JsonAttr when no better
    sub-exception exists

    Attributes
        cls -- name of the class that raised the error
        attr -- name of the attribute that raised the error
        msg -- explanation of the error
    """

    cls: str
    attr: str
    detail: str = ""
    msg: str = "Error in '{cls}' at attribute '{attr}'{detail}"


@autoformat
@exception
class JsonAttrValueError(JSONValueError, JsonAttrError):
    """
    Raised when the default value of a json attribute is not a valid json

    Attributes
        cls -- name of the class that raised the error
        attr -- name of the attribute that raised the error
        value -- invalid value
        msg -- explanation of the error
    """

    msg: str = "Invalid JSON value '{value}' in class '{cls}' at attribute '{attr}'"


@autoformat
@exception
class JsonAttrTypeError(JSONTypeError, JsonAttrError):
    """
    Raised when the default value of a json attribute doesn't have the correct type

    Attributes:
        type -- expected type
        value -- invalid value
        msg -- explanation of the error
    """

    msg: str = "{value} doesn't have type {type} in class '{cls}' at attribute '{attr}'"


##################
# DECODING ERROR #
##################


@exception
class DecodeError:
    """
    Raised on decoding problem in Pheres when no better exception exists

    Attributes:
        msg -- explanation of the error
    """

    msg: str


class TypedJSONDecodeError(json.JSONDecodeError, DecodeError):
    """
    Raised when the decoded type is not the expected one
    """

    def __init__(self, msg, doc, pos):
        """
        Special case when the decoded document is an object
        """
        if not isinstance(doc, str):
            # quote the string keys only
            pos = ['"%s"' % p if isinstance(p, str) else str(p) for p in pos]
            keys = " -> ".join(("<base object>", *pos))
            errmsg = "%s: at %s" % (msg, keys)
            ValueError.__init__(self, errmsg)
            self.msg = msg
            self.doc = doc
            self.pos = pos
            self.lineno = None
            self.colno = None
        else:
            super().__init__(msg, doc, pos)
