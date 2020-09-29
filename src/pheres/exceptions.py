"""
Pheres module for all exceptions
"""
import functools
from typing import Any, Tuple

from attr import attrib
from attr import dataclass as attrs
from graphlib import CycleError as _CycleError

from .utils import TypeHint, autoformat

exception = functools.partial(attrs, auto_exc=True)


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
class TypeHintError(TypeError, PheresError):
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
class JSONTypeError(TypeError, PheresError):
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
class JSONValueError(ValueError, PheresError):
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
class CycleError(_CycleError, PheresError):
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
