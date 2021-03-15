"""Module containing all the exceptions used in `pheres`

All exception are subclasses of `PheresError`, except for `PheresInternalError`.
This is intended, as `PheresInternalError` is only raised in case of bugs and
should not be catched. Please make a bug report if you ever encounter `PheresInternalError` 
"""
from __future__ import annotations

import functools
import graphlib
import json
from typing import Any, Union

import attr
from attr import attrib

from pheres.aliases import TypeForm
from pheres.utils import autoformat

__all__ = [
    "PheresInternalError",
    "PheresError",
    "CycleError",
]


@attr.dataclass(auto_exc=True)
class PheresInternalError(BaseException):
    """
    Raised when pheres encounters an internal error

    If you see this exception, please make a bug report

    Attributes:
        msg: explanation of the error
    """

    msg: str


class PheresError(Exception):
    """
    Base exception in pheres for all other exceptions
    """

    msg: str

    def __str__(self):
        return self.msg


@autoformat
@attr.dataclass(auto_exc=True)
class CycleError(PheresError, graphlib.CycleError):
    """
    Raised when a value has cycles in it

    Attributes:
        obj: object containing the cycle
        cycle: detected cycle
        msg: explanation of the error
    """

    obj: Any
    cycle: tuple[Any]
    msg: str = "{obj} contains the cycle {cycle}"
