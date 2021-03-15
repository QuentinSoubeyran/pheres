from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, TypeVar, Union

from pheres.utils import MISSING, Singleton

__all__ = [
    "TypeForm",
    "OneOrMore",
    "NoneOrMore",
    "Maybe",
]

T = TypeVar("T")

TypeForm = Any
"""Type alias representing a Type form object, as created by the `typing` module"""
TypeOrig = Optional[TypeForm]  # pylint: disable=unsubscriptable-object
"""Type alias for the return value of `typing.get_origin()`"""
TypeArgs = Tuple[Optional[TypeForm], ...]  # pylint: disable=unsubscriptable-object
"""Type alias for the return value if `typing.get_args()`"""

OneOrMore = Union[T, Iterable[T]]  # pylint: disable=unsubscriptable-object
NoneOrMore = Optional[OneOrMore[T]]  # pylint: disable=unsubscriptable-object
Maybe = Union[T, Singleton]
