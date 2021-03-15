"""
Type aliases useful to type JSON
"""
from __future__ import annotations

from abc import abstractmethod
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    List,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pheres.utils import singleton

__all__ = [
    "FilePosition",
    "Inline",
    "JSON",
    "Jsonable",
    "NotRequired",
    "Position",
    "RawJSON",
    "RawPosition",
    "Required",
]

_JSONValueTypes: Tuple[type, ...] = (type(None), int, bool, float, str)
_JSONArrayTypes: Tuple[type, ...] = (tuple, list)
_JSONObjectTypes: Tuple[type, ...] = (dict,)

REQUIRED = singleton("required")
NOT_REQUIRED = singleton("not_required")
INLINE = singleton("inline")

T = TypeVar("T")
JSON_contra = TypeVar("T_contra", contravariant=True, bound="JSON")

Required = Annotated[T, REQUIRED]
NotRequired = Annotated[T, NOT_REQUIRED]
Inline = Annotated[T, INLINE]

FilePosition = int
RawPosition = Tuple[Union[int, str], ...]
Position = Union[FilePosition, RawPosition]


class Jsonable(Protocol[JSON_contra]):
    """
    Protocol for Jsonable classes
    """

    @classmethod
    @abstractmethod
    def from_json(cls: Type[T], obj: JSON_contra) -> T:
        """
        Deserialize an instance of the class from Json
        """
        raise NotImplementedError

    @abstractmethod
    def to_json(self) -> RawJSON:
        """
        Serializes an instance of the class to Json
        """
        raise NotImplementedError


# NOTE: Mypy doesn't yet support recursive types, those are placeholders
RawJSON = Any
JSON = Any
#
# RawJson = Union[  # pylint: disable=unsubscriptable-object
#     None, bool, int, float, str, List["RawJson"], Tuple["RawJson"], Dict[str, "RawJson"]
# ]
# """Type hint for any raw JSON (composed only of builtin types)"""

# Json = Union[ # pylint: disable=unsubscriptable-object
#     None, bool, int, float, str, List["RawJson"], Tuple["RawJson"], Dict[str, "RawJson"], Jsonable
# ]
# """Type hint for any JSON, including Jsonable classes"""
