"""
Module for improved JSON objects
"""
from __future__ import annotations

from pathlib import Path
from typing import Generic, Iterable, Optional, Tuple, Type, TypeVar

import attr

from pheres.utils import NoneOrMore

AnyClass = TypeVar("AnyClass")


@attr.dataclass(frozen=True)
class FileSource:
    file: Path
    line: int
    col: int

    def __str__(self):
        return f"{self.file!s} at line {self.line!s}, col {self.col!s}"


def _parse_sources(sources: NoneOrMore[FileSource]) -> Tuple[FileSource, ...]:
    if not sources:
        return tuple()
    if isinstance(sources, Iterable):
        return tuple(sources)
    return (sources,)


class _JsonBase(Generic[AnyClass]):
    """
    Shared base class for advanced JSON object
    """

    _sources_: Tuple[FileSource, ...]

    def __new__(cls, *args, sources: NoneOrMore[FileSource] = None, **kwargs):
        inst = super().__new__(cls, *args, **kwargs)  # type: ignore
        inst._sources_ = _parse_sources(sources)
        return inst

    def __str__(self) -> str:
        if super(_JsonBase, type(self)).__str__ is object.__str__:
            return super().__repr__()
        return super().__str__()

    def __repr__(self) -> str:
        cls = type(self)
        return "%s%s(%s%s)" % (
            "" if cls.__module__ == "__main__" else cls.__module__ + ".",
            cls.__qualname__,
            super().__repr__(),
            "" if not self.sources else f"sources={self.sources!r}",
        )

    @property
    def sources(self) -> Tuple[FileSource, ...]:
        return self._sources_

    @property
    def formatted_sources(self) -> str:
        return ", ".join(str(s) for s in self.sources)

    def raw(self) -> AnyClass:
        # __self__ is documented to be the class defining the method
        # https://docs.python.org/3/reference/datamodel.html
        # mypy doesn't seems to know that
        cls = super(_JsonBase, type(self)).__new__.__self__  # type: ignore
        return cls(self)


class Null(_JsonBase[None], object):
    """
    Json null values
    """

    def __repr__(self):
        cls = type(self)
        return "%s%s(%s)" % (
            "" if cls.__module__ == "__main__" else cls.__module__ + ".",
            cls.__qualname__,
            "" if not self.sources else f"sources={self.sources!r}",
        )

    def __bool__(self) -> bool:
        return False

    def raw(self) -> None:
        return None


class Bool(_JsonBase[bool], int):
    """Json booleans"""

    def __new__(cls, x, sources: NoneOrMore[FileSource] = None):
        inst = int.__new__(cls, bool(x))
        inst.sources = _parse_sources(sources)
        return inst

    def raw(self) -> bool:
        return bool(self)


class Int(_JsonBase[int], int):
    """Json integer"""


class Float(_JsonBase[float], float):
    """Json floats"""


class Array(_JsonBase[list], list):
    """Json arrays"""


class Object(_JsonBase[dict], dict):
    """Json objects"""
