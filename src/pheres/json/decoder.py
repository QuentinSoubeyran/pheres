"""
Typed JSON decoding
"""
from __future__ import annotations

import functools
import json
import re
import types
import typing
from abc import ABC, ABCMeta, abstractmethod
from json import JSONDecodeError, JSONDecoder
from json.decoder import (  # type: ignore[attr-defined]
    WHITESPACE,
    WHITESPACE_STR,
    scanstring,
)
from json.scanner import NUMBER_RE  # type: ignore[import]
from os import PathLike
from typing import (
    IO,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr

from pheres.aliases import Maybe, TypeArgs, TypeForm, TypeOrig
from pheres.core import EvalForwardRef, get_outer_namespaces
from pheres.data import get_data_json
from pheres.exceptions import PheresInternalError
from pheres.json.aliases import (
    INLINE,
    JSON,
    NOT_REQUIRED,
    REQUIRED,
    FilePosition,
    RawJSON,
    RawPosition,
    _JSONArrayTypes,
    _JSONObjectTypes,
    _JSONValueTypes,
)
from pheres.json.exceptions import (
    JsonableAttrError,
    JsonableTypeFormError,
    TypedJSONDecodeError,
    UnparametrizedDecoderError,
)
from pheres.utils import MISSING, Commented, docstring, exec_body_factory

_TypedDictMeta: type = typing._TypedDictMeta  # type: ignore[attr-defined]

__all__ = [
    "FileSource",
    "TypedDecoder",
    "UsableDecoder",
]


T = TypeVar("T")
_DocT = TypeVar("_DocT", str, RawJSON)
_PosT = TypeVar("_PosT", FilePosition, RawPosition)
_CursorT = TypeVar("_CursorT", bound="_CursorABC")


def _unpack_union_helper(
    eval_fref: Callable[[TypeForm], TypeForm]
) -> Callable[[TypeForm], Generator[TypeForm, None, None]]:
    def aux(form: TypeForm):
        form = eval_fref(form)
        if typing.get_origin(form) is Union:
            for g in map(aux, typing.get_args(form)):
                yield from g
        else:
            yield form

    return aux


def _unpack_union(
    form: TypeForm, eval_fref: Callable[[TypeForm], TypeForm]
) -> Tuple[TypeForm, ...]:
    """
    Recursively unpack Union type hints and resolve ForwardRef
    """
    return tuple(_unpack_union_helper(eval_fref)(form))


def _unpack_annotated(
    form: TypeForm, eval_fref: Callable[[TypeForm], TypeForm]
) -> Tuple[TypeForm, List[Any]]:
    annotations: List[Any] = []
    form = eval_fref(form)
    while typing.get_origin(form) is typing.Annotated:
        form, *annots = typing.get_args(form)
        annotations.extend(annots)
    return form, annotations


def _unpack_as_typeinfos(
    form: TypeForm, index: int, eval_fref: EvalForwardRef
) -> Generator["_TypeInfo", None, None]:
    """
    Unpacks Unions and jsonable classes into _TypeInfo objects
    """
    for tp in _unpack_union(form, eval_fref):
        # Unpack jsonable class of a Union
        if isinstance(tp, type) and (data := get_data_json(tp)) is not None:
            for inner_type in _unpack_union(data.typeform, eval_fref):
                yield _TypeInfo(index, inner_type, cls=tp)
        else:
            yield _TypeInfo(index, tp)


@functools.lru_cache(maxsize=256)
def _get_object_keys(typeform: type) -> Tuple[set[str], set[str]]:
    """Return the required and not required keys of a TypedDict"""
    if isinstance(typeform, type) and (data := get_data_json(typeform)) is not None:
        typed_dict = typing.cast(type, data.typeform)
    elif isinstance(typeform, _TypedDictMeta):
        typed_dict = typeform
    else:
        raise PheresInternalError("{typeform!r} is not a JSON Object typeform")
    required: set[str] = set()
    not_required: set[str] = set()
    # start by the root of the class hierachy
    for base in (
        base
        for base in reversed(typed_dict.__mro__)
        if isinstance(base, _TypedDictMeta)
    ):
        default = required if base.__total__ else not_required  # type: ignore[attr-defined]
        hints = typing.get_type_hints(base, include_extras=True)
        # get_type_hints also returns hints from parent classes, ignore those
        for key in hints.keys() - required - not_required:
            form, annots = _unpack_annotated(
                hints[key], EvalForwardRef.from_class(base)
            )
            if REQUIRED in annots and NOT_REQUIRED in annots:
                raise JsonableTypeFormError(
                    base,
                    msg=f"Key '{key}' in {base.__name__} is marked as required and not required",
                )
            elif INLINE in annots:
                req, unreq = _get_object_keys(form)
                not_required.update(unreq)
                if REQUIRED in annots:
                    required.update(req)
                elif NOT_REQUIRED in annots:
                    not_required.update(req)
                else:
                    default.update(req)
            elif REQUIRED in annots:
                required.add(key)
            elif NOT_REQUIRED in annots:
                not_required.add(key)
            else:
                default.add(key)
    return required, not_required


@functools.lru_cache(maxsize=256)
def _get_flat_hints(typeform: type) -> Dict[str, TypeForm]:
    """
    Returns the annotations of TypedDict with flattened Inlined types
    """
    if isinstance(typeform, type) and (data := get_data_json(typeform)) is not None:
        typed_dict = typing.cast(type, data.typeform)
    elif isinstance(typeform, _TypedDictMeta):
        typed_dict = typeform
    else:
        raise PheresInternalError("{typeform!r} is not a JSON Object typeform")
    hints: Dict[str, TypeForm] = {}
    for key, hint in typing.get_type_hints(typed_dict, include_extras=True).items():
        form, annots = _unpack_annotated(hint, lambda x: x)
        if INLINE in annots:
            inner_hints = _get_flat_hints(form)
            if inner_hints.keys() & hints.keys():
                raise JsonableAttrError(
                    typed_dict.__name__,
                    inner_hints.keys() & hints.keys(),
                    detail=f"conflict with inlined {form.__name__}",
                )
            hints.update(inner_hints)
        else:
            hints[key] = form
    return hints


@functools.lru_cache(maxsize=256)
def _get_inlined(typed_dict: type) -> Dict[str, Tuple[type, Iterable[str]]]:
    inlined: Dict[str, Tuple[type, Iterable[str]]] = {}
    for key, typeform in typing.get_type_hints(typed_dict, include_extras=True).items():
        form, annots = _unpack_annotated(
            typeform, EvalForwardRef.from_class(typed_dict)
        )
        if INLINE in annots:
            inlined[key] = (form, _get_flat_hints(form).keys())
    return inlined


@functools.lru_cache(maxsize=256)
def _get_default_attrs(typed_dict: type) -> Dict[str, Any]:
    return {
        k: getattr(typed_dict, k)
        for k in getattr(typed_dict, "__annotations__", {}).keys()
        if hasattr(typed_dict, k)
    }


def _instanciate_typed_dict(typeform: TypeForm, obj: dict) -> JSON:
    if isinstance(typeform, _TypedDictMeta):
        cls: type = object
        typed_dict = typeform
    elif isinstance(typeform, type) and (data := get_data_json(typeform)) is not None:
        cls = typeform
        typed_dict = data.typeform
    else:
        raise PheresInternalError("Unhandled typed dict typeform {typeform!r}")
    for key, (inlined_type, inlined_keys) in _get_inlined(typed_dict).items():
        inlined_obj = {k: obj.pop(k) for k in inlined_keys if k in obj}
        obj[key] = _instanciate_typed_dict(inlined_type, inlined_obj)
    obj = _get_default_attrs(typed_dict) | obj
    if cls is not object:
        return cls(**obj)
    else:
        return obj


@attr.dataclass(slots=True)
class FileSource:
    filename: str
    char: int
    lineno: int
    colno: int


class _CursorABC(Generic[_DocT, _PosT], ABC):
    s: _DocT
    pos: _PosT
    lineno: int = MISSING
    colno: int = MISSING

    def decode_error(self, msg: str) -> TypedJSONDecodeError:
        return TypedJSONDecodeError(
            msg=msg, doc=self.s, pos=self.pos, lineno=self.lineno, colno=self.colno
        )


# Modified regex similar to json.decoder.WHITESPACE
# This one counts whitespaces after the last line return
# this is necessary for keeping track of the column number
_match_whitespace: Callable = re.compile("([ \t\r]*\n)*([ \t\n\r]*)").match


@attr.dataclass(slots=True)
class _Cursor(_CursorABC[str, FilePosition]):

    src: str
    s: str = attr.ib(repr=False)  # too long
    pos: FilePosition
    lineno: int = MISSING
    colno: int = MISSING
    nextchar: str = MISSING

    def __attrs_post_init__(self):
        if self.lineno is MISSING:
            self.lineno = 1 + self.s.count("\n", 0, self.pos)
        if self.colno is MISSING:
            self.colno = self.pos - self.s.rfind("\n", 0, self.pos)
        if self.nextchar is MISSING:
            self.nextchar = self.s[self.pos : self.pos + 1]

    def skip_whitespace(self, match_whitespace=_match_whitespace):
        m = match_whitespace(self.s, self.pos)
        count = self.s.count("\n", self.pos, m.end())
        self.pos = m.end()
        self.lineno += count
        self.colno = (1 if count else self.colno) + len(m[2])
        self.nextchar = self.s[self.pos : self.pos + 1]

    def advance(self, to: int = None) -> "_Cursor":
        if to is None:
            self.pos += 1
            if self.nextchar == "\n":
                self.lineno += 1
                self.colno = 1
            else:
                self.colno += 1
        else:
            count = self.s.count("\n", self.pos, to)
            self.lineno += count
            if count:
                self.colno = to - self.s.rfind("\n", self.pos, to)
            else:
                self.colno += to - self.pos
            self.pos = to
        self.nextchar = self.s[self.pos : self.pos + 1]
        return self

    def comment(self, value: T) -> Union[T, Commented[T]]:
        if self.s:
            source = FileSource(self.src, self.pos, self.lineno, self.colno)
            return Commented(value, sources=[source])
        return value


@attr.dataclass(slots=True, auto_detect=True)
class _TypeInfo:
    """
    Internal class to store types and related infos

    Attributes:
        index: index in the parent _DecodeContext that created this type
        typeform: the type accepted by this _TypeInfo
        cls: the jsonable class that was unpacked into this type, if any
    """

    index: int
    typeform: TypeForm
    cls: Optional[type] = None

    # caches
    _orig: Maybe[TypeOrig] = attr.ib(
        MISSING, init=False, repr=False, eq=False, order=False
    )
    _args: Maybe[TypeArgs] = attr.ib(
        MISSING, init=False, repr=False, eq=False, order=False
    )

    def __attrs_post_init__(self):
        if self.orig is Union:
            raise PheresInternalError("TypeInfo typeform cannot be Union")

    @property
    def orig(self) -> TypeOrig:
        if self._orig is MISSING:
            self._orig = typing.get_origin(self.typeform)
        return typing.cast(TypeOrig, self._orig)

    @property
    def args(self) -> TypeArgs:
        if self._args is MISSING:
            self._args = typing.get_args(self.typeform)
        return typing.cast(TypeArgs, self._args)

    @property
    def type(self) -> type:
        """
        The base type of this TypeInfo
        """
        if isinstance(self.typeform, type):
            return self.typeform
        elif isinstance(self.orig, type):
            return self.orig
        else:
            raise PheresInternalError(f"{self!r} doesn't have a type")

    def get_subtype(self, key: Union[int, str]) -> Maybe[TypeForm]:
        """
        If this type is a recursive one (e.g. list, dict...), the subtype at the given key
        """
        if (
            isinstance(key, int)
            and isinstance(self.orig, type)
            and issubclass(self.orig, _JSONArrayTypes)
        ):
            if issubclass(self.orig, tuple):
                if len(self.args) == 2 and self.args[1] == Ellipsis:
                    return self.args[0]
                else:
                    return self.args[key] if key < len(self.args) else MISSING
            elif issubclass(self.orig, list):
                return self.args[0]
            else:
                raise PheresInternalError(
                    f"Unhandled array type {self.typeform!r} in TypeInfo.get_subtypes()"
                )
        elif isinstance(key, str):
            if isinstance(self.typeform, _TypedDictMeta):
                return _get_flat_hints(self.typeform).get(key, MISSING)
            elif isinstance(self.orig, type) and issubclass(
                self.orig, _JSONObjectTypes
            ):
                if issubclass(self.orig, dict):
                    return self.args[1]
                else:
                    raise PheresInternalError(
                        f"Unhandled object type {self.typeform!r} in TypeInfo.get_subtypes()"
                    )
        return MISSING

    def validate(self, obj: JSON) -> bool:
        """
        Check that an object is valid

        Only performs superficial tests -- that is, does not recurse into list or
        dict. This is unnecessary as those checks where already performed during
        the decoding process
        """
        if self.orig is Literal:
            return obj in self.args
        elif isinstance(self.type, _TypedDictMeta):
            required, not_required = _get_object_keys(self.type)
            return (
                required <= typing.cast(dict, obj).keys() <= (required | not_required)
            )
        elif issubclass(self.type, tuple):
            return isinstance(obj, list) and (
                (len(self.args) == 2 and self.args[1] == Ellipsis)
                or (len(typing.cast(tuple, obj)) == len(self.args))
            )
        else:
            return isinstance(obj, self.type)

    def instanciate(self, obj: JSON) -> JSON:
        """
        Instanciate a jsonable class, if any
        """
        if self.cls is None:
            return obj
        elif isinstance(self.type, _JSONValueTypes):
            return self.cls(obj)
        elif isinstance(self.type, _TypedDictMeta):
            return _instanciate_typed_dict(self.cls, obj)
        elif issubclass(self.type, tuple):
            return self.cls(*obj)
        elif issubclass(self.type, list):
            return self.cls(obj)
        elif issubclass(self.type, dict):
            return self.cls(obj)
        else:
            raise PheresInternalError(
                f"Unhandled type {self.type!r} (from {self.typeform!r}) in TypeInfo.instanciate()"
            )


def _typeinfo_priority(ti: _TypeInfo) -> Tuple[bool, bool]:
    """
    Defines the priority of _TypeInfo objects
    """
    # False < True so invert tests
    return (
        # jsonable classes have priority
        ti.cls is None,
        # tuples have priority
        not (isinstance(ti.orig, type) and issubclass(ti.orig, tuple)),
    )


class _ContextABC(Generic[_DocT, _PosT], ABC):
    @abstractmethod
    def get_subcontext(
        self,
        /,
        key: Union[int, str],
        cursor: _CursorABC[_DocT, _PosT],
    ) -> _ContextABC[_DocT, _PosT]:
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        /,
        obj: JSON,
        start_cursor: _CursorABC[_DocT, _PosT],
        end_cursor: _CursorT,
    ) -> Tuple[Any, _CursorT, _ContextABC[_DocT, _PosT]]:
        raise NotImplementedError


class _DummyContext(_ContextABC[_DocT, _PosT]):
    def get_subcontext(
        self,
        /,
        key: Union[int, str],
        cursor: _CursorABC[_DocT, _PosT],
    ) -> "_DummyContext[_DocT, _PosT]":
        return self

    def validate(
        self,
        /,
        obj: JSON,
        start_cursor: _CursorABC[_DocT, _PosT],
        end_cursor: _CursorT,
    ) -> Tuple[Any, _CursorT, "_DummyContext[_DocT, _PosT]"]:
        return obj, end_cursor, self


@attr.dataclass(frozen=True)
class _DecodeContext(_ContextABC[_DocT, _PosT]):
    """
    Internal class to infer types during JSON decoding

    DecodeContext are immutable

    Attributes:
        type_infos: type form still valid, given what was decoded so far
        eval_fref: callable to resolve ForwardRef in typeforms
        parent: parent DecodeContext that created this one
    """

    type_infos: Tuple[_TypeInfo, ...]
    eval_fref: EvalForwardRef

    parent: Optional["_DecodeContext[_DocT, _PosT]"] = None

    @classmethod
    def from_typeform(
        cls,
        /,
        typeform: TypeForm,
        eval_fref: EvalForwardRef,
    ) -> "_DecodeContext[_DocT, _PosT]":
        return _DecodeContext(
            type_infos=tuple(_unpack_as_typeinfos(typeform, -1, eval_fref)),
            eval_fref=eval_fref,
        )

    def err_msg(
        self, /, value: Any = MISSING, *, pre: str = None, post: str = None
    ) -> str:
        parts = []
        if value is MISSING and not pre and not post:
            pre = "expected"
        if pre:
            parts.append(pre)
        if value is not MISSING:
            parts.append(f"{value!r} (type {type(value).__name__}) doesn't match")
        typeforms = tuple(ti.typeform for ti in self.type_infos)
        parts.append(f"inferred type {Union[typeforms]!r}")
        if post:
            parts.append(post)
        msg = " ".join(parts)
        return msg[:1].upper() + msg[1:]

    def get_subcontext(
        self,
        /,
        key: Union[int, str],
        cursor: _CursorABC[_DocT, _PosT],
    ) -> "_DecodeContext[_DocT, _PosT]":
        # Filter parent types for those that are valid
        type_infos: List[_TypeInfo] = []
        subtypes: List[_TypeInfo] = []
        for ti in self.type_infos:
            subtype = ti.get_subtype(key)
            if subtype is not MISSING:
                type_infos.append(ti)
                subtypes.extend(
                    _unpack_as_typeinfos(subtype, len(type_infos) - 1, self.eval_fref)
                )
        # Check for errors
        if not type_infos:
            if isinstance(key, int):
                errmsg = self.err_msg(f"<Array of length {key+1} or longer>")
            else:
                errmsg = self.err_msg(post=f"doesn't have a key '{key}'")
            raise cursor.decode_error(errmsg)
        return _DecodeContext(
            type_infos=tuple(subtypes),
            eval_fref=self.eval_fref,
            parent=attr.evolve(self, type_infos=tuple(type_infos)),
        )

    def validate(
        self,
        /,
        obj: JSON,
        start_cursor: _CursorABC[_DocT, _PosT],
        end_cursor: _CursorT,
    ) -> Tuple[JSON, _CursorT, "_DecodeContext[_DocT, _PosT]"]:
        # Find valid types
        type_infos = [ti for ti in self.type_infos if ti.validate(obj)]
        # Check for no valid types
        if not type_infos:
            raise start_cursor.decode_error(self.err_msg(value=obj))
        # Check for conflicting jsonable classes
        classes = tuple(ti.cls for ti in type_infos if ti.cls is not None)
        classes = tuple(
            c for i, c in enumerate(classes) if not issubclass(c, classes[i + 1 :])
        )
        if len(classes) > 1:
            msg = ", ".join(c.__name__ for c in classes)
            raise start_cursor.decode_error(
                f"Inferred mutiple jsonable classes ({msg}) for {obj!r}"
            )
        # Instanciate the object
        obj = sorted(type_infos, key=_typeinfo_priority)[0].instanciate(obj)
        # Infer parent types
        if self.parent is not None:
            indexes = {ti.index for ti in type_infos}
            parent = attr.evolve(
                self.parent,
                type_infos=tuple(
                    self.parent.type_infos[index] for index in sorted(indexes)
                ),
            )
            return obj, end_cursor, parent
        else:
            return obj, end_cursor, attr.evolve(self, type_infos=())


#################################
# Builtin json module overrides #
#################################

# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/scanner.py
# See also LICENSE file


# Has to be a protocol, because mypy doesn't support callable attributes
# and the scanner ultimately ends as an attribute to TypedDecoder
class _ScannerProto(Protocol[_DocT, _PosT]):
    def __call__(
        self, cursor: _Cursor, ctx: _ContextABC[_DocT, _PosT]
    ) -> Tuple[JSON, _Cursor, _ContextABC[_DocT, _PosT]]:
        ...


def _make_string_scanner(
    decoder: "_UsableDecoderProto",
) -> _ScannerProto[str, FilePosition]:
    parse_object = decoder.parse_object
    parse_array = decoder.parse_array
    parse_string = decoder.parse_string
    match_number = NUMBER_RE.match
    strict = decoder.strict
    parse_float = decoder.parse_float
    parse_int = decoder.parse_int
    parse_constant = decoder.parse_constant
    object_hook = decoder.object_hook
    object_pairs_hook = decoder.object_pairs_hook
    memo = decoder.memo

    def _scan_once(
        cursor: _Cursor, ctx: _ContextABC[str, FilePosition]
    ) -> Tuple[JSON, _Cursor, _ContextABC[str, FilePosition]]:
        string = cursor.s
        idx = cursor.pos
        nextchar = cursor.nextchar
        if nextchar == "":
            raise StopIteration(cursor.pos)
        if nextchar == '"':
            s, end = parse_string(string, cursor.pos + 1, strict)
            # Python guarantees left to right evaluation - mutable cursor is OK
            return ctx.validate(
                cursor.comment(s), attr.evolve(cursor), cursor.advance(end)
            )
        elif nextchar == "{":
            return parse_object(
                cursor,
                strict,
                _scan_once,
                ctx,
                object_hook,
                object_pairs_hook,
                memo,
            )
        elif nextchar == "[":
            return parse_array(cursor, _scan_once, ctx)
        elif nextchar == "n" and string[idx : idx + 4] == "null":
            return ctx.validate(cursor.comment(None), cursor, cursor.advance(idx + 4))
        elif nextchar == "t" and string[idx : idx + 4] == "true":
            return ctx.validate(cursor.comment(True), cursor, cursor.advance(idx + 4))
        elif nextchar == "f" and string[idx : idx + 5] == "false":
            return ctx.validate(cursor.comment(False), cursor, cursor.advance(idx + 5))

        m = match_number(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or "") + (exp or ""))
            else:
                res = parse_int(integer)
            return ctx.validate(cursor.comment(res), cursor, cursor.advance(m.end()))
        elif nextchar == "N" and string[idx : idx + 3] == "NaN":
            return ctx.validate(
                cursor.comment(parse_constant("NaN")), cursor, cursor.advance(idx + 3)
            )
        elif nextchar == "I" and string[idx : idx + 8] == "Infinity":
            return ctx.validate(
                cursor.comment(parse_constant("Infinity")),
                cursor,
                cursor.advance(idx + 8),
            )
        elif nextchar == "-" and string[idx : idx + 9] == "-Infinity":
            return ctx.validate(
                cursor.comment(parse_constant("-Infinity")),
                cursor,
                cursor.advance(idx + 9),
            )
        else:
            raise StopIteration(idx)

    def scan_once(
        cursor: _Cursor, ctx: _ContextABC[str, FilePosition]
    ) -> Tuple[JSON, _Cursor, _ContextABC[str, FilePosition]]:
        try:
            return _scan_once(cursor, ctx)
        finally:
            memo.clear()

    return scan_once


# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
# See also LICENSE file


class _JSONObjectParserProto(Protocol):
    def __call__(
        self,
        cursor: _Cursor,
        strict: bool,
        scan_once: _ScannerProto[str, FilePosition],
        ctx: _ContextABC[str, FilePosition],
        object_hook: "_ObjectHookProto",
        object_pairs_hook: "_ObjectPairsHookProto",
        memo: Optional[dict] = None,  # pylint: disable=unsubscriptable-object
        _ws: str = WHITESPACE_STR,
    ) -> Tuple[JSON, _Cursor, _ContextABC[str, FilePosition]]:
        ...


def _ParseJSONObject(
    cursor: _Cursor,
    strict: bool,
    scan_once: _ScannerProto[str, FilePosition],
    ctx: _ContextABC[str, FilePosition],
    object_hook: "_ObjectHookProto",
    object_pairs_hook: "_ObjectPairsHookProto",
    memo: Optional[dict] = None,  # pylint: disable=unsubscriptable-object
    _ws: str = WHITESPACE_STR,
) -> Tuple[JSON, _Cursor, _ContextABC[str, FilePosition]]:
    pairs: List[Tuple[str, JSON]] = []
    pairs_append = pairs.append
    # Backwards compatibility
    if memo is None:
        memo = {}
    memo_get = memo.setdefault
    start_cursor = attr.evolve(cursor)
    cursor.advance()
    # Normally we expect nextchar == '"'
    if cursor.nextchar != '"':
        cursor.skip_whitespace()
        # Trivial empty object
        if cursor.nextchar == "}":
            if object_pairs_hook is not None:
                result = start_cursor.comment(object_pairs_hook(pairs))
                return ctx.validate(result, start_cursor, cursor.advance())
            result = {}
            if object_hook is not None:
                result = start_cursor.comment(object_hook(result))
            return ctx.validate(result, start_cursor, cursor.advance())
        elif cursor.nextchar != '"':
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes",
                cursor.s,
                cursor.pos,
            )
    key_cursor = attr.evolve(cursor)
    cursor.advance()
    while True:
        key, end = scanstring(cursor.s, cursor.pos, strict)
        cursor.advance(end)
        key = memo_get(key, key)
        key = key_cursor.comment(key)
        # To skip some function call overhead we optimize the fast paths where
        # the JSON key separator is ": " or just ":".
        if cursor.nextchar != ":":
            cursor.skip_whitespace()
            if cursor.nextchar != ":":
                raise JSONDecodeError("Expecting ':' delimiter", cursor.s, cursor.pos)
        cursor.advance()

        try:
            if cursor.nextchar in _ws:
                cursor.advance()
                if cursor.nextchar in _ws:
                    cursor.skip_whitespace()
        except IndexError:
            pass
        # value_cursor = attr.evolve(cursor)
        try:
            value, cursor, ctx = scan_once(
                cursor, ctx.get_subcontext(key, key_cursor)
            )
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", cursor.s, err.value) from None
        # value = value_cursor.source(value)
        pairs_append((key, value))
        if cursor.nextchar in _ws:
            cursor.skip_whitespace()
        if cursor.nextchar == "}":
            cursor.advance()
            break
        elif cursor.nextchar != ",":
            raise JSONDecodeError("Expecting ',' delimiter", cursor.s, cursor.pos)
        cursor.advance()
        cursor.skip_whitespace()
        if cursor.nextchar != '"':
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes",
                cursor.s,
                cursor.pos,
            )
        key_cursor = attr.evolve(cursor)
        cursor.advance()
    if object_pairs_hook is not None:
        result = object_pairs_hook(pairs)
        return ctx.validate(start_cursor.comment(result), start_cursor, cursor)
    result = dict(pairs)
    if object_hook is not None:
        result = object_hook(pairs)
    return ctx.validate(start_cursor.comment(result), start_cursor, cursor)


# Original Source code at:
# https://github.com/python/cpython/blob/3.8/Lib/json/decoder.py
# See also LICENSE file
class _JSONArrayParserProto(Protocol):
    def __call__(
        self,
        cursor: _Cursor,
        scan_once: _ScannerProto[str, FilePosition],
        ctx: _ContextABC[str, FilePosition],
        _ws: str = WHITESPACE_STR,
    ) -> Tuple[JSON, _Cursor, _ContextABC[str, FilePosition]]:
        ...


def _ParseJSONArray(
    cursor: _Cursor,
    scan_once: _ScannerProto[str, FilePosition],
    ctx: _ContextABC[str, FilePosition],
    _ws: str = WHITESPACE_STR,
) -> Tuple[JSON, _Cursor, _ContextABC[str, FilePosition]]:
    # Wrap the list in a Commented object
    values: Union[List[JSON], Commented[List[JSON]]] = cursor.comment([])
    start_cursor = attr.evolve(cursor)
    cursor.advance()
    if cursor.nextchar in _ws:
        cursor.skip_whitespace()
    # Look-ahead for trivial empty array
    if cursor.nextchar == "]":
        return ctx.validate(values, start_cursor, cursor.advance())
    _append = values.append  # method still accessible through Commented wrapper
    i = -1
    while True:
        try:
            i += 1
            # value_cursor = attr.evolve(cursor)
            value, cursor, ctx = scan_once(cursor, ctx.get_subcontext(i, cursor))
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", cursor.s, err.value) from None
        # value = value_cursor.source(value)
        _append(value)
        if cursor.nextchar in _ws:
            cursor.skip_whitespace()
        if cursor.nextchar == "]":
            cursor.advance()
            break
        elif cursor.nextchar != ",":
            raise JSONDecodeError("Expecting ',' delimiter", cursor.s, cursor.pos)
        cursor.advance()
        if cursor.nextchar in _ws:
            cursor.advance()
            if cursor.nextchar in _ws:
                cursor.skip_whitespace()
    return ctx.validate(values, start_cursor, cursor)


##################
# Typed decoding #
##################


class _ObjectHookProto(Protocol):
    def __call__(self, obj: JSON) -> Any:
        ...


class _ObjectPairsHookProto(Protocol):
    def __call__(self, pairs: List[Tuple[str, JSON]]) -> Any:
        ...


class _IntParserProto(Protocol):
    def __call__(self, obj: str) -> int:
        ...


class _FloatParserProto(Protocol):
    def __call__(self, obj: str) -> float:
        ...


class _ConstantParserProto(Protocol):
    def __call__(self, key: str) -> JSON:
        ...


class _StringParserProto(Protocol):
    def __call__(
        self, s: str, end: FilePosition, strict: bool = True
    ) -> Tuple[str, FilePosition]:
        ...


class _UsableDecoderProto(Protocol):
    strict: bool
    memo: dict

    parse_int: _IntParserProto
    parse_float: _FloatParserProto
    parse_constant: _ConstantParserProto
    parse_string: _StringParserProto
    parse_array: _JSONArrayParserProto
    parse_object: _JSONObjectParserProto

    object_hook: _ObjectHookProto
    object_pairs_hook: _ObjectPairsHookProto


class UsableDecoder(json.JSONDecoder):
    """
    `json.JSONDecoder` subclass with wrapper methods for ``load()`` and
    ``loads()`` using itself for the decoder class
    """

    src: str
    scan_once: _ScannerProto[str, FilePosition]

    def __init__(self, *args, **kwargs):
        self.src = kwargs.pop("src", "")
        super().__init__(*args, **kwargs)
        # Replace default decoder by our contextualized decoder
        self.parse_object = _ParseJSONObject
        self.parse_array = _ParseJSONArray
        self.scan_once = _make_string_scanner(self)

    @classmethod
    @docstring(
        pre="""
            Load a json file using this class as the Decoder.
            
            Supports all the arguments supported by `open`.
            Wrapped function docstring:\n    """
    )
    @functools.wraps(json.load)
    def loadf(
        cls,
        file: Union[str, bytes, int, PathLike[Any]],
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        closefd: bool = True,
        opener: Optional[Callable[[str, int], int]] = None,
        **kwargs,
    ) -> Union[JSON, Commented[JSON]]:
        if cls is TypedDecoder:
            raise TypeError(f"You must parametrize {cls.__name__} before using it")
        with open(
            file,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        ) as fp:
            return cls.load(fp=fp, **kwargs)

    @classmethod
    @docstring(
        pre="""
            Wrapper of `json.load` that use this class as the Decoder.
            
            Wrapped function docstring:\n    """
    )
    @functools.wraps(json.load)
    def load(cls, fp: IO[str], *args, **kwargs) -> Union[JSON, Commented[JSON]]:
        if cls is TypedDecoder:
            raise TypeError(f"You must parametrize {cls.__name__} before using it")
        kwargs["src"] = fp.name
        return json.load(fp, *args, cls=cls, **kwargs)

    @classmethod
    @docstring(
        pre="""
        Wrapper around `json.loads` that use this class as the Decoder.
        
        Wrapped function docstring:\n    """
    )
    @functools.wraps(json.loads)
    def loads(cls, *args, **kwargs) -> Union[JSON, Commented[JSON]]:
        if cls is TypedDecoder:
            raise TypeError(f"You must parametrize {cls.__name__} before using it")
        return json.loads(*args, cls=cls, **kwargs)

    @docstring(post="\n:meta private:")
    @functools.wraps(JSONDecoder.raw_decode)
    def raw_decode(self, s: str, idx: int = 0):
        cursor = _Cursor(src=self.src, s=s, pos=idx)
        context = _DummyContext()
        try:
            obj, cursor, _ = self.scan_once(
                cursor,
                ctx=context,
            )
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        return obj, cursor.pos


def _tp_cache(func: T) -> T:
    """Wrapper caching __class_getitem__ on type hints

    Provides a fallback if arguments are not hashable
    """
    cache = functools.lru_cache()(func)  # type: ignore[arg-type, var-annotated]

    @functools.wraps(func)  # type: ignore[arg-type]
    def wrapper(*args, **kwargs):
        try:
            return cache(*args, **kwargs)
        except TypeError as err:  # unhashable args
            print(err)  # TODO: remove
            pass
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class _ParametrizedTypedDecoderMeta(ABCMeta):
    """
    Metaclass for parametrized TypedJSONDecoder classes -- provides a nice repr()

    :meta private:
    """

    def __repr__(self):
        tp = self.typeform
        if tp is MISSING:
            return "TypedJSONDecoder"
        elif isinstance(tp, type):
            return f"TypedJSONDecoder[{tp.__name__}]"
        else:
            return f"TypedJSONDecoder[{self.typeform!r}]"


class TypedDecoder(ABC, UsableDecoder):
    """
    `jUsableJSONDecoder` subclass for typed JSON decoding

    The type to decode must be provided by indexing this class with the
    type as key (like in the 'typing' module). The type hint must be
    valid in a JSON context.

    jsonables are supported, as this is the whole point
    of that class

    Example::

        # type check that all values are str or int
        json.load(..., cls=JSONableDecoder[Dict[str, int]])
        # The class exposed a `load` method
        JSONableDecoder[Dict[str, int]].load(...)
    """

    typeform: ClassVar[Maybe[TypeForm]] = MISSING
    eval_fref: ClassVar[Optional[EvalForwardRef]] = None

    @classmethod
    @_tp_cache
    def _cached_class_getitem_(
        cls, typeform: TypeForm, eval_fref: EvalForwardRef = None
    ) -> Type["TypedDecoder"]:
        return types.new_class(
            "ParametrizedTypedJSONDecoder",
            (cls,),
            kwds={"metaclass": _ParametrizedTypedDecoderMeta},
            exec_body=exec_body_factory(typeform=typeform, eval_fref=eval_fref),
        )

    def __class_getitem__(cls, typeform: TypeForm) -> Type["TypedDecoder"]:
        """Parametrize the TypedJSONDecoder to decode the provided type hint

        Jsonable subclasses are supported

        Args:
            tp: type hint to parametrize a TypedJSONDecoder for

        Returns:
            A special subclass of TypedJSONDecoder for use with
            `json.load`
        """
        if isinstance(typeform, type) and (
            isinstance(typeform, _TypedDictMeta) or get_data_json(typeform) is not None
        ):
            eval_fref = None  # delayed
        else:
            eval_fref = EvalForwardRef(*get_outer_namespaces())
        return cls._cached_class_getitem_(typeform, eval_fref)

    def __init__(self, *args, **kwargs):
        if self.typeform is MISSING:
            raise UnparametrizedDecoderError(
                f"{type(self).__name__} must be parametrized before use"
                f"(hint: {type(self).__name__}[...])"
            )
        super().__init__(*args, **kwargs)

    @docstring(post="\n:meta private:")
    @functools.wraps(JSONDecoder.raw_decode)
    def raw_decode(self, s: str, idx: int = 0):
        if self.typeform is MISSING:
            raise UnparametrizedDecoderError(
                f"{type(self).__name__} must be parametrized before use"
                f"(hint: {type(self).__name__}[...])"
            )
        else:
            if self.eval_fref is None:
                if isinstance(self.typeform, type):
                    eval_fref = EvalForwardRef.from_class(self.typeform)
                else:
                    raise PheresInternalError(
                        f"Could not find globalns and localns from {self.typeform}"
                    )
            else:
                eval_fref = self.eval_fref
            cursor = _Cursor(src=self.src, s=s, pos=idx)
            context = _DecodeContext.from_typeform(self.typeform, eval_fref)
            try:
                obj, cursor, _ = self.scan_once(
                    cursor,
                    ctx=context,
                )
            except StopIteration as err:
                raise JSONDecodeError("Expecting value", s, err.value) from None
            return obj, cursor.pos
