"""
Typed JSON decoding
"""
import typing
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import attr

from pheres.aliases import TypeArgs, TypeForm, TypeOrig
from pheres.core import EvalForwardRef, _typecheck
from pheres.data import get_data_json
from pheres.exceptions import PheresInternalError
from pheres.typed_json.aliases import (
    JSON,
    FilePosition,
    RawJSON,
    RawPosition,
    _JSONArrayTypes,
    _JSONObjectTypes,
    _JSONValueTypes,
)
from pheres.typed_json.exceptions import TypedJSONDecodeError
from pheres.utils import MISSING, Singleton, singleton, sync_filter

T = TypeVar("T")
PosT = TypeVar("PosT", FilePosition, RawPosition)
TypeFormFilter = Callable[[TypeForm, Optional[TypeOrig], Optional[TypeArgs]], bool]
_TypedDictMeta: type = typing._TypedDictMeta  # type: ignore[attr-defined]

identity: Callable[[T], T] = singleton("identity", True, lambda x: x)  # type: ignore[assignment]


def _unpack_union(
    form: TypeForm, transform: Callable[[TypeForm], TypeForm] = identity
) -> Tuple[TypeForm, ...]:
    """
    Unpack Union type hints and apply a transform to the unpacked type forms

    Returns a 1-tuple of other values
    """
    if typing.get_origin(form) is Union:
        return tuple(map(transform, typing.get_args(form)))
    return (form,)


def _filter_array_forms(
    tp: TypeForm, orig: TypeOrig = None, args: TypeArgs = None
) -> bool:
    """
    Filter typeforms that can typecheck arrays
    """
    if isinstance(tp, type):
        if (data := get_data_json(tp)) is not None:
            return _filter_array_forms(data.type, data.orig, data.args)
    if orig is None:
        orig = typing.get_origin(tp)
    if orig is Union:
        if args is None:
            args = typing.get_args(tp)
        return any(_filter_array_forms(arg) for arg in args)
    return isinstance(orig, type) and issubclass(orig, _JSONArrayTypes)


def _filter_index_form_factory(index: int) -> TypeFormFilter:
    """
    Factory to filtering types that typecheck an array with a specific index
    """

    def filter_index_form(
        tp: TypeForm, orig: TypeOrig = None, args: TypeArgs = None
    ) -> bool:
        if isinstance(tp, type):
            if (data := get_data_json(tp)) is not None:
                return filter_index_form(data.type, data.orig, data.args)
        if orig is None:
            orig = typing.get_origin(tp)
        if orig is Union:
            if args is None:
                args = typing.get_args(tp)
            return any(filter_index_form(arg) for arg in args)
        if isinstance(orig, type):
            if issubclass(orig, tuple):
                if args is None:
                    args = typing.get_args(tp)
                return (len(args) == 2 and args[1] is Ellipsis) or len(args) > index
            elif issubclass(orig, list):
                return True
        raise PheresInternalError(
            f"Unhandled JSON array type in filter_index_form(): {orig}"
        )

    return filter_index_form


def _filter_indexvalue_form_factory(
    index: int, value: Any, eval_fref: EvalForwardRef
) -> TypeFormFilter:
    """
    Factory for filtering typeform that accept the value at the given index
    """

    def filter_indexvalue_form(
        tp: TypeForm, orig: TypeOrig = None, args: TypeArgs = None
    ) -> bool:
        if isinstance(tp, type):
            if (data := get_data_json(tp)) is not None:
                return filter_indexvalue_form(data.type, data.orig, data.args)
        if orig is None:
            orig = typing.get_origin(tp)
        if args is None:
            args = typing.get_args(tp)
        if orig is Union:
            return any(filter_indexvalue_form(arg) for arg in args)
        if isinstance(orig, type):
            if issubclass(orig, tuple):
                t = args[0] if len(args) == 2 and args[1] is Ellipsis else args[index]
                return _typecheck(value, t, eval_fref)
            elif issubclass(orig, list):
                return _typecheck(value, args[0], eval_fref)
        raise PheresInternalError(
            f"Unhandled JSON array type in filter_valueindex_form(): {orig}"
        )

    return filter_indexvalue_form


def _filter_length_form_factory(length: int) -> TypeFormFilter:
    """
    Factory for filtering array typeforms of a given length
    """

    def filter_length_form(
        tp: TypeForm, orig: TypeOrig = None, args: TypeArgs = None
    ) -> bool:
        if isinstance(tp, type):
            if (data := get_data_json(tp)) is not None:
                return filter_length_form(data.type, data.orig, data.args)
        if orig is None:
            orig = typing.get_origin(tp)
        if orig is Union:
            if args is None:
                args = typing.get_args(tp)
            return any(filter_length_form(arg) for arg in args)
        if isinstance(orig, type):
            if issubclass(orig, tuple):
                if args is None:
                    args = typing.get_args(tp)
                return (len(args) == 2 and args[1] is Ellipsis) or len(args) == length
            elif issubclass(orig, list):
                return True
        raise PheresInternalError(
            f"Unhandled JSON array type in filter_length_form(): {orig}"
        )

    return filter_length_form


def _filter_object_forms(
    tp: TypeForm, orig: TypeOrig = None, args: TypeArgs = None
) -> bool:
    """
    Filters typeform that accepts JSON objects
    """
    if isinstance(tp, type):
        if (data := get_data_json(tp)) is not None:
            return _filter_object_forms(data.type, data.orig, data.args)
    if orig is None:
        orig = typing.get_origin(tp)
    if orig is Union:
        if args is None:
            args = typing.get_args(tp)
        return any(_filter_object_forms(arg) for arg in args)
    return isinstance(tp, _TypedDictMeta) or (
        isinstance(orig, type) and issubclass(orig, _JSONObjectTypes)
    )


def _filter_key_form(key: str) -> TypeFormFilter:
    """
    Factory to filter typeform accepting a key
    """

    def filter_key_form(
        tp: TypeForm, orig: TypeOrig = None, args: TypeArgs = None
    ) -> bool:
        if isinstance(tp, type):
            if (data := get_data_json(tp)) is not None:
                return filter_key_form(data.type, data.orig, data.args)
            elif isinstance(tp, _TypedDictMeta):
                return key in tp.__annotations__
        if orig is None:
            orig = typing.get_origin(tp)
        if orig is Union:
            if args is None:
                args = typing.get_args(tp)
            return any(filter_key_form(arg) for arg in args)
        if isinstance(orig, type):
            if issubclass(orig, dict):
                return True
        raise PheresInternalError(
            f"Unhandled JSON array type in filter_key_form(): {orig}"
        )

    return filter_key_form


def _filter_keyvalue_form_factory(
    key: str, value: Any, eval_fref: EvalForwardRef
) -> TypeFormFilter:
    """
    Factory for filtering typeforms that accept a value for a key
    """

    def filter_keyvalue_form(
        tp: TypeForm, orig: TypeOrig = None, args: TypeArgs = None
    ) -> bool:
        if isinstance(tp, type):
            if (data := get_data_json(tp)) is not None:
                return filter_keyvalue_form(data.type, data.orig, data.args)
            elif isinstance(tp, _TypedDictMeta):
                return _typecheck(value, typing.get_type_hints(tp)[key], eval_fref)
        if orig is None:
            orig = typing.get_origin(tp)
        if args is None:
            args = typing.get_args(tp)
        if isinstance(orig, type):
            if issubclass(orig, dict):
                return _typecheck(value, args[1], eval_fref)
        raise PheresInternalError(
            f"Unhandled JSON array type in filter_keyvalue_form(): {orig}"
        )

    return filter_keyvalue_form


def _instanciate(
    tp: TypeForm,
    cls: Type[T],
    obj: Any,
    *,
    orig: TypeOrig = None,
    args: TypeArgs = None,
) -> T:
    if isinstance(tp, type) and issubclass(tp, _JSONValueTypes):
        return cls(obj)  # type: ignore[call-arg]
    elif isinstance(tp, _TypedDictMeta):
        return cls(**cast(dict, obj))  # type: ignore[call-arg]
    # compute orig if not provided
    if orig is None:
        orig = typing.get_origin(tp)
    # check other JSON types
    if orig is Union:
        if args is None:
            args = typing.get_args(tp)
        for arg in args:
            try:
                return _instanciate(arg, cls, obj)
            except PheresInternalError:
                raise
            except Exception:
                pass
    elif isinstance(orig, type) and issubclass(orig, tuple):
        return cls(*cast(tuple, obj))
    elif isinstance(orig, type) and issubclass(orig, list):
        return cls(cast(list, obj))  # type: ignore[call-arg]
    elif isinstance(orig, type) and issubclass(orig, dict):
        return cls(cast(dict, obj))  # type: ignore[call-arg]
    # Sanity checks
    elif isinstance(orig, type) and issubclass(
        orig, (_JSONArrayTypes, *_JSONObjectTypes)
    ):
        raise PheresInternalError(f"Unhandled JSON type in _instanciate(): {orig!r}")
    raise PheresInternalError(f"Invalid type in _instanciate(): {tp!r}")


@attr.dataclass(frozen=True)
class _DecodeContext(Generic[PosT]):
    """
    Internal class to keep track of valid types during JSON decoding
    """

    doc: Union[str, JSON]
    pos: PosT
    eval_fref: EvalForwardRef

    forms: Tuple[TypeForm, ...]
    origs: Tuple[TypeOrig, ...] = ()
    args_t: Tuple[TypeArgs, ...] = ()

    parent: Optional["_DecodeContext[PosT]"] = None
    parent_key: Optional[Union[int, str]] = None

    def __attrs_post_init__(self, /):
        # cache types origins and arguments
        if not self.origs:
            self.__dict__["origs"] = tuple(typing.get_origin(tp) for tp in self.forms)
        if not self.args_t:
            self.__dict__["args_t"] = tuple(
                tuple(map(self.eval_fref, typing.get_args(tp))) for tp in self.forms
            )
        if self.parent is not None and self.parent_key is None:
            raise PheresInternalError(
                "_DecodeContext with a parent must be given a parent key"
            )

    def err_msg(self, /, *, msg: str = None, value: Any = MISSING) -> str:
        parts = []
        if msg:
            parts.append(msg)
        else:
            parts.append(f"Expected type {Union[self.forms]}")
        if value is not MISSING:
            parts.append(f"got {value}")
        parts.append("at")
        return ", ".join(parts)

    def filtered(
        self, /, filter: TypeFormFilter, err_msg: str, *, err_pos: PosT = None
    ) -> "_DecodeContext[PosT]":
        forms, origs, args_t = sync_filter(filter, self.forms, self.origs, self.args_t)
        if not forms:
            raise TypedJSONDecodeError(
                msg=err_msg,
                doc=self.doc,
                pos=self.pos if err_pos is None else err_pos,
            )
        return attr.evolve(self, forms=forms, origs=origs, args_t=args_t)

    def array_filtered(self) -> "_DecodeContext[PosT]":
        return self.filtered(_filter_array_forms, self.err_msg(value="'Array'"))

    def object_filtered(self) -> "_DecodeContext[PosT]":
        return self.filtered(_filter_object_forms, self.err_msg(value="'Object'"))

    def get_subtypes_at_index(self, /, index: int) -> Tuple[TypeForm, ...]:
        """
        Returns the subtypes at index ``index`` from its array types

        Types should already be filtered for array types
        """
        subtypes: List[TypeForm] = []
        for form, orig, args in zip(self.forms, self.origs, self.args_t):
            if isinstance(form, type) and (data := get_data_json(form)) is not None:
                form, orig, args = data.type, data.orig, data.args
            if isinstance(orig, type) and issubclass(orig, tuple):
                subtypes.extend(_unpack_union(args[index], self.eval_fref))
            elif isinstance(orig, type) and issubclass(orig, list):
                subtypes.extend(_unpack_union(args[0], self.eval_fref))
            else:
                raise PheresInternalError(
                    f"Unhandled array type {form!r} in get_subtypes_at()"
                )
        return tuple(subtypes)

    def get_subtypes_at_key(self, /, key: str) -> Tuple[TypeForm, ...]:
        """
        Returns the subtypes under key ``key`` from object-like types

        Types should already be filtered for object types
        """
        subtypes: List[TypeForm] = []
        for form, orig, args in zip(self.forms, self.origs, self.args_t):
            if isinstance(form, type) and (data := get_data_json(form)) is not None:
                form, orig, args = data.type, data.orig, data.args
            if isinstance(form, _TypedDictMeta):
                annotations = typing.get_type_hints(form, include_extras=True)
                eval_fref = EvalForwardRef.from_class(typing.cast(type, form))
                subtypes.extend(_unpack_union(annotations[key], eval_fref))
            elif isinstance(orig, type) and issubclass(orig, dict):
                subtypes.extend(_unpack_union(args[1], self.eval_fref))
            else:
                raise PheresInternalError(
                    f"Unhandled object type {form!r} in get_subtypes_under()"
                )
        return tuple(subtypes)

    def get_subcontext_at_index(
        self, /, index: int, pos: PosT
    ) -> "_DecodeContext[PosT]":
        parent = self.filtered(
            _filter_index_form_factory(index),
            self.err_msg(value=f"'Array' of length >={index+1}"),
        )
        return _DecodeContext(
            doc=self.doc,
            pos=pos,
            eval_fref=self.eval_fref,
            forms=parent.get_subtypes_at_index(index),
            parent=parent,
            parent_key=index,
        )

    def get_subcontext_at_key(self, /, key: str, pos: PosT) -> "_DecodeContext[PosT]":
        parent = self.filtered(
            _filter_key_form(key),
            self.err_msg(msg=f"Inferred type {Union[self.forms]} has no key '{key}'"),
            err_pos=pos,
        )
        return _DecodeContext(
            doc=self.doc,
            pos=pos,
            eval_fref=self.eval_fref,
            forms=parent.get_subtypes_at_key(key),
            parent=parent,
            parent_key=key,
        )

    def validate_value(
        self, /, value: RawJSON, start_pos: PosT, end_pos: PosT
    ) -> Tuple[Any, PosT, "_DecodeContext[PosT]"]:
        # filter the remaining types and jsonable classes
        types: List[TypeForm] = []
        classes: List[type] = []
        for tp in self.forms:
            if (data := get_data_json(tp)) is not None:
                if _typecheck(value, data.type, self.eval_fref):
                    classes.append(tp)
            elif _typecheck(value, tp, self.eval_fref):
                types.append(tp)
        if not types and not classes:
            raise TypedJSONDecodeError(
                msg=self.err_msg(value=value), doc=self.doc, pos=start_pos
            )
        # instanciate the jsonable class, if any
        if classes:
            classes = [
                c for i, c in enumerate(classes) if not issubclass(c, classes[i + 1 :])
            ]
            if len(classes) > 1:
                raise TypedJSONDecodeError(
                    msg=self.err_msg(
                        msg=f"Multiple jsonable classes found for value {value}"
                    ),
                    doc=self.doc,
                    pos=self.pos,
                )
            data = get_data_json(classes[0])
            if data is not None:
                instance = _instanciate(
                    data.type, classes[0], value, orig=data.orig, args=data.args
                )
            else:
                raise PheresInternalError(
                    "Inconsistent jsonable class: pheres data was lost"
                )
        else:
            instance = value
        # Notify parent
        parent = None
        if self.parent is not None:
            if isinstance(self.parent_key, int):
                form_filter = _filter_indexvalue_form_factory(
                    self.parent_key, value, self.eval_fref
                )
            elif isinstance(self.parent_key, str):
                form_filter = _filter_keyvalue_form_factory(
                    self.parent_key, value, self.eval_fref
                )
            else:
                raise PheresInternalError(
                    f"Unhandled parent key {self.parent_key} of type {type(self.parent_key)}"
                )
            parent = self.parent.filtered(
                form_filter, self.parent.err_msg(value=f"{value} of type {type(value)}")
            )
        return instance, end_pos, parent
