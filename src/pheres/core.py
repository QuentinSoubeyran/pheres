"""
Code functionalities of pheres
"""
from __future__ import annotations

import collections
import collections.abc
import inspect
import types
import typing
from typing import Any, Callable, Dict, ForwardRef, Optional, Tuple

from pheres.aliases import TypeForm
from pheres.exceptions import PheresInternalError
from pheres.utils import get_class_namespaces, get_outer_namespaces, singleton

Namespace = Dict[str, Any]
TypeOrigin = TypeForm
TypeArgs = Tuple[TypeForm, ...]
Annotations = Tuple[Any, ...]

_TypedDictMeta: type = typing._TypedDictMeta  # type: ignore[attr-defined]
_FUNCTION_TYPES = (
    types.FunctionType,
    types.LambdaType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.BuiltinMethodType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
    types.MethodDescriptorType,
)
NO_TYPE_HINTS = singleton("NO_TYPE_HINTS", True)
_eval_type: Callable = typing._eval_type  # type: ignore[attr-defined]


class EvalForwardRef:
    __slots__ = ("globalns", "localns", "cache")

    def __init__(
        self,
        globalns: Namespace,
        localns: Namespace,
        *,
        cache: Dict[ForwardRef, TypeForm] = None,
    ) -> None:
        self.globalns = globalns
        self.localns = localns
        self.cache = cache or {}

    @staticmethod
    def from_class(
        cls: type, *, cache: Dict[ForwardRef, TypeForm] = None
    ) -> EvalForwardRef:
        return EvalForwardRef(*get_class_namespaces(cls), cache=cache)

    def copy(self) -> EvalForwardRef:
        return EvalForwardRef(self.globalns, self.localns)

    def __call__(self, tp: TypeForm) -> TypeForm:
        if isinstance(tp, str):
            tp = ForwardRef(tp)
        if isinstance(tp, ForwardRef):
            tp = self.cache.get(tp, _eval_type(tp, self.globalns, self.localns))
        if isinstance(tp, ForwardRef):
            raise PheresInternalError(f"Could not resolve {tp!r}")
        return tp


def _unpack_annotated(
    tp: TypeForm, eval_fref: EvalForwardRef
) -> Tuple[TypeOrigin, TypeArgs, Annotations]:
    tp = eval_fref(tp)
    orig = typing.get_origin(tp)
    if orig is typing.Annotated:
        tp, *annots = typing.get_args(tp)
        tp, args, inner_annots = _unpack_annotated(tp, eval_fref)
        return tp, args, inner_annots + tuple(annots)
    return orig, typing.get_args(tp), ()


def _typecheck(
    value: Any,
    tp: TypeForm,
    eval_fref: EvalForwardRef,
) -> bool:
    """
    Typechecks a value against a type (python class or type annotations from the
    `typing` module).

    Arguments:
        value: value to typecheck
        tp: type to typecheck against
        eval_fref: context to use to evaluate forward references in type annotations

    Returns:
        A boolean, whether ``value`` is valid for the ``tp`` type
    """
    tp = eval_fref(tp)
    orig, args, annots = _unpack_annotated(tp, eval_fref)
    # Same order as in typing.__all__
    if tp is typing.Any:
        return True
    elif orig is collections.abc.Callable:
        if not callable(value):
            return False
        args, ret = args
        if isinstance(value, _FUNCTION_TYPES):
            hints = typing.get_type_hints(
                value, eval_fref.globalns, eval_fref.localns, include_extras=True
            )
        else:
            # Callable instance, get_type_hints() doesn't work straight away
            hints = hints = typing.get_type_hints(
                type(value).__call__,
                eval_fref.globalns,
                eval_fref.localns,
                include_extras=True,
            )
        params = [
            p
            for p in inspect.signature(value).parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]

        pass
    elif orig is typing.ClassVar:
        return _typecheck(value, args[0], eval_fref)
    elif orig is typing.Final:
        return _typecheck(value, args[0], eval_fref)
    elif orig is typing.Generic:  # doesn't work !
        pass
    elif orig is typing.Literal:
        return value in args
    elif isinstance(tp, type) and typing.Protocol in tp.__bases__:
        pass
    elif orig is tuple:
        if isinstance(value, tuple):
            if len(args) == 2 and args[1] == Ellipsis:
                return all(_typecheck(v, args[0], eval_fref) for v in value)
            else:
                return len(args) == len(value) and all(
                    _typecheck(v, t, eval_fref) for v, t in zip(value, args)
                )
        return False
    elif orig is type:
        pass
    elif orig is typing.TypeVar:
        pass
    elif orig is typing.Union:
        return any(_typecheck(value, arg, eval_fref) for arg in args)

    # collections.abc not yet suppored
    # protocols are handled by the Protocol check above

    elif tp in (None, type(None)):
        return value is None
    elif orig is collections.ChainMap:
        dict_tp = dict[args]  # type: ignore
        return isinstance(value, collections.ChainMap) and all(
            _typecheck(subdict, dict_tp, eval_fref) for subdict in value.maps
        )
    elif orig is collections.Counter:
        pass
    elif orig is collections.deque:
        pass
    elif orig is dict:
        return isinstance(value, dict) and all(
            _typecheck(k, args[0], eval_fref) and _typecheck(v, args[1], eval_fref)
            for k, v in value.items()
        )
    elif orig is collections.defaultdict:
        pass
    elif orig is list:
        return isinstance(value, list) and all(
            _typecheck(v, args[0], eval_fref) for v in value
        )
    elif orig is collections.OrderedDict:
        pass
    elif orig is set:
        return isinstance(value, set) and all(
            _typecheck(v, args[0], eval_fref) for v in value
        )
    elif orig is frozenset:
        return isinstance(value, frozenset) and all(
            _typecheck(v, args[0], eval_fref) for v in value
        )
    # NamedTuple isn't a type, the check relies on implementation details
    elif isinstance(tp, type) and typing.NamedTuple in getattr(
        tp, "__orig_bases__", tp.__bases__
    ):
        pass
    # TypedDict isn't a type, the check relies on implementation details
    elif isinstance(tp, _TypedDictMeta):
        pass
    elif orig is collections.abc.Generator:
        pass
    elif isinstance(tp, type):
        return isinstance(value, tp)
    raise TypeError(f"Unsupported type {tp!r}")


def typecheck(value: Any, tp: TypeForm) -> bool:
    """
    Typechecks a value against a type hint

    Arguments:
        value: python object to typecheck
        tp: type to typecheck against

    Returns:
        if ``value`` is valid value for type ``tp``

    Raises:
        TypeError: the provided type is not supported
    """
    namespaces = get_outer_namespaces()
    return _typecheck(value, tp, EvalForwardRef(*namespaces))
