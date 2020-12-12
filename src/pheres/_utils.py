"""
Various internal utilities for Pheres
"""
from __future__ import annotations

import functools
import inspect
import json
import re
import types
import typing
from contextlib import contextmanager
from types import FrameType
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

# Type Aliases
AnyClass = TypeVar("AnyClass", bound=type)
TypeHint = Union[  # pylint: disable=unsubscriptable-object
    Type[type],
    Type[Any],
    Type[TypeVar],
    # Type[Generic],
    Type[Annotated],
    Type[Tuple],
    Type[Callable],
]
Namespace = dict[str, Any]
TypeT = TypeVar("TypeT", *typing.get_args(TypeHint))
U = TypeVar("U")
V = TypeVar("V")


class Virtual:
    """
    Mixin class to make a class non-heritable and non-instanciable
    """

    __slots__ = ("__weakref__",)

    def __init__(self):
        raise TypeError("Cannot instanciate virtual class")

    def __init_subclass__(cls, *args, **kwargs):
        if Virtual not in cls.__bases__:
            raise TypeError("Cannot subclass virtual class")
        super().__init_subclass__(*args, **kwargs)


class Subscriptable(Generic[U, V]):
    """
    Decorator to make a subscriptable instance from a __getitem__ function

    Usage:
        @Subscriptable
        def my_subscriptable(key):
            return key

    assert my_subscriptable[8] == 8
    """

    __slots__ = ("_func",)

    def __init__(self, func: Callable[[U], V]) -> None:
        self._func = func
        # self.__doc__ = func.__doc__

    def __call__(self):
        raise SyntaxError("Use brackets '[]' instead")

    def __getitem__(self, arg: U) -> V:
        return self._func(arg)


def docstring(
    docstring: str = None, *, pre: str = None, post: str = None
) -> Callable[[U], U]:
    """
    Decorator to modify the docstring of an object.

    For all provided strings, unused empty lines are removed, and the indentation
    of the first non-empty line is removed from all lines if possible. This allows
    better indentation when used as a decorator.

    Unused empty lines means initial enpty lines for ``pre``, and final empty lines
    for ``post``.

    Arguments:
        docstring: replaces the docstring of the object
        pre: adds the string at the start of the object original docstring
        post: adds the strings at the end of the object original docstring
    """

    def edit_docstring(obj: U) -> U:
        obj.__doc__ = "".join(
            (
                clean_docstring(pre or "", unused="pre"),
                clean_docstring(docstring or (obj.__doc__ or "")),
                clean_docstring(post or "", unused="post"),
            )
        )
        return obj

    return edit_docstring


# from https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
class ClassPropertyDescriptor:
    """
    Descriptor for class properties
    """

    __slots__ = ("fget", "fset")

    def __init__(
        self,
        fget: Union[classmethod, staticmethod],
        fset: Union[classmethod, staticmethod] = None,
    ):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj: U, cls: Type[U] = None) -> V:
        if cls is None:
            cls = type(obj)
        return self.fget.__get__(obj, cls)()

    def __set__(self, obj: U, value: V):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    @property
    def __isabstractmethod__(self):
        return any(
            getattr(f, "__isabstractmethod__", False) for f in (self.fget, self.fset)
        )

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(
    func: Union[Callable, classmethod, staticmethod]
) -> ClassPropertyDescriptor:
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return ClassPropertyDescriptor(func)


@overload
def autoformat(
    cls: None, /, params: Union[str, Iterable[str]] = ("message", "msg")
) -> Callable[[Type[U]], Type[U]]:
    ...


@overload
def autoformat(
    cls: Type[U], /, params: Union[str, Iterable[str]] = ("message", "msg")
) -> Type[U]:
    ...


def autoformat(
    cls: Type[U] = None,
    /,
    params: Union[str, Iterable[str]] = (  # pylint: disable=unsubscriptable-object
        "message",
        "msg",
    ),
):
    """
    Class decorator to autoformat string arguments in the __init__ method

    Modify the class __init__ method in place by wrapping it. The wrapped class
    will call the format() method of arguments specified in `params` that exist
    in the original signature, passing all other arguments are dictionary to
    str.format()

    Arguments:
        params -- names of the arguments to autoformats

    Usage:
        @autoformat
        class MyException(Exception):
            def __init__(self, elem, msg="{elem} is invalid"):
                super().__init__(msg)
                self.msg = msg
                self.elem = elem

        assert MyException(8).msg == "8 is invalid"
    """
    if isinstance(params, str):
        params = (params,)

    if cls is None:
        return functools.partial(autoformat, params=params)

    orig_init = cls.__init__
    signature = inspect.signature(orig_init)
    params = signature.parameters.keys() & set(params)

    @functools.wraps(orig_init)
    def init(*args, **kwargs):
        bounds = signature.bind(*args, **kwargs)
        bounds.apply_defaults()
        pre_formatted = {
            name: bounds.arguments.pop(name)
            for name in params
            if name in bounds.arguments
        }
        formatted = {
            name: string.format(**bounds.arguments)
            for name, string in pre_formatted.items()
        }
        for name, arg in formatted.items():
            bounds.arguments[name] = arg
        return orig_init(*bounds.args, **bounds.kwargs)

    # init.__signature__ = signature
    setattr(cls, "__init__", init)
    return cls


class Variable(str):
    def __repr__(self) -> str:
        return self

    def __str__(self) -> str:
        return self


def _sig_without(sig: inspect.Signature, param: Union[int, str]) -> inspect.Signature:
    """Removes a parameter from a Signature object

    If param is an int, remove the parameter at that position, else
    remove any paramater with that name
    """
    if isinstance(param, int):
        params = list(sig.parameters.values())
        params.pop(param)
    else:
        params = [p for name, p in sig.parameters.items() if name != param]
    return sig.replace(parameters=params)


def _sig_merge(lsig: inspect.Signature, rsig: inspect.Signature) -> inspect.Signature:
    """Merges two signature object, dropping the return annotations"""
    return inspect.Signature(
        sorted(
            list(lsig.parameters.values()) + list(rsig.parameters.values()),
            key=lambda param: param.kind,
        )
    )


def _sig_to_def(sig: inspect.Signature) -> str:
    return str(sig).split("->", 1)[0].strip()[1:-1]


def _sig_to_call(sig: inspect.Signature) -> str:
    l = []
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            l.append(f"{p.name}={p.name}")
        else:
            l.append(p.name)
    return ", ".join(l)


def post_init(cls: Type[U]) -> Type[U]:
    """
    Class decorator to automatically support __post_init__() on classes

    This is useful for @attr.s decorated classes, because __attr_post_init__() doesn't
    support additional arguments.

    This decorators wraps the class __init__ in a new function that accept merged arguments,
    and dispatch them to __init__ and then __post_init__()
    """
    if not isinstance(cls, type):
        raise TypeError("Can only decorate classes")
    if not hasattr(cls, "__post_init__"):
        raise TypeError("The class must have a __post_init__() method")
    # Ignore the first argument which is the "self" argument
    sig = init_sig = _sig_without(inspect.signature(cls.__init__), 0)
    previous = [(cls, "__init__", sig)]
    for parent in reversed(cls.__mro__):
        if hasattr(parent, "__post_init__"):
            post_sig = _sig_without(
                inspect.signature(getattr(parent, "__post_init__")), 0
            )
            try:
                sig = _sig_merge(sig, post_sig)
            except Exception as err:
                # find the incompatibility
                for parent, method, psig in previous:
                    try:
                        _sig_merge(psig, post_sig)
                    except Exception:
                        break
                else:
                    raise TypeError(
                        "__post_init__ signature is incompatible with the class"
                    ) from err
                raise TypeError(
                    f"__post_init__() is incompatible with {parent.__qualname__}{method}()"
                ) from err
            # No exception
            previous.append((parent, "__post_init__", post_sig))
    # handles type annotations and defaults
    # inspired by the dataclasses modules
    params = list(sig.parameters.values())
    localns = (
        {
            f"__type_{p.name}": p.annotation
            for p in params
            if p.annotation is not inspect.Parameter.empty
        }
        | {
            f"__default_{p.name}": p.default
            for p in params
            if p.default is not inspect.Parameter.empty
        }
        | cls.__dict__
    )
    for i, p in enumerate(params):
        if p.default is not inspect.Parameter.empty:
            p = p.replace(default=Variable(f"__default_{p.name}"))
        if p.annotation is not inspect.Parameter.empty:
            p = p.replace(annotation=f"__type_{p.name}")
        params[i] = p
    new_sig = inspect.Signature(params)
    # Build the new __init__ source code
    self_ = "self" if "self" not in sig.parameters else "__post_init_self"
    init_lines = [
        f"def __init__({self_}, {_sig_to_def(new_sig)}) -> None:",
        f"__original_init({self_}, {_sig_to_call(init_sig)})",
    ]
    for parent, method, psig in previous[1:]:
        if hasattr(parent, "__post_init__"):
            if parent is not cls:
                init_lines.append(
                    f"super({parent.__qualname__}, {self_}).{method}({_sig_to_call(psig)})"
                )
            else:
                init_lines.append(f"{self_}.{method}({_sig_to_call(psig)})")
    init_src = "\n  ".join(init_lines)
    # Build the factory function source code
    local_vars = ", ".join(localns.keys())
    factory_src = (
        f"def __make_init__(__original_init, {local_vars}):\n"
        f" {init_src}\n"
        " return __init__"
    )
    # Create new __init__ with the factory
    globalns = inspect.getmodule(cls).__dict__
    ns: dict[str, Any] = {}
    exec(factory_src, globalns, ns)
    init = ns["__make_init__"](cls.__init__, **localns)
    self_param = inspect.Parameter(self_, inspect.Parameter.POSITIONAL_ONLY)
    init.__signature__ = inspect.Signature(
        parameters=[self_param] + list(sig.parameters.values()), return_annotation=None
    )
    setattr(cls, "__init__", init)
    return cls


@contextmanager
def on_error(func, *args, yield_=None, **kwargs):
    """
    Context manager that calls a function if the managed code doesn't raise
    """
    try:
        yield yield_
    except Exception:
        func(*args, **kwargs)
        raise


@contextmanager
def on_success(func, *args, yield_=None, **kwargs):
    """
    Context manager that calls a function if the managed code raises an Exception
    """
    try:
        yield yield_
    except Exception:
        raise
    else:
        func(*args, **kwargs)


def clean_docstring(doc: str, unused: Literal["pre", "post"] = None) -> str:
    """
    Removes initial empty lines and shared indentation

    Arguments:
        doc: docstring to clean up
        unused: whether to remove statring or endind empty lines
    Returns:
        The cleaned docstring
    """
    doc = doc.split("\n")
    if unused == "pre":
        try:
            index = next(i for i, l in enumerate(doc) if l.strip())
            doc = doc[index:]
        except StopIteration:
            doc = []
    elif unused == "post":
        try:
            index = next(i for i, l in enumerate(reversed(doc)) if l.strip())
            doc = doc[: len(doc) - index]
        except StopIteration:
            doc = []
    if doc:
        first_line = doc[0]
        index = len(first_line) - len(first_line.lstrip())
        indent = first_line[:index]
        if all(l.startswith(indent) for l in doc if l.strip()):
            doc = [(l[index:] if l.strip() else l) for l in doc]
    return "\n".join(doc)


def split(func, iterable):
    """split an iterable based on the truth value of the function for element

    Arguments
        func -- a callable to apply to each element in the iterable
        iterable -- an iterable of element to split

    Returns
        falsy, truthy - two tuple, the first with element e of the itrable where
        func(e) return false, the second with element of the iterable that are True
    """
    falsy, truthy = [], []
    it = iter(iterable)
    for e in it:
        if func(e):
            truthy.append(e)
        else:
            falsy.append(e)
    return tuple(falsy), tuple(truthy)


def sync_filter(func, *iterables):
    """
    Filter multiple iterable at once, selecting values at index i
    such that func(iterables[0][i], iterables[1][i], ...) is True
    """
    return tuple(zip(*tuple(i for i in zip(*iterables) if func(*i)))) or ((),) * len(
        iterables
    )


def get_outer_frame() -> Optional[FrameType]:
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back
        if frame is not None:
            return frame.f_back


def get_outer_namespaces() -> Tuple[Namespace, Namespace]:
    """
    Get the globals and locals from the context that called the function
    calling this utility

    Returns:
        globals, locals
    """
    frame = inspect.currentframe()
    if frame:
        frame = frame.f_back
        if frame:
            frame = frame.f_back
            if frame:
                return frame.f_globals or {}, frame.f_locals or {}
    return {}, {}


def get_args(
    tp: TypeHint, *, globalns: Namespace = None, localns: Namespace = None
) -> Tuple[TypeHint, ...]:
    if globalns is not None or localns is not None:
        return typing.get_args(typing._eval_type(tp, globalns, localns))
    return typing.get_args(tp)


# Adapted version of typing._type_repr
def type_repr(tp) -> str:
    """Return the repr() of objects, special casing types and tuples"""
    from pheres._typing import JSONArray, JSONObject, JSONValue

    if isinstance(tp, tuple):
        return ", ".join(map(type_repr, tp))
    if isinstance(tp, type):
        if tp.__module__ == "builtins":
            return tp.__qualname__
        return f"{tp.__module__}.{tp.__qualname__}"
    if tp is Ellipsis:
        return "..."
    if isinstance(tp, types.FunctionType):
        return tp.__name__
    if tp is JSONValue:
        return "JSONValue"
    if tp is JSONArray:
        return "JSONArray"
    if tp is JSONObject:
        return "JSONObject"
    return repr(tp)


def get_class_namespaces(cls: type) -> tuple[Namespace, Namespace]:
    """
    Return the module a class is defined in and its internal dictionary

    Returns:
        globals, locals
    """
    return inspect.getmodule(cls).__dict__, cls.__dict__ | {cls.__name__: cls}


def get_updated_class(cls: AnyClass) -> AnyClass:
    module = inspect.getmodule(cls)
    if module is not None:
        return getattr(module, cls.__name__)
    return cls
