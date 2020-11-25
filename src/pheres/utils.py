"""
Various internal utilities for Pheres
"""
from __future__ import annotations

import functools
import inspect
import json
import types
import typing
from contextlib import contextmanager
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# Type Aliases
AnyClass = TypeVar("AnyClass", bound=type)
TypeHint = Union[  # pylint: disable=unsubscriptable-object
    type,
    Type[Any],
    Type[TypeVar],
    type(Generic),
    Type[Annotated],
    Type[Tuple],
    Type[Callable],
]
TypeT = TypeVar("TypeT", *typing.get_args(TypeHint))
U = TypeVar("U")


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


class Subscriptable:
    """
    Decorator to make a subscriptable instance from a __getitem__ function

    Usage:
        @Subscriptable
        def my_subscriptable(key):
            return key

    assert my_subscriptable[8] == 8
    """

    __slots__ = ("_func",)

    def __init__(self, func):
        self._func = func

    def __getitem__(self, arg):
        return self._func(arg)


def append_doc(s: str) -> Callable[[U], U]:
    """
    Decorator appending in-place to the docstring of the decorated object
    """

    def append(obj: U) -> U:
        obj.__doc__ += s
        return obj

    return append


# from https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
class ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.fget.__get__(obj, cls)()

    def __set__(self, obj, value):
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


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def autoformat(
    cls,
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
    init = cls.__init__
    signature = inspect.signature(init)
    params = signature.parameters.keys() & set(params)

    @functools.wraps(init)
    def __init__(*args, **kwargs):
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
        return init(*bounds.args, **bounds.kwargs)

    # __init__.__signature__ = signature
    cls.__init__ = __init__
    return cls


class Variable(str):
    def __repr__(self):
        return self

    def __str__(self):
        return self


def _sig_without(sig: inspect.Signature, param: Union[int, str]):
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


def _sig_merge(lsig: inspect.Signature, rsig: inspect.Signature):
    """Merges two signature object, dropping the return annotations"""
    return inspect.Signature(
        sorted(
            list(lsig.parameters.values()) + list(rsig.parameters.values()),
            key=lambda param: param.kind,
        )
    )


def _sig_to_def(sig: inspect.Signature):
    return str(sig).split("->", 1)[0].strip()[1:-1]


def _sig_to_call(sig: inspect.Signature):
    l = []
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            l.append(f"{p.name}={p.name}")
        else:
            l.append(p.name)
    return ", ".join(l)


def post_init(cls):
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
            post_sig = _sig_without(inspect.signature(parent.__post_init__), 0)
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
    self = "self" if "self" not in sig.parameters else "__post_init_self"
    init_lines = [
        f"def __init__({self}, {_sig_to_def(new_sig)}) -> None:",
        f"__original_init({self}, {_sig_to_call(init_sig)})",
    ]
    for parent, method, psig in previous[1:]:
        if hasattr(parent, "__post_init__"):
            if parent is not cls:
                init_lines.append(
                    f"super({parent.__qualname__}, {self}).{method}({_sig_to_call(psig)})"
                )
            else:
                init_lines.append(f"{self}.{method}({_sig_to_call(psig)})")
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
    exec(factory_src, globalns, (ns := {}))
    cls.__init__ = ns["__make_init__"](cls.__init__, **localns)
    self_param = inspect.Parameter(self, inspect.Parameter.POSITIONAL_ONLY)
    cls.__init__.__signature__ = inspect.Signature(
        parameters=[self_param] + list(sig.parameters.values()), return_annotation=None
    )
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


def get_outer_frame():
    return inspect.currentframe().f_back.f_back


def get_outer_namespaces() -> Tuple:
    """
    Get the globals and locals from the context that called the function
    calling this utility

    Returns
        globals, locals
    """
    frame = inspect.currentframe().f_back.f_back
    return frame.f_globals, frame.f_locals


def get_args(tp, *, globalns=None, localns=None) -> Tuple:
    if globalns is not None or localns is not None:
        return typing.get_args(typing._eval_type(tp, globalns, localns))
    return typing.get_args(tp)


# Adapted version of typing._type_repr
def type_repr(tp):
    """Return the repr() of objects, special casing types and tuples"""
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
    return repr(tp)


def get_class_namespaces(cls):
    return inspect.getmodule(cls).__dict__, cls.__dict__ | {cls.__name__: cls}


def get_updated_class(cls):
    module = inspect.getmodule(cls)
    if module is not None:
        return getattr(module, cls.__name__)
    return cls
