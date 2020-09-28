"""
Various internal utilities for Pheres
"""
import functools
import inspect
import typing
from contextlib import contextmanager
from typing import Iterable, List, Tuple, Type, TypeVar, Union

# Type Aliases
AnyClass = TypeVar("AnyClass", bound=type)
TypeHint = Union[  # pylint: disable=unsubscriptable-object
    type, type(Union), type(Type), type(List)
]


class Virtual:
    """
    Mixin class to make a class non-heritable and non-instanciable
    """

    __slots__ = ("__weakref__",)

    def __init__(self, *args, **kwargs):
        raise TypeError("Cannot instanciate virtual class")

    def __init_subclass__(cls, *args, **kwargs):
        if Virtual not in cls.__bases__:
            raise TypeError("Cannot subclass virtual class")
        super().__init_subclass__(*args, **kwargs)


def autoformat(
    cls,
    /,
    params: Union[str, Iterable[str]] = (  # pylint: disable=unsubscriptable-object
        "message",
        "msg",
    ),
):
    """
    Class decorator to autoformat string arguments in it's init method

    Modify the class __init__ method in place by wrapping it. The wrapped class
    will call the format() method of arguments specified in `params` that exist
    in the original signature, passing all other arguments are dictionary

    Arguments
        params -- names of the arguments to autoformats

    Usage
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
    params = signature.parameters & set(params)

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
            name: string.format(**bounds.arguments) for name, string in pre_formatted
        }
        for name, arg in formatted:
            bounds.arguments[name] = arg
        return init(*bounds.args, **bounds.kwargs)

    __init__.__signature__ = init.__signature__
    cls.__init__ = __init__
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
