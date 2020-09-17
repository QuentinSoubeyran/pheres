# -*- coding: utf-8 -*-
"""
Module with various internal miscs

Part of the Pheres package
"""

import functools
import inspect
from typing import Callable, Dict, Iterable, Optional, TypeVar

__all__ = ["JSONError"]

U = TypeVar("U")
V = TypeVar("V")


class AutoFormatMixin(Exception):
    """Mixin class for exception to auto-format the error message

    Automatically wraps the __init__ method of subclass to call
    str.format() on the 'message' argument (if any), passing the dictionary of
    other arguments to __init__
    """

    @staticmethod
    def _init_wrapper(init_method):
        init_sig = inspect.signature(init_method)

        @functools.wraps(init_method)
        def wrapped_init(*args, **kwargs):
            bound_args = init_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            if "message" in bound_args.arguments:
                msg = bound_args.arguments.pop("message")
                bound_args.arguments["message"] = msg.format(**bound_args.arguments)
            return init_method(*bound_args.args, **bound_args.kwargs)

        wrapped_init.__signature__ = init_sig
        return wrapped_init

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__init__ = AutoFormatMixin._init_wrapper(cls.__init__)


class JSONError(Exception):
    f"""Base exception for the pheres module

    Raised as-is in case of bug. Only subclasses are normaly raised
    """


class Subscriptable:
    """Decorator to make a subscriptable object from a function"""

    __slots__ = ("_func",)

    def __init__(self, func):
        self._func = func

    def __getitem__(self, arg):
        return self._func(arg)


class FallbackRegister:
    def __init__(self, register_func, unregister_func):
        self.register = register_func
        self.unregister = unregister_func

    def __enter__(self):
        self.register()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.unregister()


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

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def split(function, iterable):
    """split an iterable based on the boolean value of the function

    return two tuples"""
    falsy, truthy = [], []
    functools.reduce(
        lambda appends, next: appends[bool(function(next))](next) or appends,
        iterable,
        (falsy.append, truthy.append),
    )
    return tuple(falsy), tuple(truthy)


def _test_injection(matches, unassigned, availables, acc):
    """internal helper for find_injection"""
    if len(unassigned) == 0:
        return acc
    elem, *unassigned = unassigned
    for match in matches[elem] & availables:
        if (
            result := _test_injection(
                matches, unassigned, availables - {match}, {**{elem: match}, **acc}
            )
        ) is not None:
            return result
    # No match worked, or no match availabe anymore
    return None


def find_injection(
    A: Iterable[U],
    B: Iterable[V],
    match_func: Callable[[U, V], bool],
    validator_func: Callable[[Dict[U, V]], bool] = lambda _: True,
) -> Optional[Dict[U, V]]:
    """Assign an element b of B to each element a of A such that test_f(a, b) is True
    and no element of B is used more than once

    Arguments:
        A -- iterable of element to match from. All element in A will be matched
        B -- iterable if element to match to. Some elements may not be matched
        match_func -- callable returning if an element from A can be matched with an element from B
        validator_func -- (Optional) function validating the found(s) matching. If it return False,
            the found matching is abandonned and another one is tried

    Returns
        The first matching found from A to B, such that all pairs satisfy match_func and the matching
        satisfy validator_func

        return None if no such matching exists
    """
    A = list(A)
    B = list(B)
    # Quick test
    if len(A) > len(B) or len(A) == 0:
        return None
    # Find possible matches
    matches = {
        i: {j for j, b in enumerate(B) if match_func(a, b)} for i, a in enumerate(A)
    }
    injection = _test_injection(matches, list(range(len(A))), set(range(len(B))), {})
    if injection is not None:
        return {A[a_index]: B[b_index] for a_index, b_index in injection.items()}
    return None
