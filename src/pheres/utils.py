"""
Various internal utilities for Pheres
"""
import functools
import inspect
import json
import types
import typing
from contextlib import contextmanager
from typing import Iterable, List, Tuple, Type, TypeVar, Union

# Type Aliases
AnyClass = TypeVar("AnyClass", bound=type)
TypeHint = Union[  # pylint: disable=unsubscriptable-object
    type, type(Union), type(Type), type(List)
]
TypeT = TypeVar("TypeT", *typing.get_args(TypeHint))


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


class UsableDecoder(json.JSONDecoder):
    """
    JSONDecoder subclass with method to use itself as a decoder
    """

    @functools.wraps(json.load)
    @classmethod
    def load(cls, *args, **kwargs):
        return json.load(*args, cls=cls, **kwargs)

    @functools.wraps(json.loads)
    @classmethod
    def loads(cls, *args, **kwargs):
        return json.loads(*args, cls=cls, **kwargs)


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
    init_sig = inspect.Signature(
        parameters=list(inspect.signature(cls.__init__).parameters.values())[1:]
    )
    post_sig = inspect.Signature(
        parameters=list(inspect.signature(cls.__post_init__).parameters.values())[1:]
    )
    params = list(init_sig.parameters.values()) + list(post_sig.parameters.values())
    params = sorted(params, key=lambda param: param.kind)
    localns = {
        f"__type_{p.name}": p.annotation
        for p in params
        if p.annotation is not inspect.Parameter.empty
    } | {
        f"__default_{p.name}": p.default
        for p in params
        if p.default is not inspect.Parameter.empty
    }
    for i, p in enumerate(params):
        if p.default is not inspect.Parameter.empty:
            p = p.replace(default=Variable(f"__default_{p.name}"))
        if p.annotation is not inspect.Parameter.empty:
            p = p.replace(annotation=f"__type_{p.name}")
        params[i] = p
    try:
        new_sig = inspect.Signature(params)
    except Exception as err:
        raise TypeError("Incompatible __init__ and __post_init__") from err
    globalns = inspect.getmodule(cls).__dict__
    localns = cls.__dict__ | localns
    local_vars = ", ".join(localns.keys())
    init_src = (
        f"def __init__(self, {_sig_to_def(new_sig)}) -> None:\n"
        f"  __original_init(self, {_sig_to_call(init_sig)})\n"
        f"  self.__post_init__({_sig_to_call(post_sig)})"
    )
    factory_src = (
        f"def __make_init__(__original_init, {local_vars}):\n"
        f" {init_src}\n"
        " return __init__"
    )
    exec(factory_src, globalns, (ns := {}))
    cls.__init__ = ns["__make_init__"](cls.__init__, **localns)
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
