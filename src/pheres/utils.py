"""
Various utilities used in pheres yet unrelated to pheres funtionalities
"""
from __future__ import annotations

import functools
import inspect
import json
import re
import types
import typing
from contextlib import contextmanager
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

__all__ = [
    # Constants
    "Singleton",
    "singleton",
    "MISSING",
    # Mixin
    "Commented",
    # Decorators
    "subscriptable",
    "docstring",
    "classproperty",
    "autoformat",
    # Ontext managers
    "on_error",
    "on_success",
    # functions
    "exec_body_factory",
    "clean_docstring",
    "split",
    "sync_filter",
]

U = TypeVar("U")
V = TypeVar("V")
Namespace = Dict[str, Any]


def get_outer_namespaces() -> Tuple[Namespace, Namespace]:
    """
    Get the globals and locals from the context that called the function
    calling this utility

    Returns:
        globals, locals
    """
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back
        if frame is not None:
            frame = frame.f_back
            if frame:
                return frame.f_globals or {}, frame.f_locals or {}
    return {}, {}


def get_class_namespaces(cls: type) -> Tuple[Namespace, Namespace]:
    """
    Return the namespace the class is defined in and its own namespace

    Arguments:
        cls: class to retrieve namespaces for
    Returns:
        globals, locals
    """
    return inspect.getmodule(cls).__dict__, cls.__dict__ | {cls.__name__: cls}


class Singleton:
    """
    Base class of singleton produced by `singleton()`

    Exposed for typing purposes only
    """

    _classes_: ClassVar[dict[str, Type[Singleton]]] = {}

    __slots__ = ()

    _instance_: ClassVar[  # pylint: disable=unsubscriptable-object
        Optional[Singleton]
    ] = None
    _name_: ClassVar[str]
    _truthy_: ClassVar[bool]

    def __new__(cls) -> Singleton:
        if cls._instance_ is None:
            cls._instance_ = super().__new__(cls)
        return cls._instance_

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "Cannot instanciate Singleton class. Use the singleton function instead"
        )

    def __bool__(self) -> bool:
        return self._truthy_

    def __str__(self) -> str:
        return self._name_

    def __repr__(self) -> str:
        return self.__module__ + "." + self._name_


def exec_body_factory(
    ns: Dict[str, Any] = None, /, **kwargs: Any
) -> Callable[[Namespace], None]:
    """
    Helper factory for the ``exec_body`` argument of ``types.new_class``

    Arguments:
        kwargs:
    """
    values = (ns or {}) | kwargs

    def exec_body(namespace: Namespace):
        for key, value in values.items():
            namespace[key] = value

    return exec_body


def _no_subclass(cls, *args, **kwargs) -> None:
    raise TypeError(f"{cls.__name__} cannot be subclassed")


def singleton(
    name: str, truthy: bool = False, call_method: Callable = None
) -> Singleton:
    """
    Factory function for singleton values with prettier str() and repr()

    The created object (and its dynamic class) are marked as comming from the
    module that called `singleton`, if possible.

    Arguments:
        name: name of the singleton in the calling module
        truthy: truth value of the singleton object
        callable (Optional): __call__ implementation for the
            singleton object. Automatically converted to a staticmethod
            if it isn't one already. Do not include ``cls`` or ``self``
            as first argument.

    Returns:
        A singleton object that is the unique instance of a dynamic singleton
        class.

    Note:
        The dynamic class' module is the module that called `singleton` and
        ``name`` must be unique within that module. Calling `singleton` again
        from the same module with the same name will return the same object.
    """
    globalns, _ = get_outer_namespaces()
    module = globalns.get("__name__", __name__)
    classname = f"{module}.{name}"
    if classname not in Singleton._classes_:
        members = dict(
            __module__=module,
            __slots__=(),
            __init_subclass__=_no_subclass,
            __init__=lambda self: None,  # Override Singleton.__init__ which errors out
            _name_=name,
            _truthy_=truthy,
        )
        if call_method is not None:
            if not isinstance(call_method, staticmethod):
                members["__call__"] = staticmethod(call_method)
            else:
                members["__call__"] = call_method
        Singleton._classes_[f"{module}.{name}"] = types.new_class(
            name, (Singleton,), exec_body=exec_body_factory(**members)
        )
    return Singleton._classes_[f"{module}.{name}"]()


MISSING: Any = singleton("MISSING")
"""Sentinel object used when `None` cannot be used"""


class Virtual:
    """
    Mixin class to make a class non-inheritable and non-instanciable
    """

    __slots__ = ("__weakref__",)

    def __init__(self):
        raise TypeError("Cannot instanciate virtual class")

    def __init_subclass__(cls, *args, **kwargs):
        if Virtual not in cls.__bases__:
            raise TypeError("Cannot subclass virtual class")
        super().__init_subclass__(*args, **kwargs)


class _Commented:
    """
    Mixin class to add runtime comments to python objects

    Arguments:
        obj: python object to decorate
        **kwargs: comments to add on the object

    Returns:
        An instance of a special class that is a subclass of both `Commented`
        and of ``type(obj)``. The instance is a copy of ``obj`` for all intents
        and purposes related to the original type of ``obj``.

    Attributes:
        _comments_: Comments added on the object

    Warning:
        The type of ``obj`` must support creating a copy by passing an instance
        to its class (e.g. ``type(obj)(obj)`` should make a copy of ``obj``).
        Most builtins types do support such initialization.

    Note:
        Commenting an already commented object is akin to commenting the
        non-commented version.

    Usage::

        # Comment an object
        myint = Commented(5, source=("./myfile", 5))

        # Retrieve the class of a commented object
        type(myint) is Commented[int] # True

        # Use comments stored in the _comments_ attribute
        myint._comments_.values() # view of the comments dict values
    """

    _comments_: Dict[str, Any]

    @overload
    def __new__(cls: Type[_Commented], obj: U, **kwargs) -> U:  # type: ignore[misc]
        ...

    @overload
    def __new__(cls: Type[U], *args, **kwargs) -> U:
        ...

    def __new__(cls, *args, **kwargs):
        if cls is _Commented:
            return cls[type(args[0])](args[0], _comments_=kwargs)
        _comments_ = kwargs.pop("_comments_", {})
        instance = super().__new__(cls, *args, **kwargs)
        instance._comments_ = _comments_
        return instance

    def __class_getitem__(cls, key: type) -> Any:
        if cls is _Commented:
            if not isinstance(key, type):
                raise TypeError(f"{key} is not a class")
            if issubclass(key, _Commented):
                key = getattr(key, "_commented_class_", key)
            return cls._get_commented_class_(key)
        else:
            return super().__class_getitem__(key)  # type: ignore[misc] # pylint: disable=no-member

    @classmethod
    @functools.lru_cache
    def _get_commented_class_(cls, base: type) -> type:
        ns = {
            "__module__": cls.__module__,
            "_commented_class_": base,
        }
        return types.new_class(
            name="Commented_" + base.__name__,
            bases=(cls, base),
            exec_body=exec_body_factory(ns),
        )

    @staticmethod
    def repr(inst) -> str:
        """
        Equivalent of repr() that reveals commented instances

        By default, the `repr` of commented instance falls back to their class
        repr, which hides their commented nature, to make them behave as much
        as possible as a standard instance. This method provides a repr() that
        reveals commented method as such.

        Note:
            You can make this the default behavior of commented instances by
            binding this method as `Commented` 's ``__repr__``::
                Commented.__repr__ = Commented.repr
        """
        if not isinstance(inst, _Commented):
            return repr(inst)
        else:
            return "%s%s(%s%s)" % (
                ""
                if _Commented.__module__ == "__main__"
                else _Commented.__module__ + ".",
                _Commented.__qualname__,
                # Force skipping Commented.__repr__
                # safer if someone does Commented.__repr__ = Commented.repr
                super(_Commented, inst).__repr__(),
                (
                    ", " + ", ".join(f"{k!s}={v!r}" for k, v in inst._comments_.items())
                    if inst._comments_
                    else ""
                ),
            )


class Commented(Generic[U]):
    """
    Wrapper class to add runtime comments to python objects

    The wrapper object delegates most method call and attributes to the wrapped
    object. The actual type of the created object is still `Commented`, so using
    `isinstance()`, `type()` and so one does not work.

    Arguments:
        obj: python object to decorate
        **kwargs: comments to add on the object

    Attributes:
        _obj_: the wrapped object
        _comments_: Comments added on the object

    Usage::

        # Comment an object
        myint = Commented(5, source=("./myfile", 5))

        # Use comments stored in the _comments_ attribute
        myint._comments_.values() # view of the comments dict values
    """

    _obj_: U
    _comments_: Dict[str, Any]

    def __init__(self, obj, **comments):
        object.__setattr__(
            self, "_obj_", obj._obj_ if isinstance(obj, Commented) else obj
        )
        object.__setattr__(self, "_comments_", comments)

    def __getattribute__(self, name):
        if name in ("_obj_", "_comment_"):
            return object.__getattribute__(self, name)
        return getattr(self._obj_, name)

    def __setattr__(self, name, value):
        if name in ("_obj_", "_comments_"):
            object.__setattr__(self, name, value)
        return setattr(self._obj_, name, value)

    def __delattr__(self, name):
        if name in ("_obj_", "_comments_"):
            object.__delattr__(self, name)
        return delattr(self._obj_, name)

    # Those special methods are defined on `object` and not handled by __getattribute__
    # We need to fallback explicitely
    for method in (
        "dir",
        "eq",
        "format",
        "ge",
        "gt",
        "hash",
        "le",
        "lt",
        "ne",
        "reduce",
        "reduce_ex",
        "repr",
        "str",
    ):
        method = "__%s__" % method
        exec(
            f"def {method}(self, *args, **kwargs):\n"
            f"    return self._obj_.{method}(*args, **kwargs)"
        )
    del method


class ClassPropertyDescriptor(Generic[U, V]):
    """
    Descriptor for class properties

    Courtesy of https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    """

    __slots__ = ("fget", "fset")

    def __init__(
        self,
        fget: Union[
            classmethod, staticmethod
        ],  # pylint: disable=unsubscriptable-object
        fset: Union[
            classmethod, staticmethod
        ] = None,  # pylint: disable=unsubscriptable-object
    ):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj: U, cls: Type[U] = None) -> Any:
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


class subscriptable(Generic[U, V]):

    __slots__ = ("func", "__doc__")

    def __init__(self, func: Callable[[U], V]) -> None:
        """
        Function decorator to make a subscriptable object from a
        ``__getitem__`` implementation.

        Args:
            func: callable to use as ``__getitem__`

        Returns:
            An object that can be subscribed to call ``func``

        Notes:
            `func` should not include ``self`` as a first argument

        Usage::

            @subscriptable
            def my_subscriptable(key):
                return key

            assert my_subscriptable[8] == 8
        """
        self.func = func
        self.__doc__ = getattr(func, "__doc__", None)

    def __repr__(self):
        cls = type(self)
        return "%s%s(%s)" % (
            "" if cls.__module__ == "__main__" else cls.__module__ + ".",
            cls.__qualname__,
            repr(self.func),
        )

    def __call__(self) -> NoReturn:
        raise TypeError(f"{self} is not callable, use {self}[arg] instead")

    def __getitem__(self, arg: U) -> V:
        return self.func(arg)


def docstring(
    docstring: str = None, *, pre: str = None, post: str = None
) -> Callable[[U], U]:
    """
    Decorator to modify the docstring of an object.

    For all provided strings, unused empty lines are removed, and the indentation
    of the first non-empty line is removed from all lines if possible. This allows
    prettier code indentation when `docstring` is used as a decorator.

    Unused empty lines means initial empty lines for ``pre``, and final empty lines
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


def classproperty(
    func: Union[
        Callable, classmethod, staticmethod
    ]  # pylint: disable=unsubscriptable-object
) -> ClassPropertyDescriptor:
    """
    Decorator to make a class property

    This is similar to the builtin `property` but operates on classes.
    Courtesy of https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    """
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return ClassPropertyDescriptor(func)


@overload
def autoformat(
    cls: None,
    /,
    params: Union[str, Iterable[str]] = (
        "message",
        "msg",
    ),  # pylint: disable=unsubscriptable-object
) -> Callable[[type], type]:
    ...


@overload
def autoformat(
    cls: type,
    /,
    params: Union[str, Iterable[str]] = (
        "message",
        "msg",
    ),  # pylint: disable=unsubscriptable-object
) -> type:
    ...


def autoformat(
    cls: type = None,
    /,
    params: Union[str, Iterable[str]] = (  # pylint: disable=unsubscriptable-object
        "message",
        "msg",
    ),
):
    """
    Class decorator to autoformat string arguments in the ``__init__`` method.

    Modifies the class ``__init__`` method in place by wrapping it. The wrapped
    method will call the str.format() method on arguments specified in the ``params``
    argument of the decorator, if they exist in the decorated class's __init__
    function signature. All other arguments are passed to str.format() as a dict.

    Arguments:
        params: names of the arguments to autoformat

    Returns:
        The decorated class (same object), with a wrapped ``__init__``

    Usage::

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

    if cls is not None:
        orig_init = getattr(cls, "__init__")
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
    else:
        return functools.partial(autoformat, params=params)


class InlineString(str):
    def __repr__(self) -> str:
        return self

    def __str__(self) -> str:
        return self


def _sig_without(
    sig: inspect.Signature,
    param: Union[int, str],  # pylint: disable=unsubscriptable-object
) -> inspect.Signature:
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
    """
    Transform a signature to a parameter definition string
    """
    return str(sig).split("->", 1)[0].strip()[1:-1]


def _sig_to_call(sig: inspect.Signature) -> str:
    """
    Transform a signature to a call string
    """
    l = []
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            l.append(f"{p.name}={p.name}")
        else:
            l.append(p.name)
    return ", ".join(l)


def post_init(cls: type) -> type:
    """
    Class decorator to support ``__post_init__`` on classes

    This is useful for classes decorated with `attr`, because ``__attr_post_init__``
    doesn't support arguments.

    This decorator wraps the class ``__init__`` method in a new method that accepts
    merged arguments from ``__init__`` and ``__post_init__`` (including those from
    base classes) and dispatch them to ``__init__ ``and then ``__post_init__`` (starting
    from the highest class in the inheritance tree).
    """
    if not isinstance(cls, type):
        raise TypeError("Can only decorate classes")
    if not hasattr(cls, "__post_init__"):
        raise TypeError("The decorated class must have a __post_init__ method")
    # Ignore the first argument which is the "self" argument
    sig = init_sig = _sig_without(inspect.signature(getattr(cls, "__init__")), 0)
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
                    f"__post_init__ signature is incompatible with {parent.__qualname__}.{method}()"
                ) from err
            # No exception
            previous.append((parent, "__post_init__", post_sig))
    # handles type annotations and defaults
    # inspired by the dataclasses module
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
            p = p.replace(default=InlineString(f"__default_{p.name}"))
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
    init = ns["__make_init__"](getattr(cls, "__init__"), **localns)
    self_param = inspect.Parameter(self_, inspect.Parameter.POSITIONAL_ONLY)
    init.__signature__ = inspect.Signature(
        parameters=[self_param] + list(sig.parameters.values()), return_annotation=None
    )
    setattr(cls, "__init__", init)
    return cls


@contextmanager
def on_error(func, *args, yield_=None, **kwargs):
    """
    Context manager that calls a function if the managed code raises
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


def clean_docstring(
    doc: str,
    unused: Literal["pre", "post"] = None,  # pylint: disable=unsubscriptable-object
) -> str:
    """
    Removes initial empty lines and shared indentation

    Arguments:
        doc: docstring to clean up
        unused: whether to remove starting or ending empty lines
    Returns:
        The cleaned docstring
    """
    lines = doc.split("\n")
    if unused == "pre":
        try:
            index = next(i for i, l in enumerate(doc) if l.strip())
            lines = lines[index:]
        except StopIteration:
            lines = []
    elif unused == "post":
        try:
            index = next(i for i, l in enumerate(reversed(doc)) if l.strip())
            doc = doc[: len(doc) - index]
        except StopIteration:
            lines = []
    if lines:
        first_line = lines[0]
        index = len(first_line) - len(first_line.lstrip())
        indent = first_line[:index]
        if all(l.startswith(indent) for l in lines if l.strip()):
            lines = [(l[index:] if l.strip() else l) for l in doc]
    return "\n".join(doc)


def split(
    func: Callable[[U], bool], iterable: Iterable[U]
) -> Tuple[Tuple[U, ...], Tuple[U, ...]]:
    """split an iterable based on the truth value of the function for element

    Arguments
        func -- a callable to apply to each element in the iterable
        iterable -- an iterable of element to split

    Returns
        falsy, truthy: 2-tuple, the first with elements of the iterable where
        func(e) is false, the second with element of the iterable where func(e)
        is true
    """
    falsy, truthy = [], []
    for e in iterable:
        if func(e):
            truthy.append(e)
        else:
            falsy.append(e)
    return tuple(falsy), tuple(truthy)


def sync_filter(func: Callable, *iterables: Iterable[U]) -> Tuple:
    """
    Filter multiple iterable at once, selecting values at index i
    such that func(iterables[0][i], iterables[1][i], ...) is True
    """
    return tuple(zip(*tuple(i for i in zip(*iterables) if func(*i)))) or ((),) * len(
        iterables
    )
