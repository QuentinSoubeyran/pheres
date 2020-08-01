# -*- coding: utf-8 -*-
"""
Module with various internal miscs

Part of the JsonPlus package
"""

import functools
import inspect


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

        return wrapped_init

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__init__ = AutoFormatMixin._init_wrapper(cls.__init__)


class JsonError(Exception):
    """
    Base exception for the jsonplus module
    """
