from __future__ import annotations

import typing
from typing import Union

import attr

from pheres.aliases import TypeArgs, TypeForm, TypeOrig
from pheres.json.aliases import _JSONArrayTypes
from pheres.utils import MISSING, Singleton, post_init


@attr.dataclass
class JSONData:
    typeform: TypeForm
    _orig: Union[Singleton, TypeOrig] = attr.ib(default=MISSING, init=False)
    _args: Union[Singleton, TypeArgs] = attr.ib(default=MISSING, init=False)
    is_class_initialized: bool = attr.ib(default=False, init=False)

    @property
    def orig(self):
        if self._orig is MISSING:
            self._orig = typing.get_origin(self.typeform)
        return self._orig

    @property
    def args(self):
        if self._args is MISSING:
            self._args = typing.get_args(self.typeform)
        return self._args
