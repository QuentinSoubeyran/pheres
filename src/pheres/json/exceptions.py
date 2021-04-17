import json
from typing import Any, Union

import attr

from pheres.aliases import TypeForm
from pheres.exceptions import PheresError
from pheres.json.aliases import Position, RawJSON
from pheres.utils import MISSING, autoformat

__all__ = [
    "JSONTypeError",
    "JSONValueError",
    "JsonableError",
    "JsonableAttrError",
    "JsonableAttrValueError",
    "JsonableAttrTypeError",
    "DecodeError",
    "TypedJSONDecodeError",
]


@autoformat
@attr.dataclass(auto_exc=True)
class JSONTypeHintError(PheresError, TypeError):
    """
    Raised for invalid JSON type hints

    Attributes:
        type: invalid type hint
        msg: explanation of the error
    """

    type: TypeForm
    msg: str = "{type} is not a valid JSON type hint"


@autoformat
@attr.dataclass(auto_exc=True)
class JSONTypeError(PheresError, TypeError):
    """
    Raised when a value doesn't have the excepted type

    Attributes:
        type: expected type
        value: invalid value
        msg: explanation of the error
    """

    type: TypeForm
    value: Any
    msg: str = "{value} doesn't have type {type}"


@autoformat
@attr.dataclass(auto_exc=True)
class JSONValueError(PheresError, ValueError):
    """
    Raised when an invalid json value is encountered

    Attributes:
        value: invalid value
        msg: explanation of the error
    """

    value: Any
    msg: str = "invalid JSON {value}"


@autoformat
@attr.dataclass(auto_exc=True)
class JSONKeyError(PheresError, KeyError):
    """
    Raised on KeyError on a JSON object

    Attributes:
        obj: object with missing key
        key: missing key
        msg: explanation of the error
    """

    obj: Any
    key: Union[int, str, tuple[Union[int, str], ...]]
    msg: str = "{obj} has no key '{key}'"


###################
# JSONABLE ERRORS #
###################


@attr.dataclass(auto_exc=True)
class JsonableError(PheresError):
    """
    Raised on problems with `@jsonable <jsonable>` when no better sub-exception
    exists

    Attributes:
        msg: explanation of the error
    """

    msg: str


@attr.dataclass(auto_exc=True)
class JsonableTypeFormError(PheresError):
    """
    Raised on problems with type forms used with the @jsonable <jsonable>`
    decorator

    Attributes:
        typeform: errored type form
        msg: explanation of the error
    """

    typeform: TypeForm
    msg: str = "{typeform!s} is not a valid JSON type form"


@autoformat
@attr.dataclass(auto_exc=True)
class JsonableAttrError(JsonableError):
    """
    Raised on problems with `@jsonable <jsonable>` due attribute
    problems

    Attributes:
        cls: name of the class that raised the error
        attrs: set of problematic attributes
        msg: explanation of the error
    """

    cls: str
    attrs: set[str]
    detail: str = ""
    msg: str = "Error in '{cls}' with attributes {attrs}: {detail}"


@autoformat
@attr.dataclass(auto_exc=True)
class JsonableAttrValueError(JSONValueError, JsonableAttrError):
    """
    Raised when the default value of a json attribute is not a valid json

    Attributes:
        cls: name of the class that raised the error
        attr: name of the attribute that raised the error
        value: invalid value
        msg: explanation of the error
    """

    msg: str = "Invalid JSON value '{value}' in class '{cls}' at attribute '{attr}'"


@autoformat
@attr.dataclass(auto_exc=True)
class JsonableAttrTypeError(JSONTypeError, JsonableAttrError):
    """
    Raised when the default value of a json attribute doesn't have the correct type

    Attributes:
        type: expected type
        value: invalid value
        msg: explanation of the error
    """

    msg: str = "{value} doesn't have type {type} in class '{cls}' at attribute '{attr}'"


##################
# DECODING ERROR #
##################


@attr.dataclass(auto_exc=True)
class DecodeError(PheresError):
    """
    Raised on decoding problem in Pheres when no better exception exists

    Attributes:
        msg: explanation of the error
    """

    msg: str


@attr.dataclass(auto_exc=True)
class UnparametrizedDecoderError(DecodeError):
    """
    Raised when `TypedJSONDecoder` is used without type parametrization

    Attributes:
        msg: explanation of the error
    """


class TypedJSONDecodeError(json.JSONDecodeError, DecodeError):
    """
    Raised when the decoded type is not the expected one

    Attributes:
        doc (Union[str, JSONType]): JSON document that contains the error
        pos (Union[int, Tuple[str, ...]]): position of the error in 'doc'
            can be either an string index, or a list of keys in the json
        lineno (Optional[int]): line number of pos
        colno (Optional[int]): column number of pos
        msg: explanation of the error
    """

    pos: Position  # type: ignore[assignment]

    def __init__(
        self,
        msg: str,
        doc: Union[str, RawJSON],
        pos: Position,
        lineno: int = MISSING,
        colno: int = MISSING,
    ):
        """
        Special case when the decoded document is an object
        """
        if isinstance(pos, int):
            if lineno is MISSING:
                lineno = doc.count("\n", 0, pos) + 1
            if colno is MISSING:
                colno = pos - doc.rfind("\n", 0, pos)
            errmsg = "%s, at line %d, column %d (char %d)" % (msg, lineno, colno, pos)
        else:
            # quote string keys
            path = ['"%s"' % p if isinstance(p, str) else str(p) for p in pos]
            path = ["[%s]" % k for k in path]
            keys = "".join(("<python object>", *path))
            errmsg = "%s, at %s" % (msg, keys)
            lineno = -1
            colno = -1
        ValueError.__init__(self, errmsg)
        self.msg = errmsg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno
