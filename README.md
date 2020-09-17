<h1 align="center">Pheres</h1>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
---

In greek methology, Pheres is (one of) the son of Jason. In python, Pheres expands the `json` builtin-module with useful utilities.
It is a drop-in replacement for `json` and provides several utilities to work with JSON in python.

# Disclaimer
*Pheres* is currently in alpha. It might have bugs or features not working properly, and the API might change.

# Installation

> Not yet available on PyPI

Install directly from GitHub:

```bash
pip install "git+https://github.com/QuentinSoubeyran/pheres#egg=pheres"
```


# Features

This is a quick introduction. See the [wiki](https://github.com/QuentinSoubeyran/pheres/wiki) for details.

Pheres is a drop-in replacement for `json`. It provides the complete `json` module content, tweaks the loading and dumping functions (see below) and adds many utilities.

## Jsonable classes

*Pheres* adds **jsonable classes**. Jsonable classes are classes that can readily be serialized and deserialized in the JSON format. (Almost) no code user code is required.
Jsonable classes are easy to make, akin to dataclasses: pass the class to the `@pheres.jsonable` decorator and annotate the attributes with type-hints.

```python
from typing import *
from pheres import jsonable, loads, dumps
from dataclasses import dataclass, field

# Use type hint to specify the attributes to jsonize
# The decorator order does not matter
@dataclass
@jsonable
class ExampleJSONable:
    x: int
    y: Union[Tuple[int, int, int], Dict[str, float]] # complex type hints are supported
    name: str = "A" # default value can be provided
    list_of_things: List[str] = field(default_factory=list) #fields are suported

# Get an instance
ex = ExampleJSONable(5, (1,2,3))

# Serialize to JSON
jstring = dumps(ex)

# Load from JSON
ex2 = ExampleJSONable.Decoder.loads(jstring)
assert ex == ex2

# Type error reporting
# X should be an 'int', not 'float'
jstring = """{
    "x": 15.006,
    "y": {
        "gravity": 9.81,
        "mass": 75.0
    }
}"""

# Raises TypedJSONDecodeError that reporst were exactly
# the type error is
ExampleJSONable.Decoder.loads(jstring)
```

Jsonable classes are fully *typed*. Type errors will be detected early on during JSON decoding, and their exact location in the file or string reported.

Jsonable classes are fully compatible with [dataclasses](https://docs.python.org/3/library/dataclasses.html) and can be nested. This provides a powerfull API to define structured data as python classes whose methods use/handle the data. *Pheres* will provide seamless loading and dumping to JSON. The obvious example are configuration files.

It is also possible to create classes that serialize to (deserialize from) simple JSON values or arrays.

### Serializing

Jsonable classes automatically gain a `to_json()` method, that return a python object suitable for `json.dump()`. `pheres.JSONableEncoder` can encode Jsonable classes to a string when used in `json.dumps(..., cls=pheres.JSONableEncoder)`, and `pheres.dumps()` defaults to that encoder.

### Deserialization

All Jsonable class can be deserialized from JSON object, string or files using the `from_josn()` method of any Jsonable class. Deserialization is __typed__: values are type-checked against the type hint used in the definition, and errors are reported. 

Additionally, *Pheres* provides a `jsonable_hook` to pass to `json.load()` (or `pheres.load` directly) that detects any Jsonable class serialization in the JSON and replace them by a proper instance. The drawbacks is that Jsonable classes cannot have serialization that could correspond to more than a single class, but *Pheres* provides a mechanism to resolve such conflicts.

## Typing for JSON

Pheres provides utilities to inspect and validate the types of JSON object, such as `typeof()`, `typecheck()` and `is_json()`. It also provides utilities for working with type-hints themselves.

## Various Utilities

There are also useful small functions, such as `get`, `has` and `set` that accept nested keys (`int` or `str`) at once and can navigate in python JSON object. Those are ultimately `dict`s and `list`s, but adding a key-value pair deep down an arborescent in one go is convenient.

The function `flatten`, `expand` and `compact` provide transformations on JSON objects that make handling them easier in certain contexts.