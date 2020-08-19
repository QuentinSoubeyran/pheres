# Pheres

In greek methology, Pheres is (one of) the son of Jason. In python, Pheres expands the `json` builtin-module with useful utilities.
It is a drop-in replacement for `json` and provides several utilities to work with JSON in python.

# Installation

> Not yet available on PyPI

```bash
pip install pheres
```

# Features

This is a quick introduction. See the [wiki](https://github.com/QuentinSoubeyran/pheres/wiki) for details.

Pheres is a drop-in replacement for `json`. It provides the complete `json` module content, tweaks the loading and dumping functions (see below) and adds many utilities.

## JSONable classes

*Pheres* defines **JSONable classes** by passing the class to its `@pheres.jsonable` class decorator, or if the class inherits from `pheres.JSONable`. JSONable classes define their JSON representation using python's type hints, and are very easily serializable to and deserializable from JSON:

```python
from typing import *
from pheres import jsonable, loads, dumps
from dataclasses import dataclass, field

# Use type hint to specify the attributes to jsonize
# The decorator order does not matter
@jsonable
@dataclass
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

The main sell for JSONable classes is that they are fully *typed*. Type error will be detected early on during JSON decoding, and their exact location in the file or string reported.

JSONable classes are fully compatible with the rest of the module, including as type-hint for future JSONable class definition. You can therefore describe a whole typed json in python. JSONable classes are also fully compatible with dataclasses from the `dataclasses` builtin module.

This work really well for configuration file, for instance. You can declare the different parts of your configuration file are JSONable class, using the attributes. *Pheres* handles the deserialization for you. You then use methods to access and/or process your file.

### Serializing

Just call the `to_json()` method of your JSONable class. If you need to encode, `pheres.JSONableEncoder` does that for you. Use it as the encoder `json.dumps(..., cls=pheres.JSONableEncoder)`. Or you can use `pheres.dumps()` that uses it by default.

### Deserialization

Pheres is really useful for deserializing complex typed data: all JSONable class can be easily deserialized from JSON, including from files. All deserialization are __completely typed__: if the JSON has a string were an array is required, or is missing a key to properly define your complex JSONable class, Pheres will find out exactly were the problem is (as in, the line and column in the file were things got wrong).

You can deserialize:
- With `json.load` or `loads`, by providing the `json.JSONDecoder` subclass built into each JSONable class. You need to deserialize a complex type, `MyComplexConfig`, with recursive definitions ? It's as easy as `json.load(file, cls=MyComplexConfig.Decoder)`
- As a shortcut to the above, each JSONable class have methods `from_json_file`, `from_json_str` and `from_json` that do just that, shorter
- Or you can use `pheres.jsonable_hook()` as the `object_hook` argument of `json.load()`. This hook will recognize and convert any JSONable class present in the JSON. It finds the correct type by examining the keys and the types. `pheres.load()` uses that hook by default, so its quicker to use

The (little) price for that easy deserialization, is that you cannot declare JSONable class that have common representation (unless they are subclasses of each other). See the wiki for details.

## Typing for JSON

Pheres provides utilities to inspect and test the types of JSON object, such as `typeof`, `typecheck` and `is_json`. It also provides utilities for working with type-hints themselves: `normalize_json_tp` and `have_common_value` work on type hints objects.

## Various Utilities

There are also useful small functions, such as `get`, `has` and `set` that accept several keys (`int` or `str`) at once and can navigate in python JSON object. Those are ultimately `dict`s and `list`s, but creating a key-value down an arborescent in one go is convinient.

The function `flatten`, `expand` and `compact` provide transformation on JSON objects to make handling them easier in certain contexts.