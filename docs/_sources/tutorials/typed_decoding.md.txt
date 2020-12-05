Typed JSON decoding
===================

**NOTE**: This tutorial is outdated and needs to be rewritten

The `TypedJSONDecoder` allows to decode/deserialize JSON files or string with typechecking.To use it, pass it as the `cls` argument of [`json.load()`](https://docs.python.org/3/library/json.html#json.load) (and `json.loads()`), after parametrizing it with a type.

Let us go throught an example:
```python
from typing import *
import pheres as ph

# Build a *parametrized* version of TypedJSONDecoder to decode typed JSON
ArrayDecoder = ph.TypedJSONDecoder[List[ph.JSONObject]]

# JSON string to decode
jstring = """[
    {
        "name": "first object in the array!"
    },
    {
        "name": "second object",
        "bool-key": false,
        "nested-object": {
            "name": "a nested object"
        }
    }
]"""

# Decode with type-checking
array = ArrayDecoder.loads(jstring)

# Invalid JSON string: only objects are allowed, not another array
jstring = """[
    {
        "key": "value"
    },
    [
        "i'm an array, not an object"
    ]
]"""

# raises TypedJSONDecodeError
ArrayDecoder.loads(jstring)
```
First, we import necessary modules.

Then, we parametrize a `TypedJSONDecoder` for the type we want to decode. This is done by indexing `TypedJSONDecoder` with a type-hint. The supported type-hints are defined in the [jtyping](jtyping#type-hint-utilities) category.

Then, you simply used the parametrized `TypedJSONDecoder` in [`json.load()`](https://docs.python.org/3.9/library/json.html#json.load), as the `cls` argument (that is the class that is used to decode).
As a short-hand, **parametrized** `TypedJSONDecoder` have `load()` and `loads()` methods, that simply wraps their `json` counterpart but using the parametrized decoder by default.

If the type-checking fails, the decoding process raises a `TypedJSONDecodeError` that tells what is wrong and were.

> *NOTE*
>
> You cannot use `TypedJSONDecoder` directly. Internally, it is an Abstract Base Class, and parametrizing it dynamically creates a subclass. This is because in the `json.load` function, the decoder is provided as a class and not an instance.