# Pheres

In greek methology, Pheres is (one of) the son of Jason. In python, Pheres expands the `json` builtin-module with useful utilities.
It is a drop-in replacement for `json` and provides several utilities to work with JSON in python.

# Installation

> Upcoming !

```bash
pip install pheres
```

# Features

This is a quick introduction. See the [wiki](https://github.com/QuentinSoubeyran/pheres/wiki) for fill details

## Easy Serialization and deserialization

Pheres has class decorator, `@jsonize`, as well as an Abstract Base Class, `JSONable`, that do exactly the same thing: they make your python class serializable and deserializable to JSON, using type hints.

They are fully compatible with dataclasses and will handle fields just fine.

```python3
from typing import *
from pheres import jsonize
from dataclasses import dataclass, field

# Use type hint to specify the attributes to jsonize
# The decorator order does not matter
@jsonize
@dataclass
class A:
    x: int
    y: Union[Tuple[int, int, int], Dict[str, float]] # complex type hints are supported
    name: str = "A" # default value can be provided
    list_of_things: List[str] = field(default_factory=list) #fields are suported


a = A(x=3, y = (1,2,3))
print(a) # A(x=3, y=(1,2,3))

# Serialize with the method
assert a.to_json() == {}

```