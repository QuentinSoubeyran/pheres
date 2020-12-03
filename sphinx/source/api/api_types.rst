Types
=====

This module contains all values used in type-hints in pheres.
It should be imported in the local namespace with ``from pheres.types import *``.
While this kind of import is not recommended in python, ForwardRef in type hints needs the value to be accessible from the namespace they are used in, not the one they are defined in.
This means that is you used e.g. `JSONType`, `JSONValue`, `JSONArray` or `JSONObject`, some values must be availabe in the current namespace.
This modules simplifies that by provided all values necessary for type-hints, and only those (as to not pollute the namespace)

``pheres.types``
----------------

.. automodule:: pheres.types
    