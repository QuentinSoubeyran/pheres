.. _pheres:

======
Pheres
======

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

Pheres is an extension of python's `json` module, that is almost a drop-in replacement.
It allows typed JSON decoding/encoding and features the `@jsonable`
that makes python classes automatically (de)serializable to JSON format.

The name comes from greek mythology: Pheres is a son of Jason (JSON).

For more information, refer to the `documentation`__.

**NOTE**: Pheres is currently in *alpha*. There are planned features not
yet implemented, tests are far from complete, features may have bugs and
the API might change. Please report bugs and suggestions on the `bug tracker`__.

.. TODO: update when Sphinx doc is available

.. __: https://github.com/QuentinSoubeyran/pheres/wiki
.. __: https://github.com/QuentinSoubeyran/pheres/issues

Installation
============
.. TODO: update when available

Pheres is not yet available on `PyPI`__, so you must intall it from github::

    pip install git+https://github.com/QuentinSoubeyran/pheres

We are working on releasing Pheres to `PyPI`__.

.. __: https://pypi.org
.. __: https://pypi.org

Documentation
=============
.. TODO: update

The documentation is available from `github wiki`__.

.. __: https://github.com/QuentinSoubeyran/pheres/wiki


Overview
========

Pheres is typed and i, many places will require types.
Those are specified with the objects from the builtin `typing` module.

**Note**: Starting with `Python 3.9`__, you can also use builtin types
for generic type annotations thanks to `PEP 585`__.

Pheres provide the `pheres.types` submodule for quick imports of all
the required types in the current namespace. See the `documentation`__
for why this is necessary.

.. __: https://docs.python.org/3.9/whatsnew/3.9.html
.. __: https://www.python.org/dev/peps/pep-0585/

.. TODO: Update when Sphinx doc is available

.. __: https://github.com/QuentinSoubeyran/pheres/wiki

Typed JSON
----------

The `TypedJSONDecoder` class is a typed version of the builtin `json.JSONDecoder`,
with a few tweaks to make it easier to use::

    import pheres as ph
    from pheres.types import *

    GraphDecoder = ph.TypedJSONDecoder[dict[str, list[str]]]

    graph = GraphDecoder.load("graph.json")

    not_a_graph = """{
        "v1": ["v1", "v2"],
        "v2": "v2"
    }"""
    GraphDecoder.loads(not_a_graph) # Raises pheres.TypedJSONDecodeError

Jsonable classes
----------------

The `@jsonable <pheres._jsonable.jsonable>` decorator is an easy way to make
a class serializable and deserializable to JSON. It is compatible with
`dataclasses.dataclass` and `attr.s` which makes creating classes seamless::

    from dataclasses import dataclass
    
    import pheres as ph
    from pheres.types import *
    
    @ph.jsonable.Array["People", ...](after="People")
    class Contacts:
        contacts: list["People"]

        def __init__(self, *contacts) -> None:
            self.contacts = contacts
        
        def has(self, p: People) -> bool:
            return p in self.contacts
        
        def __iter__(self):
            return iter(self.contacts)
    
    @dataclass
    @ph.jsonable
    class People:
        name: str
        surname: str
        number: int
        contacts: Contacts

        def phone(name: str) -> None:
            for contact in self.contacts:
                if contact.name == name:
                    print("Calling %s" % contact.number)
                    break
            else:
                print("I do not know %s's number" % name)
    
    database = ph.TypedJSONDecoder[list[People]].load("data/database.json")
    database[0].phone("Bob")

This overly simple example highlights the main features of `@jsonable <pheres._jsonable.jsonable>`:

* Definition similar to those of `dataclasses.dataclass`, that can be combined
* Different types of jsonable classes depending on the JSON representation
* The ability to nest jsonable classes together, and to create cyclic definitions

Typing
------

Pheres also contains some utilities to analyse the type of loaded JSON::

    import pheres as ph

    jdata = ph.load("data/my_file.json")

    if ph.typeof(jdata) is ph.JSONObject:
        print("Root document found!")

See the `documentation`__ for details.

.. TODO: update when Sphinx documentation is available

.. __: https://github.com/QuentinSoubeyran/pheres/wiki
