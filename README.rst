.. _pheres:

======
Pheres
======

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

Pheres is a typed extension of python's ``json`` module.
By typed, I mean that it can typecheck values at runtime.
It provides:

* typed JSON decoding with ``TypedJSONDecoder`` (`link`__)
* JSON type analysis with ``typeof`` (`link`__)
* runtime typechecking with ``typecheck`` (`link`__)
* easy typechecked conversion between python objects, custom classes and JSON str or files
  with the ``@jsonable`` decorator  (`link`__)

The name comes from greek mythology: Pheres is a son of Jason (JSON).

For more information, refer to the `documentation`__.

.. __: https://quentinsoubeyran.github.io/pheres/api/api_decoder.html#pheres._decoder.TypedJSONDecoder
.. __: https://quentinsoubeyran.github.io/pheres/api/api_typing.html#pheres._typing.typecheck
.. __: https://quentinsoubeyran.github.io/pheres/api/api_typing.html#pheres._typing.typeof
.. __: https://quentinsoubeyran.github.io/pheres/api/api_jsonable.html#pheres._jsonable.jsonable
.. __: https://quentinsoubeyran.github.io/pheres/

Development status
==================

Pheres is currently in *alpha*.
This is essentially a personal project I'm using to learn to do (hopefully) better code,
and python tools I find interesting (``typing`` module, ``mypy``, ``pytest``,
``hypothesis``, ``black`` ...).
This means that:

* features are incomplete and new features may appear
* type annotations are not complete
* the API might change (but I try to avoid that)
* the tests are incomplete
* there may be bugs

If you find bugs or have suggestions, you are welcome to report them on the  `bug tracker`__.

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

The documentation is available on `github pages`__.

.. __:  https://quentinsoubeyran.github.io/pheres/


Overview
========

Pheres is typed and will require types in many places.
Types are specified with type hints from the builtin ``typing`` `module`__.

**Note**: Starting with `Python 3.9`__, you can also use builtin types
for generic type annotations thanks to `PEP 585`__.

Pheres provide the ``pheres.types`` submodule for quick imports of all
the required types in the current namespace. See the `documentation`__
for why this is necessary.

.. __: https://docs.python.org/3/library/typing.html
.. __: https://docs.python.org/3.9/whatsnew/3.9.html
.. __: https://www.python.org/dev/peps/pep-0585/
.. __: https://quentinsoubeyran.github.io/pheres/

Typed JSON
----------

The ``TypedJSONDecoder`` class is a typed version of the builtin ``json.JSONDecoder``,
with a few tweaks to make it easier to use::

    import pheres as ph
    from pheres.types import *

    GraphT = dict[str, list[str]]
    GraphDecoder = ph.TypedJSONDecoder[GraphT]

    graph = GraphDecoder.load("graph.json")

    not_a_graph = """{
        "v1": ["v1", "v2"],
        "v2": "v2"
    }"""

    # Raises pheres.TypedJSONDecodeError
    GraphDecoder.loads(not_a_graph) 

Jsonable classes
----------------

The ``@jsonable`` `decorator`__ is an easy way to make
a class serializable and deserializable to JSON. It is compatible with
``dataclasses.dataclass`` and ``attr.s``::

    from dataclasses import dataclass
    
    import pheres as ph
    from pheres.types import *

    @dataclass
    @ph.jsonable(after="Contacts")
    class People:
        name: str
        surname: str
        number: int
        contacts: "Contacts"

        def phone(name: str) -> None:
            for contact in self.contacts:
                if contact.name == name:
                    print("Calling %s" % contact.number)
                    break
            else:
                print("I do not know %s's number" % name)
    
    @ph.jsonable.Array["People", ...](internal="contacts")
    class Contacts:
        contacts: list[People]

        def __init__(self, *contacts) -> None:
            self.contacts = list(contacts)
        
        def has(self, p: People) -> bool:
            return p in self.contacts
        
        def __iter__(self):
            return iter(self.contacts)
    
    alice = People.from_json("""{
        "name": "alice",
        "surname": "",
        "number": 9999999,
        "contacts": [
            {
                "name": "Bob",
                "surname": "",
                "number": 12345678,
                "contacts": []
            }
        ]
    }""")

    print(alice.to_json())
    assert alice == People.from_json(alice.to_json())
    people_list = ph.dumps([alice])

    database = ph.TypedJSONDecoder[list[People]].loads(people_list)
    database[0].phone("Bob")

While this example is overly simple, it highlights the main features of ``@jsonable``:

* Definitions similar to those of, and compatible with, ``dataclasses.dataclass``
* Different types of jsonable classes, depending on the JSON representation
* The ability to nest jsonable classes together, and to create cyclic definitions

.. __: https://quentinsoubeyran.github.io/pheres/api/api_jsonable.html#pheres._jsonable.jsonable

Typing
------

Pheres also contains some utilities to analyse the types of loaded JSON::

    import pheres as ph

    jdata = ph.load("data/my_file.json")

    if ph.typeof(jdata) is ph.JSONObject:
        print("Root document found!")

See the `documentation`__ for details.

.. __: https://quentinsoubeyran.github.io/pheres/
