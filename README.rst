.. _pheres:

======
Pheres
======

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

Pheres is an extension of python's ``json`` module, that is almost a drop-in replacement.
It allows typed JSON decoding/encoding and features the ``@jsonable`` `decorator`__
that makes python classes (de)serializable to JSON format.

The name comes from greek mythology: Pheres is a son of Jason (JSON).

For more information, refer to the `documentation`__.

**NOTE**: Pheres is currently in *alpha*. There are planned features not
yet implemented, tests are far from complete so features may have bugs, and
the API might change. Please report bugs and suggestions on the `bug tracker`__.

.. __: https://quentinsoubeyran.github.io/pheres/api/api_jsonable.html#pheres._jsonable.jsonable
.. __: https://quentinsoubeyran.github.io/pheres/
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

Pheres is typed and i, many places will require types.
Those are specified with the objects from the builtin ``typing`` `module`__.

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
    

    
    database = ph.TypedJSONDecoder[list[People]].load("data/database.json")
    database[0].phone("Bob")

While this example is overly simple, it highlights the main features of ``@jsonable``:

* Definition similar to those of `dataclasses.dataclass`, that can be combined
* Different types of jsonable classes depending on the JSON representation
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
