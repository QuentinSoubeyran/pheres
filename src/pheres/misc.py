from typing import Dict, List, Tuple, Type, Union

from .typing import JSONValue

# Types used in his module
FlatKey = Tuple[Union[int, str], ...]  # pylint: disable=unsubscriptable-object
FlatJSON = Dict[FlatKey, JSONValue]

T = Type[List]


def mock(arg: T):
    """
    Test for Union autodoc
    """
