from typing import Dict, List, Tuple, Union

from .typing import JSONValue

# Types used in his module
FlatKey = Tuple[Union[int, str], ...]  # pylint: disable=unsubscriptable-object
FlatJSON = Dict[FlatKey, JSONValue]
