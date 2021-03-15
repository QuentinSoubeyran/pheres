import pheres.json as json
from typing import *

JSONLit = Union[int, str]
JSONArr = List["JSON"]
JSONObj = Dict[str, "JSON"]
JSON = Union[JSONLit, JSONArr, JSONObj]

JSDecoder = json.TypedDecoder[JSON]

q = JSDecoder.loadf("../../Documents/Campagne ArM5/PJ/Querneus/Querneus-Ex-Misc.json")

print(q)