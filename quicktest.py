import json
from typing import *

import pheres.typed_json.decoder as decoder

typeform = Tuple[None, None]
example = [None]

class DT(TypedDict):
    a: int = 0

print("TYPEFORM:", typeform)
print("EXAMPLE :", example)
print("TEST    :", decoder.TypedJSONDecoder[typeform].loads(json.dumps(example)))
