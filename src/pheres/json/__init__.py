"""
Typed version of the `json` builtin module
"""
from json import *

from pheres.json.decoder import *

loadf = UsableDecoder.loads
load = UsableDecoder.load
loads = UsableDecoder.loads