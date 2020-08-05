# -*- coding: utf-8 -*-
"""
Utility module that expands the builtin module 'json'
"""
from json import *

from .misc import *
from .jtyping import *
from .utils import *
from .decoder import *
from .jsonize import *

from . import metadata

__version__ = metadata.version
__status__ = metadata.status
__author__ = metadata.author
__copyright__ = metadata.copyright
__license__ = metadata.license
__maintainer__ = metadata.maintainer
__email__ = metadata.email
