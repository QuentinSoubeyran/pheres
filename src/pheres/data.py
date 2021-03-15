from __future__ import annotations

from typing import Final, Optional, Union

import attr

from pheres.json.data import JSONData
from pheres.utils import MISSING, Singleton

PHERES_ATTR: Final[str] = "__pheres_data__"


@attr.dataclass
class PheresData:
    """
    Dataclass to store the data on a class decorated by Pheres
    """

    json: Union[Singleton, JSONData] = MISSING


def get_data(cls: type) -> Optional[PheresData]:
    return getattr(cls, PHERES_ATTR, None)


def get_data_json(cls: type) -> Optional[JSONData]:
    if (data := getattr(cls, PHERES_ATTR, None) is not None):
        if data.json is not MISSING:
            return data.json
