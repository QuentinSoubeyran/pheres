import json
from pprint import pprint

import pytest
from hypothesis import given

from pheres.typed_json.decoder import TypedJSONDecoder

from .strategies import typed_jsons


@given(typed_jsons())
def test_typed_decoder(typeform_example):
    typeform, example = typeform_example
    # pprint(typeform_example)
    assert TypedJSONDecoder[typeform].loads(json.dumps(example)) == example
