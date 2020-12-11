from dataclasses import dataclass

import pytest

# from pheres.utils import slotted_dataclass

# @slotted_dataclass(
#     dataclass,
#     b = 0
# )
# class Slots:
#     __slots__ = ("a", "b")

#     a: int
#     b: int

# def test_slotted_dataclass():
#     obj = Slots(1)
#     assert obj.a == 1
#     assert obj.b == 0

#     obj = Slots(1, 1)
#     assert obj.a == 1
#     assert obj.b == 1

#     with pytest.raises(AttributeError):
#         obj.c = 1
