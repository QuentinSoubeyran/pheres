from string import printable
from typing import Dict, List, Literal, Tuple

from hypothesis import strategies as st

json_value: st.SearchStrategy = (
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(printable)
)


def jsons(max_size: int = 10, *, max_leaves: int = 15) -> st.SearchStrategy:
    """
    Return a recursive stategie for generating JSON inputs

    Arguments:
        max_size: maximum size of the lists and dicts created
        max_leaves: maximum number of call to the base json_value strategy

    Returns:
        a strategy for generating JSON intput
    """
    return st.recursive(
        json_value,
        lambda children: (
            st.lists(children, max_size=max_size)
            | st.dictionaries(st.text(printable), children, max_size=max_size)
        ),
        max_leaves=max_leaves,
    )


def json_arrays(
    max_length: int = 15, *, max_size: int = 10, max_leaves: int = 15
) -> st.SearchStrategy:
    return st.lists(
        jsons(max_size=max_size, max_leaves=max_leaves), max_size=max_length
    )


def json_objects(
    max_length: int = 15, *, max_size: int = 10, max_leaves: int = 15
) -> st.SearchStrategy:
    return st.dictionaries(
        st.text(printable), jsons(max_size=max_size, max_leaves=max_leaves)
    )


json_value_type: st.SearchStrategy = (
    st.just(type(None)) | st.just(bool) | st.just(int) | st.just(float) | st.just(str)
)


def _build_list(tp):
    return List[tp]


def _build_tuple(tp_tuple):
    return Tuple[tuple(tp_tuple)]


def _build_var_tuple(tp):
    return Tuple[tp, ...]


def _build_dict(tp):
    return Dict[str, tp]


def _build_literal(values):
    return Literal[tuple(values)]

def _json_literals(max_size: int = 15):
    return st.builds(
        _build_literal,
        st.lists(st.booleans() | st.integers() | st.text(printable), max_size=max_size),
    )


def json_types(max_size: int = 10, *, max_leaves: int = 15) -> st.SearchStrategy:
    """
    Return a recursive stategie for generating JSON type-hints inputs

    Arguments:
        max_size: maximum size of the lists and dicts created
        max_leaves: maximum number of call to the base json_value strategy

    Returns:
        a strategy for generating JSON type-hints intput
    """
    return st.recursive(
        json_value_type | _json_literals(max_size=max_size),
        lambda children: (
            st.builds(_build_list, children)
            | st.builds(_build_var_tuple, children)
            | st.builds(_build_tuple, st.lists(children, max_size=max_size))
            | st.builds(_build_dict, children)
        ),
        max_leaves=max_leaves,
    )

typed_json_value = (
    st.tuples(st.none(), st.just(type(None))),
    st.tuples(st.booleans(), st.just(bool)),
    st.tuples(st.integers(), st.just(int)),
    st.tuples(st.floats(), st.just(float)),
    st.tuples(st.text(printable), st.just(str))
)

def typed_json_literals(max_size=5):
    return st.builds(
    lambda l: (l[0][0], Literal[tuple(val_tp[1] for val_tp in l)]),
    st.lists(typed_json_value, max_size=max_size)
)


def typed_jsons(max_size: int = 10, *, max_leaves: int = 15) -> st.SearchStrategy:
    return st.recursive(
        typed_json_value | typed_json_literals,
        lambda children: (
            # list
            # fixed-len tuple
            # variadic tuple
            # dict
        ),
        max_leaves=max_leaves
    )