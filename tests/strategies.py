from functools import partial
from string import printable
from typing import Any, Dict, List, Literal, Tuple, Union

from hypothesis import strategies as st

json_value: st.SearchStrategy = (
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(printable)
)


def jsons(max_size: int = 5, *, max_leaves: int = 30) -> st.SearchStrategy:
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
    max_length: int = 5, *, max_size: int = 5, max_leaves: int = 20
) -> st.SearchStrategy:
    return st.lists(
        jsons(max_size=max_size, max_leaves=max_leaves), max_size=max_length
    )


def json_objects(
    max_length: int = 5, *, max_size: int = 5, max_leaves: int = 20
) -> st.SearchStrategy:
    return st.dictionaries(
        st.text(printable),
        jsons(max_size=max_size, max_leaves=max_leaves),
        max_size=max_length,
    )


json_value_type: st.SearchStrategy = (
    st.just(type(None)) | st.just(bool) | st.just(int) | st.just(float) | st.just(str)
)


def _build_list(tp):
    return List[tp]


def _subscribe_from_list(obj, l):
    return obj[tuple(l)]


def _build_var_tuple(tp):
    return Tuple[tp, ...]


def _build_dict(tp):
    return Dict[str, tp]


def _json_literals(max_size: int = 15):
    return st.builds(
        partial(_subscribe_from_list, Literal),
        st.lists(st.booleans() | st.integers() | st.text(printable), max_size=max_size),
    )


def json_types(max_size: int = 5, *, max_leaves: int = 25) -> st.SearchStrategy:
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
            | st.builds(
                partial(_subscribe_from_list, Tuple),
                st.lists(children, max_size=max_size, min_size=1),
            )
            | st.builds(
                partial(_subscribe_from_list, Union),
                st.lists(children, max_size=max_size, min_size=1),
            )
            | st.builds(_build_dict, children)
        ),
        max_leaves=max_leaves,
    )


# To generate a JSON and its type, we use a two-stage approach:
# 1) Recursively builds meta strategies that, when drawn from, returns
#     - a type hint
#     - a strategy to draw from the type hint
# 2) We draw from the above recursive strategy, then draw again from the nested one

_meta_value = (
    st.tuples(st.just(type(None)), st.just(st.none()))
    | st.tuples(st.just(bool), st.just(st.booleans()))
    | st.tuples(st.just(int), st.just(st.integers()))
    | st.tuples(st.just(float), st.just(st.floats()))
    | st.tuples(st.just(str), st.just(st.text(printable)))
)


def _meta_literal(max_size: int = 5):
    return st.builds(
        lambda values: (
            Literal[tuple(values)],  # pylint: disable=unsubscriptable-object
            st.one_of(st.just(v) for v in values),
        ),
        st.lists(
            st.booleans() | st.integers() | st.text(printable),
            min_size=1,
            max_size=max_size,
        ),
    )


@st.composite
def _meta_list(draw, meta, max_size: int = 5):
    tp, strategy = draw(meta)
    return (List[tp], st.lists(strategy, max_size=max_size))


@st.composite
def _meta_vtuple(draw, meta, max_size: int = 5):
    tp, strategy = draw(meta)
    return (Tuple[tp, ...], st.lists(strategy, max_size=max_size))


@st.composite
def _meta_union(draw, meta, max_size: int = 5):
    tps, strategies = zip(*draw(st.lists(meta, min_size=1, max_size=max_size)))
    index = draw(st.integers(min_value=0, max_value=len(strategies) - 1))
    return (Union[tps], strategies[index])  # pylint: disable=unsubscriptable-object


@st.composite
def _meta_ftuple(draw, meta, max_size: int = 5):
    tps, strategies = zip(*draw(st.lists(meta, min_size=1, max_size=max_size)))
    return (Tuple[tps], st.builds(list, st.tuples(*strategies)))


@st.composite
def _meta_object(draw, meta, max_size: int = 5):
    tp, strategy = draw(meta)
    return (
        Dict[str, tp],
        st.dictionaries(st.text(printable), strategy, max_size=max_size),
    )


def _meta_jsons(max_size: int = 5, *, max_leaves: int = 30):
    return st.recursive(
        _meta_value | _meta_literal(max_size),
        lambda children: (
            _meta_list(children, max_size=max_size)
            | _meta_vtuple(children, max_size=max_size)
            | _meta_union(children, max_size=max_size)
            | _meta_ftuple(children, max_size=max_size)
            | _meta_object(children, max_size=max_size)
        ),
        max_leaves=max_leaves,
    )


@st.composite
def _typed_jsons(draw, max_size: int = 5, max_leaves: int = 30):
    """
    Returns a strategy that produces a JSON type hint and an example JSON
    """
    tp, strategy = draw(_meta_jsons(max_size=max_size, max_leaves=max_leaves))
    example = draw(strategy)
    return tp, example


def typed_jsons(max_size: int = 5, max_leaves: int = 30):
    """
    Returns a strategy that produces a JSON type hint and an example JSON
    """
    return _typed_jsons(max_size=max_size, max_leaves=max_leaves)
