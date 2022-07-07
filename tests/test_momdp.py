"""Test for the MOMDP module."""


import pytest

from icsoc_2022.lmdp import LMDP


def test_momdp_with_rewards_not_tuples() -> None:
    data = {
        1: {
            "a": ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            "b": ({2: 0.3, 3: 0.7}, 2.8),
            "c": ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2),
        },
        2: {
            "a": ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            "c": ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2),
        },
        3: {"a": ({3: 1.0}, 0.0), "b": ({3: 1.0}, 0.0)},
    }

    with pytest.raises(AssertionError, match="rewards.*is not a tuple!"):
        LMDP(data, 0.99)


def test_momdp_with_different_number_of_rewards() -> None:
    data = {
        1: {
            "a": ({1: 0.3, 2: 0.6, 3: 0.1}, (5.0,)),
            "b": ({2: 0.3, 3: 0.7}, (2.8,)),
            "c": ({1: 0.2, 2: 0.4, 3: 0.4}, (-7.2,)),
        },
        2: {
            "a": ({1: 0.3, 2: 0.6, 3: 0.1}, (5.0,)),
            "c": ({1: 0.2, 2: 0.4, 3: 0.4}, (-7.2,)),
        },
        3: {"a": ({3: 1.0}, (0.0,)), "b": ({3: 1.0}, (0.0, 1.0))},
    }

    with pytest.raises(AssertionError, match="expected number of rewards 1, got 2"):
        LMDP(data, 0.99)


def test_momdp_initialization() -> None:
    data = {
        1: {
            "a": ({1: 0.3, 2: 0.6, 3: 0.1}, (5.0, 1.0)),
            "b": ({2: 0.3, 3: 0.7}, (2.8, 1.0)),
            "c": ({1: 0.2, 2: 0.4, 3: 0.4}, (-7.2, 1.0)),
        },
        2: {
            "a": ({1: 0.3, 2: 0.6, 3: 0.1}, (5.0, 1.0)),
            "c": ({1: 0.2, 2: 0.4, 3: 0.4}, (-7.2, 1.0)),
        },
        3: {"a": ({3: 1.0}, (0.0, 1.0)), "b": ({3: 1.0}, (0.0, 1.0))},
    }

    momdp = LMDP(data, 0.99)

    assert momdp.rewards[3]["a"] == (0.0, 1.0)
