"""Tests for the LVI implementation."""
from typing import Any, Dict, Tuple

import numpy as np
from mdp_dp_rl.processes.mdp import MDP

from icsoc_2022.lvi import lexicographic_value_iteration, value_iteration
from icsoc_2022.lmdp import LMDP


def build_chain_mdp(n: int, gamma: float = 0.99) -> MDP:
    """Build a chain-like MDP: # 0 <-> 1 <-> 2 <-> ... <-> n-2 <-> n-1"""
    assert n >= 2
    LEFT = "left"
    RIGHT = "right"
    NOP = "nop"
    info: Dict[Any, Dict[Any, Tuple[Dict[Any, float], float]]] = {}

    for i in range(n - 1):
        out_transitions = {}

        # add transitions with action "left"
        out_transitions[LEFT] = ({max(0, i - 1): 1.0}, 0.0)

        # add transitions with action "right"
        out_transitions[RIGHT] = ({i + 1: 1.0}, 1.0 if i == n - 2 else 0.0)

        info[i] = out_transitions

    info[n - 1] = {NOP: ({n - 1: 1.0}, 0.0)}
    return MDP(info, gamma)


def test_value_iteration() -> None:
    """Run the tests for value iteration."""
    n = 10
    gamma = 0.99
    mdp = build_chain_mdp(n)

    value_function = value_iteration(mdp)
    for i in range(n - 1):
        assert np.allclose(value_function[i], gamma ** (n - i - 2))


def test_lvi() -> None:
    """Run the tests for LVI."""
    data = {
        0: {"a": ({1: 1.0}, (1.0, 0.0)), "b": ({2: 1.0}, (0.0, 1.0))},
        1: {"a": ({1: 1.0}, (0.0, 0.0)), "b": ({1: 1.0}, (0.0, 1.0))},
        2: {"a": ({2: 1.0}, (0.0, 0.0)), "b": ({2: 1.0}, (0.0, 1.0))},
    }

    momdp = LMDP(data, 0.9)
    vf_result, _ = lexicographic_value_iteration(momdp)

    assert np.allclose(vf_result[0], [1.0, 9.0])
    assert np.allclose(vf_result[1], [0.0, 10.0])
    assert np.allclose(vf_result[2], [0.0, 10.0])
