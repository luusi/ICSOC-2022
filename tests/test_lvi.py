"""Tests for the LVI implementation."""
from typing import Mapping, Tuple, Dict, Any

import numpy as np
from mdp_dp_rl.algorithms.dp.dp_analytic import DPAnalytic
from mdp_dp_rl.processes.mdp import MDP
from mdp_dp_rl.utils.generic_typevars import S, A

from icsoc_2022.lvi import value_iteration
from icsoc_2022.custom_types import State, Action


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
        out_transitions[RIGHT] = ({i + 1: 1.0}, 1.0 if i == n-2 else 0.0)

        info[i] = out_transitions

    info[n-1] = {NOP: ({n-1: 1.0}, 0.0)}
    return MDP(info, gamma)


def test_value_iteration() -> None:
    """Run the tests for LVI."""
    n = 10
    gamma = 0.99
    mdp = build_chain_mdp(n)

    value_function = value_iteration(mdp)
    for i in range(n - 1):
        assert np.allclose(value_function[i], gamma ** (n - i - 2))
