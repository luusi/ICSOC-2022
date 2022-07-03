"""
Code for Lexicographic Value Iteration (LVI).

Paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9471/9773
"""
from typing import AbstractSet, Dict, Mapping, Optional, Set, cast

import numpy as np
from mdp_dp_rl.processes.mdp import MDP
from numpy.typing import NDArray

from icsoc_2022.custom_types import Action, State
from icsoc_2022.momdp import MOMDP


def value_iteration(
    mdp: MDP,
    tol: float = 1e-20,
    allowed_actions: Optional[Mapping[State, AbstractSet[Action]]] = None,
) -> NDArray:
    nb_states = len(mdp.all_states)
    vf = np.zeros(nb_states)
    epsilon = tol * 1e4
    while epsilon >= tol:
        new_vf = np.zeros(nb_states)
        for s, v in mdp.rewards.items():
            new_vf[s] = max(
                r
                + mdp.gamma * sum(p * vf[s1] for s1, p in mdp.transitions[s][a].items())
                for a, r in v.items()
                if allowed_actions is None or a in allowed_actions[s]
            )
        epsilon = np.max(np.abs(new_vf - vf))
        vf = new_vf
    return vf


def lexicographic_value_iteration(momdp: MOMDP, tol: float = 1e-20):
    nb_states = len(momdp.all_states)
    nb_rewards = momdp.nb_rewards
    gamma = momdp.gamma
    tolerance_gamma_coefficient = gamma / (1.0 - gamma)
    vf_vec = np.zeros((nb_rewards, nb_states))
    prev_vf_vec = vf_vec
    started = False
    tolerance = tol * tolerance_gamma_coefficient
    actions = None
    while not started or np.max(np.abs(vf_vec - prev_vf_vec)) > tolerance:
        started = True
        vf_vec = prev_vf_vec
        for i in range(nb_rewards):
            mdp_i = momdp.get_mdp_i(i)
            vf_i = value_iteration(mdp_i, tolerance, allowed_actions=actions)
            vf_vec[i, :] = vf_i
            actions = get_optimal_actions(mdp_i, cast(Mapping[State, float], vf_i))
    return vf_vec


def get_optimal_actions(
    mdp: MDP, vf: Mapping[State, float]
) -> Mapping[State, AbstractSet[Action]]:
    q_function = get_act_value_func_dict_from_value_func(mdp, vf, mdp.gamma)
    optimal_actions_by_state: Dict[State, Set[Action]] = {}
    for s in range(len(vf)):
        maximum_value = max(q_function[s].items(), key=lambda pair: pair[1])[1]
        optimal_actions_by_state[s] = {
            action for action, value in q_function[s].items() if value == maximum_value
        }
    return optimal_actions_by_state


def get_q_function_from_v_function(mdp: MDP, vf: NDArray, nb_actions: int) -> NDArray:
    nb_states = len(vf)
    q_function = np.zeros((nb_states, nb_actions))
    for s, v in mdp.rewards.items():
        for a, r in v.items():
            q_function[s, a] = r + mdp.gamma * sum(
                p * vf[s1] for s1, p in mdp.transitions[s][a].items()
            )
    return q_function


def get_act_value_func_dict_from_value_func(
    mdp: MDP, value_function: Mapping[State, float], gamma: float
) -> Mapping[State, Mapping[Action, float]]:
    vf = value_function
    return {
        s: {
            a: r + gamma * sum(p * vf[s1] for s1, p in mdp.transitions[s][a].items())
            for a, r in v.items()
        }
        for s, v in mdp.rewards.items()
    }
