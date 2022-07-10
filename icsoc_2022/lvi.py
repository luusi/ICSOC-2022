"""
Code for Lexicographic Value Iteration (LVI).

Paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9471/9773
"""
import math
from typing import AbstractSet, Dict, Mapping, Optional, Set, cast, Sequence, Tuple

import numpy as np
from mdp_dp_rl.processes.mdp import MDP
from numpy.typing import NDArray

from icsoc_2022.custom_types import Action, State
from icsoc_2022.lmdp import LMDP


def value_iteration(
    mdp: MDP,
    tol: float = np.finfo(float).eps,
    id2state: Optional[Sequence[int]] = None,
    allowed_actions: Optional[Mapping[State, AbstractSet[Action]]] = None,
) -> Dict:
    nb_states = len(mdp.all_states)
    id2state = sorted(mdp.all_states) if id2state is not None else id2state
    state2id = dict(map(reversed, enumerate(id2state)))
    vf = np.zeros(nb_states)
    epsilon = tol * 1e4
    while epsilon >= tol:
        new_vf = np.zeros(nb_states)
        for s, v in mdp.rewards.items():
            new_vf[state2id[s]] = max(
                r
                + mdp.gamma
                * sum(p * vf[state2id[s1]] for s1, p in mdp.transitions[s][a].items())
                for a, r in v.items()
                if allowed_actions is None or a in allowed_actions[s]
            )
        epsilon = np.max(np.abs(new_vf - vf))
        vf = new_vf
    # inefficient, but practical
    result_vf = {id2state[i]: vf[i] for i in range(len(vf))}
    return result_vf


def _combine_optimal_actions(
    old_actions: Mapping[State, AbstractSet[Action]],
    new_actions: Mapping[State, AbstractSet[Action]],
):
    result = {}
    assert old_actions.keys() == new_actions.keys()
    for state in old_actions.keys():
        old_optimal_actions = old_actions[state]
        new_optimal_actions = new_actions[state]
        optimal_actions = set.intersection(old_optimal_actions, new_optimal_actions)
        if len(optimal_actions) == 0:
            optimal_actions = old_optimal_actions
        result[state] = optimal_actions
    return result


def lexicographic_value_iteration(
    momdp: LMDP, tol: float = np.finfo(float).eps
) -> Tuple[Mapping[State, float], Mapping[State, AbstractSet[Action]]]:
    nb_states = len(momdp.all_states)
    id2state = sorted(momdp.all_states)
    nb_rewards = momdp.nb_rewards
    vf_vec = np.zeros((nb_rewards, nb_states))
    prev_vf_vec = vf_vec
    started = False
    tolerance = tol
    actions = None
    while not started or np.max(np.abs(vf_vec - prev_vf_vec)) > tolerance:
        started = True
        vf_vec = prev_vf_vec
        for i in range(nb_rewards):
            mdp_i = momdp.get_mdp_i(i)
            print(f"Computing optimal value function for objective {i}...")
            vf_i = value_iteration(
                mdp_i, tolerance, id2state=id2state, allowed_actions=actions
            )
            vf_vec[i, :] = np.array([vf_i[s] for s in id2state])
            actions = get_optimal_actions(
                mdp_i, cast(Mapping[State, float], vf_i), allowed_actions=actions
            )
    # inefficient, but practical
    result_vf = {
        id2state[state_id]: vf_vec[:, state_id] for state_id in range(vf_vec.shape[1])
    }
    return result_vf, actions


def get_optimal_actions(
    mdp: MDP,
    vf: Mapping[State, float],
    allowed_actions: Optional[Mapping[State, AbstractSet[Action]]] = None,
) -> Mapping[State, AbstractSet[Action]]:
    q_function = get_act_value_func_dict_from_value_func(mdp, vf, mdp.gamma)
    optimal_actions_by_state: Dict[State, Set[Action]] = {}
    for s in vf.keys():
        filtered_q_values = [
            (action, value)
            for action, value in q_function[s].items()
            if (allowed_actions is None or action in allowed_actions[s])
        ]
        maximum_value = max(filtered_q_values, key=lambda pair: pair[1])[1]
        # we use math.isclose due to numerical instability
        optimal_actions_by_state[s] = {
            action for action, value in filtered_q_values if math.isclose(value, maximum_value, abs_tol=np.finfo(float).eps * 10.0)
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
