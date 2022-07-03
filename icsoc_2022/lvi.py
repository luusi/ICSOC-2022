"""
Code for Lexicographic Value Iteration (LVI).

Paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9471/9773
"""
from typing import Dict, Mapping

import numpy as np
from mdp_dp_rl.processes.mdp import MDP

from icsoc_2022.custom_types import State, Action


def value_iteration(mdp: MDP, tol: float = 1e-20) -> Dict:
    vf = {s: 0. for s in mdp.all_states}
    epsilon = tol * 1e4
    while epsilon >= tol:
        new_vf = {s: max(r + mdp.gamma * sum(p * vf[s1] for s1, p in
                                             mdp.transitions[s][a].items())
                         for a, r in v.items())
                  for s, v in mdp.rewards.items()}
        epsilon = max(abs(new_vf[s] - v) for s, v in vf.items())
        vf = new_vf
    return vf


#def value_iteration_lvi(mdp: MDP, tol: float = 1e-20) -> Dict:
    #vf = {s: 0. for s in mdp.all_states}
    #epsilon = tol * 1e4
    #while epsilon >= tol:
        #new_vf = {s: max(mdp.gamma * sum((p * vf[s1] for s1, p in
                                             #mdp.transitions[s][a].items()) * (r * vf[s1] for s1, r in mdp.transitions[s][a].items()))
                         #for a, r in v.items())
                  #for s, v in mdp.rewards.items()}
        #epsilon = max(abs(new_vf[s] - v) for s, v in vf.items())
        #vf = new_vf
    #return vf


def get_act_value_func_dict_from_value_func(mdp: MDP, value_function: Mapping[State, float], gamma: float) -> Mapping[State, Mapping[Action, float]]:
    vf = value_function
    return {s: {a: r + gamma * sum(p * vf[s1] for s1, p in mdp.transitions[s][a].items())
                for a, r in v.items()}
            for s, v in mdp.rewards.items()}