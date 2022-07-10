from collections import deque
from typing import Generic, Mapping, Sequence, Set, Tuple, List, Deque, Optional

from mdp_dp_rl.processes.mdp import MDP
from mdp_dp_rl.processes.mp_funcs import (
    get_actions_for_states,
    get_all_states,
    get_lean_transitions,
)
from mdp_dp_rl.utils.gen_utils import is_approx_eq, memoize, zip_dict_of_tuple
from mdp_dp_rl.utils.generic_typevars import A, S
from pythomata import SimpleDFA

from icsoc_2022.constants import OTHER_ACTION_SYMBOL
from icsoc_2022.custom_types import MDPDynamics, MOMDPDynamics
from icsoc_2022.dfa_target import MdpDfa
from icsoc_2022.services import Service, build_system_service


@memoize
def verify_momdp(mdp_data: MOMDPDynamics) -> None:
    _is_momdp_dynamics_consistent(mdp_data)
    all_st = get_all_states(mdp_data)
    assert all(len(v) > 0 for _, v in mdp_data.items()), "actions are not correct"
    val_seq = [v2 for _, v1 in mdp_data.items() for _, (v2, _) in v1.items()]
    verify_transitions(all_st, val_seq)


def verify_transitions(states: Set[S], tr_seq: Sequence[Mapping[S, float]]) -> None:
    b1 = set().union(*tr_seq).issubset(states)
    b2 = all(all(x >= 0 for x in d.values()) for d in tr_seq)
    b3 = all(is_approx_eq(sum(d.values()), 1.0) for d in tr_seq)
    assert b1 and b2 and b3, "transitions are not correct"


def _is_momdp_dynamics_consistent(momdp_dynamics: MOMDPDynamics) -> None:
    nb_rewards = None
    for _, out_transitions_by_action in momdp_dynamics.items():
        for _, (_, rewards) in out_transitions_by_action.items():
            assert isinstance(rewards, tuple), f"rewards {rewards} is not a tuple!"
            if nb_rewards is None:
                nb_rewards = len(rewards)
            assert nb_rewards == len(
                rewards
            ), f"expected number of rewards {nb_rewards}, got {len(rewards)}"


class LMDP(Generic[S, A]):
    def __init__(self, info: MOMDPDynamics, gamma: Optional[float] = None,
                 gammas: Optional[Sequence[float]] = None) -> None:
        verify_momdp(info)
        self._nb_objectives = self._get_nb_objectives(info)
        assert (gamma is not None) != (gammas is not None), "either one discount factor or a list of discount factors is required"
        assert not gammas or len(gammas) == self._nb_objectives, f"gammas has to be of length equal of the number of objectives, {self._nb_objectives}; got {len(gammas)}"
        self.gamma = gamma
        self.gammas = gammas

        d = {k: zip_dict_of_tuple(v) for k, v in info.items()}
        d1, d2 = zip_dict_of_tuple(d)
        self.info = info
        self.all_states: Set[S] = get_all_states(info)
        self.state_action_dict: Mapping[S, Set[A]] = get_actions_for_states(info)
        self.transitions: Mapping[S, Mapping[A, Mapping[S, float]]] = {
            s: {a: get_lean_transitions(v1) for a, v1 in v.items()}
            for s, v in d1.items()
        }
        self.rewards: Mapping[S, Mapping[A, Tuple[float, ...]]] = d2
        self.nb_rewards: int = self._nb_objectives
        self.gamma: float = gamma
        self.terminal_states: Set[S] = self.get_terminal_states()

    def get_discount_factor_i(self, i: int):
        assert 0 <= i < self._nb_objectives
        return self.gammas[i] if self.gammas is not None else self.gamma

    def _get_nb_objectives(
        self, info: MOMDPDynamics
    ) -> int:
        return len(next(iter(list(info.values())[0].values()))[1])

    def get_sink_states(self) -> Set[S]:
        return {
            k
            for k, v in self.transitions.items()
            if all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
        }

    def get_terminal_states(self) -> Set[S]:
        """
        A terminal state is a sink state (100% probability to going back
        to itself, FOR EACH ACTION) and the rewards on those transitions back
        to itself are zero.
        """
        sink = self.get_sink_states()
        return {
            s
            for s in sink
            if all(
                is_approx_eq(r, 0.0)
                for _, rewards_vec in self.rewards[s].items()
                for r in rewards_vec
            )
        }

    def get_mdp_i(self, reward_index: int) -> MDP:
        """Get the MDP with only one source of reward whose index is specified by the argument."""
        assert 0 <= reward_index < self.nb_rewards
        mdp_dynamics: MDPDynamics = {}
        for state, out_trans_by_action in self.info.items():
            mdp_dynamics.setdefault(state, {})
            for action, (next_state_dist, rewards) in out_trans_by_action.items():
                reward = rewards[reward_index]
                mdp_dynamics[state][action] = (next_state_dist, reward)

        return MDP(mdp_dynamics, self.get_discount_factor_i(reward_index))


def compute_composition_lmdp(
    dfa: SimpleDFA, services: List[Service], gammas: Sequence[float], with_all_initial_states: bool = False
) -> LMDP:
    nb_rewards = services[0].nb_rewards
    assert all(service.nb_rewards == nb_rewards for service in services)

    system_service = build_system_service(*services)

    transition_function: MOMDPDynamics = {}

    visited = set()
    to_be_visited = set()
    queue: Deque = deque()

    # add initial transitions
    initial_state = (system_service.initial_state, dfa.initial_state)
    queue.append(initial_state)
    to_be_visited.add(initial_state)
    if with_all_initial_states:
        for system_service_state in system_service.states:
            if system_service_state == system_service.initial_state:
                continue
            new_initial_state = (system_service_state, dfa.initial_state)
            queue.append(new_initial_state)
            to_be_visited.add(new_initial_state)

    while len(queue) > 0:
        cur_state = queue.popleft()
        to_be_visited.remove(cur_state)
        visited.add(cur_state)
        cur_system_state, cur_dfa_state = cur_state
        trans_dist = {}

        next_system_state_trans = system_service.transition_function[
            cur_system_state
        ].items()

        # iterate over all available actions of system service
        # in case symbol is in DFA available actions, progress DFA state component
        for (symbol, service_id), next_state_info in next_system_state_trans:
            next_system_state_distr, reward_vector = next_state_info
            system_reward = reward_vector

            # symbols not in the transition function of the target
            # are considered as "other"; however, when we add the
            # LMDP transition, we will label it with the original
            # symbol.
            if symbol in dfa.transition_function[cur_dfa_state]:
                symbol_to_next_dfa_states = dfa.transition_function[cur_dfa_state]
                next_dfa_state = symbol_to_next_dfa_states[symbol]
                goal_reward = 1.0 if dfa.is_accepting(next_dfa_state) else 0.0
            else:
                next_dfa_state = cur_dfa_state
                goal_reward = 0.0
            final_rewards = (goal_reward, *system_reward)

            for next_system_state, prob in next_system_state_distr.items():
                assert prob > 0.0
                next_state = (next_system_state, next_dfa_state)
                trans_dist.setdefault((symbol, service_id), ({}, final_rewards))[0][
                    next_state
                ] = prob
                if next_state not in visited and next_state not in to_be_visited:
                    queue.append(next_state)
                    to_be_visited.add(next_state)

        transition_function[cur_state] = trans_dist

    result = LMDP(transition_function, gammas=gammas)
    result.initial_state = initial_state
    return result
