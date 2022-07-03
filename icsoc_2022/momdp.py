from typing import Generic, Mapping, Sequence, Set, Tuple

from mdp_dp_rl.processes.mp_funcs import (
    get_actions_for_states,
    get_all_states,
    get_lean_transitions,
)
from mdp_dp_rl.utils.gen_utils import is_approx_eq, memoize, zip_dict_of_tuple
from mdp_dp_rl.utils.generic_typevars import A, S

from icsoc_2022.custom_types import MOMDPDynamics


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


class MOMDP(Generic[S, A]):
    def __init__(self, info: MOMDPDynamics, gamma: float) -> None:
        verify_momdp(info)
        d = {k: zip_dict_of_tuple(v) for k, v in info.items()}
        d1, d2 = zip_dict_of_tuple(d)
        self.all_states: Set[S] = get_all_states(info)
        self.state_action_dict: Mapping[S, Set[A]] = get_actions_for_states(info)
        self.transitions: Mapping[S, Mapping[A, Mapping[S, float]]] = {
            s: {a: get_lean_transitions(v1) for a, v1 in v.items()}
            for s, v in d1.items()
        }
        self.rewards: Mapping[S, Mapping[A, Tuple[float, ...]]] = d2
        self.gamma: float = gamma
        self.terminal_states: Set[S] = self.get_terminal_states()

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
