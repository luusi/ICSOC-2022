"""Represent a target service."""
from typing import Any, Mapping, Set, Tuple

from mdp_dp_rl.processes.mdp import MDP
from mdp_dp_rl.utils.generic_typevars import A, S
from pythomata.core import DFA
from sympy import Symbol
from sympy.logic.boolalg import And, BooleanFunction, BooleanTrue, Or

from icsoc_2022.constants import DEFAULT_GAMMA, COMPOSITION_MDP_UNDEFINED_ACTION
from icsoc_2022.custom_types import MDPDynamics


def guard_to_symbol(prop_formula: BooleanFunction) -> Set[str]:
    """From guard to symbol."""
    if isinstance(prop_formula, Symbol):
        return {str(prop_formula)}
    elif isinstance(prop_formula, And):
        symbol_args = [arg for arg in prop_formula.args if isinstance(arg, Symbol)]
        assert len(symbol_args) == 1
        return {str(symbol_args[0])}
    elif isinstance(prop_formula, Or):
        operands_as_symbols = [
            symb for arg in prop_formula.args for symb in guard_to_symbol(arg)
        ]
        operands_as_symbols = list(filter(lambda x: x is not None, operands_as_symbols))
        assert len(operands_as_symbols) > 0
        return set(operands_as_symbols)
    # None case
    return None


class MdpDfa(MDP):

    initial_state: Any
    failure_state: Any
    all_actions: Set[str]

    def __init__(
        self,
        info: Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]],
        gamma: float,
    ) -> None:
        super().__init__(info, gamma)

        self.all_actions = set(a for s, trans in info.items() for a, _ in trans.items())


def mdp_from_dfa(dfa: DFA, reward: float = 2.0, gamma: float = DEFAULT_GAMMA) -> MdpDfa:
    transition_function: MDPDynamics = {}
    failure_state = _find_failure_state(dfa)
    for _start in dfa.states:
        for start, action, end in dfa.get_transitions_from(_start):
            if end == failure_state:
                symbol = COMPOSITION_MDP_UNDEFINED_ACTION
                transition_function.setdefault(start, {}).setdefault(symbol, ({end: 1.0}, 0.0))
            else:
                symbols = guard_to_symbol(action)
                for symbol in symbols:
                    dest = ({end: 1.0}, reward if end in dfa.accepting_states else 0.0)
                    transition_function.setdefault(start, {}).setdefault(symbol, dest)

    result = MdpDfa(transition_function, gamma)
    result.initial_state = dfa.initial_state
    result.failure_state = _find_failure_state(dfa)
    return result


def _find_failure_state(dfa: DFA):
    """Find failure state, if any."""
    for state in dfa.states:
        if state in dfa.accepting_states:
            continue
        transitions = dfa.get_transitions_from(state)
        if len(transitions) == 1:
            t = list(transitions)[0]
            start, guard, end = t
            if start == end and isinstance(guard, BooleanTrue):
                # non-accepting, self-loop with true
                return start
    return None