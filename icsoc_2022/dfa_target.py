"""Represent a target service."""
from collections import deque
from typing import Any, Mapping, Set, Tuple, Deque

from mdp_dp_rl.processes.mdp import MDP
from mdp_dp_rl.utils.generic_typevars import A, S
from pythomata import SimpleDFA
from pythomata.core import DFA
from pythomata.impl.symbolic import SymbolicDFA
from sympy.logic.boolalg import BooleanTrue

from icsoc_2022.constants import DEFAULT_GAMMA
from icsoc_2022.custom_types import MDPDynamics


def from_symbolic_automaton_to_declare_automaton(
    sym_automaton: SymbolicDFA, all_symbols: Set[str]
) -> SimpleDFA:
    states = sym_automaton.states
    initial_state = sym_automaton.initial_state
    accepting_states = sym_automaton.accepting_states
    transition_function = {}

    queue: Deque = deque()
    discovered = set()
    queue.append(initial_state)
    while len(queue) != 0:
        current_state = queue.popleft()
        discovered.add(current_state)
        for symbol in all_symbols:
            next_state = sym_automaton.get_successor(current_state, {symbol: True})
            transition_function.setdefault(current_state, {})[symbol] = next_state
            if next_state not in discovered:
                queue.append(next_state)
                discovered.add(next_state)
    return SimpleDFA(
        states, all_symbols, initial_state, accepting_states, transition_function
    )


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


def mdp_from_dfa(
    dfa: SimpleDFA, reward: float = 1.0, gamma: float = DEFAULT_GAMMA
) -> MdpDfa:
    assert isinstance(dfa, SimpleDFA)
    transition_function: MDPDynamics = {}
    for _start in dfa.states:
        for start, action, end in dfa.get_transitions_from(_start):
            dest = ({end: 1.0}, reward if end in dfa.accepting_states else 0.0)
            transition_function.setdefault(start, {}).setdefault(action, dest)

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
