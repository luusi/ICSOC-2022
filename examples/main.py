from typing import cast

from logaut import ltl2dfa
from pylogics.parsers import parse_ltl
from pythomata.impl.symbolic import SymbolicDFA

from icsoc_2022.constants import OTHER_ACTION_SYMBOL
from icsoc_2022.declare_utils import build_declare_assumption, alt_response, alt_succession
from icsoc_2022.dfa_target import MdpDfa, mdp_from_dfa, from_symbolic_automaton_to_declare_automaton
from icsoc_2022.lmdp import compute_composition_lmdp
from icsoc_2022.rendering import (
    mdp_to_graphviz,
    service_to_graphviz,
)
from icsoc_2022.services import Service

if __name__ == "__main__":
    all_symbols = {"a", "b", OTHER_ACTION_SYMBOL}
    formula_str = alt_succession("a", "b")
    formula = parse_ltl(formula_str + " & " + build_declare_assumption(all_symbols))
    automaton = cast(SymbolicDFA, ltl2dfa(formula, backend="lydia"))
    mdp: MdpDfa = mdp_from_dfa(from_symbolic_automaton_to_declare_automaton(automaton, all_symbols))

    cast(SymbolicDFA, automaton).to_graphviz().render("automaton")
    mdp_to_graphviz(mdp).render("mdp")

    a = Service({"a1", "a_broken"}, {"a", "fix_a"}, {"a1"}, "a1", {"a1": {"a": ({"a1": 0.95, "a_broken": 0.05}, (1.0, 1.0))}, "a_broken": {"fix_a": ({"a1": 1.0}, (0.0, 0.0))}})
    b = Service({"b1"}, {"b"}, {"b1"}, "b1", {"b1": {"b": ({"b1": 1.0}, (2.0, 2.0))}})
    service_to_graphviz(a).render("service_a")
    service_to_graphviz(b).render("service_b")

    weights = []
    lmdp = compute_composition_lmdp(mdp, [a, b])
    mdp_to_graphviz(lmdp).render("final_lmdp")
