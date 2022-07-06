from typing import cast

from logaut import ltl2dfa
from pylogics.parsers import parse_ltl
from pythomata.impl.symbolic import SymbolicDFA

from icsoc_2022.dfa_target import MdpDfa, mdp_from_dfa
from icsoc_2022.momdp import compute_final_mdp
from icsoc_2022.rendering import (
    mdp_to_graphviz,
    service_to_graphviz,
)
from icsoc_2022.services import Service

if __name__ == "__main__":
    # without declare
    # formula = parse_ltl("a U b")
    # with declare:
    formula = parse_ltl("(a U b) & (G(a -> !b) & G(b -> !a) & G(a | b))")
    automaton = ltl2dfa(formula, backend="lydia")
    mdp: MdpDfa = mdp_from_dfa(automaton)

    cast(SymbolicDFA, automaton).to_graphviz().render("automaton")
    mdp_to_graphviz(mdp).render("mdp")

    a = Service({"a1", "a_broken"}, {"a", "act_a"}, {"a1"}, "a1", {"a1": {"a": ({"a1": 0.95, "a_broken": 0.05}, (1.0, 1.0))}, "a_broken": {"act_a": ({"a1": 1.0}, (0.0, 0.0))}})
    b = Service({"b1"}, {"b", "act_b"}, {"b1"}, "b1", {"b1": {"b": ({"b1": 1.0}, (2.0, 2.0)), "act_b": ({"b1": 1.0}, (0.0, 0.0))}})
    service_to_graphviz(a).render("service_a")
    service_to_graphviz(b).render("service_b")

    weights = []
    final_mdp = compute_final_mdp(mdp, [a, b], [10.0, 2.0, 1.0])
    mdp_to_graphviz(final_mdp).render("final_mdp")
