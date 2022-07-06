from typing import Set


def weak_until(a, b):
    return f"(G({a}) | ({a} U {b}))"


def absence_2(a):
    return f"G({a} -> XG(!{a}))"


def exactly_once(a):
    return f"(F({a}) & G({a} -> XG(!{a})))"


def alt_response(a, b):
    return f"(G({a} -> X[!](!{a} U {b})))"


def alt_precedence(a, b):
    return f"({weak_until(f'!{b}', a)} & G({b} -> X({weak_until(f'!{b}', a)})))"


def alt_succession(a, b):
    return f"({alt_response(a, b)} & {alt_precedence(a, b)})"


def not_coexistence(a, b):
    return f"G(G(!{a}) | G(!{b}))"


def build_declare_assumption(all_symbols: Set[str]) -> str:
    assert len(all_symbols) > 1
    at_least_one = f"G({' | '.join(all_symbols)})"
    at_most_one_subformulas = []
    for symbol in all_symbols:
        all_nots = " & ".join(map(lambda s: "!" + s, all_symbols.difference([symbol])))
        subformula = f"G({symbol} -> ({all_nots}))"
        at_most_one_subformulas.append(subformula)
    at_most_one = " & ".join(at_most_one_subformulas)
    return f"{at_least_one} & {at_most_one}"
