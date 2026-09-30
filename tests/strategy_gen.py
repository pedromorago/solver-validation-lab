"""Hypothesis generator of small, valid strategy files.

Valid means every rule in svlab.strategy.rules passes by construction:
distributions sum to 1, no combo is blocked or duplicated, unplayed
actions never have the best EV, overall EVs are the frequency-weighted
action EVs, ranges follow the tree, and every suit-isomorphic combo gets
the same strategy."""

from __future__ import annotations

import math

from hypothesis import strategies as st

from svlab.cards import card_str
from svlab.strategy import Action, ComboStrategy, Node, Spot, Strategy
from svlab.strategy.combos import all_combos, board_symmetries, combo_key, permute_cards


def _distribution(draw, actions: tuple[Action, ...]):
    counts = draw(st.lists(st.integers(0, 20), min_size=len(actions), max_size=len(actions)).filter(any))
    total = sum(counts)
    return {a.id: k / total for a, k in zip(actions, counts)}


def _evs(draw, freqs: dict[str, float]):
    evs = {a: draw(st.integers(-5000, 5000)) / 100 for a in freqs}
    best_played = max(evs[a] for a, f in freqs.items() if f > 0)
    return {a: (v if freqs[a] > 0 else min(v, best_played)) for a, v in evs.items()}


def _text(draw, key) -> str:
    cards = sorted(key, reverse=draw(st.booleans()))
    return "".join(card_str(c) for c in cards)


@st.composite
def valid_strategies(draw, max_classes: int = 8) -> Strategy:
    board = draw(st.lists(st.integers(0, 51), min_size=3, max_size=5, unique=True))
    stack = draw(st.integers(10, 200)) * 1.0
    pot = draw(st.integers(2, 40)) * 1.0
    small = round(min(pot / 3, stack), 2)
    big = round(min(pot, stack), 2)
    depth = draw(st.integers(1, 3))

    root_actions = (Action("check", "check"), Action("b1", "bet", small)) + (
        (Action("b2", "bet", big),) if draw(st.booleans()) and big > small else ()
    )
    tree = [((), "OOP", pot, 0.0, root_actions)]
    if depth >= 2:
        tree.append((("check",), "IP", pot, 0.0, (Action("check", "check"), Action("b1", "bet", small))))
    if depth >= 3:
        raise_to = round(min(3 * small, stack), 2)
        facing = [Action("fold", "fold"), Action("call", "call")]
        if raise_to > small:
            facing.append(Action("raise", "raise", raise_to))
        facing.append(Action("allin", "allin", stack))
        tree.append((("check", "b1"), "OOP", pot + small, small, tuple(facing)))

    symmetries = board_symmetries(board)
    playable = [combo_key(c) for c in all_combos() if not set(c) & set(board)]
    reps = draw(st.lists(st.sampled_from(playable), min_size=1, max_size=max_classes, unique=True))
    with_evs = draw(st.booleans())
    with_ev = draw(st.booleans())

    classes: list[list[frozenset[int]]] = []
    seen: set[frozenset[int]] = set()
    for key in reps:
        if key in seen:
            continue
        orbit = sorted({key} | {combo_key(permute_cards(sorted(key), p)) for p in symmetries}, key=sorted)
        seen.update(orbit)
        classes.append(orbit)

    nodes = []
    own_reach: dict[tuple[str, ...], dict[int, float]] = {}
    for path, player, node_pot, to_call, actions in tree:
        combos = []
        reach = {}
        for n, orbit in enumerate(classes):
            if path == ("check", "b1"):
                parent = nodes[0]
                weight = own_reach[()][n] * parent.combos[_first_index(classes, n)].freqs["check"]
            else:
                weight = draw(st.integers(100, 10000)) / 10000
            reach[n] = weight
            freqs = _distribution(draw, actions)
            evs = _evs(draw, freqs) if with_evs else None
            ev = math.fsum(f * evs[a] for a, f in freqs.items()) if evs is not None and with_ev else None
            for key in orbit:
                combos.append(ComboStrategy(_text(draw, key), weight, dict(freqs), dict(evs) if evs else None, ev))
        own_reach[path] = reach
        nodes.append(Node(path, player, node_pot, stack, to_call, actions, tuple(combos)))
    spot = Spot("NLHE", ("OOP", "IP"), pot, stack, "".join(card_str(c) for c in board))
    return Strategy(spot=spot, nodes=tuple(nodes), meta={"synthetic": True, "generator": "hypothesis"})


def _first_index(classes: list[list[frozenset[int]]], n: int) -> int:
    """Index in a node's combo tuple of the first combo of class n."""
    return sum(len(c) for c in classes[:n])
