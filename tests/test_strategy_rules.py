"""Each rule, on small hand-made edits of the minimal example."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from svlab.strategy import RULES, Action, Tolerances, load, validate
from svlab.strategy.combos import board_symmetries, parse_board, perm_text

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "strategies"
BASE = load(FIXTURES / "minimal_Ks7s2d.json")


def edit_combo(strategy, node, name, **changes):
    n = strategy.nodes[node]
    combos = tuple(replace(c, **changes) if c.combo == name else c for c in n.combos)
    nodes = list(strategy.nodes)
    nodes[node] = replace(n, combos=combos)
    return replace(strategy, nodes=tuple(nodes))


def edit_node(strategy, node, **changes):
    nodes = list(strategy.nodes)
    nodes[node] = replace(nodes[node], **changes)
    return replace(strategy, nodes=tuple(nodes))


def rules_hit(strategy, tol=Tolerances()):
    return {f.rule for f in validate(strategy, tol)}


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_fixtures_pass_every_rule(path):
    assert validate(load(path)) == []


def test_rule_ids_are_unique():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids)) == 13


def test_freq_sum_message_is_located():
    s = edit_combo(BASE, 0, "7h7c", freqs={"check": 0.32, "bet33": 0.75})
    findings = validate(s, only=["freq-sum"])
    assert [str(f) for f in findings] == ["[freq-sum] nodes[0].combos['7h7c'].freqs: sums to 1.07, expected 1 ± 0.001"]


def test_freq_sum_respects_the_tolerance():
    s = edit_combo(BASE, 0, "8s6s", freqs={"check": 0.6005, "bet33": 0.4})
    assert "freq-sum" not in rules_hit(s)
    assert "freq-sum" in rules_hit(s, Tolerances(freq=1e-4))


def test_freq_range():
    s = edit_combo(BASE, 0, "8s6s", freqs={"check": 1.1, "bet33": -0.1})
    assert {f.where for f in validate(s, only=["freq-range"])} == {
        "nodes[0].combos['8s6s'].freqs.check",
        "nodes[0].combos['8s6s'].freqs.bet33",
    }


def test_weight_range():
    assert "weight-range" in rules_hit(edit_combo(BASE, 1, "AsJs", weight=-0.2))


def test_ev_finite():
    s = edit_combo(BASE, 1, "AsJs", evs={"check": math.inf, "bet33": 3.3, "bet75": 3.2})
    assert "ev-finite" in rules_hit(s)


def test_combo_syntax_and_blockers():
    s = edit_combo(BASE, 1, "AsJs", combo="AsAs")
    assert "combo-syntax" in rules_hit(s)
    s = edit_combo(BASE, 1, "AsJs", combo="AsKs")
    assert [f.message for f in validate(s, only=["board-blocker"])] == ["uses a board card (Ks7s2d)"]


def test_duplicate_in_other_card_order():
    s = edit_combo(BASE, 1, "AsJs", combo="JsAs")
    assert validate(s, only=["duplicate-combo"]) == []
    extra = replace(BASE.nodes[1].combos[2], combo="JsAs")
    s = edit_node(BASE, 1, combos=BASE.nodes[1].combos + (extra,))
    assert [f.message for f in validate(s, only=["duplicate-combo"])] == ["duplicate combo (same cards as 'AsJs')"]


@pytest.mark.parametrize(
    "actions, to_call, fragment",
    [
        ((Action("check", "check"), Action("bet33", "bet", 120.0)), 0.0, "above the effective stack"),
        ((Action("check", "check"), Action("bet33", "bet", 0.0)), 0.0, "needs a positive size"),
        ((Action("check", "check"), Action("bet33", "bet")), 0.0, "needs a positive size"),
        ((Action("check", "check", 1.0), Action("bet33", "bet", 1.8)), 0.0, "takes no size"),
        ((Action("check", "check"), Action("bet33", "shove", 1.8)), 0.0, "unknown action type"),
        ((Action("check", "check"), Action("check", "bet", 1.8)), 0.0, "used more than once"),
        ((Action("check", "check"), Action("bet33", "bet", 1.8)), 2.0, "not possible facing a bet"),
        ((Action("check", "fold"), Action("bet33", "bet", 1.8)), 0.0, "nothing to call"),
        ((Action("check", "call"), Action("bet33", "raise", 1.5)), 2.0, "not above the 2 to call"),
        ((Action("check", "call"), Action("bet33", "allin", 50.0)), 2.0, "differs from the effective stack"),
    ],
)
def test_action_legal(actions, to_call, fragment):
    s = edit_node(BASE, 0, actions=actions, to_call=to_call)
    messages = [f.message for f in validate(s, only=["action-legal"])]
    assert any(fragment in m for m in messages), messages


def test_unknown_action_in_freqs_and_in_path():
    s = edit_combo(BASE, 0, "8s6s", freqs={"check": 0.6, "bet50": 0.4})
    assert [f.where for f in validate(s, only=["unknown-action"])] == ["nodes[0].combos['8s6s'].freqs.bet50"]
    s = edit_node(BASE, 2, path=("check", "bet50"))
    assert [f.where for f in validate(s, only=["unknown-action"])] == ["nodes[2].path[1]"]


def test_ev_consistency():
    s = edit_combo(BASE, 1, "AsJs", ev=3.5)
    [f] = validate(s, only=["ev-consistency"])
    assert f.where == "nodes[1].combos['AsJs'].ev" and "give 3.2100" in f.message
    assert validate(edit_combo(BASE, 1, "AsJs", ev=3.215), only=["ev-consistency"]) == []


def test_zero_freq_best_response():
    # bet33 is never played by 6h5h; an EV of 1.2 ties the best played action and is fine.
    tie = {"check": 1.2, "bet33": 1.2, "bet75": 1.2}
    assert validate(edit_combo(BASE, 1, "6h5h", evs=tie), only=["zero-freq-best-response"]) == []
    # Above every played action by more than the tolerance: flagged.
    better = {"check": 1.2, "bet33": 1.5, "bet75": 1.2}
    [f] = validate(edit_combo(BASE, 1, "6h5h", evs=better), only=["zero-freq-best-response"])
    assert f.where == "nodes[1].combos['6h5h'].evs.bet33" and "by 0.3" in f.message
    # Within the EV tolerance: not flagged.
    close = {"check": 1.2, "bet33": 1.205, "bet75": 1.2}
    assert validate(edit_combo(BASE, 1, "6h5h", evs=close), only=["zero-freq-best-response"]) == []


def test_reach_consistency():
    s = edit_combo(BASE, 2, "Jd9d", weight=0.5)
    [f] = validate(s, only=["reach-consistency"])
    assert f.where == "nodes[2].combos['Jd9d'].weight" and "expected 0.45" in f.message
    missing = edit_node(BASE, 2, combos=BASE.nodes[2].combos[:-1])
    assert "8s6s is missing" in validate(missing, only=["reach-consistency"])[0].message


def test_suit_isomorphism_within_a_file():
    s = edit_combo(BASE, 1, "6h5h", freqs={"check": 0.5, "bet33": 0.0, "bet75": 0.5})
    [f] = validate(s, only=["suit-isomorphism"])
    assert "differs from '6c5c'" in f.message or "differs from '6h5h'" in f.message
    alone = edit_node(BASE, 1, combos=tuple(c for c in BASE.nodes[1].combos if c.combo != "6c5c"))
    assert "is missing from the node" in validate(alone, only=["suit-isomorphism"])[0].message


@pytest.mark.parametrize(
    "board, swaps",
    [
        ("Ks7d2c", []),  # rainbow flop, three ranks: no symmetry
        ("Ks7s2d", ["c->h, h->c"]),  # two-tone: the two absent suits
        ("KhKd7c", ["d->h, h->d"]),  # paired: hearts and diamonds each hold a king
        ("9h6h3h", 5),  # monotone: any permutation of the three absent suits
        ("Ks7s2d5c", []),  # the turn brings the last absent suit's partner
    ],
)
def test_board_symmetries(board, swaps):
    found = [perm_text(p) for p in board_symmetries(parse_board(board))]
    if isinstance(swaps, int):
        assert len(found) == swaps
    else:
        assert sorted(found) == sorted(swaps)
