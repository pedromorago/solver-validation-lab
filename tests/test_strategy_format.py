"""The strategy file contract: strict parsing with located errors, and a
lossless round trip."""

import json
from pathlib import Path

import pytest

from svlab.strategy import FORMAT, ParseError, dumps, load, loads

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "strategies"
MINIMAL = FIXTURES / "minimal_Ks7s2d.json"


def minimal() -> dict:
    return json.loads(MINIMAL.read_text())


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_fixtures_parse_and_round_trip(path):
    strategy = load(path)
    assert strategy.format == FORMAT
    assert strategy.meta["synthetic"] is True
    assert loads(dumps(strategy)) == strategy


def test_minimal_example_is_typed():
    s = load(MINIMAL)
    assert s.spot.positions == ("BTN", "BB")
    assert [n.path for n in s.nodes] == [(), ("check",), ("check", "bet75")]
    first = s.nodes[0].combos[0]
    assert (first.combo, first.weight, first.freqs, first.ev) == ("7h7c", 1.0, {"check": 0.25, "bet33": 0.75}, 4.25)
    assert s.nodes[2].action("raise").size == 13.5


def _set(path, value):
    def edit(d):
        target = d
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _delete(path):
    def edit(d):
        target = d
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return edit


CASES = [
    (_set(["format"], "other/2"), "format: expected 'svlab-strategy/1'"),
    (_delete(["spot"]), "(root): missing key 'spot'"),
    (_set(["extra"], 1), "(root): unknown key 'extra'"),
    (_set(["spot", "board"], "Ks7sKs"), "spot.board: Duplicate card"),
    (_set(["spot", "board"], "Ks7s"), "spot.board: A board has three to five cards"),
    (_set(["spot", "positions"], ["BTN"]), "spot.positions: expected at least two distinct positions"),
    (_set(["spot", "pot"], "5.5"), 'spot.pot: expected a number, got "5.5"'),
    (_set(["nodes"], []), "nodes: expected a non-empty list"),
    (_set(["nodes", 1, "player"], "SB"), "nodes[1].player: 'SB' is not one of the spot's positions"),
    (_set(["nodes", 1, "path"], []), "nodes[1].path: same path as nodes[0]"),
    (_set(["nodes", 0, "to_call"], True), "nodes[0].to_call: expected a number, got true"),
    (_set(["nodes", 0, "actions", 1, "size"], "big"), 'nodes[0].actions[1].size: expected a number, got "big"'),
    (_delete(["nodes", 0, "actions", 1, "type"]), "nodes[0].actions[1]: missing key 'type'"),
    (_delete(["nodes", 0, "combos", "AhQh", "freqs"]), "nodes[0].combos['AhQh']: missing key 'freqs'"),
    (_set(["nodes", 0, "combos", "AhQh", "freqs", "bet33"], "x"), "nodes[0].combos['AhQh'].freqs.bet33: expected a number"),
    (_set(["nodes", 2, "combos", "7h7c", "evs"], [1, 2]), "nodes[2].combos['7h7c'].evs: expected an object, got a list"),
    (_set(["nodes", 2, "combos", "7h7c", "strategy"], {}), "nodes[2].combos['7h7c']: unknown key 'strategy'"),
]


@pytest.mark.parametrize("edit, message", CASES, ids=[m.split(":")[0] for _, m in CASES])
def test_parse_errors_say_where(edit, message):
    data = minimal()
    edit(data)
    with pytest.raises(ParseError) as e:
        loads(json.dumps(data))
    assert str(e.value).startswith(message)


def test_duplicate_keys_are_refused_outside_combos():
    text = MINIMAL.read_text().replace('"to_call": 0,', '"to_call": 0, "to_call": 1,', 1)
    with pytest.raises(ParseError, match=r"nodes\[0\]: duplicate key 'to_call'"):
        loads(text)


def test_duplicate_combo_keys_are_kept_for_the_rules():
    """`dict` would keep the last copy silently; the parser keeps both so
    the duplicate-combo rule can report it."""
    line = '"8s6s": {"weight": 1.0, "freqs": {"check": 0.6, "bet33": 0.4}}'
    text = MINIMAL.read_text().replace('"8s6s": {', line + ',\n        "8s6s": {', 1)
    assert [c.combo for c in loads(text).nodes[0].combos].count("8s6s") == 2


def test_invalid_json_says_where():
    with pytest.raises(ParseError, match=r"line 1 column \d+: invalid JSON"):
        loads('{"format": ')


def test_nan_is_read_so_a_rule_can_report_it():
    text = MINIMAL.read_text().replace('"ev": 4.25', '"ev": NaN', 1)
    assert loads(text).nodes[0].combos[0].ev != loads(text).nodes[0].combos[0].ev
