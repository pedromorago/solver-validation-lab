"""Strict parsing and serialisation of strategy files.

Parsing checks structure and types only: required and unknown keys, JSON
types, a readable board, positions that exist. Everything about whether
the numbers make sense (frequencies, EVs, legal sizes, valid combos) is
left to the rules in `rules.py`, so that a file with a wrong number still
loads and every problem in it can be reported at once.

Errors name the exact place, e.g. `nodes[1].combos['AhKh'].freqs.bet33:
expected a number, got "x"`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .combos import parse_board
from .model import FORMAT, Action, ComboStrategy, Node, Spot, Strategy


class ParseError(ValueError):
    def __init__(self, where: str, problem: str):
        self.where = where
        self.problem = problem
        super().__init__(f"{where}: {problem}")


class _Pairs(list):
    """A JSON object as its list of (key, value) pairs, so duplicate keys
    are not silently dropped the way `dict` would."""


def _describe(value: Any) -> str:
    if isinstance(value, _Pairs):
        return "an object"
    if isinstance(value, list):
        return "a list"
    return json.dumps(value)


def _object(value: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(value, _Pairs):
        raise ParseError(where, f"expected an object, got {_describe(value)}")
    out: dict[str, Any] = {}
    for key, item in value:
        if key in out:
            raise ParseError(where, f"duplicate key {key!r}")
        if key not in required and key not in optional:
            raise ParseError(where, f"unknown key {key!r}")
        out[key] = item
    missing = [k for k in required if k not in out]
    if missing:
        raise ParseError(where, f"missing key {missing[0]!r}")
    return out


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(where, f"expected a number, got {_describe(value)}")
    return float(value)


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParseError(where, f"expected a non-empty string, got {_describe(value)}")
    return value


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ParseError(where, f"expected a list of strings, got {_describe(value)}")
    return tuple(_string(v, f"{where}[{i}]") for i, v in enumerate(value))


def _number_map(value: Any, where: str) -> dict[str, float]:
    fields = _object(value, where, (), tuple(k for k, _ in value) if isinstance(value, _Pairs) else ())
    return {k: _number(v, f"{where}.{k}") for k, v in fields.items()}


def _plain(value: Any, where: str) -> Any:
    """Free-form JSON (the meta block), with duplicate keys still refused."""
    if isinstance(value, _Pairs):
        fields = _object(value, where, (), tuple(k for k, _ in value))
        return {k: _plain(v, f"{where}.{k}") for k, v in fields.items()}
    if isinstance(value, list):
        return [_plain(v, f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def _spot(value: Any) -> Spot:
    where = "spot"
    f = _object(value, where, ("game", "positions", "pot", "effective_stack", "board"), ("preflop",))
    positions = _strings(f["positions"], f"{where}.positions")
    if len(positions) < 2 or len(set(positions)) != len(positions):
        raise ParseError(f"{where}.positions", "expected at least two distinct positions")
    board = _string(f["board"], f"{where}.board")
    try:
        parse_board(board)
    except ValueError as e:
        raise ParseError(f"{where}.board", str(e)) from None
    return Spot(
        game=_string(f["game"], f"{where}.game"),
        positions=positions,
        pot=_number(f["pot"], f"{where}.pot"),
        effective_stack=_number(f["effective_stack"], f"{where}.effective_stack"),
        board=board,
        preflop=_strings(f.get("preflop", []), f"{where}.preflop"),
    )


def _action(value: Any, where: str) -> Action:
    f = _object(value, where, ("id", "type"), ("size",))
    size = f.get("size")
    return Action(
        id=_string(f["id"], f"{where}.id"),
        kind=_string(f["type"], f"{where}.type"),
        size=None if size is None else _number(size, f"{where}.size"),
    )


def _combo(text: str, value: Any, where: str) -> ComboStrategy:
    f = _object(value, where, ("weight", "freqs"), ("evs", "ev"))
    evs, ev = f.get("evs"), f.get("ev")
    return ComboStrategy(
        combo=text,
        weight=_number(f["weight"], f"{where}.weight"),
        freqs=_number_map(f["freqs"], f"{where}.freqs"),
        evs=None if evs is None else _number_map(evs, f"{where}.evs"),
        ev=None if ev is None else _number(ev, f"{where}.ev"),
    )


def _node(value: Any, where: str, positions: tuple[str, ...]) -> Node:
    f = _object(value, where, ("path", "player", "pot", "effective_stack", "to_call", "actions", "combos"))
    player = _string(f["player"], f"{where}.player")
    if player not in positions:
        raise ParseError(f"{where}.player", f"{player!r} is not one of the spot's positions {list(positions)}")
    if not isinstance(f["actions"], list):
        raise ParseError(f"{where}.actions", f"expected a list, got {_describe(f['actions'])}")
    combos = f["combos"]
    if not isinstance(combos, _Pairs):
        raise ParseError(f"{where}.combos", f"expected an object, got {_describe(combos)}")
    # Duplicate combo keys are kept: reporting them is the job of a rule.
    return Node(
        path=_strings(f["path"], f"{where}.path"),
        player=player,
        pot=_number(f["pot"], f"{where}.pot"),
        effective_stack=_number(f["effective_stack"], f"{where}.effective_stack"),
        to_call=_number(f["to_call"], f"{where}.to_call"),
        actions=tuple(_action(a, f"{where}.actions[{i}]") for i, a in enumerate(f["actions"])),
        combos=tuple(_combo(k, v, f"{where}.combos[{k!r}]") for k, v in combos),
    )


def from_json(value: Any) -> Strategy:
    """Builds a Strategy from JSON decoded with `object_pairs_hook=_Pairs`."""
    f = _object(value, "(root)", ("format", "spot", "nodes"), ("meta",))
    if f["format"] != FORMAT:
        raise ParseError("format", f"expected {FORMAT!r}, got {_describe(f['format'])}")
    spot = _spot(f["spot"])
    if not isinstance(f["nodes"], list) or not f["nodes"]:
        raise ParseError("nodes", "expected a non-empty list of nodes")
    nodes = tuple(_node(n, f"nodes[{i}]", spot.positions) for i, n in enumerate(f["nodes"]))
    seen: dict[tuple[str, ...], int] = {}
    for i, node in enumerate(nodes):
        if node.path in seen:
            raise ParseError(f"nodes[{i}].path", f"same path as nodes[{seen[node.path]}]")
        seen[node.path] = i
    return Strategy(spot=spot, nodes=nodes, meta=_plain(f.get("meta", _Pairs()), "meta"), format=FORMAT)


def loads(text: str) -> Strategy:
    try:
        value = json.loads(text, object_pairs_hook=_Pairs)
    except json.JSONDecodeError as e:
        raise ParseError(f"line {e.lineno} column {e.colno}", f"invalid JSON: {e.msg}") from None
    return from_json(value)


def load(path: str | Path) -> Strategy:
    return loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------- writing


def to_json(strategy: Strategy) -> dict[str, Any]:
    s = strategy.spot
    out: dict[str, Any] = {"format": strategy.format}
    if strategy.meta:
        out["meta"] = strategy.meta
    spot: dict[str, Any] = {
        "game": s.game,
        "positions": list(s.positions),
        "pot": s.pot,
        "effective_stack": s.effective_stack,
        "board": s.board,
    }
    if s.preflop:
        spot["preflop"] = list(s.preflop)
    out["spot"] = spot
    out["nodes"] = [
        {
            "path": list(n.path),
            "player": n.player,
            "pot": n.pot,
            "effective_stack": n.effective_stack,
            "to_call": n.to_call,
            "actions": [
                {"id": a.id, "type": a.kind} | ({} if a.size is None else {"size": a.size}) for a in n.actions
            ],
            "combos": {c.combo: _combo_json(c) for c in n.combos},
        }
        for n in strategy.nodes
    ]
    return out


def _combo_json(c: ComboStrategy) -> dict[str, Any]:
    out: dict[str, Any] = {"weight": c.weight, "freqs": dict(c.freqs)}
    if c.evs is not None:
        out["evs"] = dict(c.evs)
    if c.ev is not None:
        out["ev"] = c.ev
    return out


def dumps(strategy: Strategy) -> str:
    """JSON text with one line per combo and per action, so that files diff
    well in version control. Duplicate combos cannot be written (a dict
    can't hold them); corruptions that need them edit the text instead."""
    return _format(to_json(strategy), 0, inline=False) + "\n"


def _format(value: Any, depth: int, inline: bool) -> str:
    compact = json.dumps(value, separators=(", ", ": "))
    if inline or not isinstance(value, (dict, list)) or len(compact) <= 80:
        return compact
    pad, inner = "  " * depth, "  " * (depth + 1)
    if isinstance(value, list):
        items = [inner + _format(v, depth + 1, inline=False) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    items = []
    for key, item in value.items():
        # Combos and actions: one line each.
        child_inline = key == "actions"
        if key == "combos" and isinstance(item, dict):
            rows = [f"{inner}  {json.dumps(k)}: {_format(v, 0, inline=True)}" for k, v in item.items()]
            text = "{\n" + ",\n".join(rows) + "\n" + inner + "}" if rows else "{}"
        elif child_inline and isinstance(item, list):
            rows = [f"{inner}  {_format(v, 0, inline=True)}" for v in item]
            text = "[\n" + ",\n".join(rows) + "\n" + inner + "]" if rows else "[]"
        else:
            text = _format(item, depth + 1, inline=False)
        items.append(f"{inner}{json.dumps(key)}: {text}")
    return "{\n" + ",\n".join(items) + "\n" + pad + "}"
