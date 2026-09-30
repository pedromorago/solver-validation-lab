"""Regression diff between two versions of the same strategy.

A new build of a solver, a new abstraction or a new export pipeline should
change a strategy only where someone meant it to. The diff matches nodes
by path and combos by their cards (card order doesn't matter), measures
how far each combo moved, and compares the result with thresholds. Any
threshold exceeded is a violation, and the CLI exits non-zero.

What is measured, per node:
- per combo: the largest change in any action frequency, the largest
  change in any EV (overall or per action), and the change in range weight;
- per action: the range-weighted frequency, sum(weight * freq) / sum(weight);
- combos added and removed (with a positive weight);
- nodes added or removed, and nodes whose actions changed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .combos import combo_key, parse_board, parse_combo
from .model import ComboStrategy, Node, Strategy
from .rules import Finding


@dataclass(frozen=True)
class Thresholds:
    max_freq_shift: float = 0.05
    max_ev_shift: float = 0.25
    max_weight_shift: float = 0.05
    max_aggregate_shift: float = 0.02
    max_combos_added: int = 0
    max_combos_removed: int = 0
    # Combos below this weight in both files are left out of the per-combo
    # thresholds (they still count in the aggregates).
    min_weight: float = 0.0
    allow_tree_changes: bool = False

    @classmethod
    def from_json(cls, data: Any, where: str = "config") -> "Thresholds":
        if not isinstance(data, dict):
            raise ValueError(f"{where}: expected an object")
        known = {f.name: f.type for f in fields(cls)}
        values: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                raise ValueError(f"{where}: unknown threshold {key!r}; known: {', '.join(known)}")
            if key == "allow_tree_changes":
                if not isinstance(value, bool):
                    raise ValueError(f"{where}.{key}: expected true or false")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{where}.{key}: expected a number >= 0")
            values[key] = value
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")), str(path))


@dataclass(frozen=True)
class ComboShift:
    combo: str
    freq_shift: float
    freq_action: str
    ev_shift: float | None
    weight_shift: float


@dataclass
class NodeDiff:
    path: list[str]
    player: str
    compared: int
    added: list[str]
    removed: list[str]
    actions_changed: str | None
    aggregate: dict[str, tuple[float | None, float | None]]
    shifts: list[ComboShift] = field(default_factory=list)

    def largest(self, attr: str) -> ComboShift | None:
        candidates = [s for s in self.shifts if getattr(s, attr) is not None]
        return max(candidates, key=lambda s: getattr(s, attr), default=None)


@dataclass
class DiffReport:
    baseline: str
    candidate: str
    thresholds: Thresholds
    nodes: list[NodeDiff]
    nodes_added: list[list[str]]
    nodes_removed: list[list[str]]
    violations: list[Finding]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "ok": self.ok,
            "thresholds": asdict(self.thresholds),
            "violations": [asdict(v) for v in self.violations],
            "nodes_added": self.nodes_added,
            "nodes_removed": self.nodes_removed,
            "nodes": [
                {
                    "path": n.path,
                    "player": n.player,
                    "combos_compared": n.compared,
                    "combos_added": n.added,
                    "combos_removed": n.removed,
                    "actions_changed": n.actions_changed,
                    "aggregate": {a: {"baseline": b, "candidate": c} for a, (b, c) in n.aggregate.items()},
                    "max_freq_shift": _shift_json(n.largest("freq_shift")),
                    "max_ev_shift": _shift_json(n.largest("ev_shift")),
                    "max_weight_shift": _shift_json(n.largest("weight_shift")),
                }
                for n in self.nodes
            ],
        }

    def to_markdown(self, top: int = 5) -> str:
        t = self.thresholds
        out = [
            "# Strategy diff",
            "",
            f"Baseline: `{self.baseline}`  ",
            f"Candidate: `{self.candidate}`",
            "",
            f"**Result: {'PASS' if self.ok else 'FAIL'}**"
            + ("" if self.ok else f" ({len(self.violations)} threshold violation(s))"),
            "",
            "| Threshold | Value |",
            "|---|---|",
            *(f"| {f.name} | {getattr(t, f.name)} |" for f in fields(t)),
            "",
            "## Nodes",
            "",
            "| Node | Player | Compared | Added | Removed | Max freq shift | Max EV shift | Max weight shift |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for n in self.nodes:
            f, e, w = n.largest("freq_shift"), n.largest("ev_shift"), n.largest("weight_shift")
            out.append(
                f"| {_path(n.path)} | {n.player} | {n.compared} | {len(n.added)} | {len(n.removed)} | "
                f"{_cell(f, 'freq_shift')} | {_cell(e, 'ev_shift')} | {_cell(w, 'weight_shift')} |"
            )
        for label, paths in (("added", self.nodes_added), ("removed", self.nodes_removed)):
            if paths:
                out += ["", f"Nodes {label}: " + ", ".join(_path(p) for p in paths)]
        out += ["", "## Range-weighted action frequencies", ""]
        for n in self.nodes:
            out += [f"{_path(n.path)} ({n.player})", "", "| Action | Baseline | Candidate | Shift |", "|---|---|---|---|"]
            for action, (b, c) in n.aggregate.items():
                shift = "" if b is None or c is None else f"{c - b:+.4f}"
                out.append(f"| {action} | {_num(b)} | {_num(c)} | {shift} |")
            if n.actions_changed:
                out.append(f"\nActions changed: {n.actions_changed}")
            out.append("")
        out += ["## Largest frequency shifts", "", "| Node | Combo | Action | Shift |", "|---|---|---|---|"]
        for n in self.nodes:
            for s in sorted(n.shifts, key=lambda s: -s.freq_shift)[:top]:
                if s.freq_shift > 0:
                    out.append(f"| {_path(n.path)} | {s.combo} | {s.freq_action} | {s.freq_shift:.4f} |")
        out += ["", "## Violations", ""]
        out += [f"- `{v.rule}` {v.where}: {v.message}" for v in self.violations] or ["None."]
        return "\n".join(out) + "\n"


def _shift_json(s: ComboShift | None) -> dict[str, Any] | None:
    return None if s is None else asdict(s)


def _path(path: list[str]) -> str:
    return "root" if not path else " > ".join(path)


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def _cell(s: ComboShift | None, attr: str) -> str:
    if s is None or getattr(s, attr) is None:
        return "-"
    return f"{getattr(s, attr):.4f} ({s.combo})"


# ------------------------------------------------------------- computing


def _keyed(node: Node, board: set[int]) -> dict[frozenset[int], ComboStrategy]:
    out: dict[frozenset[int], ComboStrategy] = {}
    for c in node.combos:
        try:
            cards = parse_combo(c.combo)
        except ValueError:
            continue
        if not set(cards) & board:
            out.setdefault(combo_key(cards), c)
    return out


def _aggregate(combos: list[ComboStrategy], action: str) -> float | None:
    total = math.fsum(c.weight for c in combos)
    if total <= 0:
        return None
    return math.fsum(c.weight * c.freqs.get(action, 0.0) for c in combos) / total


def _shift(b: ComboStrategy, c: ComboStrategy) -> ComboShift:
    actions = sorted(set(b.freqs) | set(c.freqs))
    freq_action = max(actions, key=lambda a: abs(b.freqs.get(a, 0.0) - c.freqs.get(a, 0.0)), default="")
    freq_shift = abs(b.freqs.get(freq_action, 0.0) - c.freqs.get(freq_action, 0.0)) if actions else 0.0
    ev_shifts = []
    if b.ev is not None and c.ev is not None:
        ev_shifts.append(abs(b.ev - c.ev))
    for a in set(b.evs or {}) & set(c.evs or {}):
        ev_shifts.append(abs(b.evs[a] - c.evs[a]))  # type: ignore[index]
    # A NaN appearing (or disappearing) is an unbounded change.
    ev_shift = max((math.inf if math.isnan(x) else x for x in ev_shifts), default=None)
    return ComboShift(c.combo, freq_shift, freq_action, ev_shift, abs(b.weight - c.weight))


def diff(
    baseline: Strategy,
    candidate: Strategy,
    thresholds: Thresholds = Thresholds(),
    names: tuple[str, str] = ("baseline", "candidate"),
) -> DiffReport:
    t = thresholds
    violations: list[Finding] = []
    if baseline.spot != candidate.spot:
        violations.append(Finding("spot-changed", "spot", "the two files describe different spots"))
    board_b, board_c = set(parse_board(baseline.spot.board)), set(parse_board(candidate.spot.board))
    nodes: list[NodeDiff] = []
    nodes_removed = [list(n.path) for n in baseline.nodes if candidate.node(n.path) is None]
    nodes_added = [list(n.path) for n in candidate.nodes if baseline.node(n.path) is None]
    if (nodes_added or nodes_removed) and not t.allow_tree_changes:
        violations.append(
            Finding("tree-changed", "nodes", f"{len(nodes_added)} node(s) added, {len(nodes_removed)} removed")
        )
    for nb in baseline.nodes:
        nc = candidate.node(nb.path)
        if nc is None:
            continue
        where = f"node {_path(list(nb.path))}"
        actions_changed = None
        if nb.actions != nc.actions or nb.player != nc.player:
            removed_actions = [a.id for a in nb.actions if a not in nc.actions]
            added_actions = [a.id for a in nc.actions if a not in nb.actions]
            actions_changed = f"removed {removed_actions or 'none'}, added {added_actions or 'none'}"
            if not t.allow_tree_changes:
                violations.append(Finding("tree-changed", where, f"actions changed: {actions_changed}"))
        kb, kc = _keyed(nb, board_b), _keyed(nc, board_c)
        added = [kc[k].combo for k in kc if k not in kb and kc[k].weight > 0]
        removed = [kb[k].combo for k in kb if k not in kc and kb[k].weight > 0]
        shifts = {k: _shift(kb[k], kc[k]) for k in kb if k in kc}
        aggregate = {}
        for action in dict.fromkeys([*nb.action_ids, *nc.action_ids]):
            aggregate[action] = (_aggregate(list(kb.values()), action), _aggregate(list(kc.values()), action))
        nodes.append(
            NodeDiff(list(nb.path), nb.player, len(shifts), added, removed, actions_changed, aggregate, list(shifts.values()))
        )

        if len(added) > t.max_combos_added:
            violations.append(Finding("combos-added", where, f"{len(added)} combo(s) added: {', '.join(added[:8])}"))
        if len(removed) > t.max_combos_removed:
            violations.append(
                Finding("combos-removed", where, f"{len(removed)} combo(s) removed: {', '.join(removed[:8])}")
            )
        checked = [s for k, s in shifts.items() if max(kb[k].weight, kc[k].weight) >= t.min_weight]
        for rule_id, attr, limit, what in (
            ("max-freq-shift", "freq_shift", t.max_freq_shift, "frequency"),
            ("max-ev-shift", "ev_shift", t.max_ev_shift, "EV"),
            ("max-weight-shift", "weight_shift", t.max_weight_shift, "weight"),
        ):
            over = [s for s in checked if getattr(s, attr) is not None and getattr(s, attr) > limit]
            if over:
                over.sort(key=lambda s: getattr(s, attr), reverse=True)
                worst = "; ".join(
                    f"{s.combo} {s.freq_action + ' ' if attr == 'freq_shift' else ''}{what} moved {getattr(s, attr):.4g}"
                    for s in over[:3]
                )
                violations.append(Finding(rule_id, where, f"{len(over)} combo(s) above {limit:g}, largest: {worst}"))
        for action, (b, c) in aggregate.items():
            if b is not None and c is not None and abs(c - b) > t.max_aggregate_shift:
                violations.append(
                    Finding(
                        "max-aggregate-shift",
                        where,
                        f"{action} {b:.4f} -> {c:.4f} ({c - b:+.4f}, limit {t.max_aggregate_shift:g})",
                    )
                )
    return DiffReport(names[0], names[1], t, nodes, nodes_added, nodes_removed, violations)

