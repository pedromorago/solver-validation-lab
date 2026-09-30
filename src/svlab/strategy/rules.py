"""Rules a single strategy file must satisfy, whatever produced it.

None of these needs the right answer. They are relationships that any
correct export has to respect: frequencies form a distribution, EVs are
consistent with the frequencies, combos are real and not blocked by the
board, actions are legal, the ranges agree with the tree, and suits that
play the same role on the board are treated the same way.

Every rule takes a Strategy and the tolerances and yields Findings. A
finding points at the exact entry, e.g. `nodes[0].combos['AhKh'].freqs`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from .combos import board_symmetries, combo_key, parse_board, parse_combo, permute_cards, perm_text
from .model import ACTION_KINDS, SIZED_KINDS, ComboStrategy, Node, Strategy


@dataclass(frozen=True)
class Tolerances:
    # Frequencies: how far a sum may be from 1, and the level at or below
    # which an action counts as not played.
    freq: float = 1e-3
    # EVs, in the file's units (usually big blinds).
    ev: float = 0.01
    # Range weights along the tree, and between suit-isomorphic combos.
    weight: float = 1e-3


@dataclass(frozen=True)
class Finding:
    rule: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.where}: {self.message}"


@dataclass(frozen=True)
class Rule:
    id: str
    summary: str
    run: Callable[[Strategy, Tolerances], Iterable[Finding]]


RULES: list[Rule] = []


def rule(rule_id: str, summary: str):
    def register(fn: Callable[[Strategy, Tolerances], Iterable[Finding]]):
        RULES.append(Rule(rule_id, summary, fn))
        return fn

    return register


def validate(strategy: Strategy, tol: Tolerances = Tolerances(), only: Iterable[str] | None = None) -> list[Finding]:
    wanted = None if only is None else set(only)
    findings: list[Finding] = []
    for r in RULES:
        if wanted is None or r.id in wanted:
            findings.extend(r.run(strategy, tol))
    return findings


# ------------------------------------------------------------------ helpers


def _where(i: int, combo: ComboStrategy | None = None, part: str = "") -> str:
    out = f"nodes[{i}]"
    if combo is not None:
        out += f".combos[{combo.combo!r}]"
    return out + (f".{part}" if part else "")


def _combos(strategy: Strategy) -> Iterator[tuple[int, Node, ComboStrategy]]:
    for i, node in enumerate(strategy.nodes):
        for c in node.combos:
            yield i, node, c


def _cards(combo: ComboStrategy) -> tuple[int, int] | None:
    try:
        return parse_combo(combo.combo)
    except ValueError:
        return None


def _usable(strategy: Strategy, combo: ComboStrategy) -> tuple[int, int] | None:
    """The combo's cards if it is well formed and not blocked by the board.
    Rules that compare combos skip the others: they are reported elsewhere."""
    cards = _cards(combo)
    if cards is None or set(cards) & set(parse_board(strategy.spot.board)):
        return None
    return cards


def _finite(*values: float | None) -> bool:
    return all(v is None or math.isfinite(v) for v in values)


# -------------------------------------------------------- distributions


@rule("freq-range", "every action frequency is a number in [0, 1]")
def freq_range(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, _, c in _combos(strategy):
        for action, f in c.freqs.items():
            if not (0.0 <= f <= 1.0):
                yield Finding("freq-range", _where(i, c, f"freqs.{action}"), f"frequency {f} is outside [0, 1]")


@rule("freq-sum", "the frequencies of each combo add up to 1 within the frequency tolerance")
def freq_sum(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, _, c in _combos(strategy):
        total = math.fsum(c.freqs.values())
        if not abs(total - 1.0) <= tol.freq:
            yield Finding("freq-sum", _where(i, c, "freqs"), f"sums to {total:.6g}, expected 1 ± {tol.freq:g}")


@rule("weight-range", "every range weight is a number in [0, 1]")
def weight_range(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, _, c in _combos(strategy):
        if not (0.0 <= c.weight <= 1.0):
            yield Finding("weight-range", _where(i, c, "weight"), f"weight {c.weight} is outside [0, 1]")


@rule("ev-finite", "every EV is a finite number")
def ev_finite(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, _, c in _combos(strategy):
        if not _finite(c.ev):
            yield Finding("ev-finite", _where(i, c, "ev"), f"EV is {c.ev}")
        for action, v in (c.evs or {}).items():
            if not _finite(v):
                yield Finding("ev-finite", _where(i, c, f"evs.{action}"), f"EV is {v}")


# ---------------------------------------------------------------- actions


@rule("action-legal", "actions and sizes are legal at the node (sizes positive, within the effective stack)")
def action_legal(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, node in enumerate(strategy.nodes):
        where = _where(i)
        if not node.actions:
            yield Finding("action-legal", f"{where}.actions", "no actions at this node")
        if not (node.pot > 0 and node.effective_stack >= 0 and node.to_call >= 0):
            yield Finding(
                "action-legal",
                where,
                f"pot {node.pot}, effective stack {node.effective_stack}, to call {node.to_call}: "
                "the pot must be positive and the others not negative",
            )
        ids = [a.id for a in node.actions]
        facing_bet = node.to_call > 0
        for k, a in enumerate(node.actions):
            at = f"{where}.actions[{k}]"
            if ids.count(a.id) > 1:
                yield Finding("action-legal", at, f"action id {a.id!r} is used more than once")
            if a.kind not in ACTION_KINDS:
                yield Finding("action-legal", at, f"unknown action type {a.kind!r}")
                continue
            if a.kind in SIZED_KINDS:
                if a.size is None or not math.isfinite(a.size) or a.size <= 0:
                    yield Finding("action-legal", at, f"{a.kind} needs a positive size, got {a.size}")
                    continue
                if a.size > node.effective_stack + 1e-9:
                    yield Finding(
                        "action-legal", at, f"size {a.size:g} is above the effective stack {node.effective_stack:g}"
                    )
            elif a.size is not None:
                yield Finding("action-legal", at, f"{a.kind} takes no size, got {a.size}")
            if a.kind in ("check", "bet") and facing_bet:
                yield Finding("action-legal", at, f"{a.kind} is not possible facing a bet of {node.to_call:g}")
            if a.kind in ("fold", "call", "raise") and not facing_bet:
                yield Finding("action-legal", at, f"{a.kind} is not possible when there is nothing to call")
            if a.kind == "raise" and facing_bet and a.size is not None and a.size <= node.to_call:
                yield Finding("action-legal", at, f"raise to {a.size:g} is not above the {node.to_call:g} to call")
            if a.kind == "allin" and a.size is not None and abs(a.size - node.effective_stack) > 1e-9:
                yield Finding(
                    "action-legal", at, f"all-in of {a.size:g} differs from the effective stack {node.effective_stack:g}"
                )


@rule("unknown-action", "combos only use actions defined at their node, and paths follow defined actions")
def unknown_action(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, node, c in _combos(strategy):
        known = set(node.action_ids)
        for part, values in (("freqs", c.freqs), ("evs", c.evs or {})):
            for action in values:
                if action not in known:
                    yield Finding("unknown-action", _where(i, c, f"{part}.{action}"), f"no action {action!r} at this node")
    for i, node in enumerate(strategy.nodes):
        for k, step in enumerate(node.path):
            parent = strategy.node(node.path[:k])
            if parent is not None and step not in parent.action_ids:
                yield Finding(
                    "unknown-action", f"nodes[{i}].path[{k}]", f"{step!r} is not an action at the node {list(node.path[:k])}"
                )


# ----------------------------------------------------------------- combos


@rule("combo-syntax", "every combo is two distinct cards such as 'AhKh'")
def combo_syntax(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, _, c in _combos(strategy):
        try:
            parse_combo(c.combo)
        except ValueError as e:
            yield Finding("combo-syntax", _where(i, c), str(e))


@rule("duplicate-combo", "no combo appears twice in a node, in either card order")
def duplicate_combo(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, node in enumerate(strategy.nodes):
        first: dict[frozenset[int], str] = {}
        for c in node.combos:
            cards = _cards(c)
            if cards is None:
                continue
            key = combo_key(cards)
            if key in first:
                same = "written the same way" if first[key] == c.combo else f"same cards as {first[key]!r}"
                yield Finding("duplicate-combo", _where(i, c), f"duplicate combo ({same})")
            else:
                first[key] = c.combo


@rule("board-blocker", "no combo uses a card that is on the board")
def board_blocker(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    board = set(parse_board(strategy.spot.board))
    for i, _, c in _combos(strategy):
        cards = _cards(c)
        if cards is not None and set(cards) & board:
            yield Finding("board-blocker", _where(i, c), f"uses a board card ({strategy.spot.board})")


# -------------------------------------------------------------------- EVs


@rule("ev-consistency", "where both are given, the overall EV equals the frequency-weighted action EVs")
def ev_consistency(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, _, c in _combos(strategy):
        if c.ev is None or c.evs is None or not _finite(c.ev, *c.evs.values()):
            continue
        played = {a: f for a, f in c.freqs.items() if f > 0}
        missing = [a for a in played if a not in c.evs]
        if missing:
            yield Finding("ev-consistency", _where(i, c, "evs"), f"no EV for played action {missing[0]!r}")
            continue
        mixed = math.fsum(f * c.evs[a] for a, f in played.items())
        if abs(mixed - c.ev) > tol.ev:
            yield Finding(
                "ev-consistency",
                _where(i, c, "ev"),
                f"EV {c.ev:g} but the frequencies and action EVs give {mixed:.4f} (tolerance {tol.ev:g})",
            )


@rule(
    "zero-freq-best-response",
    "an action played at 0% does not have an EV above every played action by more than the EV tolerance",
)
def zero_freq_best_response(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    """A local best-response sanity check.

    The action EVs in an export are the values of each pure action against
    the opponent's strategy in the same file. At an equilibrium every
    action a combo plays is a best response, so an action it never plays
    cannot be worth strictly more than all of them. When it is, either the
    frequencies or the EVs are wrong, or the solution is far from
    converged. Passing says nothing about whether the strategy is good:
    it only checks the file against its own numbers, at one node."""
    for i, node, c in _combos(strategy):
        if c.evs is None or c.weight <= 0 or not _finite(*c.evs.values()):
            continue
        played = [a for a in node.action_ids if c.freqs.get(a, 0.0) > tol.freq]
        if not played or any(a not in c.evs for a in played):
            continue
        best = max(played, key=lambda a: c.evs[a])  # type: ignore[index]
        for a in node.action_ids:
            if a in played or a not in c.evs:
                continue
            gap = c.evs[a] - c.evs[best]
            if gap > tol.ev:
                yield Finding(
                    "zero-freq-best-response",
                    _where(i, c, f"evs.{a}"),
                    f"{a!r} is played {c.freqs.get(a, 0.0):.4g} of the time but its EV {c.evs[a]:g} beats every "
                    f"played action (best: {best!r} at {c.evs[best]:g}) by {gap:.4g}",
                )


# ------------------------------------------------------------------- tree


def previous_own_decision(strategy: Strategy, node: Node) -> tuple[Node, str] | None:
    """The same player's previous decision on this path and the action they
    took there, when every node in between is in the file and belongs to
    someone else. Otherwise None: the reach can't be checked."""
    for k in range(len(node.path) - 1, -1, -1):
        prefix = strategy.node(node.path[:k])
        if prefix is None:
            return None
        if prefix.player == node.player:
            return prefix, node.path[k]
    return None


@rule("reach-consistency", "a player's range at a node is their range at their previous decision times the action taken")
def reach_consistency(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    for i, node in enumerate(strategy.nodes):
        found = previous_own_decision(strategy, node)
        if found is None:
            continue
        parent, taken = found
        here = {combo_key(cards): c for c in node.combos if (cards := _usable(strategy, c))}
        seen: set[frozenset[int]] = set()
        for c in parent.combos:
            cards = _usable(strategy, c)
            if cards is None or combo_key(cards) in seen:
                continue  # malformed, blocked or duplicate: reported by other rules
            seen.add(combo_key(cards))
            expected = c.weight * c.freqs.get(taken, 0.0)
            child = here.pop(combo_key(cards), None)
            got = 0.0 if child is None else child.weight
            if abs(got - expected) > tol.weight:
                where = _where(i, child) if child else f"nodes[{i}].combos"
                what = f"weight {got:g}" if child else f"{c.combo} is missing"
                yield Finding(
                    "reach-consistency",
                    where + (".weight" if child else ""),
                    f"{what}, expected {expected:.4g} ({c.weight:g} × {taken!r} at {c.freqs.get(taken, 0.0):g})",
                )
        for child in here.values():
            if child.weight > tol.weight:
                yield Finding(
                    "reach-consistency",
                    _where(i, child, "weight"),
                    f"weight {child.weight:g} but the combo is not in {node.player}'s range at the previous decision",
                )


# ------------------------------------------------------------- symmetry


def _same(a: float, b: float, tol: float) -> bool:
    if math.isfinite(a) and math.isfinite(b):
        return abs(a - b) <= tol
    return a == b or (math.isnan(a) and math.isnan(b))


def strategy_differences(a: ComboStrategy, b: ComboStrategy, tol: Tolerances) -> list[str]:
    """How two combos' strategies differ, beyond tolerance. Used for suit
    isomorphism within a file and across files."""
    out = []
    if not _same(a.weight, b.weight, tol.weight):
        out.append(f"weight {a.weight:g} vs {b.weight:g}")
    for action in sorted(set(a.freqs) | set(b.freqs)):
        fa, fb = a.freqs.get(action, 0.0), b.freqs.get(action, 0.0)
        if not _same(fa, fb, tol.freq):
            out.append(f"{action} {fa:g} vs {fb:g}")
    if (a.ev is None) != (b.ev is None) or (a.ev is not None and not _same(a.ev, b.ev, tol.ev)):  # type: ignore[arg-type]
        out.append(f"EV {a.ev} vs {b.ev}")
    evs_a, evs_b = a.evs or {}, b.evs or {}
    for action in sorted(set(evs_a) | set(evs_b)):
        va, vb = evs_a.get(action), evs_b.get(action)
        if va is None or vb is None or not _same(va, vb, tol.ev):
            out.append(f"EV of {action} {va} vs {vb}")
    return out


@rule("suit-isomorphism", "combos that differ only by swapping suits with the same role on the board play the same")
def suit_isomorphism(strategy: Strategy, tol: Tolerances) -> Iterator[Finding]:
    symmetries = board_symmetries(parse_board(strategy.spot.board))
    if not symmetries:
        return
    for i, node in enumerate(strategy.nodes):
        by_key = {}
        for c in node.combos:
            cards = _usable(strategy, c)
            if cards is not None:
                by_key.setdefault(combo_key(cards), c)
        reported: set[frozenset[frozenset[int]]] = set()
        for key, c in by_key.items():
            for perm in symmetries:
                image = combo_key(permute_cards(sorted(key), perm))
                pair = frozenset((key, image))
                if image == key or pair in reported:
                    continue
                other = by_key.get(image)
                if other is None:
                    if c.weight > tol.weight:
                        reported.add(pair)
                        yield Finding(
                            "suit-isomorphism",
                            _where(i, c),
                            f"its image under {perm_text(perm)} is missing from the node",
                        )
                    continue
                diffs = strategy_differences(c, other, tol)
                if diffs:
                    reported.add(pair)
                    yield Finding(
                        "suit-isomorphism",
                        _where(i, c),
                        f"differs from {other.combo!r} (suits {perm_text(perm)}): " + "; ".join(diffs[:3]),
                    )
