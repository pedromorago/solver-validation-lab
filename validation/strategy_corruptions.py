"""Deliberately broken strategy files, one defect each.

The strategy-side counterpart of mutants.py: if an export were wrong in
this particular way, would a rule, the isomorphism check or the regression
diff notice? `scripts/run_corruptions.py` applies every corruption to a
valid fixture and reports what caught it.

Each corruption changes one thing and keeps the rest of the file as
consistent as it can: the same change is made to every suit-isomorphic
copy of the chosen combo (so the symmetry rule isn't tripped for free),
and the overall EV is recomputed when frequencies move, except where the
EV is the point. Knock-on effects that can't be avoided, such as a later
node's range no longer matching, show up in the table.

A corruption either targets the file itself (checked with `validate` and
with `diff` against the original) or a suit-relabelled copy of it
(checked with `validate` and with `iso` against the original).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Callable

from svlab.cards import card_str
from svlab.strategy import (
    ComboStrategy,
    Node,
    Strategy,
    Thresholds,
    Tolerances,
    compare_isomorphic,
    diff,
    dumps,
    loads,
    relabel,
    validate,
)
from svlab.strategy.combos import board_symmetries, combo_key, parse_board, parse_combo, permute_cards
from svlab.strategy.rules import previous_own_decision

Target = tuple[int, frozenset[int]]  # node index, combo


@dataclass(frozen=True)
class Corruption:
    name: str
    # The check that must catch it: "validate:<rule>", "diff:<rule>" or "iso:<rule>".
    expected: str
    # "file": compared with the original by diff; "twin": a relabelled copy, compared by iso.
    target: str
    apply: Callable[[Strategy, random.Random], Strategy | None]


# ------------------------------------------------------------------ helpers


def _key(c: ComboStrategy) -> frozenset[int]:
    return combo_key(parse_combo(c.combo))


def _orbit(strategy: Strategy, key: frozenset[int]) -> set[frozenset[int]]:
    perms = board_symmetries(parse_board(strategy.spot.board))
    return {key} | {combo_key(permute_cards(sorted(key), p)) for p in perms}


def _pick(strategy: Strategy, rng: random.Random, ok: Callable[[Node, ComboStrategy], bool]) -> Target | None:
    # Combos with a tiny weight are left alone: some checks ignore them on purpose.
    choices = [(i, _key(c)) for i, n in enumerate(strategy.nodes) for c in n.combos if c.weight > 0.01 and ok(n, c)]
    return rng.choice(choices) if choices else None


def _edit(strategy: Strategy, target: Target, fn: Callable[[Node, ComboStrategy], ComboStrategy], orbit: bool = True) -> Strategy:
    i, key = target
    keys = _orbit(strategy, key) if orbit else {key}
    node = strategy.nodes[i]
    combos = tuple(fn(node, c) if _key(c) in keys else c for c in node.combos)
    nodes = list(strategy.nodes)
    nodes[i] = replace(node, combos=combos)
    return replace(strategy, nodes=tuple(nodes))


def _with_ev(c: ComboStrategy, freqs: dict[str, float], evs: dict[str, float] | None = None) -> ComboStrategy:
    """New frequencies (and EVs), with the overall EV recomputed from them."""
    evs = dict(c.evs) if evs is None and c.evs is not None else evs
    ev = c.ev
    if ev is not None and evs is not None and all(a in evs for a in freqs):
        ev = math.fsum(f * evs[a] for a, f in freqs.items())
    return replace(c, freqs=freqs, evs=evs, ev=ev)


def _add_combo(strategy: Strategy, i: int, combo: ComboStrategy) -> Strategy:
    nodes = list(strategy.nodes)
    nodes[i] = replace(nodes[i], combos=nodes[i].combos + (combo,))
    return replace(strategy, nodes=tuple(nodes))


def _has_own_child(strategy: Strategy, node: Node) -> bool:
    return any((p := previous_own_decision(strategy, n)) is not None and p[0] is node for n in strategy.nodes)


# -------------------------------------------------------------- corruptions


def freqs_sum_above_one(s: Strategy, rng: random.Random) -> Strategy | None:
    t = _pick(s, rng, lambda n, c: True)
    return t and _edit(s, t, lambda n, c: _with_ev(c, {a: f * 1.07 for a, f in c.freqs.items()}))


def negative_frequency(s: Strategy, rng: random.Random) -> Strategy | None:
    def fn(n: Node, c: ComboStrategy) -> ComboStrategy:
        ordered = sorted(c.freqs, key=lambda a: c.freqs[a])
        low, high = ordered[0], ordered[-1]
        freqs = {**c.freqs, low: -0.05, high: c.freqs[high] + c.freqs[low] + 0.05}
        return _with_ev(c, freqs)

    t = _pick(s, rng, lambda n, c: len(c.freqs) >= 2)
    return t and _edit(s, t, fn)


def weight_above_one(s: Strategy, rng: random.Random) -> Strategy | None:
    t = _pick(s, rng, lambda n, c: True)
    return t and _edit(s, t, lambda n, c: replace(c, weight=1.3))


def blocked_combo(s: Strategy, rng: random.Random) -> Strategy | None:
    """Adds a combo holding one of the board's cards."""
    board = parse_board(s.spot.board)
    i = rng.randrange(len(s.nodes))
    node = s.nodes[i]
    if not node.combos:
        return None
    template = rng.choice(node.combos)
    blocked = rng.choice(board)
    other = rng.choice([c for c in range(52) if c not in board])
    return _add_combo(s, i, replace(template, combo=card_str(blocked) + card_str(other)))


def duplicate_in_other_order(s: Strategy, rng: random.Random) -> Strategy | None:
    """Adds "KhAh" to a node that already has "AhKh"."""
    t = _pick(s, rng, lambda n, c: True)
    if t is None:
        return None
    i, key = t
    original = next(c for c in s.nodes[i].combos if _key(c) == key)
    return _add_combo(s, i, replace(original, combo=original.combo[2:] + original.combo[:2]))


def malformed_combo(s: Strategy, rng: random.Random) -> Strategy | None:
    t = _pick(s, rng, lambda n, c: True)
    return t and _edit(s, t, lambda n, c: replace(c, combo=c.combo[:3] + "x"), orbit=False)


def suit_asymmetry(s: Strategy, rng: random.Random) -> Strategy | None:
    """One combo plays differently from its suit-swapped twin."""

    def fn(n: Node, c: ComboStrategy) -> ComboStrategy:
        ordered = sorted(c.freqs, key=lambda a: -c.freqs[a])
        top, other = ordered[0], ordered[1]
        moved = min(0.2, c.freqs[top] / 2)
        return _with_ev(c, {**c.freqs, top: c.freqs[top] - moved, other: c.freqs[other] + moved})

    t = _pick(s, rng, lambda n, c: len(c.freqs) >= 2 and len(_orbit(s, _key(c))) > 1)
    return t and _edit(s, t, fn, orbit=False)


def ev_inconsistent(s: Strategy, rng: random.Random) -> Strategy | None:
    t = _pick(s, rng, lambda n, c: c.ev is not None and c.evs is not None)
    return t and _edit(s, t, lambda n, c: replace(c, ev=c.ev + 0.5))  # type: ignore[operator]


def nan_ev(s: Strategy, rng: random.Random) -> Strategy | None:
    t = _pick(s, rng, lambda n, c: bool(c.evs))
    return t and _edit(s, t, lambda n, c: replace(c, evs={**c.evs, next(iter(c.evs)): math.nan}))  # type: ignore[arg-type,dict-item]


def silent_zero_freq_best_action(s: Strategy, rng: random.Random) -> Strategy | None:
    """An action is dropped to 0% while its EV is raised above every played
    action. Frequencies and overall EV stay consistent with each other."""

    def fn(n: Node, c: ComboStrategy) -> ComboStrategy:
        top = max(c.freqs, key=lambda a: c.freqs[a])
        victim = next(a for a in n.action_ids if a != top)
        freqs = {**c.freqs, top: c.freqs[top] + c.freqs.get(victim, 0.0), victim: 0.0}
        evs = dict(c.evs or {})
        played = [evs[a] for a, f in freqs.items() if f > 0]
        evs[victim] = max(played) + 1.0
        return _with_ev(c, freqs, evs)

    t = _pick(s, rng, lambda n, c: len(n.actions) >= 2 and c.evs is not None and set(n.action_ids) <= set(c.evs))
    return t and _edit(s, t, fn)


def unknown_action(s: Strategy, rng: random.Random) -> Strategy | None:
    """The exporter writes an action id that the node doesn't define."""

    def rename(values: dict[str, float] | None, old: str) -> dict[str, float] | None:
        return None if values is None else {("bet150" if a == old else a): v for a, v in values.items()}

    def fn(n: Node, c: ComboStrategy) -> ComboStrategy:
        old = max(c.freqs, key=lambda a: c.freqs[a])
        return replace(c, freqs=rename(c.freqs, old), evs=rename(c.evs, old))  # type: ignore[arg-type]

    t = _pick(s, rng, lambda n, c: True)
    return t and _edit(s, t, fn)


def _edit_action(s: Strategy, rng: random.Random, ok: Callable[[Node], bool], fn) -> Strategy | None:
    candidates = [i for i, n in enumerate(s.nodes) if ok(n)]
    if not candidates:
        return None
    i = rng.choice(candidates)
    node = s.nodes[i]
    nodes = list(s.nodes)
    nodes[i] = replace(node, actions=fn(node))
    return replace(s, nodes=tuple(nodes))


def bet_above_stack(s: Strategy, rng: random.Random) -> Strategy | None:
    def fn(node: Node):
        k = next(k for k, a in enumerate(node.actions) if a.kind in ("bet", "raise"))
        actions = list(node.actions)
        actions[k] = replace(actions[k], size=round(node.effective_stack * 1.5, 1))
        return tuple(actions)

    return _edit_action(s, rng, lambda n: any(a.kind in ("bet", "raise") for a in n.actions), fn)


def check_facing_a_bet(s: Strategy, rng: random.Random) -> Strategy | None:
    def fn(node: Node):
        return tuple(replace(a, kind="check") if a.kind == "fold" else a for a in node.actions)

    return _edit_action(s, rng, lambda n: n.to_call > 0 and any(a.kind == "fold" for a in n.actions), fn)


def reach_off_the_tree(s: Strategy, rng: random.Random) -> Strategy | None:
    """At a node that follows the same player's earlier decision, a combo's
    weight no longer equals its earlier weight times the action taken."""
    later = {id(n) for n in s.nodes if previous_own_decision(s, n) is not None}
    t = _pick(s, rng, lambda n, c: id(n) in later and c.weight <= 0.8)
    return t and _edit(s, t, lambda n, c: replace(c, weight=round(c.weight + 0.2, 4)))


def large_regression_shift(s: Strategy, rng: random.Random) -> Strategy | None:
    """A valid file that plays very differently at one node: 30% of every
    combo's most-played action moves to another action. Only the
    regression diff can see this."""
    candidates = [i for i, n in enumerate(s.nodes) if len(n.actions) >= 2 and not _has_own_child(s, n)]
    if not candidates:
        return None
    i = rng.choice(candidates)

    def fn(n: Node, c: ComboStrategy) -> ComboStrategy:
        top = max(n.action_ids, key=lambda a: c.freqs.get(a, 0.0))
        other = max((a for a in n.action_ids if a != top), key=lambda a: (c.evs or {}).get(a, 0.0))
        moved = 0.3 * c.freqs.get(top, 0.0)
        freqs = {**c.freqs, top: c.freqs.get(top, 0.0) - moved, other: c.freqs.get(other, 0.0) + moved}
        return _with_ev(c, freqs)

    node = s.nodes[i]
    nodes = list(s.nodes)
    nodes[i] = replace(node, combos=tuple(fn(node, c) for c in node.combos))
    return replace(s, nodes=tuple(nodes))


# ---------------------------------------------------- relabelled copies


def _relabelling(s: Strategy, rng: random.Random):
    perms = [p for p in permutations(range(4)) if p != (0, 1, 2, 3)]
    return rng.choice(perms)


def twin_keeps_old_suits(s: Strategy, rng: random.Random) -> Strategy | None:
    """A relabelled copy where one combo was written with its old suits."""
    twin = relabel(s, _relabelling(s, rng))
    moved = {
        (i, _key(c)): old.combo
        for i, (node, new) in enumerate(zip(s.nodes, twin.nodes))
        for old, c in zip(node.combos, new.combos)
        if _key(old) != _key(c)
    }
    index = {id(n): i for i, n in enumerate(twin.nodes)}
    t = _pick(twin, rng, lambda n, c: (index[id(n)], _key(c)) in moved)
    return t and _edit(twin, t, lambda n, c: replace(c, combo=moved[t]), orbit=False)


def twin_drifts(s: Strategy, rng: random.Random) -> Strategy | None:
    """A relabelled copy whose strategy for one hand drifted by 0.1, the
    same way for all of its suit-isomorphic combos. The copy is a valid
    file on its own; only comparing it with the original shows the change."""
    twin = relabel(s, _relabelling(s, rng))

    def fn(n: Node, c: ComboStrategy) -> ComboStrategy:
        ordered = sorted(n.action_ids, key=lambda a: -c.freqs.get(a, 0.0))
        top, other = ordered[0], ordered[1]
        moved = min(0.1, c.freqs.get(top, 0.0))
        return _with_ev(c, {**c.freqs, top: c.freqs.get(top, 0.0) - moved, other: c.freqs.get(other, 0.0) + moved})

    ok = lambda n, c: len(n.actions) >= 2 and c.freqs.get(max(c.freqs, key=lambda a: c.freqs[a]), 0) >= 0.2  # noqa: E731
    t = _pick(twin, rng, lambda n, c: ok(n, c) and not _has_own_child(twin, n))
    return t and _edit(twin, t, fn)


CORRUPTIONS: list[Corruption] = [
    Corruption("frequencies sum to 1.07", "validate:freq-sum", "file", freqs_sum_above_one),
    Corruption("negative frequency, sum still 1", "validate:freq-range", "file", negative_frequency),
    Corruption("range weight of 1.3", "validate:weight-range", "file", weight_above_one),
    Corruption("combo holding a board card", "validate:board-blocker", "file", blocked_combo),
    Corruption("duplicate combo in the other card order", "validate:duplicate-combo", "file", duplicate_in_other_order),
    Corruption("malformed combo string", "validate:combo-syntax", "file", malformed_combo),
    Corruption("one combo differs from its suit-swapped twin", "validate:suit-isomorphism", "file", suit_asymmetry),
    Corruption("overall EV off by 0.5", "validate:ev-consistency", "file", ev_inconsistent),
    Corruption("NaN action EV", "validate:ev-finite", "file", nan_ev),
    Corruption("0% action with the best EV", "validate:zero-freq-best-response", "file", silent_zero_freq_best_action),
    Corruption("action id the node doesn't define", "validate:unknown-action", "file", unknown_action),
    Corruption("bet size above the effective stack", "validate:action-legal", "file", bet_above_stack),
    Corruption("check offered facing a bet", "validate:action-legal", "file", check_facing_a_bet),
    Corruption("range weight off the tree", "validate:reach-consistency", "file", reach_off_the_tree),
    Corruption("large strategy shift in a valid file", "diff:max-freq-shift", "file", large_regression_shift),
    Corruption("relabelled copy: one combo keeps its old suits", "iso:iso-combo", "twin", twin_keeps_old_suits),
    Corruption("relabelled copy: one hand's strategy drifts", "iso:iso-combo", "twin", twin_drifts),
]


def caught_by(
    corruption: Corruption,
    baseline: Strategy,
    seed: int = 0,
    thresholds: Thresholds = Thresholds(),
    tol: Tolerances = Tolerances(),
) -> list[str] | None:
    """Applies the corruption and returns every check that objects, as
    "validate:<rule>", "diff:<rule>" or "iso:<rule>". None when the
    corruption has nothing to act on in this file."""
    broken = corruption.apply(baseline, random.Random(seed))
    if broken is None:
        return None
    broken = loads(dumps(broken))  # the defect has to survive being written to a file
    found = [f"validate:{f.rule}" for f in validate(broken, tol)]
    if corruption.target == "file":
        found += [f"diff:{v.rule}" for v in diff(baseline, broken, thresholds).violations]
    else:
        found += [f"iso:{f.rule}" for f in compare_isomorphic(baseline, broken, tol).findings]
    return list(dict.fromkeys(found))
