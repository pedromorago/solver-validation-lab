"""Suit isomorphism across two files.

Relabelling suits changes nothing about poker. A strategy for Ks7s2d and
one for Kh7h2c are the same strategy with hearts written as spades and
clubs as diamonds, so every combo in one file must map onto a combo in the
other with the same weight, frequencies and EVs (within tolerance).

The within-file version of this property is the `suit-isomorphism` rule in
rules.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from svlab.cards import card_str

from .combos import Perm, board_isomorphisms, combo_key, combo_text, parse_board, parse_combo, perm_text, permute_cards
from .model import ComboStrategy, Node, Strategy
from .rules import Finding, Tolerances, strategy_differences


@dataclass(frozen=True)
class IsoResult:
    perm: Perm | None
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        return self.perm is not None and not self.findings


def _usable_combos(node: Node, board: set[int]) -> dict[frozenset[int], ComboStrategy]:
    out = {}
    for c in node.combos:
        try:
            cards = parse_combo(c.combo)
        except ValueError:
            continue
        if not set(cards) & board:
            out.setdefault(combo_key(cards), c)
    return out


def _compare(a: Strategy, b: Strategy, perm: Perm, tol: Tolerances) -> list[Finding]:
    findings: list[Finding] = []
    sa, sb = a.spot, b.spot
    for name in ("game", "positions", "pot", "effective_stack", "preflop"):
        va, vb = getattr(sa, name), getattr(sb, name)
        if va != vb:
            findings.append(Finding("iso-spot", f"spot.{name}", f"{va!r} vs {vb!r}"))
    board_a, board_b = set(parse_board(sa.board)), set(parse_board(sb.board))
    for i, na in enumerate(a.nodes):
        j = b.node_index(na.path)
        if j is None:
            findings.append(Finding("iso-tree", f"nodes[{i}]", f"path {list(na.path)} is missing from the second file"))
            continue
        nb = b.nodes[j]
        if (na.player, na.pot, na.effective_stack, na.to_call, na.actions) != (
            nb.player,
            nb.pot,
            nb.effective_stack,
            nb.to_call,
            nb.actions,
        ):
            findings.append(Finding("iso-tree", f"nodes[{i}]", f"node {list(na.path)} differs apart from the combos"))
            continue
        combos_b = _usable_combos(nb, board_b)
        for key, ca in _usable_combos(na, board_a).items():
            image = combo_key(permute_cards(sorted(key), perm))
            cb = combos_b.pop(image, None)
            where = f"nodes[{i}].combos[{ca.combo!r}]"
            if cb is None:
                if ca.weight > tol.weight:
                    findings.append(
                        Finding("iso-combo", where, f"maps to {combo_text(sorted(image))}, which the second file lacks")
                    )
                continue
            diffs = strategy_differences(ca, cb, tol)
            if diffs:
                findings.append(Finding("iso-combo", where, f"vs {cb.combo!r}: " + "; ".join(diffs[:3])))
        for cb in combos_b.values():
            if cb.weight > tol.weight:
                findings.append(
                    Finding("iso-combo", f"nodes[{i}] (second file)", f"{cb.combo!r} has no counterpart in the first file")
                )
    for j, nb in enumerate(b.nodes):
        if a.node(nb.path) is None:
            findings.append(Finding("iso-tree", f"nodes[{j}] (second file)", f"path {list(nb.path)} is missing from the first file"))
    return findings


def compare_isomorphic(a: Strategy, b: Strategy, tol: Tolerances = Tolerances()) -> IsoResult:
    """Finds the suit relabelling from a's board to b's board and checks that
    it maps a's strategy onto b's. When several relabellings exist (the
    board has symmetries of its own), the one with the fewest findings is
    reported."""
    perms = board_isomorphisms(parse_board(a.spot.board), parse_board(b.spot.board))
    if not perms:
        return IsoResult(
            None,
            [Finding("iso-board", "spot.board", f"no suit relabelling turns {a.spot.board} into {b.spot.board}")],
        )
    results = [IsoResult(p, _compare(a, b, p, tol)) for p in perms]
    return min(results, key=lambda r: len(r.findings))


def relabel(strategy: Strategy, perm: Perm) -> Strategy:
    """The same strategy with suits renamed: the board, every combo, and
    nothing else."""
    board = "".join(card_str(c) for c in permute_cards(parse_board(strategy.spot.board), perm))
    nodes = []
    for node in strategy.nodes:
        combos = []
        for c in node.combos:
            try:
                cards = parse_combo(c.combo)
            except ValueError:
                combos.append(c)
                continue
            combos.append(replace(c, combo="".join(card_str(x) for x in permute_cards(cards, perm))))
        nodes.append(replace(node, combos=tuple(combos)))
    meta = dict(strategy.meta)
    if meta:
        meta["relabelled"] = perm_text(perm)
    return replace(strategy, spot=replace(strategy.spot, board=board), nodes=tuple(nodes), meta=meta)

