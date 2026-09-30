"""Hand combos, boards and suit relabelling for strategy files.

A combo is two distinct cards, written as four characters such as "AhKh".
The same combo can be written in either card order ("KhAh"), so anything
that compares combos goes through `combo_key`.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

from svlab.cards import RANKS, SUITS, card_str, parse, parse_many, rank, suit

Perm = tuple[int, int, int, int]  # perm[old_suit] = new_suit
IDENTITY: Perm = (0, 1, 2, 3)


def parse_combo(text: str) -> tuple[int, int]:
    """Parses "AhKh" into two cards. Raises ValueError for anything else."""
    if not isinstance(text, str) or len(text) != 4:
        raise ValueError(f"Not a combo: {text!r} (expected four characters such as 'AhKh')")
    a, b = parse(text[:2]), parse(text[2:])
    if a == b:
        raise ValueError(f"Not a combo: {text!r} uses the same card twice")
    return a, b


def combo_key(cards: Sequence[int]) -> frozenset[int]:
    """Identity of a combo regardless of card order."""
    return frozenset(cards)


def combo_text(cards: Sequence[int]) -> str:
    """Canonical spelling: higher rank first, then suit order c, d, h, s."""
    hi, lo = sorted(cards, reverse=True)
    return card_str(hi) + card_str(lo)


def parse_board(text: str) -> list[int]:
    """Parses a board of three to five distinct cards ("Ks7s2d" or "Ks 7s 2d")."""
    cards = parse_many(text)
    if not 3 <= len(cards) <= 5:
        raise ValueError(f"A board has three to five cards, got {len(cards)} in {text!r}")
    return cards


def board_text(cards: Sequence[int]) -> str:
    return "".join(card_str(c) for c in cards)


def permute_card(card: int, perm: Perm) -> int:
    return rank(card) * 4 + perm[suit(card)]


def permute_cards(cards: Sequence[int], perm: Perm) -> list[int]:
    return [permute_card(c, perm) for c in cards]


def perm_text(perm: Perm) -> str:
    """Readable form of a suit relabelling, e.g. "h->c, c->h"."""
    moved = [f"{SUITS[s]}->{SUITS[t]}" for s, t in enumerate(perm) if s != t]
    return ", ".join(moved) or "identity"


def _board_maps(src: Sequence[int], dst: Sequence[int], perm: Perm) -> bool:
    """Does `perm` send board `src` to board `dst`? The flop is a set, the
    turn and river are compared card by card because they came later."""
    if len(src) != len(dst):
        return False
    moved = permute_cards(src, perm)
    return set(moved[:3]) == set(dst[:3]) and moved[3:] == list(dst[3:])


def board_isomorphisms(src: Sequence[int], dst: Sequence[int]) -> list[Perm]:
    """Every suit relabelling that turns board `src` into board `dst`."""
    return [p for p in permutations(range(4)) if _board_maps(src, dst, p)]  # type: ignore[misc]


def board_symmetries(board: Sequence[int]) -> list[Perm]:
    """Suit relabellings, other than the identity, that leave the board as it is.

    Two suits can be swapped when they play the same role on the board, for
    example both absent from a two-tone flop, or each holding one card of a
    pair. A rainbow flop with three different ranks has no such symmetry."""
    return [p for p in board_isomorphisms(board, board) if p != IDENTITY]


def all_combos() -> list[tuple[int, int]]:
    return [(a, b) for a in range(52) for b in range(a)]


def hand_class_combos(label: str) -> list[tuple[int, int]]:
    """Expands a hand class such as "AKs", "AKo", "AK" or "77" into combos."""
    r1, r2 = RANKS.index(label[0]), RANKS.index(label[1])
    kind = label[2:] if len(label) > 2 else ""
    out = []
    for s1 in range(4):
        for s2 in range(4):
            a, b = r1 * 4 + s1, r2 * 4 + s2
            if r1 == r2:
                if s1 < s2:
                    out.append((b, a))
                continue
            if kind == "s" and s1 != s2 or kind == "o" and s1 == s2:
                continue
            out.append((a, b))
    return out
