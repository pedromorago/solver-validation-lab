"""Best five-card hand out of five to seven cards.

`evaluate` returns a tuple that compares correctly with the built-in
operators: a bigger tuple is a better hand, equal tuples are a split.
"""

from collections import Counter
from enum import IntEnum
from typing import Iterable

from .cards import rank, suit


class HandCategory(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    TRIPS = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    QUADS = 7
    STRAIGHT_FLUSH = 8


WHEEL = frozenset({12, 0, 1, 2, 3})  # A-2-3-4-5, where the ace plays low


def straight_high(ranks: Iterable[int]) -> int | None:
    """The top rank of the best straight in `ranks`, or None. The wheel counts as a five-high straight."""
    present = set(ranks)
    for high in range(12, 3, -1):
        if all(r in present for r in range(high - 4, high + 1)):
            return high
    if WHEEL <= present:
        return 3
    return None


def evaluate(cards: Iterable[int]) -> tuple[int, ...]:
    cards = list(cards)
    if not 5 <= len(cards) <= 7:
        raise ValueError(f"Need 5 to 7 cards, got {len(cards)}")

    by_suit: dict[int, list[int]] = {}
    for c in cards:
        by_suit.setdefault(suit(c), []).append(rank(c))
    flush_ranks = next((sorted(rs, reverse=True) for rs in by_suit.values() if len(rs) >= 5), None)

    if flush_ranks is not None:
        high = straight_high(flush_ranks)
        if high is not None:
            return (HandCategory.STRAIGHT_FLUSH, high)

    counts = Counter(rank(c) for c in cards)
    # Ranks ordered by how many times they appear, then by rank.
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    ranks_desc = sorted(counts, reverse=True)

    top_rank, top_count = groups[0]
    if top_count == 4:
        kicker = max(r for r in ranks_desc if r != top_rank)
        return (HandCategory.QUADS, top_rank, kicker)

    if top_count == 3:
        pair_ranks = [r for r, n in groups[1:] if n >= 2]
        if pair_ranks:
            return (HandCategory.FULL_HOUSE, top_rank, max(pair_ranks))

    if flush_ranks is not None:
        return (HandCategory.FLUSH, *flush_ranks[:5])

    high = straight_high(ranks_desc)
    if high is not None:
        return (HandCategory.STRAIGHT, high)

    if top_count == 3:
        kickers = [r for r in ranks_desc if r != top_rank][:2]
        return (HandCategory.TRIPS, top_rank, *kickers)

    pairs = [r for r, n in groups if n == 2]
    if len(pairs) >= 2:
        high_pair, low_pair = pairs[0], pairs[1]
        kicker = max(r for r in ranks_desc if r not in (high_pair, low_pair))
        return (HandCategory.TWO_PAIR, high_pair, low_pair, kicker)

    if len(pairs) == 1:
        kickers = [r for r in ranks_desc if r != pairs[0]][:3]
        return (HandCategory.PAIR, pairs[0], *kickers)

    return (HandCategory.HIGH_CARD, *ranks_desc[:5])
