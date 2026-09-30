"""Deliberately broken versions of the engine, one bug each.

They answer the question a validation suite has to answer about itself:
if the engine were wrong in this particular way, would anything notice?
`scripts/run_mutants.py` runs every check against every mutant."""

from __future__ import annotations

import random
from itertools import combinations
from typing import Sequence

from svlab import equity
from svlab.cards import FULL_DECK, rank, suit
from svlab.evaluator import HandCategory, evaluate as real_evaluate

from .engine import Engine


def _flush_below_straight(cards):
    value = real_evaluate(cards)
    swap = {HandCategory.FLUSH: HandCategory.STRAIGHT, HandCategory.STRAIGHT: HandCategory.FLUSH}
    return (swap.get(value[0], value[0]), *value[1:])


def _no_wheel(cards):
    value = real_evaluate(cards)
    if value[0] in (HandCategory.STRAIGHT, HandCategory.STRAIGHT_FLUSH) and value[1] == 3:
        ranks = sorted({rank(c) for c in cards}, reverse=True)[:5]
        return (HandCategory.HIGH_CARD, *ranks)
    return value


def _spades_never_flush(cards):
    """Counts suits 0-2 only when looking for a flush."""
    value = real_evaluate(cards)
    if value[0] in (HandCategory.FLUSH, HandCategory.STRAIGHT_FLUSH):
        flush_suit = max(range(4), key=lambda s: sum(suit(c) == s for c in cards))
        if flush_suit == 3:
            ranks = sorted({rank(c) for c in cards}, reverse=True)[:5]
            return (HandCategory.HIGH_CARD, *ranks)
    return value


def _ignores_kickers(cards):
    value = real_evaluate(cards)
    if value[0] == HandCategory.PAIR:
        return value[:2]
    return value


def _order_dependent(cards):
    """Only looks at the first five cards it is given."""
    return real_evaluate(list(cards)[:5])


def _ties_go_to_first_player(hands, board=(), dead=(), *, evaluate=real_evaluate, max_runouts=200_000):
    result = equity.exact_equity(hands, board, dead, evaluate=evaluate, max_runouts=max_runouts)
    deck = equity.remaining_deck(hands, board, dead)
    unit = result.units_per_runout
    units = [0] * len(hands)
    for extra in combinations(deck, 5 - len(board)):
        values = [evaluate([*h, *board, *extra]) for h in hands]
        units[values.index(max(values))] += unit
    total = result.runouts * unit
    return equity.EquityResult(tuple(u / total for u in units), result.runouts, tuple(units), unit)


def _skips_last_card(hands, board=(), dead=(), *, evaluate=real_evaluate, max_runouts=200_000):
    """An off-by-one: the last card of the deck is never dealt."""
    deck = equity.remaining_deck(hands, board, dead)
    dead = [*dead, deck[-1]]
    return equity.exact_equity(hands, board, dead, evaluate=evaluate, max_runouts=max_runouts)


def _with_replacement(deck: Sequence[int], missing: int, rng: random.Random) -> list[int]:
    """Draws each board card independently, so a card can appear twice."""
    return [rng.choice(list(deck)) for _ in range(missing)]


def _full_deck_sampler(deck: Sequence[int], missing: int, rng: random.Random) -> list[int]:
    """Forgets to remove the players' cards from the deck."""
    return rng.sample(FULL_DECK, missing)


MUTANTS: list[tuple[Engine, str]] = [
    (Engine("flush ranked below straight", evaluate=_flush_below_straight), "evaluator"),
    (Engine("A-2-3-4-5 not a straight", evaluate=_no_wheel), "evaluator"),
    (Engine("spade flushes missed", evaluate=_spades_never_flush), "evaluator"),
    (Engine("pair kickers ignored", evaluate=_ignores_kickers), "evaluator"),
    (Engine("only the first five cards read", evaluate=_order_dependent), "evaluator"),
    (Engine("ties awarded to the first player", exact_impl=_ties_go_to_first_player), "exact equity"),
    (Engine("last card of the deck never dealt", exact_impl=_skips_last_card), "exact equity"),
    (Engine("board sampled with replacement", sampler=_with_replacement), "Monte Carlo"),
    (Engine("players' cards left in the deck", sampler=_full_deck_sampler), "Monte Carlo"),
]
