"""All-in equity: the share of the pot each hand wins over every runout.

Two ways to compute it:
- `exact_equity` enumerates every remaining board. It is exact, and slow
  preflop, so it refuses more than `max_runouts` boards.
- `monte_carlo_equity` samples boards and reports a standard error with
  each estimate.

Equity is accumulated in integer "units" so that exactness can be tested:
each runout is worth `lcm(1..players)` units, split evenly among the hands
that tie for the best result.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

from .cards import FULL_DECK
from .evaluator import evaluate as default_evaluate

Evaluator = Callable[[Sequence[int]], tuple[int, ...]]
Hand = Sequence[int]


@dataclass(frozen=True)
class EquityResult:
    equities: tuple[float, ...]
    runouts: int
    # Exact results only: integer shares of the pot per player, out of
    # runouts * units_per_runout.
    units: tuple[int, ...] | None = None
    units_per_runout: int | None = None
    # Monte Carlo results only: the standard error of each estimate.
    std_errors: tuple[float, ...] | None = None


def _validate(hands: Sequence[Hand], board: Sequence[int], dead: Sequence[int]) -> None:
    if len(hands) < 2:
        raise ValueError("Need at least two hands")
    if any(len(h) != 2 for h in hands):
        raise ValueError("Each hand needs exactly two cards")
    if len(board) > 5:
        raise ValueError("A board has at most five cards")
    used = [c for h in hands for c in h] + list(board) + list(dead)
    if len(set(used)) != len(used):
        raise ValueError("A card appears twice")


def remaining_deck(hands: Sequence[Hand], board: Sequence[int], dead: Sequence[int] = ()) -> list[int]:
    used = {c for h in hands for c in h} | set(board) | set(dead)
    return [c for c in FULL_DECK if c not in used]


def _score(hands: Sequence[Hand], full_board: Sequence[int], evaluate: Evaluator) -> list[int]:
    """Indices of the hands that share the pot on a complete board."""
    values = [evaluate([*h, *full_board]) for h in hands]
    best = max(values)
    return [i for i, v in enumerate(values) if v == best]


def exact_equity(
    hands: Sequence[Hand],
    board: Sequence[int] = (),
    dead: Sequence[int] = (),
    *,
    evaluate: Evaluator = default_evaluate,
    max_runouts: int = 200_000,
) -> EquityResult:
    _validate(hands, board, dead)
    deck = remaining_deck(hands, board, dead)
    missing = 5 - len(board)
    runouts = math.comb(len(deck), missing)
    if runouts > max_runouts:
        raise ValueError(f"{runouts} runouts to enumerate; use monte_carlo_equity or raise max_runouts")

    unit = math.lcm(*range(1, len(hands) + 1))
    units = [0] * len(hands)
    for extra in combinations(deck, missing):
        winners = _score(hands, [*board, *extra], evaluate)
        share = unit // len(winners)
        for i in winners:
            units[i] += share

    total = runouts * unit
    return EquityResult(
        equities=tuple(u / total for u in units),
        runouts=runouts,
        units=tuple(units),
        units_per_runout=unit,
    )


def sample_runout(deck: Sequence[int], missing: int, rng: random.Random) -> list[int]:
    """Draws the missing board cards without replacement."""
    return rng.sample(list(deck), missing)


def monte_carlo_equity(
    hands: Sequence[Hand],
    board: Sequence[int] = (),
    dead: Sequence[int] = (),
    *,
    iterations: int = 20_000,
    seed: int = 0,
    evaluate: Evaluator = default_evaluate,
    sampler: Callable[[Sequence[int], int, random.Random], list[int]] = sample_runout,
) -> EquityResult:
    _validate(hands, board, dead)
    rng = random.Random(seed)
    deck = remaining_deck(hands, board, dead)
    missing = 5 - len(board)

    sums = [0.0] * len(hands)
    squares = [0.0] * len(hands)
    for _ in range(iterations):
        extra = sampler(deck, missing, rng)
        winners = _score(hands, [*board, *extra], evaluate)
        share = 1 / len(winners)
        for i in winners:
            sums[i] += share
            squares[i] += share * share

    means = [s / iterations for s in sums]
    # Standard error of the mean of each player's per-runout share.
    errors = [math.sqrt(max(sq / iterations - m * m, 0.0) / iterations) for sq, m in zip(squares, means)]
    return EquityResult(equities=tuple(means), runouts=iterations, std_errors=tuple(errors))
