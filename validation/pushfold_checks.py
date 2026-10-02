"""Checks for the push/fold solver, each named after the oracle it relies on.

Every check calls the solver through the `svlab.pushfold` module, so a seeded
bug patched into that module (see `pushfold_mutants.py`) is what the checks see.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np

from svlab import preflop, pushfold
from svlab.cards import FULL_DECK

STACK = 10.0
TOL = 1e-6


@dataclass(frozen=True)
class Check:
    name: str
    oracle: str
    run: Callable[[], None]


@cache
def _real_table() -> preflop.PreflopTable:
    return preflop.load_table()


def _solve(stack: float = STACK, table: preflop.PreflopTable | None = None):
    game = pushfold.Game.from_table(stack, table or _real_table())
    return game, pushfold.solve(game, tolerance=TOL)


def nash_gap_closes() -> None:
    game, sol = _solve()
    gap = pushfold.nash_gap(game, sol.shove, sol.call)
    assert gap <= TOL, f"Nash gap {gap:.2e} bb per hand"


def regret_failures(game, sol) -> list[str]:
    """Per-class regrets, weighted by how often each class reaches its decision, must be
    non-negative and add up to the Nash gap: the per-class EVs and the best responses
    are two computations of the same thing."""
    v = sol.values
    sb_prior = game.weights.sum(axis=1)
    sb_mixed = sol.shove * v.sb_shove + (1 - sol.shove) * pushfold.SB_FOLD
    sb_regret = sb_prior * (np.maximum(v.sb_shove, pushfold.SB_FOLD) - sb_mixed)
    reach = (game.weights * sol.shove[:, None]).sum(axis=0)
    bb_call = np.nan_to_num(v.bb_call, nan=pushfold.BB_FOLD)
    bb_mixed = sol.call * bb_call + (1 - sol.call) * pushfold.BB_FOLD
    bb_regret = reach * (np.maximum(bb_call, pushfold.BB_FOLD) - bb_mixed)
    failures = []
    if (sb_regret < -1e-12).any() or (bb_regret < -1e-12).any():
        failures.append("a negative regret")
    total = float(sb_regret.sum() + bb_regret.sum())
    if abs(total - sol.nash_gap) > 1e-9:
        failures.append(f"regrets add up to {total:.3e}, the reported Nash gap is {sol.nash_gap:.3e}")
    return failures


def regrets_add_up_to_the_gap() -> None:
    game, sol = _solve()
    failures = regret_failures(game, sol)
    assert not failures, "; ".join(failures)


def fictitious_play_agrees() -> None:
    game, sol = _solve()
    shove, call = pushfold.fictitious_play(game, iterations=3_000)
    other = pushfold.sb_value(game, shove, call)
    assert abs(other - sol.sb_value) < 0.01, f"CFR+ {sol.sb_value:.4f}, fictitious play {other:.4f}"


def coin_flips_are_shoved_and_called() -> None:
    real = _real_table()
    flips = preflop.PreflopTable(real.boards, real.pairs, tuple(tuple(p * real.boards for p in row) for row in real.pairs))
    _, sol = _solve(table=flips)
    assert np.all(sol.shove > 0.999) and np.all(sol.call > 0.999), "a coin flip is folded"
    assert abs(sol.sb_value) < 1e-3, f"SB value {sol.sb_value:.4f} on coin flips"


def aces_shove_and_call() -> None:
    for stack in (3.0, 10.0, 20.0):
        _, sol = _solve(stack)
        aa = preflop.INDEX["AA"]
        assert sol.shove[aa] > 0.999 and sol.call[aa] > 0.999, f"AA folds at {stack} bb"


def zero_sum() -> None:
    game = pushfold.Game.from_table(STACK, _real_table())
    rng = np.random.default_rng(1)
    shove, call = rng.random(169), rng.random(169)
    v = pushfold.action_values(game, shove, call)
    w = game.weights
    reach = (w * shove[:, None]).sum(axis=0)
    bb = (w * (1 - shove)[:, None]).sum() * 0.5 + (reach * (call * v.bb_call + (1 - call) * -1.0)).sum()
    assert abs(pushfold.sb_value(game, shove, call) + bb) < 1e-9, "the players' values don't cancel"


@cache
def _dealt_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Class of every two-card hand, and which pairs of hands can be dealt together."""
    hands = [(a, b) for a in FULL_DECK for b in FULL_DECK if a > b]
    classes = np.array([preflop.class_of(a, b) for a, b in hands])
    cards = np.zeros((len(hands), 52), dtype=bool)
    for k, (a, b) in enumerate(hands):
        cards[k, [a, b]] = True
    disjoint = ~(cards.astype(np.int8) @ cards.T.astype(np.int8)).astype(bool)
    return classes, disjoint


def value_over_dealt_hands() -> None:
    """The SB's EV computed from scratch: every pair of hands that can be dealt, the rules of
    the game written out again, and only the table's equities shared with the solver."""
    table = _real_table()
    game, sol = _solve(table=table)
    classes, disjoint = _dealt_pairs()
    equity = np.array(table.units, dtype=float) / (np.array(table.pairs, dtype=float) * table.boards * 2)
    s, c = sol.shove[classes][:, None], sol.call[classes][None, :]
    eq = equity[np.ix_(classes, classes)]
    payoff = (1 - s) * -0.5 + s * ((1 - c) * 1.0 + c * STACK * (2 * eq - 1))
    expected = float((payoff * disjoint).sum() / disjoint.sum())
    assert abs(expected - sol.sb_value) < 1e-9, f"solver says {sol.sb_value:.6f}, dealt hands say {expected:.6f}"


CHECKS = [
    Check("Nash gap closes", "exactness", nash_gap_closes),
    Check("regrets add up to the gap", "invariant", regrets_add_up_to_the_gap),
    Check("fictitious play agrees", "differential", fictitious_play_agrees),
    Check("value over dealt hands", "differential", value_over_dealt_hands),
    Check("coin flips shoved and called", "reference", coin_flips_are_shoved_and_called),
    Check("AA shoves and calls", "reference", aces_shove_and_call),
    Check("zero-sum", "invariant", zero_sum),
]
