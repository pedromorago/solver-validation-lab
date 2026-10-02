"""Heads-up push/fold: the small blind shoves or folds, the big blind calls or folds.

This is the one spot in this lab with a real oracle. The game is small
enough to solve, and a solution can be checked by computing best responses:
the Nash gap (how much two players who knew each other's strategy could gain
by deviating) is zero exactly at an equilibrium.

The model: blinds of 0.5 and 1, no antes, both players start with
``stack`` big blinds, and the only options are the ones above. EVs are in
big blinds, as the change in the player's stack from before the blinds.
Folding the small blind is worth -0.5, a shove that gets folded to is worth
+1, and a called shove is worth ``stack * (2 * equity - 1)``. Chips are
what count, which in a winner-take-all Spin & Go is also what the prize
depends on.

Hands are the 169 classes of `svlab.preflop`. How often class i meets class
j comes from the number of pairs of hands they can hold without sharing a
card, so card removal is part of the game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .preflop import CLASSES, PreflopTable, load_table

SB_FOLD = -0.5
BB_FOLD = -1.0


@dataclass(frozen=True)
class Game:
    stack: float
    weights: np.ndarray  # weights[i, j]: hand pairs with SB in class i and BB in class j
    called: np.ndarray  # called[i, j]: SB's EV when SB with i shoves and BB with j calls

    @classmethod
    def from_table(cls, stack: float, table: PreflopTable | None = None) -> "Game":
        if not stack > 0:
            raise ValueError("The stack must be positive")
        table = table or load_table()
        pairs = np.array(table.pairs, dtype=float)
        equity = np.array(table.units, dtype=float) / (pairs * table.boards * 2)
        return cls(stack, pairs / pairs.sum(), stack * (2 * equity - 1))


@dataclass(frozen=True)
class Values:
    """Each action's EV against the other player's strategy, per class."""

    sb_shove: np.ndarray
    sb_fold: np.ndarray
    bb_call: np.ndarray  # given that SB shoved; NaN where SB never shoves into that class
    bb_fold: np.ndarray


def action_values(game: Game, shove: np.ndarray, call: np.ndarray) -> Values:
    w = game.weights
    sb_prior = w.sum(axis=1)
    sb_shove = (w * ((1 - call) * 1.0 + call * game.called)).sum(axis=1) / sb_prior

    reach = (w * shove[:, None]).sum(axis=0)  # weight of each BB class facing a shove
    bb_called = (w * shove[:, None] * -game.called).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bb_call = np.where(reach > 0, bb_called / reach, np.nan)
    return Values(
        sb_shove=sb_shove,
        sb_fold=np.full(len(CLASSES), SB_FOLD),
        bb_call=bb_call,
        bb_fold=np.full(len(CLASSES), BB_FOLD),
    )


def sb_value(game: Game, shove: np.ndarray, call: np.ndarray) -> float:
    """The small blind's EV per hand dealt; the big blind's is its negative."""
    v = action_values(game, shove, call)
    sb_prior = game.weights.sum(axis=1)
    return float((sb_prior * (shove * v.sb_shove + (1 - shove) * SB_FOLD)).sum())


def best_response_values(game: Game, shove: np.ndarray, call: np.ndarray) -> tuple[float, float]:
    """What each player gets per hand by best-responding to the other's strategy."""
    v = action_values(game, shove, call)
    sb_prior = game.weights.sum(axis=1)
    sb_best = float((sb_prior * np.maximum(v.sb_shove, SB_FOLD)).sum())

    reach = (game.weights * shove[:, None]).sum(axis=0)
    folded_to_bb = float((game.weights * (1 - shove)[:, None]).sum()) * -SB_FOLD
    bb_best = folded_to_bb + float((reach * np.maximum(np.nan_to_num(v.bb_call, nan=BB_FOLD), BB_FOLD)).sum())
    return sb_best, bb_best


def nash_gap(game: Game, shove: np.ndarray, call: np.ndarray) -> float:
    """Total gain available to the two players from best-responding, in bb per hand. Zero at an equilibrium."""
    sb_best, bb_best = best_response_values(game, shove, call)
    return sb_best + bb_best  # the game is zero-sum, so the strategy's own values cancel


@dataclass(frozen=True)
class Solution:
    stack: float
    shove: np.ndarray  # SB shove frequency per class
    call: np.ndarray  # BB call frequency per class
    values: Values
    sb_value: float
    nash_gap: float
    iterations: int

    def shove_share(self, game: Game) -> float:
        """The fraction of dealt hands the small blind shoves."""
        return float(game.weights.sum(axis=1) @ self.shove)

    def call_share(self, game: Game) -> float:
        """The fraction of the big blind's hands that call a shove, over all hands it can be dealt."""
        return float(game.weights.sum(axis=0) @ self.call)

    def to_json(self) -> dict:
        def num(x: float) -> float | None:
            return None if np.isnan(x) else round(float(x), 6)

        return {
            "format": "svlab-pushfold/1",
            "stack": self.stack,
            "blinds": [0.5, 1.0],
            "units": "bb",
            "sb_value": round(self.sb_value, 6),
            "nash_gap": self.nash_gap,
            "iterations": self.iterations,
            "classes": {
                name: {
                    "shove": round(float(self.shove[i]), 6),
                    "call": round(float(self.call[i]), 6),
                    "ev_shove": num(self.values.sb_shove[i]),
                    "ev_call": num(self.values.bb_call[i]),
                }
                for i, name in enumerate(CLASSES)
            },
        }


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Probability of the first of two actions from their positive regrets; 0.5 when neither is positive."""
    pos = np.maximum(regrets, 0)
    total = pos.sum(axis=1)
    return np.where(total > 0, pos[:, 0] / np.where(total > 0, total, 1), 0.5)


def iterate(game: Game, max_iterations: int = 20_000) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """CFR+ with alternating updates. Yields the iteration number, the current
    strategies and the linearly weighted average ones; the average is what
    converges to the equilibrium."""
    n = len(CLASSES)
    sb_regret = np.zeros((n, 2))  # shove, fold
    bb_regret = np.zeros((n, 2))  # call, fold
    sb_sum = np.zeros(n)
    bb_sum = np.zeros(n)
    weight_sum = 0.0
    sb_prior = game.weights.sum(axis=1)

    for it in range(1, max_iterations + 1):
        shove = _regret_matching(sb_regret)
        v = action_values(game, shove, _regret_matching(bb_regret))
        node = shove * v.sb_shove + (1 - shove) * SB_FOLD
        sb_regret = np.maximum(sb_regret + sb_prior[:, None] * np.stack([v.sb_shove - node, SB_FOLD - node], axis=1), 0)
        shove = _regret_matching(sb_regret)

        call = _regret_matching(bb_regret)
        v = action_values(game, shove, call)
        reach = (game.weights * shove[:, None]).sum(axis=0)
        bb_call = np.nan_to_num(v.bb_call, nan=BB_FOLD)
        node = call * bb_call + (1 - call) * BB_FOLD
        bb_regret = np.maximum(bb_regret + reach[:, None] * np.stack([bb_call - node, BB_FOLD - node], axis=1), 0)
        call = _regret_matching(bb_regret)

        sb_sum += it * shove
        bb_sum += it * call
        weight_sum += it
        yield it, shove, call, sb_sum / weight_sum, bb_sum / weight_sum


def solution(game: Game, shove: np.ndarray, call: np.ndarray, iterations: int) -> Solution:
    return Solution(
        stack=game.stack,
        shove=shove,
        call=call,
        values=action_values(game, shove, call),
        sb_value=sb_value(game, shove, call),
        nash_gap=nash_gap(game, shove, call),
        iterations=iterations,
    )


def solve(game: Game, *, tolerance: float = 1e-6, max_iterations: int = 20_000, check_every: int = 50) -> Solution:
    """Runs CFR+ until the average strategies' Nash gap is at most `tolerance`."""
    for it, _, _, shove, call in iterate(game, max_iterations):
        if it % check_every == 0 and nash_gap(game, shove, call) <= tolerance:
            break
    return solution(game, shove, call, it)


def fictitious_play(game: Game, iterations: int = 5_000) -> tuple[np.ndarray, np.ndarray]:
    """A second, independent way to reach the equilibrium: each player best-responds
    to the other's average strategy so far. Slower, and used only to cross-check `solve`."""
    n = len(CLASSES)
    shove_sum, call_sum = np.zeros(n), np.zeros(n)
    shove, call = np.ones(n), np.ones(n)
    for t in range(1, iterations + 1):
        shove_sum += shove
        call_sum += call
        avg_shove, avg_call = shove_sum / t, call_sum / t
        v = action_values(game, avg_shove, avg_call)
        shove = (v.sb_shove > SB_FOLD).astype(float)
        call = (np.nan_to_num(v.bb_call, nan=BB_FOLD) > BB_FOLD).astype(float)
    return shove_sum / iterations, call_sum / iterations
