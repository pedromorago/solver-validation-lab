"""Deliberately broken versions of the push/fold solver, one bug each.

Each mutant patches `svlab.pushfold` for as long as its context is open, so
the checks in `pushfold_checks.py` run against it unchanged.
`scripts/run_pushfold_mutants.py` runs every check against every mutant."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from svlab import pushfold
from svlab.preflop import CLASSES, combos

_real = {name: getattr(pushfold, name) for name in ("Game", "action_values", "nash_gap", "solve", "SB_FOLD", "BB_FOLD")}


@dataclass(frozen=True)
class Mutant:
    name: str
    layer: str
    patches: Callable[[], dict]

    @contextmanager
    def applied(self) -> Iterator[None]:
        patches = self.patches()
        try:
            for name, value in patches.items():
                setattr(pushfold, name, value)
            yield
        finally:
            for name in patches:
                setattr(pushfold, name, _real[name])


def _game_with(weights_fn=None, equity_fn=None):
    class Broken(_real["Game"]):
        @classmethod
        def from_table(cls, stack, table=None):
            game = _real["Game"].from_table(stack, table)
            weights, called = game.weights, game.called
            if weights_fn:
                weights = weights_fn()
            if equity_fn:
                called = equity_fn(called)
            return cls(stack, weights, called)

    return Broken


def _no_card_removal():
    sizes = np.array([len(combos(c)) for c in CLASSES], dtype=float)
    w = np.outer(sizes, sizes)
    return {"Game": _game_with(weights_fn=lambda: w / w.sum())}


def _equity_transposed():
    return {"Game": _game_with(equity_fn=lambda called: called.T)}


def _blinds_swapped():
    return {"SB_FOLD": -1.0, "BB_FOLD": -0.5}


def _bb_uses_sb_equity():
    def action_values(game, shove, call):
        v = _real["action_values"](game, shove, call)
        w = game.weights
        reach = (w * shove[:, None]).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bb_call = np.where(reach > 0, (w * shove[:, None] * game.called).sum(axis=0) / reach, np.nan)
        return pushfold.Values(v.sb_shove, v.sb_fold, bb_call, v.bb_fold)

    return {"action_values": action_values}


def _gap_ignores_bb():
    def nash_gap(game, shove, call):
        sb_best, _ = pushfold.best_response_values(game, shove, call)
        return sb_best - pushfold.sb_value(game, shove, call)

    return {"nash_gap": nash_gap}


def _returns_last_iterate():
    """Reports the strategies of the final iteration instead of the average, a classic CFR slip."""

    def solve(game, *, tolerance=1e-6, max_iterations=20_000, check_every=50):
        for it, shove, call, avg_shove, avg_call in pushfold.iterate(game, max_iterations):
            if it % check_every == 0 and pushfold.nash_gap(game, avg_shove, avg_call) <= tolerance:
                break
        return pushfold.solution(game, shove, call, it)

    return {"solve": solve}


MUTANTS = [
    Mutant("Card removal ignored", "game", _no_card_removal),
    Mutant("Equity table read transposed", "game", _equity_transposed),
    Mutant("Blinds swapped", "game", _blinds_swapped),
    Mutant("BB scored with the SB's equity", "values", _bb_uses_sb_equity),
    Mutant("Nash gap leaves out the BB", "values", _gap_ignores_bb),
    Mutant("Last iterate reported instead of the average", "solver", _returns_last_iterate),
]
