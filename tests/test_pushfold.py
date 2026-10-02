"""The push/fold solver, checked against the one oracle a solved game has: best responses.

- Exactness: the Nash gap is (close to) zero, and every action a class plays
  is a best response for it.
- Differential: fictitious play, a different algorithm, reaches the same
  game value.
- Known answers: on a table where every matchup is a coin flip, the
  equilibrium follows by hand; AA always shoves and always calls.
- Invariant: the game is zero-sum.
"""

from __future__ import annotations

import numpy as np
import pytest

from svlab.preflop import CLASSES, INDEX, PreflopTable, load_table
from svlab.pushfold import BB_FOLD, Game, action_values, fictitious_play, nash_gap, sb_value, solve
from validation.pushfold_checks import regret_failures

STACKS = [1.5, 4, 8, 12, 20]
TOL = 1e-6


@pytest.fixture(scope="module")
def solutions():
    return {s: (Game.from_table(s), solve(Game.from_table(s), tolerance=TOL)) for s in STACKS}


@pytest.mark.parametrize("stack", STACKS)
def test_solution_is_an_equilibrium(solutions, stack):
    game, sol = solutions[stack]
    assert sol.nash_gap <= TOL
    assert nash_gap(game, sol.shove, sol.call) == pytest.approx(sol.nash_gap)


@pytest.mark.parametrize("stack", STACKS)
def test_regrets_add_up_to_the_gap(solutions, stack):
    # What each class could gain by switching to its best action, weighted by how
    # often it reaches its decision, is never negative and adds up to the Nash gap.
    # This ties the per-class EVs to the best-response computation.
    game, sol = solutions[stack]
    assert regret_failures(game, sol) == []


def test_fictitious_play_reaches_the_same_value(solutions):
    game, sol = solutions[12]
    shove, call = fictitious_play(game, iterations=4_000)
    assert nash_gap(game, shove, call) < 0.01
    assert sb_value(game, shove, call) == pytest.approx(sol.sb_value, abs=0.01)


@pytest.mark.parametrize("stack", STACKS)
def test_aces_always_shove_and_call(solutions, stack):
    _, sol = solutions[stack]
    assert sol.shove[INDEX["AA"]] > 0.999
    assert sol.call[INDEX["AA"]] > 0.999


def coin_flip_table() -> PreflopTable:
    real = load_table()
    units = tuple(tuple(p * real.boards for p in row) for row in real.pairs)
    return PreflopTable(real.boards, real.pairs, units)


def test_coin_flips_are_shoved_and_called():
    # Every called shove is worth exactly 0 to both players, so the SB prefers
    # shoving (0 or +1) to folding (-0.5) and the BB prefers calling (0) to folding (-1).
    game = Game.from_table(10, coin_flip_table())
    sol = solve(game, tolerance=TOL)
    assert np.all(sol.shove > 0.999) and np.all(sol.call > 0.999)
    assert sol.sb_value == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize("stack", [3, 15])
def test_zero_sum(stack):
    game = Game.from_table(stack)
    rng = np.random.default_rng(stack)
    shove, call = rng.random(len(CLASSES)), rng.random(len(CLASSES))
    v = action_values(game, shove, call)
    w = game.weights
    bb = (w * (1 - shove)[:, None]).sum() * 0.5
    reach = (w * shove[:, None]).sum(axis=0)
    bb += (reach * (call * v.bb_call + (1 - call) * BB_FOLD)).sum()
    assert sb_value(game, shove, call) == pytest.approx(-bb)


def test_the_gap_is_positive_away_from_equilibrium():
    game = Game.from_table(10)
    everything = np.ones(len(CLASSES))
    assert nash_gap(game, everything, everything) > 0.1


def test_rejects_a_bad_stack():
    with pytest.raises(ValueError):
        Game.from_table(0)
