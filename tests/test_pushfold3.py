"""The three-handed push/fold solver.

- Exactness: NashConv, the total that best responses could gain, is close to zero.
- Differential: replacing one player's strategy by its best response, and
  re-evaluating the whole game, gains exactly what the per-decision regrets say.
- Reduction: when the button folds everything, the blinds are playing the
  heads-up game, so the heads-up equilibrium must still be one.
- Known answers: AA never folds; on coin flips the blinds shove and call everything.
- Invariant: the three players' values add up to zero.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from svlab import pushfold, threeway
from svlab.preflop import CLASSES, INDEX
from svlab.pushfold3 import ACTING, DECISIONS, Game3, counterfactual, nash_conv, player_values, regrets, solve

if not threeway.path().exists():
    # The table is generated, not committed (scripts/make_threeway_table.py). CI builds it,
    # so a missing table fails there instead of skipping.
    if os.environ.get("CI"):
        raise RuntimeError("the three-way equity table was not built")
    pytest.skip("three-way equity table not built", allow_module_level=True)

STACK = 10.0
TOL = 1e-5


@pytest.fixture(scope="module")
def game():
    return Game3.from_table(STACK)


@pytest.fixture(scope="module")
def solution(game):
    return solve(game, tolerance=TOL)


def test_solution_is_an_equilibrium(game, solution):
    assert solution.nash_conv <= TOL
    assert nash_conv(game, solution.strategy) == pytest.approx(solution.nash_conv)


@pytest.mark.parametrize("player", ["BTN", "SB", "BB"])
def test_best_response_gains_match_the_regrets(game, solution, player):
    st = solution.strategy
    cf = counterfactual(game, st)
    best = dict(st)
    for d in DECISIONS:
        if ACTING[d] == player:
            aggressive, fold, _ = cf[d]
            best[d] = (aggressive > fold).astype(float)
    gain = player_values(game, best)[player] - player_values(game, st)[player]
    expected = sum(r.sum() for d, r in regrets(game, st).items() if ACTING[d] == player)
    assert gain == pytest.approx(expected, abs=1e-7)


def test_values_add_up_to_zero(game):
    rng = np.random.default_rng(5)
    st = {d: rng.random(len(CLASSES)) for d in DECISIONS}
    assert sum(player_values(game, st).values()) == pytest.approx(0, abs=1e-6)


def test_aces_never_fold(solution):
    aa = INDEX["AA"]
    for d in DECISIONS:
        assert solution.strategy[d][aa] > 0.999, d


def test_reduces_to_heads_up_when_the_button_folds(game):
    hu = pushfold.solve(pushfold.Game.from_table(STACK), tolerance=1e-7)
    st = {d: np.zeros(len(CLASSES)) for d in DECISIONS}
    st["sb_open"], st["bb_vs_sb"] = hu.shove, hu.call
    values = player_values(game, st)
    # Same game, with the button's hand as dead cards and three-way equities estimated by
    # Monte Carlo: the values agree up to that noise.
    assert values["SB"] == pytest.approx(hu.sb_value, abs=2e-3)
    blind_regret = sum(r.sum() for d, r in regrets(game, st).items() if d in ("sb_open", "bb_vs_sb"))
    assert blind_regret < 2e-3


def test_coin_flips():
    real = threeway.load()
    flips = threeway.ThreewayTable(
        real.samples, real.seed, real.weights,
        np.where(real.weights > 0, 1 / 3, 0.0), np.where(real.weights > 0, 0.5, 0.0), real.rows,
    )
    game = Game3.from_table(STACK, flips)
    sol = solve(game, tolerance=1e-6, max_iterations=500)
    # Calling a coin flip is worth 0 or more against folding's -0.5 or -1, so the blinds
    # continue at every decision they reach; the button is then indifferent, and any
    # frequency of shoving is fine (a decision it never leaves to the blinds can stay anything).
    cf = counterfactual(game, sol.strategy)
    for d in DECISIONS[1:]:
        reached = cf[d][2] > 1e-9
        assert np.all(sol.strategy[d][reached] > 0.999), d
    assert sol.values["BTN"] == pytest.approx(0, abs=1e-6)
    assert sol.nash_conv <= 1e-6


def test_rejects_a_bad_stack():
    with pytest.raises(ValueError):
        Game3.from_table(-1)
