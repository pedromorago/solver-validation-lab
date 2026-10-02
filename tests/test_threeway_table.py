"""The three-way equity table: exact invariants, and its Monte Carlo estimates
checked against the exact heads-up table and against the Python engine."""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pytest

from svlab import threeway
from svlab.evaluator import evaluate
from svlab.preflop import CLASSES, INDEX, combos, load_table
from svlab.threeway import load

if not threeway.path().exists():
    # The table is generated, not committed (scripts/make_threeway_table.py). CI builds it,
    # so a missing table fails there instead of skipping.
    if os.environ.get("CI"):
        raise RuntimeError("the three-way equity table was not built")
    pytest.skip("three-way equity table not built", allow_module_level=True)

N = len(CLASSES)


@pytest.fixture(scope="module")
def table():
    return load()


@pytest.fixture(scope="module")
def dealt(table):
    return table.weights > 0


def test_weights_follow_from_the_heads_up_table(table):
    # Dealing a third hand to every heads-up pair: C(48, 2) = 1128 ways, exactly.
    pairs = np.array(load_table().pairs, dtype=float)
    assert np.array_equal(table.weights.sum(axis=2), pairs * 1128)


def test_weights_do_not_depend_on_seat_order(table):
    w = table.weights
    for axes in [(1, 0, 2), (0, 2, 1), (2, 1, 0)]:
        assert np.array_equal(w, w.transpose(axes))


def test_three_way_equities_add_up_to_one(table, dealt):
    q = table.three
    total = q + q.transpose(1, 0, 2) + q.transpose(2, 1, 0)  # seat i, seat j, seat k of the same deal
    assert np.allclose(total[dealt], 1.0, atol=1e-12)


def test_heads_up_equities_are_complementary(table, dealt):
    hu = table.heads_up
    assert np.allclose((hu + hu.transpose(1, 0, 2))[dealt], 1.0, atol=1e-12)


def test_heads_up_estimates_average_to_the_exact_table(table):
    # Averaging the equity of i against j over the third hand, weighted by how often it is
    # dealt, must give the exact heads-up equity: the third hand is just dead cards.
    exact = load_table()
    w, n = table.weights, table.samples
    estimate = (w * table.heads_up).sum(axis=2) / w.sum(axis=2)
    # Each cell is a mean of n deals, each worth 0, 1/2 or 1, so its variance is at most 1/(4n).
    se = np.sqrt((w**2).sum(axis=2) / (4 * n)) / w.sum(axis=2)
    truth = np.array([[exact.equity(i, j) for j in range(N)] for i in range(N)])
    z = np.abs(estimate - truth) / se
    assert z.max() < 5, (CLASSES[z.argmax() // N], CLASSES[z.argmax() % N], z.max())


def test_aces_kings_queens(table):
    aa, kk, qq = INDEX["AA"], INDEX["KK"], INDEX["QQ"]
    assert 0.65 < table.three[aa, kk, qq] < 0.69
    assert 0.16 < table.three[kk, aa, qq] < 0.19


def test_triple_aces_cannot_be_dealt(table):
    aa = INDEX["AA"]
    assert table.weights[aa, aa, aa] == 0


def test_three_way_cells_within_python_monte_carlo_error(table):
    rng = random.Random(3)
    for _ in range(4):
        i, j, k = (rng.randrange(N) for _ in range(3))
        if table.weights[i, j, k] == 0:
            continue
        deals = [(x, y, z) for x in combos(CLASSES[i]) for y in combos(CLASSES[j]) for z in combos(CLASSES[k])
                 if not (set(x) & set(y) or set(x) & set(z) or set(y) & set(z))]
        total = squares = 0.0
        iterations = 2_000
        for _ in range(iterations):
            x, y, z = rng.choice(deals)
            board = rng.sample([c for c in range(52) if c not in (*x, *y, *z)], 5)
            values = [evaluate([*h, *board]) for h in (x, y, z)]
            best = max(values)
            share = (values[0] == best) / values.count(best)
            total += share
            squares += share * share
        mean = total / iterations
        se_python = math.sqrt(max(squares / iterations - mean * mean, 1e-12) / iterations)
        se_table = math.sqrt(0.25 / table.samples)
        assert abs(table.three[i, j, k] - mean) <= 4.5 * math.hypot(se_python, se_table), (CLASSES[i], CLASSES[j], CLASSES[k])
