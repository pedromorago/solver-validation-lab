"""The preflop equity table: its classes, its invariants, and three oracles for its numbers.

- Invariants that hold exactly, as integers: the pair counts follow from card
  removal, and the units of i against j and of j against i add up to the pot.
- Reference values beyond argument (AA against KK).
- Differential: the C program that built the table agrees exactly with the
  Python engine on random flops (when gcc is available).
- Statistical: the table falls within the error bars of a Monte Carlo
  estimate made with the Python engine.
"""

from __future__ import annotations

import gzip
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from svlab import exact_equity
from svlab.cards import card_str
from svlab.evaluator import evaluate
from svlab.preflop import CLASSES, INDEX, PreflopTable, class_of, combos, load_table, parse_table

ROOT = Path(__file__).resolve().parent.parent
BOARDS = math.comb(48, 5)


def table_invariant_failures(table: PreflopTable) -> list[str]:
    """Everything the table must satisfy whatever the true equities are."""
    failures = []
    if table.boards != BOARDS:
        failures.append(f"boards is {table.boards}, expected {BOARDS}")
    for i, a in enumerate(CLASSES):
        if sum(table.pairs[i]) != len(combos(a)) * 1225:
            failures.append(f"{a}: {sum(table.pairs[i])} pairs in its row, expected {len(combos(a)) * 1225}")
        for j, b in enumerate(CLASSES):
            pairs = table.pairs[i][j]
            expected = sum(1 for x in combos(a) for y in combos(b) if not set(x) & set(y))
            if pairs != expected:
                failures.append(f"{a} vs {b}: {pairs} pairs, expected {expected}")
            if table.units[i][j] + table.units[j][i] != pairs * table.boards * 2:
                failures.append(f"{a} vs {b}: units of both sides don't add up to the pot")
    return failures


@pytest.fixture(scope="module")
def table() -> PreflopTable:
    return load_table()


def test_classes():
    assert len(CLASSES) == len(set(CLASSES)) == 169
    assert sum(len(combos(c)) for c in CLASSES) == 1326
    assert {len(combos(c)) for c in CLASSES} == {4, 6, 12}


def test_class_of_matches_combos():
    for name in CLASSES:
        for c1, c2 in combos(name):
            assert CLASSES[class_of(c1, c2)] == name
            assert class_of(c2, c1) == class_of(c1, c2)


def test_table_invariants(table):
    assert table_invariant_failures(table) == []


def test_identical_classes_split_exactly(table):
    for i in range(len(CLASSES)):
        assert table.equity(i, i) == 0.5


def test_aces_against_kings(table):
    # AA against KK is the textbook 82% to 18%; every suit combination is close to it.
    assert 0.81 < table.equity(INDEX["AA"], INDEX["KK"]) < 0.83


def test_parser_rejects_a_missing_row():
    text = gzip.decompress((ROOT / "src/svlab/data/preflop_equity.txt.gz").read_bytes()).decode("ascii")
    with pytest.raises(ValueError, match="Expected"):
        parse_table("\n".join(text.splitlines()[:-1]))


def _sample_class_equity(i: int, j: int, iterations: int, rng: random.Random) -> tuple[float, float]:
    """Monte Carlo over random hand pairs from the two classes and random boards."""
    pairs = [(x, y) for x in combos(CLASSES[i]) for y in combos(CLASSES[j]) if not set(x) & set(y)]
    total = squares = 0.0
    for _ in range(iterations):
        a, b = rng.choice(pairs)
        deck = [c for c in range(52) if c not in (*a, *b)]
        board = rng.sample(deck, 5)
        va, vb = evaluate([*a, *board]), evaluate([*b, *board])
        share = 1.0 if va > vb else 0.0 if va < vb else 0.5
        total += share
        squares += share * share
    mean = total / iterations
    return mean, math.sqrt(max(squares / iterations - mean * mean, 0) / iterations)


def test_table_within_monte_carlo_error_bars(table):
    rng = random.Random(7)
    for _ in range(6):
        i, j = rng.randrange(169), rng.randrange(169)
        mean, se = _sample_class_equity(i, j, 3_000, rng)
        assert abs(table.equity(i, j) - mean) <= 4.5 * se, (CLASSES[i], CLASSES[j], table.equity(i, j), mean, se)


@pytest.fixture(scope="module")
def c_program(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    binary = tmp_path_factory.mktemp("c") / "preflop_equity"
    subprocess.run(["gcc", "-O2", "-o", str(binary), str(ROOT / "tools/preflop_equity.c")], check=True)
    return binary


def test_c_program_matches_python_exactly(c_program):
    rng = random.Random(11)
    for _ in range(40):
        cards = rng.sample(range(52), 7)
        a, b, board = cards[:2], cards[2:4], cards[4:]
        out = subprocess.run(
            [str(c_program), "matchup", "".join(map(card_str, a)), "".join(map(card_str, b)), "".join(map(card_str, board))],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        py = exact_equity([a, b], board)
        assert tuple(map(int, out)) == (py.runouts, *py.units)
