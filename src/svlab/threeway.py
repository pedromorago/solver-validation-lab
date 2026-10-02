"""Three-handed preflop all-in equities for every triple of hand classes.

Built by ``tools/threeway_equity.c`` (see ``scripts/make_threeway_table.py``):
an exact card-removal weight for each triple, and Monte Carlo estimates of the
three-way equities and of the three heads-up equities with the third hand's
cards dead. The file has one row per unordered triple; `load` expands it into
tensors indexed by ordered triples of classes, one axis per seat.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np

from .preflop import CLASSES

MAGIC = b"SVLAB3W1"
N = len(CLASSES)


@dataclass(frozen=True)
class ThreewayTable:
    samples: int
    seed: int
    weights: np.ndarray  # weights[i, j, k]: ways to deal classes i, j, k to three seats with no shared card
    three: np.ndarray  # three[i, j, k]: equity of i in a three-way all-in against j and k
    heads_up: np.ndarray  # heads_up[i, j, k]: equity of i against j, with k's cards dead
    rows: np.ndarray  # the raw rows, for checks on the file itself


def triples() -> np.ndarray:
    """The unordered triples a <= b <= c in file order."""
    return np.array([(a, b, c) for a in range(N) for b in range(a, N) for c in range(b, N)], dtype=np.int64)


def parse(data: bytes) -> ThreewayTable:
    if data[:8] != MAGIC:
        raise ValueError("Not a three-way equity table")
    samples, seed = np.frombuffer(data[8:16], dtype="<u4")
    rows = np.frombuffer(data[16:], dtype="<u2").reshape(-1, 7).astype(np.int64)
    idx = triples()
    if len(rows) != len(idx):
        raise ValueError(f"Expected {len(idx)} rows, got {len(rows)}")

    weights = np.zeros((N, N, N))
    three_sum = np.zeros((N, N, N))
    hu_sum = np.zeros((N, N, N))
    count = np.zeros((N, N, N))
    dealt = rows[:, 0] > 0
    three = rows[:, 1:4] / (6.0 * samples)
    two = rows[:, 4:7] / (2.0 * samples)
    # Heads-up equity of seat p against seat q in a row, with the third seat dead.
    pair_col = {(0, 1): (0, False), (1, 0): (0, True), (0, 2): (1, False), (2, 0): (1, True), (1, 2): (2, False), (2, 1): (2, True)}
    for perm in itertools.permutations(range(3)):
        x, y, z = idx[:, perm[0]], idx[:, perm[1]], idx[:, perm[2]]
        weights[x, y, z] = rows[:, 0]
        col, flip = pair_col[(perm[0], perm[1])]
        hu = 1 - two[:, col] if flip else two[:, col]
        np.add.at(three_sum, (x[dealt], y[dealt], z[dealt]), three[dealt, perm[0]])
        np.add.at(hu_sum, (x[dealt], y[dealt], z[dealt]), hu[dealt])
        np.add.at(count, (x[dealt], y[dealt], z[dealt]), 1)
    # When two seats hold the same class, every permutation that maps onto a cell is
    # an estimate of it; their mean keeps the seats' equities adding up exactly.
    with np.errstate(invalid="ignore"):
        three_eq = np.where(count > 0, three_sum / np.maximum(count, 1), 0.0)
        hu_eq = np.where(count > 0, hu_sum / np.maximum(count, 1), 0.0)
    return ThreewayTable(int(samples), int(seed), weights, three_eq, hu_eq, rows)


DATA = "data/threeway_equity.bin"


def path() -> Path:
    return Path(str(resources.files("svlab").joinpath(DATA)))


@cache
def load() -> ThreewayTable:
    """The table built by scripts/make_threeway_table.py."""
    p = path()
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; build it with: python scripts/make_threeway_table.py")
    return parse(p.read_bytes())
