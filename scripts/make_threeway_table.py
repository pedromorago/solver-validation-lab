"""Builds src/svlab/data/threeway_equity.bin with tools/threeway_equity.c.

The three-way table is estimated by Monte Carlo (8,000 deals for each of the
818,805 triples of classes, with a fixed seed, so every build gives the same
file), and its card-removal weights are exact. The file is written only if it
passes the exact invariants of tests/test_threeway_table.py.

    python scripts/make_threeway_table.py [--samples 8000] [--seed 1]

Needs gcc. Takes about ten minutes on four cores.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from svlab import threeway
from svlab.preflop import load_table

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tools" / "threeway_equity.c"


def invariant_failures(table: threeway.ThreewayTable) -> list[str]:
    failures = []
    pairs = np.array(load_table().pairs, dtype=float)
    if not np.array_equal(table.weights.sum(axis=2), pairs * 1128):
        failures.append("weights don't follow from the heads-up table")
    rows = table.rows
    dealt = rows[:, 0] > 0
    if not np.all(rows[dealt, 1:4].sum(axis=1) == 6 * table.samples):
        failures.append("three-way units don't add up to the pot")
    if np.any(rows[~dealt, 1:] != 0):
        failures.append("a triple that can't be dealt has samples")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if shutil.which("gcc") is None:
        sys.exit("gcc is required")

    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "threeway_equity"
        subprocess.run(["gcc", "-O3", "-fopenmp", "-o", str(binary), str(SOURCE)], check=True)
        out = Path(tmp) / "table.bin"
        subprocess.run([str(binary), "table", str(out), str(args.samples), str(args.seed)], check=True)
        data = out.read_bytes()

    failures = invariant_failures(threeway.parse(data))
    if failures:
        print("\n".join(failures))
        return 1
    target = threeway.path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
