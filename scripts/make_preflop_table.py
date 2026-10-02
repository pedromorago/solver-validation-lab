"""Builds src/svlab/data/preflop_equity.txt.gz with tools/preflop_equity.c.

The C program enumerates every board for every preflop matchup, which would
take days in pure Python. Before the table is written, the program is checked
against the Python engine: for random matchups on random flops, both must
report exactly the same units. The table's own invariants are then checked
(see tests/test_preflop_table.py), and the file is written only if all pass.

    python scripts/make_preflop_table.py [--checks 300] [--seed 0]

Needs gcc. Takes about half an hour on four cores.
"""

from __future__ import annotations

import argparse
import gzip
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from svlab import exact_equity
from svlab.cards import card_str
from svlab.preflop import parse_table
from tests.test_preflop_table import table_invariant_failures

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tools" / "preflop_equity.c"
TARGET = ROOT / "src" / "svlab" / "data" / "preflop_equity.txt.gz"


def build(out_dir: Path) -> Path:
    if shutil.which("gcc") is None:
        sys.exit("gcc is required")
    binary = out_dir / "preflop_equity"
    subprocess.run(["gcc", "-O3", "-fopenmp", "-o", str(binary), str(SOURCE)], check=True)
    return binary


def c_matchup(binary: Path, a: list[int], b: list[int], board: list[int]) -> tuple[int, int, int]:
    args = [str(binary), "matchup", "".join(map(card_str, a)), "".join(map(card_str, b))]
    if board:
        args.append("".join(map(card_str, board)))
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout.split()
    return int(out[0]), int(out[1]), int(out[2])


def differential(binary: Path, checks: int, seed: int) -> list[str]:
    """Random matchups on random three- or four-card boards, compared exactly."""
    rng = random.Random(seed)
    failures = []
    for _ in range(checks):
        cards = rng.sample(range(52), 4 + rng.choice((3, 4)))
        a, b, board = cards[:2], cards[2:4], cards[4:]
        py = exact_equity([a, b], board)
        c = c_matchup(binary, a, b, board)
        if (c[0], c[1], c[2]) != (py.runouts, *py.units):
            failures.append(f"{card_str(a[0])}{card_str(a[1])} vs {card_str(b[0])}{card_str(b[1])} on "
                            f"{''.join(map(card_str, board))}: C {c}, Python {(py.runouts, *py.units)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from", dest="source", help="use a table the C program already wrote instead of running it again")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp))
        failures = differential(binary, args.checks, args.seed)
        if failures:
            print("\n".join(failures))
            print(f"The C program disagrees with the Python engine in {len(failures)} of {args.checks} matchups")
            return 1
        print(f"C and Python agree exactly on {args.checks} random matchups")

        raw = Path(args.source) if args.source else Path(tmp) / "table.txt"
        if not args.source:
            subprocess.run([str(binary), "table", str(raw)], check=True)
        text = raw.read_text(encoding="ascii")

    failures = table_invariant_failures(parse_table(text))
    if failures:
        print("\n".join(failures[:20]))
        return 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(gzip.compress(text.encode("ascii"), mtime=0))
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
