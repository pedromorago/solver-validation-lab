"""The 169 preflop hand classes and their exact heads-up equity table.

A class is a pair (``"QQ"``), a suited hand (``"AKs"``) or an offsuit hand
(``"AKo"``). The order matches ``tools/preflop_equity.c``: the 13 pairs from
22 to AA, then the 78 suited hands, then the 78 offsuit hands, each by
(high rank, low rank).

The table holds, for every class i against every class j, the number of
pairs of hands that share no card (``pairs``) and the units won by class i
over every board (``units``), two per board, one each on a split. So
``equity(i, j) = units[i][j] / (pairs[i][j] * boards * 2)``. It was generated
by enumerating all 1,712,304 boards for every matchup; see
``scripts/make_preflop_table.py``.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from functools import cache
from importlib import resources

from .cards import RANKS, rank, suit

FORMAT = "svlab-preflop-equity/1"


def _names() -> tuple[str, ...]:
    pairs = [r + r for r in RANKS]
    non_pairs = [(hi, lo) for hi in range(13) for lo in range(hi)]
    suited = [RANKS[hi] + RANKS[lo] + "s" for hi, lo in non_pairs]
    offsuit = [RANKS[hi] + RANKS[lo] + "o" for hi, lo in non_pairs]
    return tuple(pairs + suited + offsuit)


CLASSES: tuple[str, ...] = _names()
INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


def class_of(c1: int, c2: int) -> int:
    """The class index of a two-card hand."""
    hi, lo = sorted((rank(c1), rank(c2)), reverse=True)
    if hi == lo:
        return hi
    base = 13 + hi * (hi - 1) // 2 + lo
    return base if suit(c1) == suit(c2) else base + 78


def combos(name: str) -> list[tuple[int, int]]:
    """Every two-card hand in a class: 6 for a pair, 4 suited, 12 offsuit."""
    if name not in INDEX:
        raise ValueError(f"Not a hand class: {name!r}")
    hi, lo = RANKS.index(name[0]), RANKS.index(name[1])
    out = []
    for s1 in range(4):
        for s2 in range(4):
            c1, c2 = hi * 4 + s1, lo * 4 + s2
            if c1 == c2 or (hi == lo and s1 >= s2):
                continue
            if hi != lo and (name[2] == "s") != (s1 == s2):
                continue
            out.append((c1, c2))
    return out


@dataclass(frozen=True)
class PreflopTable:
    boards: int
    pairs: tuple[tuple[int, ...], ...]
    units: tuple[tuple[int, ...], ...]

    def equity(self, i: int, j: int) -> float:
        """Class i's all-in equity against class j, averaged over every pair of hands they can hold."""
        return self.units[i][j] / (self.pairs[i][j] * self.boards * 2)


def parse_table(text: str) -> PreflopTable:
    lines = text.splitlines()
    header = lines[0].split()
    if len(header) < 3 or header[1] != FORMAT:
        raise ValueError(f"Not a {FORMAT} file")
    fields = dict(part.split("=", 1) for part in header[2:])
    if fields.get("units_per_board") != "2":
        raise ValueError("Expected two units per board")
    boards = int(fields["boards"])

    pairs = [[0] * len(CLASSES) for _ in CLASSES]
    units = [[0] * len(CLASSES) for _ in CLASSES]
    seen = set()
    for n, line in enumerate(lines[1:], start=2):
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"line {n}: expected four fields")
        i, j = INDEX[parts[0]], INDEX[parts[1]]
        if (i, j) in seen:
            raise ValueError(f"line {n}: {parts[0]} {parts[1]} appears twice")
        seen.add((i, j))
        pairs[i][j], units[i][j] = int(parts[2]), int(parts[3])
    if len(seen) != len(CLASSES) ** 2:
        raise ValueError(f"Expected {len(CLASSES) ** 2} rows, got {len(seen)}")
    return PreflopTable(boards, tuple(map(tuple, pairs)), tuple(map(tuple, units)))


@cache
def load_table() -> PreflopTable:
    """The table shipped with the package."""
    data = resources.files("svlab").joinpath("data/preflop_equity.txt.gz").read_bytes()
    return parse_table(gzip.decompress(data).decode("ascii"))
