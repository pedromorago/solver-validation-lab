"""Validation of exported solver strategies.

- `fileformat`: the JSON contract, strict parsing with located errors.
- `rules`: invariants a single file must satisfy.
- `isomorphism`: the same strategy under relabelled suits, across files.
- `diff`: regression diff between a baseline and a candidate.
- `synthetic`: seeded synthetic strategies for fixtures and demos.
"""

from .diff import DiffReport, Thresholds, diff
from .fileformat import ParseError, dumps, load, loads
from .isomorphism import IsoResult, compare_isomorphic, relabel
from .model import FORMAT, Action, ComboStrategy, Node, Spot, Strategy
from .rules import RULES, Finding, Tolerances, validate

__all__ = [
    "FORMAT",
    "Action",
    "ComboStrategy",
    "Node",
    "Spot",
    "Strategy",
    "ParseError",
    "load",
    "loads",
    "dumps",
    "RULES",
    "Finding",
    "Tolerances",
    "validate",
    "IsoResult",
    "compare_isomorphic",
    "relabel",
    "DiffReport",
    "Thresholds",
    "diff",
]
