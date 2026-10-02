"""A small poker equity engine, the reference layer of this lab. The
validator for exported solver strategies lives in `svlab.strategy`, and the
heads-up push/fold solver in `svlab.pushfold`."""

from .cards import parse, parse_many, card_str, FULL_DECK
from .evaluator import evaluate, HandCategory
from .equity import EquityResult, exact_equity, monte_carlo_equity, sample_runout

__all__ = [
    "parse",
    "parse_many",
    "card_str",
    "FULL_DECK",
    "evaluate",
    "HandCategory",
    "EquityResult",
    "exact_equity",
    "monte_carlo_equity",
    "sample_runout",
]
