"""The pieces of the system under test that the checks call.

The mutants in validation/mutants.py replace one piece at a time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from svlab import equity, evaluator


@dataclass(frozen=True)
class Engine:
    name: str
    evaluate: Callable = evaluator.evaluate
    sampler: Callable = equity.sample_runout
    exact_impl: Callable = equity.exact_equity
    monte_carlo_impl: Callable = equity.monte_carlo_equity

    def exact(self, hands, board=(), **kwargs):
        return self.exact_impl(hands, board, evaluate=self.evaluate, **kwargs)

    def monte_carlo(self, hands, board=(), **kwargs):
        return self.monte_carlo_impl(hands, board, evaluate=self.evaluate, sampler=self.sampler, **kwargs)


REAL = Engine("real engine")
