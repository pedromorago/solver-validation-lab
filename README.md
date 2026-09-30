# Solver Validation Lab

How do you test a poker engine when, for most inputs, nobody knows the right answer?

A solver or a trainer gives you numbers: equities, frequencies, expected values. For a handful of spots you can check them by hand. For the rest there is no perfect oracle, only relationships that must hold. This lab builds a small poker equity engine and a validation suite around it that uses those relationships, then seeds bugs into the engine to show which checks catch them.

It is a first step, not a solver. The engine computes all-in equity (the share of the pot each hand wins over every possible runout), which is the numeric core that range tools, trainers and solvers are built on.

![CI](https://github.com/pedromorago/solver-validation-lab/actions/workflows/ci.yml/badge.svg)

## The system under test

`src/svlab` is a pure-Python engine with no dependencies:

- `evaluator.py`: the best five-card hand out of five to seven cards, as a tuple that compares correctly (bigger wins, equal splits).
- `equity.py`: all-in equity two ways. `exact_equity` enumerates every remaining board and accumulates the pot in integer units, so exactness can be asserted without floating-point tolerance. `monte_carlo_equity` samples boards and reports a standard error with every estimate.

```python
from svlab import exact_equity, monte_carlo_equity, parse_many as P

exact_equity([P("AhKh"), P("QsQd")], P("Jh Th 2c")).equities
# (0.5606..., 0.4393...)   990 runouts, enumerated

monte_carlo_equity([P("AhKh"), P("QsQd")], iterations=20_000, seed=0)
# equities (0.4609, 0.5391), std_errors (0.0035, 0.0035)   preflop, sampled
```

## The validation suite

`validation/checks.py` holds 15 checks. Each one is written against an `Engine`, so the same suite can be pointed at the real engine or at a broken one, and each one names the kind of oracle it relies on.

| Oracle | What it means here | Checks |
|---|---|---|
| Reference | Hand-written expected values, only where they are beyond argument | Textbook hand categories; kickers decide equal pairs; identical hands split |
| Differential | An independent implementation must agree | 3,000 random seven-card matchups ordered the same way as by [treys](https://github.com/ihendley/treys) |
| Invariant | Must hold for any input | Card order doesn't matter; shares add up to the pot exactly; a complete board has no uncertainty; sampled boards use distinct, unused cards |
| Metamorphic | A known change to the input changes the output in a known way | Renaming suits changes nothing, for hands and for equity; swapping players swaps equities; flop equity is exactly the average of the equities after each turn card |
| Statistical | A sampled estimate must fall within its error bars | Monte Carlo within 4.5 standard errors of the exact value; quadrupling the samples roughly halves the error |
| Regression | Results must match recorded baselines | Exact equities for 11 spots in `validation/baselines.json` |

The baselines are not a copy of whatever the engine said on the day. `scripts/make_baselines.py` recomputes every spot with treys and refuses to write the file unless both agree exactly.

On top of the suite, `tests/test_properties.py` uses [Hypothesis](https://hypothesis.readthedocs.io/) to generate inputs: random hands for the order and suit properties, random turn spots for the pot and symmetry properties, random flops for Monte Carlo against exact, and one more metamorphic property, that adding a seventh card can never make the best hand worse.

## Testing the tests: seeded bugs

A validation suite for a system without an oracle has to answer a question about itself: if the engine were wrong in this particular way, would anything notice? `validation/mutants.py` holds nine versions of the engine with one deliberate bug each, and `scripts/run_mutants.py` runs every check against every one of them.

| Seeded bug | Layer | Caught by |
|---|---|---|
| Flush ranked below straight | evaluator | reference, differential, regression |
| A-2-3-4-5 not counted as a straight | evaluator | reference, differential, regression |
| Spade flushes missed | evaluator | reference, differential, both suit-renaming checks (metamorphic) |
| Pair kickers ignored | evaluator | reference (kickers), differential, regression |
| Only the first five cards read | evaluator | differential, card order (invariant), regression, statistical |
| Ties awarded to the first player | exact equity | player swap (metamorphic), identical hands split (reference), regression, statistical |
| Last card of the deck never dealt | exact equity | flop equals the average over turns (metamorphic), regression, statistical |
| Board sampled with replacement | Monte Carlo | sampled boards are legal (invariant), error shrinks with samples (statistical) |
| Players' cards left in the deck | Monte Carlo | sampled boards are legal (invariant), both statistical checks |

All nine are caught, and the real engine passes every check. CI runs this on every push and writes the table to the job summary.

What the table says about the checks:

- **No single layer is enough.** Sampling the board with replacement shifts the estimates so little that they still land inside their error bars. The invariant on the sampled cards catches it directly; the convergence check only notices because the error stops shrinking as it should. Giving every tie to the first player keeps the shares adding up to the pot, so that invariant passes, and the player-swap relation is what exposes it.
- **Metamorphic relations reach bugs that have no reference value.** An off-by-one that never deals the last card of the deck produces plausible equities everywhere. It breaks the law of total probability, though: the flop equity stops being the exact average of the turn equities.
- **Differential testing is strong but borrowed.** It catches every evaluator bug here, and it is only as good as the other implementation. That is why the regression baselines are cross-checked against it once, at generation, rather than trusted on their own.

## Running it

```bash
pip install -e ".[dev]"
pytest -q                       # the suite, the properties and the seeded bugs
python scripts/run_mutants.py   # the table above (needs PYTHONPATH=src:.)
```

Python 3.11 or later.

## Next

The same approach, one level up, is what I'm building toward:

- Heads-up push/fold equilibrium for short stacks, checked by exploitability: a best response against the computed strategy must gain no more than a stated tolerance.
- Range against range equity, with consistency checks between the range result and the weighted combination of its hand-against-hand parts.
- Checks for exported solver strategies: action frequencies that sum to one at every node, and EVs that are consistent with the frequencies and the tree.

## License

MIT
