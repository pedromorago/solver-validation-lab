# Solver Validation Lab

How do you test poker software when, for most inputs, nobody knows the right answer?

A solver or a trainer gives you numbers: equities, frequencies, expected values. For a handful of spots you can check them by hand. For the rest there is no perfect oracle, only relationships that must hold. This lab applies that idea at two levels:

1. **An equity engine** (`src/svlab`), a pure-Python all-in equity calculator, with a validation suite of 15 checks and 9 seeded bugs that show which checks catch what. It is the reference layer: the strategy tools below reuse its card handling, and the synthetic strategy generator uses its evaluator to estimate equities by Monte Carlo.
2. **A validator for exported solver strategies** (`src/svlab/strategy`): a documented JSON contract, 13 rules that any correct export must satisfy, a suit-isomorphism check across two files, and a regression diff between a baseline and a candidate, with 17 seeded corruptions that show which check catches what.

Neither part is a solver. The strategy files in this repository are synthetic, generated or hand-written here; none of them comes from a real solver.

![CI](https://github.com/pedromorago/solver-validation-lab/actions/workflows/ci.yml/badge.svg)

## Layer 1: the equity engine

The engine computes all-in equity (the share of the pot each hand wins over every possible runout), which is the numeric core that range tools, trainers and solvers are built on.

### The engine

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

### The validation suite

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

### Testing the tests: seeded bugs

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

## Layer 2: validating exported strategies

A solver's output is a strategy: at each decision, for each hand the player can hold, how often to take each action, and usually what each action is worth. Checking such a file against the "right" strategy is rarely possible, because computing it is the expensive part. What can be checked is that the file is internally possible and consistent, that it respects the symmetries of the game, and that a new version changed only what someone meant to change.

### The contract

[`docs/strategy-format.md`](docs/strategy-format.md) documents the format (`svlab-strategy/1`). In short, one file holds a spot (game, positions, pot, effective stack, board, earlier action) and a list of nodes. Each node has its action path on the street, the acting player, pot, stack and amount to call, the available actions with their sizes, and per combo a range weight, action frequencies, and optionally per-action EVs and an overall EV.

`fileformat.py` parses it strictly into frozen dataclasses (standard library only) and says where anything is wrong:

```
spot.board: Duplicate card in 'Ks7sKs'
nodes[0].combos['AhQh'].freqs.bet33: expected a number, got "x"
nodes[0]: duplicate key 'to_call'
```

Parsing checks only structure and types. Whether the numbers make sense is left to the rules, so a file with several problems loads and every problem is reported:

```
$ svlab validate broken.json
FAIL broken.json: 4 finding(s)
  [freq-sum] nodes[0].combos['2h2c'].freqs: sums to 1.07, expected 1 ± 0.001
  [unknown-action] nodes[0].combos['2h2c'].freqs.X: no action 'X' at this node
  [ev-consistency] nodes[0].combos['2h2c'].evs: no EV for played action 'X'
  [reach-consistency] nodes[2].combos['2h2c'].weight: weight 0.0131, expected 0.07 (1 × 'check' at 0.07)
```

### The rules

`rules.py` holds 13 rules. The oracle types are the same as for the engine, applied to a different output.

| Oracle | Rules |
|---|---|
| Invariant | `freq-range` and `freq-sum` (frequencies form a distribution, within a stated tolerance); `weight-range`; `ev-finite`; `action-legal` (known types, positive sizes no larger than the effective stack, no check facing a bet, raises above the amount to call); `unknown-action`; `combo-syntax`; `duplicate-combo` (including the same combo written in the other card order); `board-blocker` (no combo holds a board card) |
| Consistency | `ev-consistency` (overall EV equals the frequency-weighted action EVs); `reach-consistency` (a player's range at a node is their range at their previous decision times the action they took); `zero-freq-best-response` |
| Metamorphic | `suit-isomorphism` within a file; `svlab iso` across two files |
| Regression | `svlab diff` against a baseline |

`zero-freq-best-response` is a sanity check taken from best-response reasoning, and it is deliberately narrow. The action EVs in an export are the values of each pure action against the opponent's strategy in the same file. In an equilibrium every action a combo plays is a best response, so an action it plays 0% of the time cannot be worth more than every action it does play. When the file says otherwise (beyond the EV tolerance), the frequencies, the EVs or the convergence are wrong. Passing proves much less: it uses the file's own numbers at one node and says nothing about exploitability, which needs the tree and a best-response computation.

### Suit isomorphism

Renaming suits changes nothing in poker. Within a file, when two suits play the same role on the board, combos that differ only by swapping them must have the same strategy. On Ks7s2d, hearts and clubs are both absent, so AhQh and AcQc must play identically; on a monotone flop any of the three absent suits can be exchanged; on a paired board the two suits of the pair can. A rainbow flop with three different ranks has no such symmetry, because each suit plays a different role.

Across two files, `svlab iso a.json b.json` finds the suit relabelling that turns one board into the other (Ks7s2d into Kh7h2c, say) and checks that it maps every node and combo of one strategy onto the other.

### Regression diff

`svlab diff baseline.json candidate.json` matches nodes by path and combos by their cards, whatever order they are written in, and reports per node the largest per-combo shift in frequency, EV and weight, the range-weighted frequency of each action, and combos added or removed. Thresholds come from flags or a JSON file ([`fixtures/diff_thresholds.json`](fixtures/diff_thresholds.json)). It writes a Markdown report and, with `--json`, a machine-readable one, and exits 1 when any threshold is exceeded.

For the regressed demo candidate, where the button's kings move 60% of their 75% pot bets to the 33% size and suited A5 disappears, the report includes:

```
| Node | Player | Compared | Added | Removed | Max freq shift | Max EV shift | Max weight shift |
|---|---|---|---|---|---|---|---|
| root | BB | 596 | 0 | 0 | 0.0000 (2h2c) | 0.0000 (2h2c) | 0.0000 (2h2c) |
| check | BTN | 453 | 0 | 4 | 0.5855 (KhKc) | 0.8690 (KhKc) | 0.0000 (2h2c) |
| check > bet75 | BB | 596 | 0 | 0 | 0.0000 (2h2c) | 0.0000 (2h2c) | 0.0000 (2h2c) |

- `combos-removed` node check: 4 combo(s) removed: Ac5c, Ad5d, Ah5h, As5s
- `max-aggregate-shift` node check: bet33 0.2973 -> 0.4031 (+0.1058, limit 0.02)
```

### Synthetic fixtures

`fixtures/strategies/` holds:

- `minimal_Ks7s2d.json`: hand-written, fifteen combos over three nodes, to read alongside the format document.
- `srp_btn_bb_Ks7s2d.json` and `srp_btn_bb_9h6h3h.json`: generated, about 1,650 combos each over three flop nodes (big blind first to act, button after a check, big blind facing a 75% pot bet).
- `srp_btn_bb_Kh7h2c.json`: the Ks7s2d strategy with suits relabelled.
- `candidate_minor_Ks7s2d.json` and `candidate_regressed_Ks7s2d.json`: the baseline with controlled changes, one within the diff thresholds and one well outside them.

`scripts/make_fixtures.py` generates them with `svlab.strategy.synthetic`, seeded. Each combo's frequencies come from a softmax over EVs from a crude model (a share of the pot for checking, a fold share and a showdown against a stronger calling range for betting), driven by its equity against the opponent's range at the node, estimated by Monte Carlo with the engine's evaluator. Suit symmetry and reach hold by construction. The numbers vary by hand in a plausible way; they are not an equilibrium and not poker advice. Every file is marked `"synthetic": true` in its `meta` block.

### Testing the tests: seeded corruptions

`validation/strategy_corruptions.py` holds 17 corruptions, each a single defect applied to the Ks7s2d baseline. File corruptions are checked with `validate` and with `diff` against the original; corruptions of a relabelled copy are checked with `validate` and with `iso` against the original. Where it can, each corruption keeps the rest of the file consistent: the same change goes to every suit-isomorphic copy of the chosen combo, and the overall EV is recomputed when frequencies move, except where the EV is the point. `scripts/run_corruptions.py` prints what caught each one:

| Corruption | Compared with | Caught by |
|---|---|---|
| frequencies sum to 1.07 | original (diff) | `freq-sum` (validate) |
| negative frequency, sum still 1 | original (diff) | `freq-range` (validate), `ev-consistency` (validate), `max-freq-shift` (diff) |
| range weight of 1.3 | original (diff) | `weight-range` (validate), `max-weight-shift` (diff) |
| combo holding a board card | original (diff) | `board-blocker` (validate) |
| duplicate combo in the other card order | original (diff) | `duplicate-combo` (validate) |
| malformed combo string | original (diff) | `combo-syntax` (validate), `suit-isomorphism` (validate), `combos-removed` (diff) |
| one combo differs from its suit-swapped twin | original (diff) | `suit-isomorphism` (validate), `max-freq-shift` (diff), `max-ev-shift` (diff) |
| overall EV off by 0.5 | original (diff) | `ev-consistency` (validate), `max-ev-shift` (diff) |
| NaN action EV | original (diff) | `ev-finite` (validate), `max-ev-shift` (diff) |
| 0% action with the best EV | original (diff) | `zero-freq-best-response` (validate), `max-freq-shift` (diff), `max-ev-shift` (diff) |
| action id the node doesn't define | original (diff) | `unknown-action` (validate), `max-freq-shift` (diff) |
| bet size above the effective stack | original (diff) | `action-legal` (validate), `tree-changed` (diff) |
| check offered facing a bet | original (diff) | `action-legal` (validate), `tree-changed` (diff) |
| range weight off the tree | original (diff) | `reach-consistency` (validate), `max-weight-shift` (diff) |
| large strategy shift in a valid file | original (diff) | `max-freq-shift` (diff), `max-ev-shift` (diff), `max-aggregate-shift` (diff) |
| relabelled copy: one combo keeps its old suits | original (iso) | `board-blocker` (validate), `reach-consistency` (validate), `suit-isomorphism` (validate), `iso-combo` (iso) |
| relabelled copy: one hand's strategy drifts | original (iso) | `iso-combo` (iso) |

All 17 are caught, and every fixture passes (the script prints that first). CI runs this on every push and writes both tables to the job summary. The table is one seeded run; the tests repeat every corruption with five seeds on the fixture and on up to 40 Hypothesis-generated files per corruption, and require the rule it was written for to be among those that fire. Which other checks fire depends on the combo or node the corruption lands on.

What the table says about the checks:

- **Some defects are visible to one rule only.** A combo holding a board card and a duplicate written in the other card order are caught by their own rule and by nothing else. The diff can't see them: it skips combos that use a board card, and it matches combos by their cards, so a second copy is the same combo to it. That is why `validate` runs before `diff` in CI.
- **A valid file can still be wrong for you.** The large strategy shift passes all 13 rules, since the file is consistent; only the diff notices. The relabelled copy whose strategy drifted is also valid on its own; only comparing it with the original through `iso` shows the change.
- **The diff flags most corruptions too, and says less about them.** It measures how far a number moved. The rules say which property broke. A diff failure alone does not distinguish a broken candidate from a deliberately changed one.
- **Consistency rules depend on what the defect leaves consistent.** With frequencies summing to 1.07 and the overall EV recomputed from those same frequencies, `ev-consistency` holds and only `freq-sum` fires in this run; the largest change also stayed under the diff's 0.05 per-combo threshold.
- **Knock-on effects are real signals.** A malformed combo string also breaks `suit-isomorphism`, because its suit-swapped partner loses its match. A relabelled copy that kept one combo's old suits trips four checks, including `board-blocker`, since old suits can collide with the new board.
- **Aggregates can hide per-combo changes.** The large-shift corruption was first expected to trip the aggregate threshold, and Hypothesis found a counterexample: two combos that swap strategies leave every range-weighted frequency where it was. Only the per-combo threshold catches that, so the diff keeps both, and `tests/test_strategy_tools.py` has a test for it.

### Property-based tests

`tests/test_strategy_properties.py` uses a Hypothesis generator of small valid strategy files (`tests/strategy_gen.py`: random boards, one to three nodes, random legal actions, suit-isomorphic classes that share a strategy, and ranges that follow the tree). The properties: every generated file passes every rule; `parse(serialize(x)) == x`; relabelling suits gives a file that `iso` maps back onto the original; a file diffed against itself passes; and every corruption is reported by the rule it was written for.

## Running it

```bash
pip install -e ".[dev]"
pytest -q                          # both layers: checks, properties, seeded bugs and corruptions

svlab rules                        # the 13 strategy rules
svlab validate fixtures/strategies/*.json
svlab iso fixtures/strategies/srp_btn_bb_Ks7s2d.json fixtures/strategies/srp_btn_bb_Kh7h2c.json
svlab diff fixtures/strategies/srp_btn_bb_Ks7s2d.json fixtures/strategies/candidate_regressed_Ks7s2d.json \
    --config fixtures/diff_thresholds.json --markdown diff.md --json diff.json
svlab diff baseline.json candidate.json --max-freq-shift 0.1 --max-ev-shift 0.5   # flags instead of a config

PYTHONPATH=src:. python scripts/run_mutants.py       # seeded bugs in the engine
PYTHONPATH=src:. python scripts/run_corruptions.py   # seeded corruptions of strategy files
PYTHONPATH=src python scripts/make_fixtures.py       # regenerate the synthetic fixtures
```

`python -m svlab ...` works the same as `svlab ...`. Exit codes: 0 when everything passes, 1 when a rule or threshold fails, 2 when a file can't be read or parsed. Python 3.11 or later; the strategy tools use only the standard library.

## Limitations

- Every strategy file here is synthetic. The validator has not been run on a real solver's export, and a real export format would need an adapter to this contract.
- The consistency rules work within a node, plus reach between a player's own decisions. EVs are not checked across nodes (for example, that the EV of checking equals what the next node's strategies imply), which would need both players' strategies and the values at the end of the tree.
- Played actions are not required to have equal EVs, which an exact equilibrium would give, because real solver output is only approximately converged. `zero-freq-best-response` checks the one direction that tolerates that.
- The fixtures cover single flop decisions, with no turn or river cards dealt within the file.

## Next

- Heads-up push/fold equilibrium for short stacks, checked by exploitability: a best response against the computed strategy must gain no more than a stated tolerance. That would give the strategy validator a real solver to point at, with a stronger oracle than `zero-freq-best-response`.
- Range against range equity, with consistency checks between the range result and the weighted combination of its hand-against-hand parts.

## License

MIT
