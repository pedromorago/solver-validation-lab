# Solver Validation Lab

How do you test poker software when, for most inputs, nobody knows the right answer?

A solver or a trainer gives you numbers: equities, frequencies, expected values. For a handful of spots you can check them by hand. For the rest there is no perfect oracle, only relationships that must hold. This lab applies that idea at three levels:

1. **An equity engine** (`src/svlab`), a pure-Python all-in equity calculator, with a validation suite of 15 checks and 9 seeded bugs that show which checks catch what. It is the reference layer: the strategy tools below reuse its card handling, and the synthetic strategy generator uses its evaluator to estimate equities by Monte Carlo.
2. **A validator for exported solver strategies** (`src/svlab/strategy`): a documented JSON contract, 13 rules that any correct export must satisfy, a suit-isomorphism check across two files, and a regression diff between a baseline and a candidate, with 17 seeded corruptions that show which check catches what.

3. **Push/fold solvers** for heads-up (`src/svlab/pushfold.py`) and three-handed play (`src/svlab/pushfold3.py`), built on preflop equity tables for the 169 hand classes. They are the spots here small enough to have a real oracle: a best response against the solution must gain nothing. For heads-up, seven checks and six seeded bugs show which check catches what.

The first two parts are not solvers, and the strategy files in this repository are synthetic, generated or hand-written here; none of them comes from a real solver. The push/fold solutions are real equilibria of a deliberately small game, described below.

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

## Layer 3: a push/fold solver with a real oracle

With short stacks, heads-up play reduces to one decision each: the small blind shoves all in or folds, and the big blind calls or folds. That game is small enough to solve exactly, which makes it the place where a solver can be checked against the strongest oracle there is. For any pair of strategies you can compute what each player would gain by switching to their best response; the sum, the Nash gap, is zero at an equilibrium and positive anywhere else.

### An exact preflop equity table

The solver needs the all-in equity of every hand class against every other. The pure-Python engine would take days to enumerate every board for every matchup, so `tools/preflop_equity.c` does it in C: it lists the 812,175 pairs of hands that share no card, groups them into 47,008 matchups that are the same up to relabelling suits, enumerates all 1,712,304 boards for each group, and adds the results to the two classes involved. The table ships as `src/svlab/data/preflop_equity.txt.gz`, in integer units like the engine (two per board, one each on a split), so its invariants hold exactly.

`scripts/make_preflop_table.py` writes the table only after the C program agrees exactly, unit for unit, with the Python engine on 300 random matchups on random flops and turns, and after every invariant passes. `tests/test_preflop_table.py` then checks it from four sides:

| Oracle | Check |
|---|---|
| Invariant | The number of hand pairs behind every cell follows from card removal; the units of i against j and of j against i add up to the pot, exactly; a class against itself is exactly 50% |
| Reference | AA against KK is the textbook 82% |
| Differential | The C program matches the Python engine exactly on random flops (when gcc is available) |
| Statistical | Random cells fall within 4.5 standard errors of a Monte Carlo estimate made with the Python engine |

### The game and the solver

Blinds are 0.5 and 1, there are no antes, and both players start with the same stack. EVs are in big blinds: folding the small blind is worth -0.5, a shove that gets folded to +1, and a called shove `stack * (2 * equity - 1)`. Chips are what count, which in a winner-take-all Spin & Go is also what the prize depends on. How often class i meets class j comes from the number of hand pairs in the table, so card removal is part of the game: holding an ace makes it less likely that the opponent has one.

`solve` runs CFR+ (alternating updates, linearly weighted averages) until the Nash gap of the average strategies is below a tolerance, 1e-6 bb per hand by default:

```
$ svlab pushfold --stack 10
Heads-up push/fold, 10 bb, blinds 0.5/1, no ante
SB shoves 58.3% of hands; BB calls with 37.4%
SB EV -0.0454 bb per hand; Nash gap 9.78e-07 bb after 1300 iterations
```

followed by the shove and call frequencies as 13x13 grids. `--json` writes every class's frequencies and the EV of each action, the data a trainer needs to grade a decision by the EV it loses rather than by right or wrong.

### The heads-up checks and the seeded bugs

`validation/pushfold_checks.py` holds seven checks; `validation/pushfold_mutants.py` patches six bugs into the solver, one at a time, and `scripts/run_pushfold_mutants.py` runs every check against every one of them.

| Check | Oracle | What it says |
|---|---|---|
| Nash gap closes | Exactness | Best responses against the solution gain at most the tolerance |
| Regrets add up to the gap | Invariant | Each class's gain from switching to its best action, weighted by how often it reaches its decision, is non-negative, and the gains add up to the Nash gap: the per-class EVs and the best responses agree |
| Fictitious play agrees | Differential | A different algorithm reaches the same game value |
| Value over dealt hands | Differential | The SB's EV recomputed from scratch over every pair of hands that can be dealt, with the rules written out again; only the equities are shared with the solver |
| Coin flips shoved and called | Reference | On a table where every matchup is 50%, the answer follows by hand: shove and call everything, for a value of 0 |
| AA shoves and calls | Reference | The best hand never folds, at any stack |
| Zero-sum | Invariant | The two players' values cancel for random strategies |

| Seeded bug | Layer | Caught by |
|---|---|---|
| Card removal ignored | game | `value over dealt hands` (differential) |
| Equity table read transposed | game | `value over dealt hands` (differential), `AA shoves and calls` (reference) |
| Blinds swapped | game | `Nash gap closes` (exactness), `regrets add up to the gap` (invariant), `value over dealt hands` (differential), `zero-sum` (invariant) |
| BB scored with the SB's equity | values | `Nash gap closes` (exactness), `regrets add up to the gap` (invariant), `AA shoves and calls` (reference), `zero-sum` (invariant) |
| Nash gap leaves out the BB | values | `regrets add up to the gap` (invariant) |
| Last iterate reported instead of the average | solver | `Nash gap closes` (exactness) |

What the table says:

- **A converged solver can still solve the wrong game.** Ignoring card removal changes the game, not the algorithm, so the solver still finds an exact equilibrium of it: the Nash gap is zero and every internal check passes. Only the check that rebuilds the value from dealt hands, independently of the solver's model, notices. Exploitability proves the solver solved its model, not that the model is right.
- **The gap is only as good as its own code.** A Nash gap that leaves out one player's deviations reports convergence that never happened. The cross-check between the per-class regrets and the gap catches it; the gap alone cannot.
- **The averaged strategy is the answer.** CFR's current strategy keeps moving; returning it instead of the average is a common slip that leaves a measurable gap.

### Three-handed: the start of a Spin & Go

`svlab pushfold3` solves BTN, SB and BB with equal stacks. The button shoves or folds; if it folds, the blinds play the heads-up game with the button's cards dead; if it shoves, the small blind calls or folds and the big blind then faces the shove alone or the shove and a call. That is six decisions, each taken knowing only one's own hand.

Three-way all-ins need the equity of every triple of classes: 818,805 of them, too many to enumerate every board. `tools/threeway_equity.c` estimates them by Monte Carlo, 8,000 deals per triple with a fixed seed, together with the three heads-up equities with the third hand's cards dead; the weight of each triple, how many ways it can be dealt, is counted exactly. The table is about 11 MB, so it is generated (`scripts/make_threeway_table.py`, ten minutes on four cores) rather than committed, and CI caches it.

With three players, CFR+ has no convergence guarantee, so the result is judged by NashConv: the total that each player could gain by switching to a best response. Each player acts at most once on any path, so this is the sum of the regrets at every decision. At 10 bb the solver gets it below 1e-5 bb per hand in about 600 iterations:

```
$ svlab pushfold3 --stack 10
Three-handed push/fold, 10 bb each, blinds 0.5/1, no ante
EV per hand: BTN +0.2414, SB -0.0835, BB -0.1578
NashConv 9.81e-06 bb after 625 iterations
```

| Oracle | Check |
|---|---|
| Invariant | Triple weights summed over the third hand equal the heads-up pair counts times the 1,128 ways to deal it, exactly; the three seats' equities add up to 1; the three players' values add up to 0 |
| Statistical | The heads-up estimates, averaged over the third hand, match the exact heads-up table within five standard errors in all 28,561 cells; random three-way cells match a Monte Carlo run of the Python engine |
| Exactness | NashConv is below the tolerance |
| Differential | Replacing one player's strategy by its best response and re-evaluating the whole game gains exactly what the per-decision regrets say |
| Reduction | When the button folds everything, the blinds are playing the heads-up game: the heads-up equilibrium keeps its value and stays an equilibrium, up to the Monte Carlo noise |
| Reference | AA never folds; on coin flips the blinds call everything they face |

The Monte Carlo table is the weak point of the three-handed solver, and the checks say how weak: a solution is an exact equilibrium of the game the table describes, and that game differs from the true one by the table's sampling error.

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

svlab pushfold --stack 10 --json pushfold.json         # heads-up push/fold at 10 bb
PYTHONPATH=src:. python scripts/make_threeway_table.py    # build the three-way table once (gcc, about ten minutes)
svlab pushfold3 --stack 10 --json pushfold3.json       # three-handed push/fold at 10 bb

PYTHONPATH=src:. python scripts/run_mutants.py       # seeded bugs in the engine
PYTHONPATH=src:. python scripts/run_pushfold_mutants.py   # seeded bugs in the push/fold solver
PYTHONPATH=src:. python scripts/make_preflop_table.py     # rebuild the equity table (gcc, about half an hour)
PYTHONPATH=src:. python scripts/run_corruptions.py   # seeded corruptions of strategy files
PYTHONPATH=src python scripts/make_fixtures.py       # regenerate the synthetic fixtures
```

`python -m svlab ...` works the same as `svlab ...`. Exit codes: 0 when everything passes, 1 when a rule or threshold fails, 2 when a file can't be read or parsed. Python 3.11 or later; the engine and the strategy tools use only the standard library, and the push/fold solver uses NumPy.

## Limitations

- Every strategy file here is synthetic. The validator has not been run on a real solver's export, and a real export format would need an adapter to this contract.
- The consistency rules work within a node, plus reach between a player's own decisions. EVs are not checked across nodes (for example, that the EV of checking equals what the next node's strategies imply), which would need both players' strategies and the values at the end of the tree.
- Played actions are not required to have equal EVs, which an exact equilibrium would give, because real solver output is only approximately converged. `zero-freq-best-response` checks the one direction that tolerates that.
- The fixtures cover single flop decisions, with no turn or river cards dealt within the file.
- The push/fold games leave out limps, min-raises and antes, so their strategies are equilibria of those restricted games, not of full poker. Their EVs are in chips; outside winner-take-all prize structures they would need an ICM model.
- Three-handed play assumes equal stacks, so there are no side pots, and its three-way equities are Monte Carlo estimates. The seeded-bug report covers the heads-up solver only.
- Push/fold solutions are not yet exported in the strategy file format, which describes postflop spots only.

## Next

- Seeded bugs for the three-handed solver, and unequal stacks with side pots.
- Push/fold exports graded by EV loss in [Spin Trainer](https://github.com/pedromorago/spin-trainer-api), so a wrong answer says how much it costs.
- Range against range equity, with consistency checks between the range result and the weighted combination of its hand-against-hand parts.

## License

MIT
