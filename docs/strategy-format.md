# Strategy file format (`svlab-strategy/1`)

A strategy file describes one spot and the strategy at one or more decision nodes of it: for every hand combo the acting player can hold, how often it takes each action and, optionally, what each action is worth. It is the shape of data a solver or a trainer exports. The format is JSON, and the parser (`svlab.strategy.fileformat`) reads it strictly: unknown keys, missing keys, wrong types and duplicate keys are errors that name the exact place.

A complete small example is [`fixtures/strategies/minimal_Ks7s2d.json`](../fixtures/strategies/minimal_Ks7s2d.json). Every file in this repository is synthetic; none of them is solver output.

## Layout

```json
{
  "format": "svlab-strategy/1",
  "meta": {"synthetic": true, "description": "...", "units": "bb"},
  "spot": {
    "game": "NLHE",
    "positions": ["BTN", "BB"],
    "pot": 5.5,
    "effective_stack": 97.5,
    "board": "Ks7s2d",
    "preflop": ["BTN raise 2.5", "BB call"]
  },
  "nodes": [
    {
      "path": ["check"],
      "player": "BTN",
      "pot": 5.5,
      "effective_stack": 97.5,
      "to_call": 0,
      "actions": [
        {"id": "check", "type": "check"},
        {"id": "bet33", "type": "bet", "size": 1.8},
        {"id": "bet75", "type": "bet", "size": 4.1}
      ],
      "combos": {
        "AsJs": {"weight": 1.0,
                 "freqs": {"check": 0.2, "bet33": 0.5, "bet75": 0.3},
                 "evs": {"check": 3.0, "bet33": 3.3, "bet75": 3.2},
                 "ev": 3.21}
      }
    }
  ]
}
```

## Fields

Top level

| Key | Required | Meaning |
|---|---|---|
| `format` | yes | Always `"svlab-strategy/1"`. |
| `meta` | no | Free-form object: description, generator, units. Kept as is and not validated, apart from refusing duplicate keys. |
| `spot` | yes | The situation, see below. |
| `nodes` | yes | Non-empty list of decision nodes. Two nodes may not share a path. |

`spot`

| Key | Required | Meaning |
|---|---|---|
| `game` | yes | Free text, e.g. `"NLHE"`. |
| `positions` | yes | At least two distinct names. Every node's `player` must be one of them. |
| `pot`, `effective_stack` | yes | At the start of the street, in the file's units (usually big blinds). |
| `board` | yes | Three to five distinct cards, e.g. `"Ks7s2d"`. The first three are the flop; a fourth and fifth are the turn and river. |
| `preflop` | no | Free text describing the earlier action. |

`nodes[i]`

| Key | Required | Meaning |
|---|---|---|
| `path` | yes | Action ids taken on this street before this node, from the first decision. `[]` is the first decision of the street. |
| `player` | yes | The position acting here. |
| `pot` | yes | Pot at this node, including bets already made on this street. |
| `effective_stack` | yes | The most the acting player can put in on this street at this node. |
| `to_call` | yes | Amount the acting player must add to call; `0` when not facing a bet. |
| `actions` | yes | The actions available here: `id` (any string, unique in the node), `type` (`fold`, `check`, `call`, `bet`, `raise`, `allin`) and, for `bet`, `raise` and `allin`, a `size`: the chips the player puts in with this action on this street (a raise's size is the amount raised to). |
| `combos` | yes | Object keyed by combo, see below. |

`nodes[i].combos[combo]`

The key is two cards, e.g. `"AhKh"`. Either card order is accepted; the same combo in two orders is a duplicate.

| Key | Required | Meaning |
|---|---|---|
| `weight` | yes | The acting player's reach for this combo, in [0, 1]: how often they hold it and got here, relative to their range at the start of the street. |
| `freqs` | yes | Action id to frequency. An action left out is played 0% of the time. |
| `evs` | no | Action id to the EV of taking that action and then following the strategy, in the file's units. |
| `ev` | no | The combo's EV at this node under its mixed strategy. |

`NaN` and `Infinity` are read (Python's JSON reader accepts them) so that the `ev-finite` rule can report them rather than the parser failing.

## Parsing versus rules

The parser only checks what it needs to build the typed objects in `svlab.strategy.model`: structure, types, a readable board, known positions, unique paths. Everything about whether the numbers make sense is a rule, so a file with ten problems loads and all ten are reported. Duplicate keys are a parse error everywhere except inside `combos`, where both copies are kept for the `duplicate-combo` rule.

## Rules

`svlab rules` prints the list. Tolerances default to 0.001 for frequencies and weights and 0.01 for EVs, and can be changed with `--freq-tol`, `--weight-tol` and `--ev-tol`.

| Rule | Checks |
|---|---|
| `freq-range` | Every frequency is in [0, 1]. |
| `freq-sum` | Each combo's frequencies add up to 1, within the frequency tolerance. |
| `weight-range` | Every weight is in [0, 1]. |
| `ev-finite` | Every EV is a finite number. |
| `action-legal` | Types are known, ids unique; bets, raises and all-ins have a positive size no larger than the effective stack; no check or bet facing a bet, no fold, call or raise when there is nothing to call; a raise is above the amount to call; an all-in equals the effective stack. |
| `unknown-action` | `freqs` and `evs` only name actions defined at the node, and each step of a path is an action of the node it leaves (when that node is in the file). |
| `combo-syntax` | Every combo is two distinct valid cards. |
| `duplicate-combo` | No combo appears twice in a node, in either card order. |
| `board-blocker` | No combo contains a board card. |
| `ev-consistency` | Where `ev` and `evs` are both given, `ev` equals the sum of frequency times action EV over the played actions, within the EV tolerance. |
| `zero-freq-best-response` | An action a combo plays 0% of the time (at or below the frequency tolerance) does not have an EV above every action it does play by more than the EV tolerance. See below. |
| `reach-consistency` | A player's weight for a combo equals their weight at their previous decision on the path times the frequency of the action they took there. Checked when every node in between is in the file. |
| `suit-isomorphism` | When suits play the same role on the board, combos that differ only by swapping them have the same weight, frequencies and EVs, within tolerance. A combo whose image is missing counts as a difference. |

### What `zero-freq-best-response` does and does not say

The action EVs in an export are the values of each pure action against the opponent's strategy in the same file. In an equilibrium, every action a combo plays with positive frequency is a best response, so an action it never plays cannot be worth strictly more than all of them. A finding therefore means the frequencies, the EVs or the convergence are wrong. Passing proves much less: the check uses the file's own numbers at one node, so it cannot tell whether those EVs are right or whether the strategy is close to an equilibrium. It is no substitute for measuring exploitability, which needs the game tree and a best-response computation.

### When suits play the same role

Swapping two suits leaves a board unchanged when both hold the same ranks on it: both absent from a two-tone flop such as Ks7s2d (hearts and clubs), all three absent suits on a monotone flop, or the two suits of a pair on a paired board. A rainbow flop with three different ranks has no such symmetry, since three suits each hold a different card and only one is absent. For turn and river cards the relabelling must keep each card in place; the flop is compared as a set.

## Across two files

`svlab iso a.json b.json` finds the suit relabellings that turn a's board into b's and checks that one of them maps every node and combo of a onto b with the same numbers, reporting the one with the fewest differences. Spot fields, node paths and actions must match exactly.

## Regression diff

`svlab diff baseline.json candidate.json` matches nodes by path and combos by their cards, then reports per node the largest per-combo change in frequency, EV and weight, the range-weighted frequency of each action (`sum(weight * freq) / sum(weight)`), and combos added or removed. Thresholds come from flags or from a JSON file (`--config`, see [`fixtures/diff_thresholds.json`](../fixtures/diff_thresholds.json)); flags override the file.

| Threshold | Default | Violation when |
|---|---|---|
| `max_freq_shift` | 0.05 | any combo's frequency for any action moves more than this |
| `max_ev_shift` | 0.25 | any combo's overall or action EV moves more than this |
| `max_weight_shift` | 0.05 | any combo's weight moves more than this |
| `max_aggregate_shift` | 0.02 | any action's range-weighted frequency moves more than this |
| `max_combos_added`, `max_combos_removed` | 0 | more combos (with a positive weight) appear or disappear |
| `min_weight` | 0 | combos below this weight in both files are left out of the three per-combo thresholds |
| `allow_tree_changes` | false | when false, added or removed nodes and changed actions are violations |

The diff ignores combos it cannot match: malformed and blocked ones. Run `svlab validate` on both files first.

## Exit codes

All commands exit 0 when everything passes, 1 when a rule or threshold fails, and 2 when a file cannot be read or parsed.
