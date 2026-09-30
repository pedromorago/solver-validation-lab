"""Records exact equities for the fixed spots in validation/baselines.json.

Before writing anything, every spot is recomputed with an independent
evaluator (treys) and the two results must match exactly, so a baseline is
never a copy of a bug. Run it again only after a deliberate change, and
review the diff of the JSON file."""

import json
import math
from itertools import combinations
from pathlib import Path

from treys import Card, Evaluator

from svlab import card_str, exact_equity
from svlab.equity import remaining_deck
from validation.spots import FLOP_SPOTS, RIVER_SPOTS, TURN_SPOTS

OUT = Path(__file__).resolve().parent.parent / "validation" / "baselines.json"


def treys_units(hands, board):
    ev = Evaluator()
    t = lambda cards: [Card.new(card_str(c)) for c in cards]
    unit = math.lcm(*range(1, len(hands) + 1))
    units = [0] * len(hands)
    runouts = 0
    for extra in combinations(remaining_deck(hands, board), 5 - len(board)):
        full = t([*board, *extra])
        scores = [ev.evaluate(full, t(h)) for h in hands]
        best = min(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        for i in winners:
            units[i] += unit // len(winners)
        runouts += 1
    return units, runouts


def main() -> None:
    spots = []
    for name, hands, board in FLOP_SPOTS + TURN_SPOTS + RIVER_SPOTS:
        ours = exact_equity(hands, board)
        theirs, runouts = treys_units(hands, board)
        if list(ours.units) != theirs or ours.runouts != runouts:
            raise SystemExit(f"{name}: svlab {ours.units} and treys {theirs} disagree; not writing baselines")
        spots.append(
            {
                "name": name,
                "hands": ["".join(card_str(c) for c in h) for h in hands],
                "board": " ".join(card_str(c) for c in board),
                "runouts": ours.runouts,
                "units": list(ours.units),
                "equities": [round(e, 6) for e in ours.equities],
            }
        )
        print(f"{name:34s} {[f'{e:.4f}' for e in ours.equities]}  ({ours.runouts} runouts, treys agrees)")
    OUT.write_text(json.dumps({"generated_by": "scripts/make_baselines.py", "spots": spots}, indent=2) + "\n")


if __name__ == "__main__":
    main()
