"""Writes the synthetic strategy fixtures in fixtures/strategies/.

Nothing here comes from a real solver. `svlab.strategy.synthetic` builds
each strategy from Monte Carlo equities and a crude EV model; see its
docstring. The candidates are the baseline with controlled changes, for
the regression diff demo. Seeded, so running it again gives the same
files (up to floating-point differences between platforms).

    PYTHONPATH=src python scripts/make_fixtures.py
"""

from dataclasses import replace
from pathlib import Path

from svlab.cards import RANKS, rank
from svlab.strategy import dumps, relabel, synthetic
from svlab.strategy.combos import parse_combo

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "strategies"
ITERATIONS = 200


def blend(node, combo):
    """A small, harmless change: 1% of every combo's strategy spread evenly
    over the actions. Every frequency moves by less than 0.01."""
    n = len(node.actions)
    return {a: 0.99 * combo.freqs.get(a, 0.0) + 0.01 / n for a in node.action_ids}


def sizing_regression(node, combo):
    """A change a reviewer should see: after a check, the button's top pairs
    move 60% of their 75% pot bets to the 33% size, and A5s disappears
    from the button's range at that node."""
    if node.path != ("check",):
        return dict(combo.freqs)
    cards = parse_combo(combo.combo)
    ranks = sorted((RANKS[rank(c)] for c in cards), reverse=True)
    if ranks == ["A", "5"] and combo.combo[1] == combo.combo[3]:
        return None
    freqs = dict(combo.freqs)
    if "K" in ranks:
        moved = 0.6 * freqs["bet75"]
        freqs["bet75"] -= moved
        freqs["bet33"] += moved
    return freqs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = synthetic.generate("Ks7s2d", seed=1, iterations=ITERATIONS)
    # Same strategy, suits renamed (spades to hearts, diamonds to clubs and back).
    twin = relabel(base, (1, 0, 3, 2))
    twin = replace(
        twin,
        meta={
            **twin.meta,
            "description": "Synthetic: the Ks7s2d strategy with suits relabelled, board Kh7h2c. Not solver output.",
        },
    )
    files = {
        "srp_btn_bb_Ks7s2d.json": base,
        "srp_btn_bb_Kh7h2c.json": twin,
        "srp_btn_bb_9h6h3h.json": synthetic.generate("9h6h3h", seed=2, iterations=ITERATIONS),
        "candidate_minor_Ks7s2d.json": synthetic.rebuild(
            base, blend, "Synthetic candidate: the Ks7s2d baseline with 1% of every strategy spread evenly over the actions. Not solver output."
        ),
        "candidate_regressed_Ks7s2d.json": synthetic.rebuild(
            base,
            sizing_regression,
            "Synthetic candidate: the Ks7s2d baseline where, after a check, the button's kings move 60% of their "
            "75% bets to 33%, and suited A5 is missing from the button's range. Not solver output.",
        ),
    }
    for name, strategy in files.items():
        (OUT / name).write_text(dumps(strategy), encoding="utf-8")
        print(f"wrote {name}: {sum(len(n.combos) for n in strategy.nodes)} combos")


if __name__ == "__main__":
    main()
