"""Applies every seeded corruption to a valid synthetic strategy and
reports which checks caught it.

The fixtures themselves must pass: every file validates, the relabelled
twin maps onto the original, and the small candidate stays within the
diff thresholds. Every corruption must be caught, by the check it was
written for at least. Prints Markdown tables; exits 1 otherwise.

    PYTHONPATH=src:. python scripts/run_corruptions.py
"""

import sys
from pathlib import Path

from svlab.strategy import Thresholds, compare_isomorphic, diff, load, validate
from validation.strategy_corruptions import CORRUPTIONS, caught_by

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures" / "strategies"
BASELINE = FIXTURES / "srp_btn_bb_Ks7s2d.json"
THRESHOLDS = Thresholds.load(ROOT / "fixtures" / "diff_thresholds.json")


def fixture_checks() -> list[tuple[str, bool]]:
    rows = []
    for path in sorted(FIXTURES.glob("*.json")):
        rows.append((f"`svlab validate {path.name}` passes", not validate(load(path))))
    base = load(BASELINE)
    twin = load(FIXTURES / "srp_btn_bb_Kh7h2c.json")
    rows.append(("`svlab iso` maps the Ks7s2d strategy onto its Kh7h2c twin", compare_isomorphic(base, twin).ok))
    minor = diff(base, load(FIXTURES / "candidate_minor_Ks7s2d.json"), THRESHOLDS)
    rows.append(("`svlab diff` passes the minor candidate", minor.ok))
    regressed = diff(base, load(FIXTURES / "candidate_regressed_Ks7s2d.json"), THRESHOLDS)
    rows.append(("`svlab diff` fails the regressed candidate", not regressed.ok))
    return rows


def main() -> int:
    ok = True
    print("| Fixture check | Result |")
    print("|---|---|")
    for label, passed in fixture_checks():
        ok &= passed
        print(f"| {label} | {'ok' if passed else '**FAILED**'} |")

    base = load(BASELINE)
    print()
    print(f"| Corruption of `{BASELINE.name}` | Compared with | Caught by |")
    print("|---|---|---|")
    for corruption in CORRUPTIONS:
        caught = caught_by(corruption, base, seed=0, thresholds=THRESHOLDS)
        compared = "original (diff)" if corruption.target == "file" else "original (iso)"
        if caught is None:
            ok = False
            print(f"| {corruption.name} | {compared} | **could not be applied** |")
            continue
        if corruption.expected not in caught:
            ok = False
        names = ", ".join(f"`{c.split(':')[1]}` ({c.split(':')[0]})" for c in caught) or "**nothing**"
        if corruption.expected not in caught and caught:
            names += f" (**expected `{corruption.expected}`**)"
        print(f"| {corruption.name} | {compared} | {names} |")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
