"""Runs every check against the real engine and against every mutant.

The real engine must pass everything; every mutant must fail at least one
check. Prints a Markdown table of which checks caught which bug."""

import sys

from validation.checks import CHECKS
from validation.engine import REAL
from validation.mutants import MUTANTS


def failures(engine):
    caught = []
    for check in CHECKS:
        try:
            check.run(engine)
        except AssertionError:
            caught.append(check)
        except Exception as e:  # a crash counts as caught, but say so
            caught.append(check)
            print(f"  ({engine.name}: {check.name} crashed with {type(e).__name__}: {e})", file=sys.stderr)
    return caught


def main() -> int:
    ok = True
    real = failures(REAL)
    if real:
        ok = False
        print("The real engine fails: " + ", ".join(c.name for c in real))

    print("| Seeded bug | Layer | Caught by |")
    print("|---|---|---|")
    for engine, layer in MUTANTS:
        caught = failures(engine)
        if not caught:
            ok = False
        names = ", ".join(f"`{c.name}` ({c.oracle})" for c in caught) or "**nothing**"
        print(f"| {engine.name} | {layer} | {names} |")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
