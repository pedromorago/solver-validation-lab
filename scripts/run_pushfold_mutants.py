"""Runs every push/fold check against the real solver and against every mutant.

The real solver must pass everything; every mutant must fail at least one
check. Prints a Markdown table of which checks caught which bug."""

import sys

from validation.pushfold_checks import CHECKS
from validation.pushfold_mutants import MUTANTS


def failures():
    caught = []
    for check in CHECKS:
        try:
            check.run()
        except AssertionError:
            caught.append(check)
        except Exception as e:  # a crash counts as caught, but say so
            caught.append(check)
            print(f"  ({check.name} crashed with {type(e).__name__}: {e})", file=sys.stderr)
    return caught


def main() -> int:
    ok = True
    real = failures()
    if real:
        ok = False
        print("The real solver fails: " + ", ".join(c.name for c in real))

    print("| Seeded bug | Layer | Caught by |")
    print("|---|---|---|")
    for mutant in MUTANTS:
        with mutant.applied():
            caught = failures()
        if not caught:
            ok = False
        names = ", ".join(f"`{c.name}` ({c.oracle})" for c in caught) or "**nothing**"
        print(f"| {mutant.name} | {mutant.layer} | {names} |")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
