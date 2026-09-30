"""The validation suite, tested: every seeded bug must be caught."""

import pytest

from validation.checks import CHECKS
from validation.mutants import MUTANTS


@pytest.mark.slow
@pytest.mark.parametrize("mutant", [m for m, _ in MUTANTS], ids=lambda m: m.name)
def test_every_seeded_bug_is_caught(mutant):
    for check in CHECKS:
        try:
            check.run(mutant)
        except AssertionError:
            return
    pytest.fail(f"no check caught: {mutant.name}")
