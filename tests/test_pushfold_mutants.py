"""The push/fold checks, tested: the real solver passes them all and every seeded bug is caught."""

import pytest

from validation.pushfold_checks import CHECKS
from validation.pushfold_mutants import MUTANTS


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.name)
def test_real_solver_passes(check):
    check.run()


@pytest.mark.slow
@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.name)
def test_every_seeded_bug_is_caught(mutant):
    with mutant.applied():
        for check in CHECKS:
            try:
                check.run()
            except AssertionError:
                return
    pytest.fail(f"no check caught: {mutant.name}")
