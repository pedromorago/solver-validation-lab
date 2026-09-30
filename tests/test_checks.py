"""The validation suite, run against the real engine."""

import pytest

from validation.checks import CHECKS
from validation.engine import REAL


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: f"{c.oracle}:{c.name}")
def test_real_engine_passes(check):
    check.run(REAL)
