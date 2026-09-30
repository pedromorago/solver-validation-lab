"""Property-based tests for the strategy validator. Hypothesis generates
small valid strategy files (tests/strategy_gen.py); the properties are:

- a valid file passes every rule;
- parse(serialize(x)) == x;
- renaming the suits gives a file that `iso` maps back onto the original;
- a file diffed against itself passes;
- every seeded corruption is reported by the check it was written for.
"""


import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from strategy_gen import valid_strategies
from svlab.strategy import compare_isomorphic, diff, dumps, loads, relabel, validate
from validation.strategy_corruptions import CORRUPTIONS, caught_by

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(valid_strategies())
def test_generated_valid_strategies_pass(strategy):
    assert validate(strategy) == []


@SETTINGS
@given(valid_strategies())
def test_parse_of_serialize_is_identity(strategy):
    assert loads(dumps(strategy)) == strategy


@SETTINGS
@given(valid_strategies(), st.permutations(range(4)))
def test_relabelled_strategy_is_isomorphic(strategy, perm):
    moved = relabel(strategy, tuple(perm))
    assert validate(moved) == []
    result = compare_isomorphic(strategy, moved)
    assert result.ok, result.findings[:3]


@SETTINGS
@given(valid_strategies())
def test_diff_against_itself_passes(strategy):
    assert diff(strategy, strategy).ok


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.name)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(strategy=valid_strategies(), seed=st.integers(0, 2**16))
def test_every_corruption_is_reported_by_its_rule(corruption, strategy, seed):
    caught = caught_by(corruption, strategy, seed=seed)
    assume(caught is not None)  # nothing for this corruption to act on in this file
    assert corruption.expected in caught, caught
