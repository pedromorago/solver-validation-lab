"""Testing the tests, strategy side: the fixtures pass, and every seeded
corruption of the synthetic baseline is caught by the check it targets,
whichever combo or node the corruption happens to pick."""

import pytest

from scripts.run_corruptions import BASELINE, THRESHOLDS, fixture_checks
from svlab.strategy import load
from validation.strategy_corruptions import CORRUPTIONS, caught_by

BASE = load(BASELINE)


@pytest.mark.parametrize("label, passed", fixture_checks(), ids=lambda x: x if isinstance(x, str) else "")
def test_fixture_checks(label, passed):
    assert passed, label


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.name)
@pytest.mark.parametrize("seed", range(5))
def test_corruption_is_caught(corruption, seed):
    caught = caught_by(corruption, BASE, seed=seed, thresholds=THRESHOLDS)
    assert caught is not None, "the corruption found nothing to act on in the baseline"
    assert corruption.expected in caught, caught


def test_only_the_diff_sees_a_large_shift_in_a_valid_file():
    [shift] = [c for c in CORRUPTIONS if c.name == "large strategy shift in a valid file"]
    caught = caught_by(shift, BASE, thresholds=THRESHOLDS)
    assert caught and all(c.startswith("diff:") for c in caught)


def test_only_iso_sees_a_drifted_relabelled_copy():
    [drift] = [c for c in CORRUPTIONS if c.name == "relabelled copy: one hand's strategy drifts"]
    for seed in range(5):
        assert caught_by(drift, BASE, seed=seed) == ["iso:iso-combo"]


@pytest.mark.parametrize(
    "name, rule",
    [("combo holding a board card", "validate:board-blocker"), ("duplicate combo in the other card order", "validate:duplicate-combo")],
)
def test_blocked_and_duplicate_combos_are_seen_by_one_rule_only(name, rule):
    """The diff can't match these combos, so it is blind to them: validate first."""
    [corruption] = [c for c in CORRUPTIONS if c.name == name]
    for seed in range(5):
        assert caught_by(corruption, BASE, seed=seed, thresholds=THRESHOLDS) == [rule]
