"""Property-based tests: Hypothesis generates the inputs, the properties
are the oracle."""

import random

from hypothesis import given, settings, strategies as st

from svlab import evaluate, exact_equity, monte_carlo_equity
from svlab.cards import rank, suit

cards7 = st.lists(st.integers(0, 51), min_size=7, max_size=7, unique=True)
suit_perms = st.permutations(range(4))


@given(cards7, st.randoms())
def test_order_does_not_matter(cards, rnd):
    shuffled = cards[:]
    rnd.shuffle(shuffled)
    assert evaluate(cards) == evaluate(shuffled)


@given(cards7, suit_perms)
def test_suit_names_do_not_matter(cards, perm):
    renamed = [rank(c) * 4 + perm[suit(c)] for c in cards]
    assert evaluate(cards) == evaluate(renamed)


@given(cards7)
def test_an_extra_card_never_makes_the_hand_worse(cards):
    """The best five of seven is at least the best five of any six of them."""
    best = evaluate(cards)
    for i in range(7):
        assert evaluate(cards[:i] + cards[i + 1 :]) <= best


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(0, 51), min_size=8, max_size=8, unique=True))
def test_turn_equities_sum_to_one_and_mirror(cards):
    a, b, board = cards[:2], cards[2:4], cards[4:8]
    forward = exact_equity([a, b], board)
    backward = exact_equity([b, a], board)
    assert sum(forward.units) == forward.runouts * forward.units_per_runout
    assert forward.units == tuple(reversed(backward.units))


# Fixed examples: a statistical bound should not flake from run to run.
@settings(max_examples=15, deadline=None, derandomize=True)
@given(st.lists(st.integers(0, 51), min_size=7, max_size=7, unique=True), st.integers(0, 2**32 - 1))
def test_monte_carlo_agrees_with_exact_on_random_flops(cards, seed):
    a, b, board = cards[:2], cards[2:4], cards[4:7]
    exact = exact_equity([a, b], board).equities[0]
    mc = monte_carlo_equity([a, b], board, iterations=3000, seed=seed)
    # 5 standard errors plus a floor for spots with (almost) no variance.
    assert abs(mc.equities[0] - exact) <= max(5 * mc.std_errors[0], 0.005)
