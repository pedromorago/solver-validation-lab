"""Checks for a poker equity engine where most answers have no oracle to
compare against.

Every check takes an Engine and raises AssertionError with a readable
message when it fails. Each one names the kind of oracle it relies on:

- reference:     hand-written expected values, only where they are obvious
- differential:  an independent implementation (treys) must agree
- invariant:     properties that must hold for any input
- metamorphic:   a known change to the input must change the output in a known way
- statistical:   a sampled estimate must fall within its error bars
- regression:    results must match baselines recorded from an earlier version
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

from svlab import parse_many as P
from svlab.cards import FULL_DECK, rank, suit
from svlab.evaluator import HandCategory

from .engine import Engine
from .spots import FLOP_SPOTS, RIVER_SPOTS, TURN_SPOTS

BASELINES = Path(__file__).with_name("baselines.json")


@dataclass(frozen=True)
class Check:
    name: str
    oracle: str
    run: Callable[[Engine], None]


CHECKS: list[Check] = []


def check(oracle: str):
    def register(fn: Callable[[Engine], None]) -> Callable[[Engine], None]:
        CHECKS.append(Check(fn.__name__, oracle, fn))
        return fn

    return register


def _random_cards(rng: random.Random, n: int) -> list[int]:
    return rng.sample(FULL_DECK, n)


def _permute_suits(cards, perm):
    return [rank(c) * 4 + perm[suit(c)] for c in cards]


# ------------------------------------------------------------------ evaluator


@check("reference")
def categories_of_textbook_hands(engine: Engine) -> None:
    cases = {
        "As Ks Qs Js Ts 2c 3d": HandCategory.STRAIGHT_FLUSH,
        "5h 4h 3h 2h Ah Kd Qc": HandCategory.STRAIGHT_FLUSH,
        "9c 9d 9h 9s 2c 3d 4h": HandCategory.QUADS,
        "Kc Kd Kh 2s 2c 7d 8h": HandCategory.FULL_HOUSE,
        "Ah 9h 7h 4h 2h Kd Kc": HandCategory.FLUSH,
        "Ad 2c 3h 4s 5d Kc Qh": HandCategory.STRAIGHT,
        "Td Jc Qh Ks Ad 2c 3h": HandCategory.STRAIGHT,
        "7c 7d 7h As 2c 9d Jh": HandCategory.TRIPS,
        "8c 8d 3h 3s Ac 9d Jh": HandCategory.TWO_PAIR,
        "Qc Qd 3h 5s Ac 9d Jh": HandCategory.PAIR,
        "Ac Qd 9h 7s 5c 3d 2h": HandCategory.HIGH_CARD,
    }
    for text, expected in cases.items():
        got = engine.evaluate(P(text))[0]
        assert got == expected, f"{text}: expected {expected.name}, got {HandCategory(got).name}"


@check("reference")
def kickers_decide_equal_pairs(engine: Engine) -> None:
    board = P("Qc Qd 9h 5s 2c")
    ace, king = engine.evaluate(P("Ah 3d") + board), engine.evaluate(P("Kh 3s") + board)
    assert ace > king, "A-kicker should beat K-kicker with the same pair on board"


@check("differential")
def agrees_with_treys(engine: Engine, pairs: int = 3000, seed: int = 1) -> None:
    """Random pairs of seven-card hands must be ordered the same way as by an
    independent evaluator. treys ranks 1 as the best hand, so its order is reversed."""
    from treys import Card, Evaluator as TreysEvaluator

    treys = TreysEvaluator()
    to_treys = lambda cards: [Card.new(("23456789TJQKA"[rank(c)]) + "cdhs"[suit(c)]) for c in cards]
    rng = random.Random(seed)
    for _ in range(pairs):
        cards = _random_cards(rng, 9)
        board, a, b = cards[:5], cards[5:7], cards[7:9]
        ours = (engine.evaluate(a + board) > engine.evaluate(b + board)) - (
            engine.evaluate(a + board) < engine.evaluate(b + board)
        )
        ta, tb = treys.evaluate(to_treys(board), to_treys(a)), treys.evaluate(to_treys(board), to_treys(b))
        theirs = (ta < tb) - (ta > tb)
        assert ours == theirs, f"disagrees with treys on board {board} hands {a} vs {b}"


@check("invariant")
def evaluation_ignores_card_order(engine: Engine, trials: int = 500, seed: int = 2) -> None:
    rng = random.Random(seed)
    for _ in range(trials):
        cards = _random_cards(rng, 7)
        shuffled = cards[:]
        rng.shuffle(shuffled)
        assert engine.evaluate(cards) == engine.evaluate(shuffled), f"order changed the value of {cards}"


@check("metamorphic")
def evaluation_ignores_suit_names(engine: Engine, trials: int = 500, seed: int = 3) -> None:
    """Renaming the suits (hearts become spades, and so on) cannot change a hand's value."""
    rng = random.Random(seed)
    for _ in range(trials):
        cards = _random_cards(rng, 7)
        perm = rng.sample(range(4), 4)
        assert engine.evaluate(cards) == engine.evaluate(_permute_suits(cards, perm)), (
            f"suit permutation {perm} changed the value of {cards}"
        )


# --------------------------------------------------------------- exact equity


@check("invariant")
def shares_add_up_to_the_pot(engine: Engine) -> None:
    for name, hands, board in FLOP_SPOTS + TURN_SPOTS + RIVER_SPOTS:
        r = engine.exact(hands, board)
        assert sum(r.units) == r.runouts * r.units_per_runout, f"{name}: shares don't add up to the pot"


@check("invariant")
def a_complete_board_has_no_uncertainty(engine: Engine) -> None:
    allowed = {0.0, 0.5, 1.0}
    for name, hands, board in RIVER_SPOTS:
        r = engine.exact(hands, board)
        assert r.runouts == 1 and set(r.equities) <= allowed, f"{name}: river equities {r.equities}"


@check("metamorphic")
def swapping_players_swaps_equities(engine: Engine) -> None:
    for name, hands, board in FLOP_SPOTS + TURN_SPOTS:
        forward = engine.exact(hands, board)
        backward = engine.exact(list(reversed(hands)), board)
        assert forward.units == tuple(reversed(backward.units)), f"{name}: order of players changed the result"


@check("metamorphic")
def renaming_suits_keeps_equity(engine: Engine) -> None:
    rng = random.Random(4)
    for name, hands, board in FLOP_SPOTS + TURN_SPOTS:
        perm = rng.sample(range(4), 4)
        moved = engine.exact([_permute_suits(h, perm) for h in hands], _permute_suits(board, perm))
        assert moved.units == engine.exact(hands, board).units, f"{name}: suit permutation {perm} changed equity"


@check("metamorphic")
def flop_equity_is_the_average_over_turn_cards(engine: Engine) -> None:
    """Law of total probability: equity on the flop must equal the mean of
    the equities after each possible turn card, exactly."""
    from svlab.equity import remaining_deck

    for name, hands, board in FLOP_SPOTS:
        flop = engine.exact(hands, board)
        turns = remaining_deck(hands, board)
        totals = [Fraction(0)] * len(hands)
        for card in turns:
            r = engine.exact(hands, [*board, card])
            for i, u in enumerate(r.units):
                totals[i] += Fraction(u, r.runouts * r.units_per_runout)
        for i, total in enumerate(totals):
            expected = Fraction(flop.units[i], flop.runouts * flop.units_per_runout)
            assert total / len(turns) == expected, f"{name}: player {i} flop equity isn't the average over turns"


@check("reference")
def identical_hands_split(engine: Engine) -> None:
    """Same ranks, suits that can't make a flush for either: every runout is a split."""
    r = engine.exact([P("AhKd"), P("AdKh")], P("7c 7s 2c 3s"))
    assert r.equities == (0.5, 0.5), f"expected an even split, got {r.equities}"


@check("regression")
def matches_recorded_baselines(engine: Engine) -> None:
    baselines = json.loads(BASELINES.read_text())
    for spot in baselines["spots"]:
        r = engine.exact([P(h) for h in spot["hands"]], P(spot["board"]))
        assert list(r.units) == spot["units"] and r.runouts == spot["runouts"], (
            f"{spot['name']}: {r.units}/{r.runouts} differs from the baseline {spot['units']}/{spot['runouts']}"
        )


# ---------------------------------------------------------------- Monte Carlo


@check("statistical")
def monte_carlo_falls_within_its_error_bars(engine: Engine, iterations: int = 4000) -> None:
    """Each estimate must be within 4.5 standard errors of the exact value.
    Seeds are fixed, so the check is deterministic; with ~60 comparisons a
    correct engine passes this bound with overwhelming probability."""
    for seed, (name, hands, board) in enumerate(FLOP_SPOTS + TURN_SPOTS):
        exact = engine.exact(hands, board).equities
        mc = engine.monte_carlo(hands, board, iterations=iterations, seed=seed)
        for i, (e, m, se) in enumerate(zip(exact, mc.equities, mc.std_errors)):
            tolerance = max(4.5 * se, 0.004)
            assert abs(m - e) <= tolerance, f"{name}: player {i} estimate {m:.4f} vs exact {e:.4f} (±{tolerance:.4f})"


@check("statistical")
def monte_carlo_error_shrinks_with_more_samples(engine: Engine) -> None:
    """Quadrupling the samples should roughly halve the error, averaged over seeds."""
    name, hands, board = FLOP_SPOTS[0]
    exact = engine.exact(hands, board).equities[0]

    def mean_error(n: int) -> float:
        return sum(abs(engine.monte_carlo(hands, board, iterations=n, seed=s).equities[0] - exact) for s in range(12)) / 12

    small, large = mean_error(500), mean_error(8000)
    assert large < small * 0.5, f"{name}: error went from {small:.4f} to {large:.4f} with 16x the samples"


@check("invariant")
def sampled_boards_are_legal(engine: Engine, iterations: int = 2000) -> None:
    """Every sampled runout must use distinct cards that aren't in anyone's hand."""
    from svlab.equity import remaining_deck

    seen: list[list[int]] = []

    def spy(deck, missing, rng):
        runout = engine.sampler(deck, missing, rng)
        seen.append(runout)
        return runout

    hands, board = [P("AhKh"), P("QsQd")], []
    engine.monte_carlo_impl(hands, board, iterations=iterations, seed=7, evaluate=engine.evaluate, sampler=spy)
    legal = set(remaining_deck(hands, board))
    for runout in seen:
        assert len(runout) == 5, f"runout of {len(runout)} cards"
        assert len(set(runout)) == len(runout) and set(runout) <= legal, f"illegal runout {runout}"
