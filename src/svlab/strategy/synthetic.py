"""Synthetic strategies that look like a solver export. They are not
solver output and not an equilibrium.

The fixtures in this repository come from here. Each combo's strategy is
driven by its equity against the opponent's range at the node, estimated
with the lab's own Monte Carlo engine, and a crude EV model per action:

- check: a share of the pot proportional to equity;
- bet or raise: a fold share that grows with the size, and a showdown
  against a calling range that gets stronger as the size grows (equity
  raised to a power above 1);
- call: equity times the final pot, minus the call; fold: 0.

Frequencies are a softmax of those EVs, so higher-EV actions are played
more and close calls are mixed. The point is data with the shape and the
internal consistency of an export (distributions, reach, EVs, suit
symmetry), with numbers that vary by hand in a plausible way. Nothing here
should be read as poker advice.

Everything is seeded. Suit symmetry holds by construction: one estimate is
made per class of suit-isomorphic combos and copied to every member.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import replace
from typing import Callable, Sequence

from svlab.cards import FULL_DECK, RANKS
from svlab.evaluator import evaluate

from .combos import board_symmetries, combo_key, combo_text, hand_class_combos, parse_board, parse_combo, permute_cards
from .model import Action, ComboStrategy, Node, Spot, Strategy
from .rules import previous_own_decision

Range = dict[frozenset[int], float]


# ------------------------------------------------------------------ ranges


def _suited_down_to(high: str, low: str, kind: str) -> list[str]:
    hi = RANKS.index(high)
    return [f"{high}{RANKS[r]}{kind}" for r in range(RANKS.index(low), hi)]


def _build(classes: dict[str, float], board: set[int]) -> Range:
    out: Range = {}
    for label, weight in classes.items():
        for cards in hand_class_combos(label):
            if weight > 0 and not set(cards) & board:
                out[combo_key(cards)] = weight
    return out


def button_open(board: set[int]) -> Range:
    """A rough button opening range, about 45% of hands."""
    c: dict[str, float] = {f"{r}{r}": 1.0 for r in RANKS}
    for high, low in (("A", "2"), ("K", "2"), ("Q", "5"), ("J", "7"), ("T", "7"), ("9", "6"), ("8", "6"), ("7", "5"), ("6", "4"), ("5", "4")):
        c.update(dict.fromkeys(_suited_down_to(high, low, "s"), 1.0))
    for high, low, w in (("A", "8", 1.0), ("K", "T", 1.0), ("Q", "T", 1.0), ("J", "T", 1.0)):
        c.update(dict.fromkeys(_suited_down_to(high, low, "o"), w))
    c.update(dict.fromkeys(["A7o", "A6o", "A5o", "A4o", "K9o", "Q9o", "J9o", "T9o"], 0.5))
    return _build(c, board)


def big_blind_call(board: set[int]) -> Range:
    """A rough big blind calling range against a button open, with most of
    the strongest hands removed (they would have re-raised)."""
    c: dict[str, float] = {f"{r}{r}": 1.0 for r in "23456789TJ"}
    c["QQ"] = 0.3
    for high, low in (("A", "2"), ("K", "2"), ("Q", "2"), ("J", "4"), ("T", "6"), ("9", "6"), ("8", "5"), ("7", "4"), ("6", "3"), ("5", "3"), ("4", "3")):
        c.update(dict.fromkeys(_suited_down_to(high, low, "s"), 1.0))
    for high, low in (("A", "2"), ("K", "7"), ("Q", "8"), ("J", "8"), ("T", "8"), ("9", "8"), ("8", "7"), ("7", "6")):
        c.update(dict.fromkeys(_suited_down_to(high, low, "o"), 1.0))
    c["AKs"], c["AKo"] = 0.3, 0.3
    return _build(c, board)


# ---------------------------------------------------------------- equity


def _equity(hero: Sequence[int], villain: Range, board: Sequence[int], iterations: int, seed: int) -> float:
    """Monte Carlo equity of one combo against a weighted range, using the
    lab's evaluator. Villain combos that collide with the hero are skipped."""
    rng = random.Random(seed)
    pool = [(sorted(k), w) for k, w in villain.items() if not k & set(hero) and w > 0]
    hands, weights = [h for h, _ in pool], [w for _, w in pool]
    missing = 5 - len(board)
    total = 0.0
    for _ in range(iterations):
        v = rng.choices(hands, weights)[0]
        deck = [c for c in FULL_DECK if c not in hero and c not in v and c not in board]
        runout = [*board, *rng.sample(deck, missing)]
        a, b = evaluate([*hero, *runout]), evaluate([*v, *runout])
        total += 1.0 if a > b else 0.5 if a == b else 0.0
    return total / iterations


# -------------------------------------------------------------- EV model


def _action_evs(node: Node, equity: float, in_position: bool) -> dict[str, float]:
    pot, call = node.pot, node.to_call
    realise = 0.95 if in_position else 0.85
    leading = call == 0 and not in_position
    evs = {}
    for a in node.actions:
        if a.kind == "check":
            ev = realise * equity * pot
        elif a.kind == "fold":
            ev = 0.0
        elif a.kind == "call":
            ev = equity * (pot + call) - call
        else:
            size = a.size or 0.0
            ratio = (size - call) / pot
            # Bigger bets fold out more hands and get called by stronger ones.
            fold = 0.45 * (size - call) / (pot + size) * (0.5 if leading else 1.0)
            called = equity ** (1 + 0.5 * ratio + (0.3 if leading else 0.0))
            ev = fold * pot + (1 - fold) * (called * (pot + 2 * size - call) - size)
        evs[a.id] = ev
    return evs


def _softmax(evs: dict[str, float], temperature: float) -> dict[str, float]:
    top = max(evs.values())
    raw = {a: math.exp((v - top) / temperature) for a, v in evs.items()}
    total = sum(raw.values())
    return {a: v / total for a, v in raw.items()}


def _rounded_distribution(freqs: dict[str, float], digits: int = 4) -> dict[str, float]:
    """Rounds and puts the rounding error on the largest entry, so the
    written frequencies still add up to 1."""
    out = {a: round(f, digits) for a, f in freqs.items()}
    top = max(out, key=lambda a: out[a])
    out[top] = round(1.0 - sum(v for a, v in out.items() if a != top), digits)
    return out


def _combo(cards: Sequence[int], weight: float, freqs: dict[str, float], evs: dict[str, float]) -> ComboStrategy:
    freqs = _rounded_distribution(freqs)
    evs = {a: round(v, 3) for a, v in evs.items()}
    ev = round(math.fsum(freqs[a] * evs[a] for a in freqs), 3)
    return ComboStrategy(combo_text(list(cards)), round(weight, 4), freqs, evs, ev)


# ------------------------------------------------------------------- tree


def _flop_tree(pot: float, stack: float) -> list[tuple[tuple[str, ...], str, float, float, tuple[Action, ...]]]:
    third, three_quarters = round(pot / 3, 1), round(pot * 0.75, 1)
    return [
        ((), "BB", pot, 0.0, (Action("check", "check"), Action("bet33", "bet", third))),
        (("check",), "BTN", pot, 0.0, (Action("check", "check"), Action("bet33", "bet", third), Action("bet75", "bet", three_quarters))),
        (
            ("check", "bet75"),
            "BB",
            round(pot + three_quarters, 1),
            three_quarters,
            (Action("fold", "fold"), Action("call", "call"), Action("raise", "raise", round(3.3 * three_quarters, 1)), Action("allin", "allin", stack)),
        ),
    ]


def generate(
    board_text: str,
    *,
    seed: int = 1,
    iterations: int = 250,
    temperature: float = 0.4,
    pot: float = 5.5,
    stack: float = 97.5,
    description: str = "",
) -> Strategy:
    """A synthetic single-raised-pot flop strategy, button against big blind:
    the big blind acts first, the button acts after a check, and the big
    blind faces a 75% pot bet."""
    board = parse_board(board_text)
    dead = set(board)
    starting = {"BTN": button_open(dead), "BB": big_blind_call(dead)}
    symmetries = board_symmetries(board)
    nodes: list[Node] = []
    # Each player's reach so far: the starting range times the actions taken.
    reach = {p: dict(r) for p, r in starting.items()}
    for index, (path, player, node_pot, to_call, actions) in enumerate(_flop_tree(pot, stack)):
        node = Node(path, player, node_pot, stack, to_call, actions, ())
        villain = reach["BTN" if player == "BB" else "BB"]
        if path:
            # The opponent's range was narrowed by the action that led here.
            previous = nodes[-1]
            villain = {
                k: round(w * _freqs_of(previous, k).get(path[-1], 0.0), 6) for k, w in reach[previous.player].items()
            }
            reach[previous.player] = villain
        combos: dict[frozenset[int], ComboStrategy] = {}
        for key, weight in sorted(reach[player].items(), key=lambda kv: sorted(kv[0], reverse=True), reverse=True):
            if key in combos or round(weight, 4) <= 0:
                continue
            orbit = {key} | {combo_key(permute_cards(sorted(key), p)) for p in symmetries}
            rep = min(orbit, key=lambda k: combo_text(sorted(k)))
            label = combo_text(sorted(rep))
            equity = _equity(sorted(rep), villain, board, iterations, zlib.crc32(f"{seed}:{index}:{label}".encode()))
            evs = _action_evs(node, equity, in_position=player == "BTN")
            freqs = _softmax(evs, temperature)
            for member in orbit:
                if round(reach[player].get(member, 0.0), 4) > 0:
                    combos[member] = _combo(sorted(member), reach[player][member], freqs, evs)
        node = replace(node, combos=tuple(combos[k] for k in sorted(combos, key=_order)))
        nodes.append(node)
        # Written weights are rounded; carry the rounded values forward so
        # that the file's reach is consistent with its own numbers.
        reach[player] = {combo_key(parse_combo(c.combo)): c.weight for c in node.combos}
    spot = Spot("NLHE", ("BTN", "BB"), pot, stack, board_text, ("BTN raise 2.5", "BB call"))
    meta = {
        "synthetic": True,
        "description": description or f"Synthetic single-raised-pot flop strategy on {board_text}. Not solver output.",
        "generator": "svlab.strategy.synthetic.generate",
        "seed": seed,
        "units": "bb",
    }
    return Strategy(spot=spot, nodes=tuple(nodes), meta=meta)


def _freqs_of(node: Node, key: frozenset[int]) -> dict[str, float]:
    for c in node.combos:
        if combo_key(parse_combo(c.combo)) == key:
            return c.freqs
    return {}


def _order(key: frozenset[int]) -> tuple[int, ...]:
    return tuple(sorted(key, reverse=True))


# ----------------------------------------------------- controlled changes


FreqChange = Callable[[Node, ComboStrategy], "dict[str, float] | None"]


def rebuild(strategy: Strategy, change: FreqChange, description: str) -> Strategy:
    """Applies `change` to every combo's frequencies (None removes the
    combo), then recomputes what depends on them so the result is still a
    consistent file: overall EVs, and the ranges at later nodes."""
    nodes: list[Node] = []
    for node in strategy.nodes:
        parent = previous_own_decision(replace(strategy, nodes=tuple(nodes)), node)
        reach: dict[frozenset[int], float] | None = None
        if parent is not None:
            parent_node, taken = parent
            reach = {
                combo_key(parse_combo(p.combo)): round(p.weight * p.freqs.get(taken, 0.0), 4) for p in parent_node.combos
            }
        combos = []
        for c in node.combos:
            key = combo_key(parse_combo(c.combo))
            weight = c.weight if reach is None else reach.get(key, 0.0)
            freqs = change(node, c)
            if weight <= 0 or freqs is None:
                continue
            combos.append(_combo(sorted(key), weight, freqs, c.evs or {}))
        nodes.append(replace(node, combos=tuple(combos)))
    return replace(strategy, nodes=tuple(nodes), meta={**strategy.meta, "description": description})
