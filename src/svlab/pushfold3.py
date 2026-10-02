"""Three-handed push/fold, the start of a Spin & Go: BTN, SB and BB with equal stacks.

The button shoves or folds. If it folds, the blinds play heads-up push/fold
(with the button's cards dead). If it shoves, the small blind calls or folds,
and then the big blind calls or folds, facing either the shove alone or the
shove and a call. Blinds are 0.5 and 1, no antes; EVs are in big blinds, as
each player's change in stack. With equal stacks there are no side pots, and
in a winner-take-all Spin & Go chips are what the prize depends on.

Six decisions, each taken knowing only one's own hand class:

    btn          BTN shoves
    sb_vs_shove  SB calls the BTN's shove
    sb_open      SB shoves after the BTN folds
    bb_vs_btn    BB calls the BTN's shove after the SB folds
    bb_vs_both   BB calls after the BTN shoves and the SB calls
    bb_vs_sb     BB calls the SB's shove after the BTN folds

Each player acts at most once on any path, so a player's best response is
the best action at each of their decisions separately, and the total gain
from best responses (NashConv) is the sum of the regrets at every decision.
NashConv is zero exactly at a Nash equilibrium. With three players the
solver is not guaranteed to converge, which is why the result reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preflop import CLASSES
from .threeway import ThreewayTable, load

DECISIONS = ("btn", "sb_vs_shove", "sb_open", "bb_vs_btn", "bb_vs_both", "bb_vs_sb")
ACTING = {"btn": "BTN", "sb_vs_shove": "SB", "sb_open": "SB", "bb_vs_btn": "BB", "bb_vs_both": "BB", "bb_vs_sb": "BB"}
# The value of the passive action (fold) at each decision, per unit of reach.
FOLD = {"btn": 0.0, "sb_vs_shove": -0.5, "sb_open": -0.5, "bb_vs_btn": -1.0, "bb_vs_both": -1.0, "bb_vs_sb": -1.0}
N = len(CLASSES)


def _keep(t: np.ndarray, axis: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Contracts a (seat i, j, k) tensor with a vector on each of the two other axes."""
    u, v = u.astype(t.dtype, copy=False), v.astype(t.dtype, copy=False)
    if axis == 0:
        out = (t.reshape(N * N, N) @ v).reshape(N, N) @ u  # u on j, v on k
    elif axis == 1:
        out = u @ (t.reshape(N * N, N) @ v).reshape(N, N)  # u on i, v on k
    else:
        out = v @ (u @ t.reshape(N, N * N)).reshape(N, N)  # u on i, v on j
    return out.astype(np.float64)


@dataclass(frozen=True)
class Game3:
    stack: float
    w: np.ndarray  # deal weights, summing to 1
    btn5: np.ndarray  # w times BTN's EV when BTN shoves, SB folds, BB calls
    bb5: np.ndarray
    btn6: np.ndarray  # BTN shoves, SB calls, BB folds
    sb6: np.ndarray
    btn7: np.ndarray  # all three in
    sb7: np.ndarray
    bb7: np.ndarray
    sb3: np.ndarray  # BTN folds, SB shoves, BB calls: SB's EV (BB's is its negative)

    @classmethod
    def from_table(cls, stack: float, table: ThreewayTable | None = None) -> "Game3":
        if not stack > 0:
            raise ValueError("The stack must be positive")
        t = table or load()
        w = t.weights / t.weights.sum()
        hu, q, s = t.heads_up, t.three, stack
        btn_vs_bb = hu.transpose(0, 2, 1)  # [i, j, k] -> BTN i against BB k, SB j dead
        btn_vs_sb = hu  # BTN i against SB j, BB k dead
        sb_vs_bb = hu.transpose(2, 0, 1)  # SB j against BB k, BTN i dead
        pot5, pot6, pot7 = 2 * s + 0.5, 2 * s + 1, 3 * s

        def f32(x: np.ndarray) -> np.ndarray:
            # Single precision halves the memory each contraction reads; the sums come back in double.
            return np.ascontiguousarray(x, dtype=np.float32)

        return cls(
            stack=s,
            w=f32(w),
            btn5=f32(w * (btn_vs_bb * pot5 - s)),
            bb5=f32(w * ((1 - btn_vs_bb) * pot5 - s)),
            btn6=f32(w * (btn_vs_sb * pot6 - s)),
            sb6=f32(w * ((1 - btn_vs_sb) * pot6 - s)),
            btn7=f32(w * (q * pot7 - s)),
            sb7=f32(w * (q.transpose(1, 0, 2) * pot7 - s)),
            bb7=f32(w * (q.transpose(1, 2, 0) * pot7 - s)),
            sb3=f32(w * (sb_vs_bb * 2 * s - s)),
        )


Strategy = dict[str, np.ndarray]  # decision -> probability of the aggressive action (shove or call) per class


def counterfactual(game: Game3, st: Strategy, only: str | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per decision: (value of the aggressive action, value of folding, reach), per class,
    each weighted by how often that class gets to the decision. `only` limits it to one
    player's decisions."""
    b, s1, s2 = st["btn"], st["sb_vs_shove"], st["sb_open"]
    c1, c2, c3 = st["bb_vs_btn"], st["bb_vs_both"], st["bb_vs_sb"]
    one = np.ones(N)
    w = game.w
    out = {}

    if only in (None, "BTN"):
        shove = (_keep(w, 0, 1 - s1, 1 - c1) * 1.5 + _keep(game.btn5, 0, 1 - s1, c1)
                 + _keep(game.btn6, 0, s1, 1 - c2) + _keep(game.btn7, 0, s1, c2))
        reach = _keep(w, 0, one, one)
        out["btn"] = (shove, FOLD["btn"] * reach, reach)

    if only in (None, "SB"):
        reach = _keep(w, 1, b, one)
        out["sb_vs_shove"] = (_keep(game.sb6, 1, b, 1 - c2) + _keep(game.sb7, 1, b, c2), FOLD["sb_vs_shove"] * reach, reach)
        reach = _keep(w, 1, 1 - b, one)
        out["sb_open"] = (_keep(w, 1, 1 - b, 1 - c3) * 1.0 + _keep(game.sb3, 1, 1 - b, c3), FOLD["sb_open"] * reach, reach)

    if only in (None, "BB"):
        reach = _keep(w, 2, b, 1 - s1)
        out["bb_vs_btn"] = (_keep(game.bb5, 2, b, 1 - s1), FOLD["bb_vs_btn"] * reach, reach)
        reach = _keep(w, 2, b, s1)
        out["bb_vs_both"] = (_keep(game.bb7, 2, b, s1), FOLD["bb_vs_both"] * reach, reach)
        reach = _keep(w, 2, 1 - b, s2)
        out["bb_vs_sb"] = (-_keep(game.sb3, 2, 1 - b, s2), FOLD["bb_vs_sb"] * reach, reach)
    return out


def player_values(game: Game3, st: Strategy) -> dict[str, float]:
    """Each player's EV per hand dealt. The three add up to zero."""
    cf = counterfactual(game, st)
    btn_fold = 1 - st["btn"]
    # Every terminal payoff is counted at the decision of the player it belongs to, except one:
    # when the BTN and the SB both fold, the BB wins the small blind without deciding anything.
    values = {"BTN": float((st["btn"] * cf["btn"][0]).sum())}
    values["SB"] = sum(float((st[d] * cf[d][0] + (1 - st[d]) * cf[d][1]).sum()) for d in ("sb_vs_shove", "sb_open"))
    walk = float(btn_fold @ _keep(game.w, 0, 1 - st["sb_open"], np.ones(N)))
    values["BB"] = 0.5 * walk + sum(float((st[d] * cf[d][0] + (1 - st[d]) * cf[d][1]).sum()) for d in ("bb_vs_btn", "bb_vs_both", "bb_vs_sb"))
    return values


def regrets(game: Game3, st: Strategy) -> dict[str, np.ndarray]:
    """What each class could gain at each decision by switching to its best action."""
    out = {}
    for d, (aggressive, fold, _) in counterfactual(game, st).items():
        mixed = st[d] * aggressive + (1 - st[d]) * fold
        out[d] = np.maximum(aggressive, fold) - mixed
    return out


def nash_conv(game: Game3, st: Strategy) -> float:
    """Total gain available to the three players from best responses, in bb per hand."""
    return float(sum(r.sum() for r in regrets(game, st).values()))


@dataclass(frozen=True)
class Solution3:
    stack: float
    strategy: Strategy
    values: dict[str, float]
    nash_conv: float
    iterations: int

    def ev(self, game: Game3) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per decision and class: EV of the aggressive action and of folding, given that the class
        gets to the decision (NaN where it never does)."""
        out = {}
        for d, (aggressive, fold, reach) in counterfactual(game, self.strategy).items():
            with np.errstate(invalid="ignore", divide="ignore"):
                out[d] = (np.where(reach > 0, aggressive / reach, np.nan), np.where(reach > 0, fold / reach, np.nan))
        return out

    def to_json(self, game: Game3) -> dict:
        evs = self.ev(game)

        def num(x: float) -> float | None:
            return None if np.isnan(x) else round(float(x), 6)

        return {
            "format": "svlab-pushfold3/1",
            "stack": self.stack,
            "blinds": [0.5, 1.0],
            "units": "bb",
            "values": {p: round(v, 6) for p, v in self.values.items()},
            "nash_conv": self.nash_conv,
            "iterations": self.iterations,
            "decisions": {
                d: {
                    name: {"freq": round(float(self.strategy[d][i]), 6), "ev": num(evs[d][0][i]), "ev_fold": num(evs[d][1][i])}
                    for i, name in enumerate(CLASSES)
                }
                for d in DECISIONS
            },
        }


def _matching(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0)
    total = pos.sum(axis=1)
    return np.where(total > 0, pos[:, 0] / np.where(total > 0, total, 1), 0.5)


def solve(game: Game3, *, tolerance: float = 1e-5, max_iterations: int = 5_000, check_every: int = 25) -> Solution3:
    """CFR+ with alternating updates by player and linearly weighted averages, until the
    average strategy's NashConv is at most `tolerance` or the iterations run out."""
    regret = {d: np.zeros((N, 2)) for d in DECISIONS}
    total = {d: np.zeros(N) for d in DECISIONS}
    weight = 0.0
    current = {d: np.full(N, 0.5) for d in DECISIONS}
    average = dict(current)
    it = 0
    for it in range(1, max_iterations + 1):
        for player in ("BTN", "SB", "BB"):
            for d, (aggressive, fold, _) in counterfactual(game, current, only=player).items():
                node = current[d] * aggressive + (1 - current[d]) * fold
                regret[d] = np.maximum(regret[d] + np.stack([aggressive - node, fold - node], axis=1), 0)
                current[d] = _matching(regret[d])
        weight += it
        for d in DECISIONS:
            total[d] += it * current[d]
        if it % check_every == 0:
            average = {d: total[d] / weight for d in DECISIONS}
            if nash_conv(game, average) <= tolerance:
                break
    average = {d: total[d] / weight for d in DECISIONS}
    return Solution3(game.stack, average, player_values(game, average), nash_conv(game, average), it)
