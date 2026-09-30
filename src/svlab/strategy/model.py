"""Typed, immutable representation of an exported strategy.

The JSON format is documented in docs/strategy-format.md; `fileformat.py`
converts between the two. Values are kept as written in the file (combo
strings, action ids) so that findings can point at the exact entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FORMAT = "svlab-strategy/1"

ACTION_KINDS = ("fold", "check", "call", "bet", "raise", "allin")
SIZED_KINDS = ("bet", "raise", "allin")


@dataclass(frozen=True)
class Action:
    id: str
    kind: str
    # Chips the player puts in with this action on this street, in the
    # file's units. Only bets, raises and all-ins have a size.
    size: float | None = None


@dataclass(frozen=True)
class ComboStrategy:
    combo: str
    weight: float
    freqs: dict[str, float]
    evs: dict[str, float] | None = None
    ev: float | None = None


@dataclass(frozen=True)
class Node:
    path: tuple[str, ...]
    player: str
    pot: float
    effective_stack: float
    to_call: float
    actions: tuple[Action, ...]
    combos: tuple[ComboStrategy, ...]

    def action(self, action_id: str) -> Action | None:
        return next((a for a in self.actions if a.id == action_id), None)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.actions)


@dataclass(frozen=True)
class Spot:
    game: str
    positions: tuple[str, ...]
    pot: float
    effective_stack: float
    board: str
    preflop: tuple[str, ...] = ()


@dataclass(frozen=True)
class Strategy:
    spot: Spot
    nodes: tuple[Node, ...]
    meta: dict[str, Any] = field(default_factory=dict)
    format: str = FORMAT

    def node(self, path: tuple[str, ...]) -> Node | None:
        return next((n for n in self.nodes if n.path == path), None)

    def node_index(self, path: tuple[str, ...]) -> int | None:
        return next((i for i, n in enumerate(self.nodes) if n.path == path), None)
