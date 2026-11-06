"""Interactive-prop selection and the four catalog transition rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .catalog import PROP_BY_TOKEN
from .geometry import distance, nearest_point
from .rules import PROFILE

if TYPE_CHECKING:
    from .layout import Layout, PlacedProp


def select(layout: Layout, position: tuple[float, float]) -> PlacedProp | None:
    """Return the nearest reachable visible prop, using layout order to break ties."""
    candidates = []
    for index, prop in enumerate(layout.props):
        nearest = nearest_point(position, layout.shape_for(prop))
        if distance(position, nearest) <= PROFILE.prop_reach and layout.line_clear(position, nearest):
            candidates.append((distance(position, nearest), index, prop))
    return min(candidates, default=(0.0, 0, None), key=lambda item: (item[0], item[1]))[2]


def apply(
    states: dict[str, str],
    holders: dict[str, str | None],
    last_held: dict[str, int],
    layout: Layout,
    users: dict[str, str],
    tick: int,
) -> None:
    """Advance every prop's state from the tick's already resolved users."""
    for prop in layout.props:
        kind = PROP_BY_TOKEN[prop.type]
        old_holder = holders[prop.id]
        holder = users.get(prop.id)
        holders[prop.id] = holder
        if kind.transition == "toggle":
            if holder is not None and old_holder != holder:
                states[prop.id] = kind.active_state if states[prop.id] == kind.start else kind.start
        elif kind.transition == "occupancy":
            states[prop.id] = kind.active_state if holder is not None else kind.start
        elif kind.transition == "timed":
            if holder is not None:
                states[prop.id] = kind.active_state
                last_held[prop.id] = tick
            elif tick - last_held[prop.id] >= (kind.duration or 0):
                states[prop.id] = kind.start
