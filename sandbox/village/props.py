"""Read the village's interactive props and preview use selection."""

# pyright: reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

from collections.abc import Mapping
from math import hypot
from typing import cast

from . import me
from ._model import CATALOG, distance_to_shape, line_clear, model, nearest
from .geometry import PROP_REACH

TYPES = tuple(item["token"] for item in CATALOG["props"])


def all(observation: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return every static prop placement in canonical layout order."""
    return cast(tuple[Mapping[str, object], ...], observation["village"]["props"])


def find(observation: Mapping[str, object], prop_id: object) -> Mapping[str, object] | None:
    """Return the static prop placement with this id, or ``None`` when there is no such prop."""
    return next((prop for prop in all(observation) if prop["id"] == prop_id), None)


def seen(observation: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return the dynamic states of props in the current vision cone."""
    return cast(tuple[Mapping[str, object], ...], observation["props"])


def state(observation: Mapping[str, object], prop_id: str) -> str | None:
    """Return a visible prop's state, or None when it is not visible."""
    return next((cast(str, prop["state"]) for prop in seen(observation) if prop["prop"] == prop_id), None)


def in_reach(observation: Mapping[str, object]):
    """Return static props within geometric use reach, without considering walls."""
    village_model = model(observation)
    position = _point(me.position(observation))
    return tuple(
        prop
        for prop, shape in zip(all(observation), village_model.prop_shapes, strict=True)
        if distance_to_shape(position, shape) <= PROP_REACH
    )


def usable(observation: Mapping[str, object]):
    """Preview the engine's nearest wall-clear use target, ignoring current holders."""
    village_model = model(observation)
    current = me.position(observation)
    position = _point(current)
    candidates = []
    for index, (prop, shape) in enumerate(zip(all(observation), village_model.prop_shapes, strict=True)):
        nearest_point = nearest(position, shape)
        distance = hypot(position[0] - nearest_point[0], position[1] - nearest_point[1])
        if distance <= PROP_REACH and line_clear(
            village_model, current, {"x": nearest_point[0], "y": nearest_point[1]}
        ):
            candidates.append((distance, index, prop))
    return min(candidates, default=(0.0, 0, None), key=lambda candidate: (candidate[0], candidate[1]))[2]


def _point(value: Mapping[str, object]) -> tuple[float, float]:
    return float(value["x"]), float(value["y"])
