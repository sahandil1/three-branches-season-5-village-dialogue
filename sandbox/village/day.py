"""Read the current point in the village day."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def tick(observation: Mapping[str, object]) -> int:
    """Return the current tick of the day, counting up from ``1`` at dawn toward the day's end.

    Use it for anything time-based: scheduling, idling for a fixed number of steps, or noticing
    that the hour of the day has passed.
    """
    return cast(int, observation["tick"])


def phase(observation: Mapping[str, object]) -> str:
    """Return the current phase of the day: ``"dawn"``, ``"morning"``, ``"midday"``, ``"evening"``,
    or ``"night"``. A season with day and night off reports ``"day"``."""
    return cast(str, observation["phase"])


def bell_ringing(observation: Mapping[str, object]) -> bool:
    """Return whether the village bell is ringing right now. Use it to meet at the bell."""
    return bool(observation["bell"])


def parameters(observation: Mapping[str, object]) -> Mapping[str, object]:
    """Return the resolved season settings for this day, such as the ``seat_plan`` and whether
    day and night phases are on."""
    return cast(Mapping[str, object], observation["parameters"])
