"""Read your villager's current state."""

# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import cast


def player_id(observation: Mapping[str, object]) -> str:
    """Return your own player id, such as ``"player_1"``. Stable for the whole day and the same
    id chat messages address you by."""
    return str(_self(observation)["id"])


def position(observation: Mapping[str, object]) -> Mapping[str, float]:
    """Return your current position in metres from the village southwest corner, as an
    ``{"x": float, "y": float}`` mapping.

    Both values arrive as NumPy ``float32`` scalars, which behave like floats in arithmetic and
    formatting. Call ``float()`` on one when you need a plain Python float, such as for JSON.
    """
    return cast(Mapping[str, float], _self(observation)["position"])


def heading(observation: Mapping[str, object]) -> float:
    """Return the direction you are facing in degrees, from ``0.0`` (east) counter-clockwise
    through ``360.0``, so ``90.0`` is north, ``180.0`` west, and ``270.0`` south.

    Feed it to ``action.walk`` or ``action.stand``. It arrives as a NumPy ``float32`` scalar; call
    ``float()`` when you need a plain Python float.
    """
    return cast(float, _self(observation)["heading"])


def moved(observation: Mapping[str, object]) -> float:
    """Return how far you moved on the previous tick, in metres, as a NumPy ``float32`` scalar.
    Zero when you stood still."""
    return cast(float, _self(observation)["moved"])


def expression(observation: Mapping[str, object]) -> Mapping[str, str]:
    """Return the expression you are currently showing, as a ``{"type": str, "target": str}``
    mapping.

    ``type`` is ``"none"``, ``"use"``, or one of the emote names. ``target`` is the id of the prop
    you are using when ``type`` is ``"use"``, and ``"none"`` otherwise.
    """
    return cast(Mapping[str, str], _self(observation)["expression"])


def home(observation: Mapping[str, object]) -> str:
    """Return the id of the building you live in, read from the roster. Raises ``KeyError`` for
    an id the roster does not carry."""
    own_id = player_id(observation)
    for entry in observation["roster"]:
        if entry["id"] == own_id:
            return str(entry["home"])
    raise KeyError(own_id)


def rng(observation: Mapping[str, object], session_seed: object) -> random.Random:
    """Return the stable private stream for this seed and player id."""
    return random.Random(f"three_branches:{session_seed!r}:{player_id(observation)}")


def _self(observation: Mapping[str, object]):
    return observation["self"]
