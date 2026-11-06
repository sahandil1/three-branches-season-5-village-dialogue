"""A small wandering baseline for Days at Three Branches."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, cast

_new_rng = random.Random


def _heading(value: object) -> float:
    """Read Python and scalar-array numbers without importing a numeric package."""
    try:
        return float(cast(Any, value)) % 360.0
    except (TypeError, ValueError):
        return 0.0


def _number(value: object) -> float | None:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


class Agent:
    """Walk forward, occasionally turn, and turn sooner when blocked."""

    def reset(self, seed: object, observation: object) -> None:
        """Start a fresh walk. Builtins deliberately do not use the session seed."""
        del seed
        self._rng = _new_rng()
        self._heading = _heading(_self_record(observation).get("heading"))
        self._remaining = self._rng.randint(24, 48)
        self._stalls = 0

    def act(self, observation: object) -> dict[str, float | int]:
        """Return a movement-only order."""
        state = _self_record(observation)
        moved = _number(state.get("moved"))
        if moved is not None and moved < 0.05:
            self._stalls += 1
        else:
            self._stalls = 0

        self._remaining -= 1
        if self._stalls >= 3:
            self._heading = (self._heading + self._rng.choice((90.0, 135.0, 180.0, 225.0))) % 360.0
            self._remaining = self._rng.randint(18, 36)
            self._stalls = 0
        elif self._remaining <= 0:
            self._heading = (self._heading + self._rng.choice((-75.0, -45.0, 45.0, 75.0))) % 360.0
            self._remaining = self._rng.randint(24, 48)

        return {"heading": self._heading, "speed": 0.65, "action": 0}


def _self_record(observation: object) -> Mapping[str, Any]:
    """Read the small part of a plain observation this agent needs."""
    if isinstance(observation, Mapping):
        record = observation.get("self")
        if isinstance(record, Mapping):
            return record
    return {}
