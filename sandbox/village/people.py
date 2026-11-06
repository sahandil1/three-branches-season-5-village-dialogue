"""Read villagers in sight, nearby, and in the roster."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

_NPC_ID = re.compile(r"player_[1-9][0-9]*\Z")


def seen(observation: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return every character currently in your vision cone, each with an id, position, heading,
    last tick's movement, and expression. Order is stable."""
    return cast(tuple[Mapping[str, object], ...], observation["seen"])


def nearby(observation: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return every character currently within hearing range and line, each with only an id and
    position. This is the cheaper check for "is someone around"."""
    return cast(tuple[Mapping[str, object], ...], observation["nearby"])


def find(observation: Mapping[str, object], player_id: object) -> Mapping[str, object] | None:
    """Return the character with this id from ``seen``, else from ``nearby``, or ``None`` when you
    can neither see nor hear them. Only a seen record carries heading, movement, and expression."""
    return next(
        (person for person in (*seen(observation), *nearby(observation)) if person["id"] == player_id), None
    )


def roster(observation: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return every character's stable id and home building id. Standing knowledge, identical for
    all characters and constant for the whole day."""
    return cast(tuple[Mapping[str, object], ...], observation["roster"])


def is_visitor(player_id: str) -> bool:
    """Return whether an id is exactly the visitor's player id."""
    return player_id == "player_0"


def is_npc(player_id: str) -> bool:
    """Return whether an id has the canonical positive player-number form."""
    return _NPC_ID.fullmatch(player_id) is not None
