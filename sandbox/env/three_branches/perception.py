"""Ruleset perception projections from the mutable day state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .catalog import PROP_BY_TOKEN
from .geometry import distance, nearest_point, point_in_cone
from .rules import PROFILE

_BELL = "bell"

if TYPE_CHECKING:
    from .engine import Character, Day


def expression(character: Character) -> dict[str, str]:
    return {"type": character.expression_type, "target": character.expression_target}


def character_record(character: Character) -> dict[str, object]:
    return {
        "id": character.id,
        "position": {"x": character.position[0], "y": character.position[1]},
        "heading": character.heading,
        "moved": character.moved,
        "expression": expression(character),
    }


def _visible(day: Day, observer: Character, point: tuple[float, float]) -> bool:
    return point_in_cone(
        observer.position,
        observer.heading,
        point,
        PROFILE.vision_degrees,
        PROFILE.vision_range,
    ) and day.layout.line_clear(observer.position, point)


def observe(day: Day, character_id: str) -> dict[str, object]:
    """Return the dynamic perception leaves for one character."""
    observer = day.characters[character_id]
    seen = tuple(
        character_record(other)
        for other in day.characters.values()
        if other.id != character_id and _visible(day, observer, other.position)
    )
    nearby = tuple(
        {"id": other.id, "position": {"x": other.position[0], "y": other.position[1]}}
        for other in day.characters.values()
        if other.id != character_id
        and distance(observer.position, other.position) <= PROFILE.hearing_range
        and day.layout.line_clear(observer.position, other.position)
    )
    props = tuple(
        {"prop": prop.id, "state": day.prop_states[prop.id]}
        for prop in day.layout.props
        if _visible(day, observer, nearest_point(observer.position, day.layout.shape_for(prop)))
    )
    return {
        "self": character_record(observer),
        "seen": seen,
        "nearby": nearby,
        "props": props,
        "bell": int(bell_rings(day)),
    }


def bell_rings(day: Day) -> bool:
    """Report the village-wide signal, which every character hears at any distance."""
    ringing = PROP_BY_TOKEN[_BELL].active_state
    return any(day.prop_states[prop.id] == ringing for prop in day.layout.props if prop.type == _BELL)


def can_hear(day: Day, sender_id: str, recipient_id: str) -> bool:
    sender = day.characters[sender_id]
    recipient = day.characters[recipient_id]
    return distance(sender.position, recipient.position) <= PROFILE.hearing_range and day.layout.line_clear(
        sender.position, recipient.position
    )
