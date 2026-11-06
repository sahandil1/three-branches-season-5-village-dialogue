"""JSON-safe recording overlays for the temporary Three Branches renderer."""

from __future__ import annotations

from typing import Any


def extract_overlay_static(env: Any) -> dict[str, object]:
    """Return the immutable village once for the recording header."""
    # Observations use tuples because their Gymnasium spaces do. A recording crosses a JSON
    # boundary, so the header states those sequences as lists instead of leaving every reader
    # to normalize them.
    village = env.day.layout.village()
    for key in ("ground", "buildings", "props", "scenery"):
        village[key] = list(village[key])
    return village


def extract_overlay(env: Any) -> dict[str, object]:
    """Project changing state only, rounding movement data to its recording precision."""
    day = env.day
    return {
        "tick": max(1, day.tick),
        "phase": day.phase,
        "characters": [
            {
                "id": character.id,
                "x": round(character.position[0], 2),
                "y": round(character.position[1], 2),
                "heading": round(character.heading, 1),
                "moved": round(character.moved, 2),
                "expression": {"type": character.expression_type, "target": character.expression_target},
            }
            for character in day.characters.values()
        ],
        "props": dict(day.prop_states),
        "terminal": day.terminal,
    }
