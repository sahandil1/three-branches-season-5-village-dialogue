"""Validated static rules for Days at Three Branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any

from .validation import boolean, mapping, nonnegative_number, positive_int, positive_number, text


@dataclass(frozen=True, slots=True)
class Frame:
    cells_x: int
    cells_y: int
    cell_size: float

    @property
    def width(self) -> float:
        return self.cells_x * self.cell_size

    @property
    def height(self) -> float:
        return self.cells_y * self.cell_size


@dataclass(frozen=True, slots=True)
class Ground:
    code: str
    name: str
    speed: float
    passable: bool
    blocks_sight: bool


@dataclass(frozen=True, slots=True)
class Profile:
    body_radius: float
    vision_degrees: float
    vision_range: float
    hearing_range: float
    prop_reach: float


@dataclass(frozen=True, slots=True)
class Phase:
    name: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Rules:
    frame: Frame
    grounds: tuple[Ground, ...]
    fill: str
    emotes: tuple[str, ...]
    profile: Profile
    phases: tuple[Phase, ...]
    off_phase: str
    day_ticks: int
    physics_substeps: int

    @property
    def ground_by_code(self) -> MappingProxyType[str, Ground]:
        return MappingProxyType({ground.code: ground for ground in self.grounds})


def load(data: Any) -> Rules:
    """Validate a decoded rules document without accepting unrecognised data."""
    root = mapping(
        data,
        "rules",
        {
            "frame",
            "grounds",
            "fill",
            "emotes",
            "profile",
            "phases",
            "off_phase",
            "day_ticks",
            "physics_substeps",
        },
    )
    frame_data = mapping(root["frame"], "rules.frame", {"cells_x", "cells_y", "cell_size"})
    cell_size = positive_number(frame_data["cell_size"], "rules.frame.cell_size")
    # Layout shapes, blocked rectangles, and residence poses read a cell index as a metre
    # coordinate, so the village measures exactly one metre per cell.
    if cell_size != 1.0:
        raise ValueError("rules.frame.cell_size must be 1.0")
    frame = Frame(
        positive_int(frame_data["cells_x"], "rules.frame.cells_x"),
        positive_int(frame_data["cells_y"], "rules.frame.cells_y"),
        cell_size,
    )
    if not isinstance(root["grounds"], list) or not root["grounds"]:
        raise ValueError("rules.grounds must be a non-empty array")
    ground_entries = (
        mapping(item, "rules.ground", {"code", "name", "speed", "passable", "blocks_sight"})
        for item in root["grounds"]
    )
    grounds = tuple(
        Ground(
            text(entry["code"], "rules.ground.code", length=1),
            text(entry["name"], "rules.ground.name"),
            nonnegative_number(entry["speed"], "rules.ground.speed"),
            boolean(entry["passable"], "rules.ground.passable"),
            boolean(entry["blocks_sight"], "rules.ground.blocks_sight"),
        )
        for entry in ground_entries
    )
    for ground in grounds:
        if ground.passable and ground.speed <= 0:
            raise ValueError(f"rules.ground {ground.code} is passable, so it needs a speed above zero")
    if len({ground.code for ground in grounds}) != len(grounds):
        raise ValueError("rules.ground codes must be unique")
    fill = text(root["fill"], "rules.fill", length=1)
    by_code = {ground.code: ground for ground in grounds}
    if fill not in by_code or not by_code[fill].passable:
        raise ValueError("rules.fill must name a passable ground")
    if not any(not ground.passable for ground in grounds):
        raise ValueError("rules.grounds needs an impassable ground")
    emotes_data = root["emotes"]
    if not isinstance(emotes_data, list) or len(emotes_data) != 9:
        raise ValueError("rules.emotes must list exactly nine emotes")
    emotes = tuple(text(emote, "rules.emote") for emote in emotes_data)
    if len(set(emotes)) != len(emotes):
        raise ValueError("rules.emotes must be unique")
    profile_data = mapping(
        root["profile"],
        "rules.profile",
        {"body_radius", "vision_degrees", "vision_range", "hearing_range", "prop_reach"},
    )
    profile = Profile(
        *(positive_number(profile_data[key], f"rules.profile.{key}") for key in Profile.__dataclass_fields__)
    )
    phases_data = root["phases"]
    if not isinstance(phases_data, list) or not phases_data:
        raise ValueError("rules.phases must be a non-empty array")
    phases = tuple(
        Phase(
            text(entry["name"], "rules.phase.name"),
            positive_int(entry["start"], "rules.phase.start"),
            positive_int(entry["end"], "rules.phase.end"),
        )
        for entry in (mapping(item, "rules.phase", {"name", "start", "end"}) for item in phases_data)
    )
    if len({phase.name for phase in phases}) != len(phases) or any(
        phase.end < phase.start for phase in phases
    ):
        raise ValueError("rules.phases must be uniquely named and forward running")
    if phases[0].start != 1 or any(
        left.end + 1 != right.start for left, right in zip(phases, phases[1:], strict=False)
    ):
        raise ValueError("rules.phases must be contiguous from tick 1")
    day_ticks = positive_int(root["day_ticks"], "rules.day_ticks")
    if phases[-1].end != day_ticks:
        raise ValueError("rules.phases must end with the day")
    off_phase = text(root["off_phase"], "rules.off_phase")
    return Rules(
        frame,
        grounds,
        fill,
        emotes,
        profile,
        phases,
        off_phase,
        day_ticks,
        positive_int(root["physics_substeps"], "rules.physics_substeps"),
    )


RULES = load(json.loads(resources.files(__package__).joinpath("rules.json").read_text(encoding="utf-8")))
FRAME = RULES.frame
GROUND_BY_CODE = RULES.ground_by_code
EMOTES = RULES.emotes
PROFILE = RULES.profile
