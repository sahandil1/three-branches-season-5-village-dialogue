"""Validated building and object definitions for Days at Three Branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from math import isfinite
from types import MappingProxyType
from typing import Any

from .rules import GROUND_BY_CODE
from .validation import mapping, positive_int, positive_number, token

_SHAPES = {"box", "circle"}
_MOVING_TRANSITIONS = {"toggle", "occupancy", "timed"}
_TRANSITIONS = _MOVING_TRANSITIONS | {"none"}


@dataclass(frozen=True, slots=True)
class BuildingType:
    token: str
    width: int
    height: int
    count: int
    door_width: int
    interior_props: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PropType:
    token: str
    width: int
    height: int
    shape: str
    collision_scale: float
    activity: str
    states: tuple[str, ...]
    start: str
    transition: str
    duration: int | None

    @property
    def active_state(self) -> str:
        """Name the state a use produces. Only a prop that transitions has one."""
        return next(state for state in self.states if state != self.start)


@dataclass(frozen=True, slots=True)
class SceneryType:
    token: str
    width: int
    height: int
    shape: str


@dataclass(frozen=True, slots=True)
class Catalog:
    buildings: tuple[BuildingType, ...]
    props: tuple[PropType, ...]
    scenery: tuple[SceneryType, ...]

    @property
    def building_by_token(self) -> MappingProxyType[str, BuildingType]:
        return MappingProxyType({item.token: item for item in self.buildings})

    @property
    def prop_by_token(self) -> MappingProxyType[str, PropType]:
        return MappingProxyType({item.token: item for item in self.props})

    @property
    def scenery_by_token(self) -> MappingProxyType[str, SceneryType]:
        return MappingProxyType({item.token: item for item in self.scenery})


def load(data: Any) -> Catalog:
    """Validate the compact catalog document and return immutable tables."""
    root = mapping(data, "catalog", {"buildings", "props", "scenery"})
    if any(not isinstance(root[key], list) or not root[key] for key in root):
        raise ValueError("catalog sections must each be a non-empty array")
    buildings = tuple(
        BuildingType(
            token(entry["token"], "catalog.building.token"),
            positive_int(entry["width"], "catalog.building.width"),
            positive_int(entry["height"], "catalog.building.height"),
            positive_int(entry["count"], "catalog.building.count"),
            positive_int(entry["door_width"], "catalog.building.door_width"),
            tuple(
                token(item, "catalog.building.interior_props", max_length=12)
                for item in entry["interior_props"]
            ),
        )
        for entry in (
            mapping(
                item,
                "catalog.building",
                {"token", "width", "height", "count", "door_width", "interior_props"},
            )
            for item in root["buildings"]
        )
    )
    props = tuple(
        PropType(
            token(entry["token"], "catalog.prop.token", max_length=12),
            positive_int(entry["width"], "catalog.prop.width"),
            positive_int(entry["height"], "catalog.prop.height"),
            entry["shape"],
            _collision_scale(entry["collision_scale"]),
            token(entry["activity"], "catalog.prop.activity"),
            tuple(token(state, "catalog.prop.states", max_length=9) for state in entry["states"]),
            token(entry["start"], "catalog.prop.start", max_length=9),
            entry["transition"],
            entry["duration"],
        )
        for entry in (
            mapping(
                item,
                "catalog.prop",
                {
                    "token",
                    "width",
                    "height",
                    "shape",
                    "collision_scale",
                    "activity",
                    "states",
                    "start",
                    "transition",
                    "duration",
                },
            )
            for item in root["props"]
        )
    )
    scenery = tuple(
        SceneryType(
            token(entry["token"], "catalog.scenery.token", max_length=12),
            positive_int(entry["width"], "catalog.scenery.width"),
            positive_int(entry["height"], "catalog.scenery.height"),
            entry["shape"],
        )
        for entry in (
            mapping(item, "catalog.scenery", {"token", "width", "height", "shape"})
            for item in root["scenery"]
        )
    )
    names = [item.token for item in (*buildings, *props, *scenery)]
    if len(set(names)) != len(names):
        raise ValueError("catalog tokens must be unique")
    prop_tokens = {item.token for item in props}
    for building in buildings:
        if building.door_width > max(building.width, building.height):
            raise ValueError(f"catalog.building {building.token} has a doorway that does not fit")
        if any(name not in prop_tokens for name in building.interior_props):
            raise ValueError(f"catalog.building {building.token} names an unknown interior prop")
    for prop in props:
        if prop.shape not in _SHAPES or prop.transition not in _TRANSITIONS:
            raise ValueError(f"catalog.prop {prop.token} has an invalid shape or transition")
        if prop.start not in prop.states or len(set(prop.states)) != len(prop.states):
            raise ValueError(f"catalog.prop {prop.token} must start in one of its unique states")
        # A transition moves between exactly two states, which is what makes `active_state` the
        # single other one. A prop that never transitions holds its one state for the whole day.
        if len(prop.states) != (2 if prop.transition in _MOVING_TRANSITIONS else 1):
            raise ValueError(f"catalog.prop {prop.token} state count does not match its transition")
        if prop.transition == "timed":
            if type(prop.duration) is not int or prop.duration <= 0:
                raise ValueError(f"catalog.prop {prop.token} is timed, so it needs a positive duration")
        elif prop.duration is not None:
            raise ValueError(f"catalog.prop {prop.token} is not timed, so it carries no duration")
    if any(item.shape not in _SHAPES for item in scenery):
        raise ValueError("catalog.scenery shapes must be a box or a circle")
    if not {"i", "d", "x"} <= set(GROUND_BY_CODE):
        raise ValueError("catalog needs the interior, doorway, and wall ground codes from the rules")
    return Catalog(buildings, props, scenery)


def _collision_scale(value: Any) -> float:
    scale = positive_number(value, "catalog.prop.collision_scale")
    if not isfinite(scale) or scale > 1:
        raise ValueError("catalog.prop.collision_scale must be finite and no greater than one")
    return scale


CATALOG = load(json.loads(resources.files(__package__).joinpath("catalog.json").read_text(encoding="utf-8")))
BUILDING_BY_TOKEN = CATALOG.building_by_token
PROP_BY_TOKEN = CATALOG.prop_by_token
SCENERY_BY_TOKEN = CATALOG.scenery_by_token
