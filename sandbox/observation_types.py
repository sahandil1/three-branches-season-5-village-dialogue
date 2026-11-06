"""Stdlib-only action and observation types for Days at Three Branches."""

from __future__ import annotations

from typing import TypedDict


class Position(TypedDict):
    """One position in metres from the village southwest corner."""

    x: float
    y: float


class Cell(TypedDict):
    """One zero-based village grid cell."""

    x: int
    y: int


class Expression(TypedDict):
    """A character's current emote or prop interaction."""

    type: str
    target: str


class SeenCharacter(TypedDict):
    """The state revealed for a character in the vision cone."""

    id: str
    position: Position
    heading: float
    moved: float
    expression: Expression


class NearbyCharacter(TypedDict):
    """The presence revealed for a character within hearing range."""

    id: str
    position: Position


class SeenProp(TypedDict):
    """One visible interactive prop and its current state."""

    prop: str
    state: str


class VillageSize(TypedDict):
    """The village grid dimensions and cell scale."""

    cells_x: int
    cells_y: int
    cell_size: float


class VillageBuilding(TypedDict):
    """One semantic building placement."""

    id: str
    type: str
    cell: Cell


class VillageProp(TypedDict):
    """One interactive prop placement."""

    id: str
    type: str
    cell: Cell
    facing: str


class VillageScenery(TypedDict):
    """One solid scenery placement."""

    type: str
    cell: Cell


class Village(TypedDict):
    """The immutable village knowledge included in every observation."""

    size: VillageSize
    ground: tuple[str, ...]
    buildings: tuple[VillageBuilding, ...]
    props: tuple[VillageProp, ...]
    scenery: tuple[VillageScenery, ...]
    spawn: Position


class RosterEntry(TypedDict):
    """One character's stable identity and residence."""

    id: str
    home: str


class Parameters(TypedDict):
    """Resolved settings for this day."""

    seat_plan: str
    daynight: int


class ThreeBranchesObservation(TypedDict):
    """The complete observation given to one character."""

    self: SeenCharacter
    seen: tuple[SeenCharacter, ...]
    nearby: tuple[NearbyCharacter, ...]
    props: tuple[SeenProp, ...]
    bell: int
    tick: int
    phase: str
    village: Village
    roster: tuple[RosterEntry, ...]
    parameters: Parameters


class ThreeBranchesAction(TypedDict):
    """A simultaneous locomotion and expression command."""

    heading: float
    speed: float
    action: int
