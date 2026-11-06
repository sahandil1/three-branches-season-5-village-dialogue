"""The road, its bridges, and the spawn.

The road is carved the way a water course is: a brush on a heading that blends momentum, a pull
toward wherever it is going next, a climb toward drier ground, repulsion from water, a push off the
band edges, a meander, and a little wobble. What makes it a road rather than a river is what it is
not allowed to do. It never paints water except inside a bridge it already proved, and it never runs
back over its own trail. One deck per channel and one connected road follow from the walk itself
rather than from a check afterwards.

Bridges are chosen before the walk. A crossing is the shortest straight cut across a channel inside
the band that lands on dry bank at both ends, preferring the row nearest the crossing before it so
the road does not zigzag between channels. The walker then runs straight through each one, which is
what makes a deck come out square at the width the road was carved with.

The road reads moisture and not elevation for its dry ground. The elevation field carries a
southward bias that runs one way across the band, so climbing it would only pin the road to an edge.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from math import dist, pi

from ..geometry import Point
from ..grid import Cell
from ..rules import FRAME
from . import carve, sites, walk
from .config import Network, Retry
from .sites import Settlement
from .water import Water


@dataclass(frozen=True, slots=True)
class Crossing:
    """One proved bridge: where it cuts, how far it reaches, and the deck it paints."""

    row: int
    span: tuple[int, int]
    corridor: frozenset[Cell]
    deck: frozenset[Cell]

    @property
    def entry(self) -> Point:
        return (self.span[0] + 0.5, self.row + 0.5)


@dataclass(frozen=True, slots=True)
class Road:
    """Everything the road stage committed."""

    width: int
    centreline: tuple[Point, ...]
    cells: frozenset[Cell]
    crossings: tuple[Crossing, ...]
    spawn: Point


def lay_road(
    stream: random.Random,
    rows: list[list[str]],
    moisture: list[list[float]],
    courses: Water,
    settlement: Settlement,
    tuning: Network,
) -> Road:
    """Choose every bridge, then walk the road across the frame through all of them."""
    road = tuning.road
    half = road.width // 2
    crossings = _crossings(rows, courses, settlement, tuning)
    entry_row = stream.randint(road.band[0] + half, road.band[1] - half)
    exit_row = stream.randint(road.band[0] + half, road.band[1] - half)
    swings = [stream.randint(-road.anchor_swing, road.anchor_swing) for _ in settlement.anchors]
    inside = sites.inner_band(tuning)
    anchors = tuple(
        (cell[0] + 0.5, min(max(cell[1] + swing, inside[0]), inside[1]) + 0.5)
        for cell, swing in zip(settlement.anchors, swings, strict=True)
    )
    walker = _Walk(rows, moisture, courses, settlement, tuning, crossings, anchors)
    return walker.run(stream, entry_row, exit_row)


def _crossings(
    rows: list[list[str]], courses: Water, settlement: Settlement, tuning: Network
) -> tuple[Crossing, ...]:
    """Prove one bridge per channel, west to east, each near the one before it."""
    road = tuning.road
    band = road.band
    order = sorted(
        range(len(courses.channels)),
        key=lambda index: _mean_column(courses.channels[index], band),
    )
    found: list[Crossing] = []
    previous = (band[0] + band[1]) // 2
    for channel in order:
        crossing = _crossing(rows, courses, settlement, tuning, channel, previous)
        if crossing is None:
            raise Retry(f"channel {channel} offered no straight crossing inside the road band")
        found.append(crossing)
        previous = crossing.row
    if any(left.span[1] >= right.span[0] for left, right in zip(found, found[1:], strict=False)):
        raise Retry("two bridges overlapped along the road band")
    return tuple(found)


def _crossing(
    rows: list[list[str]],
    courses: Water,
    settlement: Settlement,
    tuning: Network,
    channel: int,
    previous: int,
) -> Crossing | None:
    """Score every admissible deck row for one channel and take the shortest, nearest cut."""
    road = tuning.road
    half = road.width // 2
    water = courses.channels[channel]
    siblings = courses.channels[:channel] + courses.channels[channel + 1 :]
    others = courses.trunk.union(*siblings)
    best: tuple[tuple[int, int, int], Crossing] | None = None
    for row in range(road.band[0] + half, road.band[1] - half + 1):
        deck = range(row - half, row + half + 1)
        columns: list[tuple[int, int]] = []
        for line in deck:
            wet = [x for x in range(FRAME.cells_x) if (x, line) in water]
            # A row the channel enters twice would be a cut across two bends, not one crossing.
            if not wet or wet[-1] - wet[0] + 1 != len(wet):
                columns = []
                break
            columns.append((wet[0], wet[-1]))
        if not columns:
            continue
        cut = (min(low for low, _ in columns), max(high for _, high in columns))
        span = (cut[0] - road.apron, cut[1] + road.apron)
        if span[0] < 1 or span[1] >= FRAME.cells_x - 1 or span[1] - span[0] + 1 > road.crossing_run:
            continue
        corridor = frozenset((x, line) for line in deck for x in range(span[0], span[1] + 1))
        if any(cell in settlement.reserved or cell in others for cell in corridor):
            continue
        # The aprons are what a deck lands on, so they have to be bank rather than more water.
        aprons = tuple(cell for cell in corridor if not cut[0] <= cell[0] <= cut[1])
        if any(rows[line][x] == "w" for x, line in aprons):
            continue
        deck_cells = frozenset(cell for cell in corridor if cell in water)
        score = (span[1] - span[0], abs(row - previous), row)
        if best is None or score < best[0]:
            best = (score, Crossing(row, span, corridor, deck_cells))
    return None if best is None else best[1]


def _mean_column(channel: frozenset[Cell], band: tuple[int, int]) -> float:
    inside = [x for x, y in channel if band[0] <= y <= band[1]]
    return sum(inside) / len(inside) if inside else float(FRAME.cells_x)


class _Walk:
    """One road being carried across the frame."""

    def __init__(
        self,
        rows: list[list[str]],
        moisture: list[list[float]],
        courses: Water,
        settlement: Settlement,
        tuning: Network,
        crossings: tuple[Crossing, ...],
        anchors: tuple[Point, ...],
    ) -> None:
        self.rows = rows
        self.moisture = moisture
        self.courses = courses
        self.settlement = settlement
        self.tuning = tuning
        self.crossings = crossings
        self.anchors = anchors
        self.offsets = carve.disc_offsets(tuning.road.width / 2)
        self.trail: dict[Cell, int] = {}
        self.cells: set[Cell] = set()
        self.closest = [float(FRAME.cells_x)] * len(anchors)

    def run(self, stream: random.Random, entry_row: int, exit_row: int) -> Road:
        """Walk from the west edge to the east one, through every bridge on the way."""
        road = self.tuning.road
        walker = road.walker
        # Anchors and bridges make one west to east list of places the road has to be.
        stops = sorted(
            [(point[0], index, "anchor") for index, point in enumerate(self.anchors)]
            + [(crossing.entry[0], index, "bridge") for index, crossing in enumerate(self.crossings)]
        )
        wavelength = float(stream.randint(*walker.meander_wavelength))
        phase = stream.uniform(0.0, 2.0 * pi)
        momentum = stream.uniform(*walker.momentum)
        dry = stream.uniform(*walker.dry)
        pull = stream.uniform(*walker.pull)
        position = (0.5, entry_row + 0.5)
        heading = (1.0, 0.0)
        centreline = [position]
        self._paint(position, None, 0)
        locked: Crossing | None = None
        index = 0
        step = 0
        while position[0] < FRAME.cells_x:
            if step >= walker.step_budget:
                raise Retry("the road never reached the east edge")
            target, kind = self._target(stops, index, exit_row)
            if locked is not None:
                heading = (1.0, 0.0)
                if position[0] > locked.span[1] + 1:
                    locked = None
                    index += 1
            elif kind == "bridge" and dist(position, target) <= walker.step:
                crossing = self.crossings[stops[index][1]]
                position, heading, locked = crossing.entry, (1.0, 0.0), crossing
            elif self._straight(position):
                heading = (1.0, 0.0)
            elif kind == "bridge" and position[0] > target[0] + road.width:
                raise Retry("the road slid past a bridge instead of taking it")
            elif 0.0 <= target[0] - position[0] <= walker.look_ahead:
                # The last stretch before a stop aims square at it, so the road arrives at the
                # bridge it proved and passes beside the anchor rather than sliding by either.
                heading = walk.unit((target[0] - position[0], target[1] - position[1]))
            else:
                sway = walk.sway(step, walker.step, wavelength, phase)
                heading = self._steer(stream, position, heading, target, momentum, dry, pull, sway)
            position, heading = self._advance(position, heading, step, locked)
            centreline.append(position)
            self._measure(position)
            if locked is None and kind == "anchor" and position[0] >= target[0]:
                index += 1
            step += 1
        if index < len(stops):
            raise Retry("the road left the frame before it had been everywhere it had to go")
        if any(gap > road.anchor_reach for gap in self.closest):
            raise Retry("the road never came within reach of a district anchor")
        spawn = (self.tuning.spawn.edge_inset + 0.5, entry_row + 0.5)
        if self.rows[int(spawn[1])][int(spawn[0])] != "r":
            raise Retry("the road left no road cell under the spawn")
        return Road(
            road.width,
            tuple(centreline),
            frozenset(self.cells),
            self.crossings,
            spawn,
        )

    def _target(self, stops: list[tuple[float, int, str]], index: int, exit_row: int) -> tuple[Point, str]:
        if index >= len(stops):
            return (float(FRAME.cells_x), exit_row + 0.5), "exit"
        _, which, kind = stops[index]
        return (self.anchors[which] if kind == "anchor" else self.crossings[which].entry), kind

    def _advance(
        self, position: Point, heading: Point, step: int, locked: Crossing | None
    ) -> tuple[Point, Point]:
        walker = self.tuning.road.walker
        # A locked road is running through a bridge it already proved, so it holds its heading and
        # paints the deck rather than testing the water under it.
        moved = walk.advance(
            position,
            heading,
            walker.step,
            walker.reroute_attempts,
            walker.reroute_degrees,
            lambda candidate: locked is not None or not self._blocked(candidate, step),
        )
        if moved is None:
            raise Retry(f"the road ran out of room to steer at {position}")
        self._paint(moved[0], locked, step)
        return moved

    def _straight(self, position: Point) -> bool:
        straight = self.tuning.road.edge_straight
        return position[0] <= straight or position[0] >= FRAME.cells_x - straight

    def _paint(self, position: Point, locked: Crossing | None, step: int) -> None:
        for spot in carve.brush((int(position[0]), int(position[1])), self.offsets):
            code = self.rows[spot[1]][spot[0]]
            if code == "w":
                # Water is only ever painted inside a bridge, and only the deck of one.
                if locked is None or spot not in locked.corridor:
                    continue
                self.rows[spot[1]][spot[0]] = "b"
            elif code != "b":
                # Brush strokes overlap, so a later one must not take a deck back to plain road.
                self.rows[spot[1]][spot[0]] = "r"
            self.trail.setdefault(spot, step)
            self.cells.add(spot)

    def _blocked(self, position: Point, step: int) -> bool:
        road = self.tuning.road
        cell = (int(position[0]), int(position[1]))
        painted = carve.brush(cell, self.offsets)
        # The brush is clipped at the frame, so the centre is bounded too. The last step is allowed
        # to carry the road off the east edge, which is how it finishes rather than stalls against
        # the frame. The band is a hard wall it is turned back from instead.
        if not 0 <= cell[0] <= FRAME.cells_x or not road.band[0] <= cell[1] <= road.band[1]:
            return True
        if any(not road.band[0] <= row <= road.band[1] for _, row in painted):
            return True
        stale = step - road.walker.self_ignore
        for spot in painted:
            if self.rows[spot[1]][spot[0]] == "w":
                return True
            if spot in self.settlement.reserved or spot in self.settlement.keep_clear:
                return True
            if self.trail.get(spot, step) < stale:
                return True
        return False

    def _steer(
        self,
        stream: random.Random,
        position: Point,
        heading: Point,
        target: Point,
        momentum: float,
        dry: float,
        pull: float,
        sway: float,
    ) -> Point:
        walker = self.tuning.road.walker
        toward = walk.unit((target[0] - position[0], target[1] - position[1]))
        return walk.steer(
            stream,
            (
                (momentum, heading),
                (pull, toward),
                (dry, walk.downhill(self.moisture, position)),
                (walker.band_push, self._band_push(position)),
                (walker.water_push, self._water_push(position)),
            ),
            toward,
            walker.meander,
            sway,
            walker.wobble,
        )

    def _band_push(self, position: Point) -> Point:
        band = self.tuning.road.band
        soft = self.tuning.road.width
        if position[1] < band[0] + soft:
            return (0.0, 1.0)
        if position[1] > band[1] - soft:
            return (0.0, -1.0)
        return (0.0, 0.0)

    def _water_push(self, position: Point) -> Point:
        """Turn away from water before the hard clearance check has to stop the road."""
        road = self.tuning.road
        span = road.width / 2 + road.water_clearance + road.walker.look_ahead
        return walk.push_away(position, self._water_near(position, span), span)

    def _water_near(self, position: Point, span: float) -> Iterable[Cell]:
        cell = (int(position[0]), int(position[1]))
        return (
            spot for spot in carve.brush(cell, carve.disc_offsets(span)) if self.rows[spot[1]][spot[0]] == "w"
        )

    def _measure(self, position: Point) -> None:
        for index, anchor in enumerate(self.anchors):
            self.closest[index] = min(self.closest[index], dist(position, anchor))
