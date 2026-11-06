"""The trunk, its fork, and the three channels that fan to the south edge.

A cell walker carries a round brush down the map. Its heading blends momentum, the downhill slope of
the elevation field, a pull toward where it is going, repulsion from the side edges, repulsion from
courses already carved, and a little wobble. Courses are carved one at a time from west to east, so
each new channel only has to keep away from what is already there.

Inside the fork mask the courses share their water, which is what makes one river into three. Outside
it, a course that would come within the configured clearance of another course, or of its own older
trail, steers away, and gives up on the whole layout when steering runs out of room.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil, pi

from ..grid import Cell, Point
from ..rules import FRAME
from . import carve, walk
from .config import Retry
from .config import Water as WaterTuning

TRUNK = -1
# Water runs down the map, so that is the way a course faces when its urges cancel out.
_SOUTH: Point = (0.0, -1.0)


@dataclass(frozen=True, slots=True)
class Water:
    """Where the water went. The masks are what tests and later stages read."""

    trunk: frozenset[Cell]
    channels: tuple[frozenset[Cell], ...]
    fork_mask: frozenset[Cell]
    # Where the trunk ended and the channels begin, which is the depth the guarantee names.
    fork: Cell


def carve_water(
    stream: random.Random, rows: list[list[str]], elevation: list[list[float]], tuning: WaterTuning
) -> Water:
    """Carve the trunk and its three channels, painting water as it goes."""
    walker = tuning.walker
    trunk_width = _odd_width(stream, tuning.trunk_width)
    # The entry band bounds the run the trunk paints, not just where its centre starts.
    entry_x = stream.randint(tuning.entry_band[0] + trunk_width // 2, tuning.entry_band[1] - trunk_width // 2)
    fork_y = FRAME.cells_y - 1 - stream.randint(*tuning.fork_band)
    # The fork sits where the stream puts it rather than straight below the entry, so the trunk
    # has somewhere to lean on its way down instead of falling in a line.
    fork_x = stream.randint(*tuning.entry_band)
    channel_widths = tuple(_odd_width(stream, tuning.channel_width) for _ in range(3))
    momentum = stream.uniform(*walker.momentum)
    downhill = stream.uniform(*walker.downhill)
    pull = stream.uniform(*walker.pull)
    mouths = _draw_mouths(stream, tuning, max(channel_widths))

    course = _Course(rows, elevation, tuning, momentum, downhill, pull)
    fork = course.walk(
        stream,
        TRUNK,
        (entry_x + 0.5, FRAME.cells_y - 0.5),
        (0.0, -1.0),
        trunk_width / 2.0,
        (float(fork_x), float(fork_y)),
        lambda point: point[1] <= fork_y,
    )
    course.fork_mask = frozenset(
        carve.brush((int(fork[0]), int(fork[1])), carve.disc_offsets(tuning.fork_radius))
    )
    # The three channels leave the fork fanned apart, and their opening reaches join the shared
    # fork area. That is what lets them share one pool and still stand clear of each other below it.
    middle = (len(mouths) - 1) / 2.0
    for index, (mouth, width) in enumerate(zip(mouths, channel_widths, strict=True)):
        target = (float(mouth), -1.0)
        course.walk(
            stream,
            index,
            fork,
            walk.rotate(
                walk.unit((target[0] - fork[0], target[1] - fork[1]), _SOUTH),
                (index - middle) * tuning.fan_degrees,
            ),
            width / 2.0,
            target,
            lambda point: point[1] < 0.5,
        )
    _flood_pockets(course)
    channels = tuple(course.mask(index) for index in range(len(channel_widths)))
    _check_mouths(rows, channels, tuning, channel_widths)
    _check_contact((course.mask(TRUNK), *channels), course.fork_mask)
    return Water(course.mask(TRUNK), channels, course.fork_mask, (int(fork[0]), int(fork[1])))


def _odd_width(stream: random.Random, bounds: tuple[int, int]) -> int:
    """Draw an odd brush width, the only kind a cell-centred round brush can carve exactly."""
    first = bounds[0] if bounds[0] % 2 else bounds[0] + 1
    return stream.randrange(first, bounds[1] + 1, 2)


def _draw_mouths(stream: random.Random, tuning: WaterTuning, widest: int) -> tuple[int, ...]:
    """Draw three well separated south-edge targets, leaving slack for the walk to wander."""
    reach = ceil(widest / 2)
    low = tuning.edge_margin + reach
    high = FRAME.cells_x - 1 - tuning.edge_margin - reach
    wanted = tuning.mouth_separation + tuning.mouth_slack
    for _ in range(tuning.mouth_budget):
        targets = sorted(stream.randint(low, high) for _ in range(3))
        if all(right - left >= wanted for left, right in zip(targets, targets[1:], strict=False)):
            return tuple(targets)
    raise Retry("no three south-edge targets kept their separation")


def _check_mouths(
    rows: list[list[str]],
    channels: tuple[frozenset[Cell], ...],
    tuning: WaterTuning,
    widths: tuple[int, ...],
) -> None:
    """Confirm the south edge carries exactly three separated runs, one per channel, west to east."""
    runs = _south_runs(rows)
    if len(runs) != len(channels):
        raise Retry(f"the south edge carries {len(runs)} water runs")
    centres: list[int] = []
    for index, (run, width) in enumerate(zip(runs, widths, strict=True)):
        if any((x, 0) not in channels[index] for x in run) or not width <= len(run) <= width + 1:
            raise Retry("a south-edge run does not belong to its own channel at the carved width")
        centres.append(sum(run) // len(run))
    if any(right - left < tuning.mouth_separation for left, right in zip(centres, centres[1:], strict=False)):
        raise Retry("two mouths came out closer than the configured separation")


def _check_contact(courses: tuple[frozenset[Cell], ...], fork_mask: frozenset[Cell]) -> None:
    """Confirm no two courses meet outside the water they share.

    The carver steers away from contact as it goes, but it cannot see everything: the shared area
    grows while the channels are carved, and flooding a pocket afterwards moves cells between
    courses. This reads the courses as they finally stand, which is what the guarantee is about.
    """
    for left in range(len(courses)):
        for right in range(left + 1, len(courses)):
            for x, y in courses[left] - fork_mask:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        spot = (x + dx, y + dy)
                        if spot in courses[right] and spot not in fork_mask:
                            raise Retry("two courses met outside the water they share")


def _flood_pockets(course: _Course) -> None:
    """Flood land the water closed off, because no villager could ever reach it.

    The courses themselves cut the land into banks, and the road's bridges are what join those back
    together, so a bank that reaches the frame is fine. A pocket is land that reaches no frame edge
    at all: where courses braid, a cell or two can end up ringed by water. Such a cell is part of the
    river rather than a place to stand, so it is flooded and joins whichever course already holds
    most of its edge. That keeps the banks whole without discarding the draw.
    """
    rows = course.rows
    land = {(x, y) for y, row in enumerate(rows) for x, code in enumerate(row) if code != "w"}
    reached = {
        cell for cell in land if 0 in cell or cell[0] == FRAME.cells_x - 1 or cell[1] == FRAME.cells_y - 1
    }
    if not reached:
        raise Retry("the water left no land against the frame")
    pending = list(reached)
    while pending:
        x, y = pending.pop()
        for spot in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
            if spot in land and spot not in reached:
                reached.add(spot)
                pending.append(spot)
    for x, y in sorted(land - reached):
        holders = [
            course.owner[spot]
            for spot in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1))
            if spot in course.owner
        ]
        course.flood((x, y), max(sorted(set(holders)), key=holders.count) if holders else TRUNK)


def _south_runs(rows: list[list[str]]) -> tuple[tuple[int, ...], ...]:
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for x, code in enumerate(rows[0]):
        if code == "w":
            current.append(x)
        elif current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return tuple(runs)


class _Course:
    """One shared carving surface: the rows being painted plus who owns each water cell."""

    def __init__(
        self,
        rows: list[list[str]],
        elevation: list[list[float]],
        tuning: WaterTuning,
        momentum: float,
        downhill: float,
        pull: float,
    ) -> None:
        self.rows = rows
        self.elevation = elevation
        self.tuning = tuning
        self.momentum = momentum
        self.downhill = downhill
        self.pull = pull
        self.fork_mask: frozenset[Cell] = frozenset()
        # Courses overlap where they share the fork, so ownership goes to whoever claimed a cell
        # first and is only used to tell courses apart. Each course also keeps its whole trail, so a
        # channel's mask stays the unbroken ribbon that tests and the road crossing check read.
        self.owner: dict[Cell, int] = {}
        self.trails: dict[int, dict[Cell, int]] = {}

    def mask(self, owner: int) -> frozenset[Cell]:
        return frozenset(self.trails.get(owner, {}))

    def flood(self, cell: Cell, owner: int) -> None:
        """Turn one closed-off land cell into water belonging to a course."""
        self.rows[cell[1]][cell[0]] = "w"
        self.owner.setdefault(cell, owner)
        self.trails.setdefault(owner, {})[cell] = 0

    def walk(
        self,
        stream: random.Random,
        owner: int,
        start: Point,
        heading: Point,
        radius: float,
        target: Point,
        done: Callable[[Point], bool],
    ) -> Point:
        """Carry a brush from a start point until the stop test passes, and report where it ended."""
        walker = self.tuning.walker
        offsets = carve.disc_offsets(radius)
        reach = carve.disc_offsets(radius + self.tuning.clearance)
        # Steering senses further than the hard check blocks, so a course turns away from another
        # course while it still has room to turn instead of stopping dead against it.
        span = radius + self.tuning.clearance + walker.look_ahead
        sense = carve.disc_offsets(span)
        # Every course meanders on its own wavelength and phase, which is what keeps it from
        # running to its mouth in a line.
        wavelength = float(stream.randint(*walker.meander_wavelength))
        phase = stream.uniform(0.0, 2.0 * pi)
        position = start
        self._paint(carve.brush((int(start[0]), int(start[1])), offsets), owner, 0)
        step = 0
        while not done(position):
            if step >= walker.step_budget:
                raise Retry("a water course never reached its stop line")
            if self._at_edge(position, owner):
                # A course runs straight where it meets a frame edge, so the run it paints there
                # comes out at exactly the width it was carved with.
                heading = _SOUTH
            else:
                sway = walk.sway(step, walker.step, wavelength, phase)
                heading = self._steer(stream, position, heading, target, sense, span, owner, sway)
            position, heading = self._advance(position, heading, offsets, reach, owner, step)
            step += 1
        return position

    def _advance(
        self,
        position: Point,
        heading: Point,
        offsets: tuple[Cell, ...],
        reach: tuple[Cell, ...],
        owner: int,
        step: int,
    ) -> tuple[Point, Point]:
        walker = self.tuning.walker
        # A channel's opening reach belongs to the shared fork area, so it carves without checking.
        sharing = owner != TRUNK and step < self.tuning.fork_steps
        moved = walk.advance(
            position,
            heading,
            walker.step,
            walker.reroute_attempts,
            walker.reroute_degrees,
            lambda candidate: sharing or not self._blocked(candidate, offsets, reach, owner, step),
        )
        if moved is None:
            raise Retry(f"water course {owner} ran out of room to steer at {position}")
        candidate, turned = moved
        painted = carve.brush((int(candidate[0]), int(candidate[1])), offsets)
        if sharing:
            self.fork_mask |= frozenset(painted)
        self._paint(painted, owner, step)
        return candidate, turned

    def _at_edge(self, position: Point, owner: int) -> bool:
        straight = self.tuning.edge_straight
        if owner == TRUNK:
            return position[1] > FRAME.cells_y - 1 - straight
        return position[1] <= straight

    def _paint(self, cells: tuple[Cell, ...], owner: int, step: int) -> None:
        trail = self.trails.setdefault(owner, {})
        for spot in cells:
            self.rows[spot[1]][spot[0]] = "w"
            self.owner.setdefault(spot, owner)
            trail[spot] = step

    def _blocked(
        self, candidate: Point, offsets: tuple[Cell, ...], reach: tuple[Cell, ...], owner: int, step: int
    ) -> bool:
        margin = self.tuning.edge_margin
        cell = (int(candidate[0]), int(candidate[1]))
        painted = carve.brush(cell, offsets)
        # The brush is clipped at the frame, so the centre is bounded too. Without that a course
        # could wander off the map, where nothing is painted and nothing ever blocks it.
        if not margin <= cell[0] < FRAME.cells_x - margin or not -1 <= cell[1] < FRAME.cells_y:
            return True
        if any(x < margin or x >= FRAME.cells_x - margin for x, _ in painted):
            return True
        stale = step - self.tuning.walker.self_ignore
        mine = self.trails.setdefault(owner, {})
        for spot in carve.brush(cell, reach):
            if spot in self.fork_mask:
                continue
            holder = self.owner.get(spot)
            if holder is None:
                continue
            if holder != owner or mine.get(spot, step) < stale:
                return True
        return False

    def _steer(
        self,
        stream: random.Random,
        position: Point,
        heading: Point,
        target: Point,
        reach: tuple[Cell, ...],
        span: float,
        owner: int,
        sway: float,
    ) -> Point:
        walker = self.tuning.walker
        pull = walk.unit((target[0] - position[0], target[1] - position[1]), _SOUTH)
        return walk.steer(
            stream,
            (
                (self.momentum, heading),
                (self.downhill, walk.downhill(self.elevation, position, _SOUTH)),
                (self.pull, pull),
                (walker.edge_push, self._edge_push(position)),
                (walker.separation, self._separation(position, reach, span, owner)),
            ),
            pull,
            walker.meander,
            sway,
            walker.wobble,
            _SOUTH,
        )

    def _edge_push(self, position: Point) -> Point:
        soft = self.tuning.edge_margin + 4
        if position[0] < soft:
            return (1.0, 0.0)
        if position[0] > FRAME.cells_x - soft:
            return (-1.0, 0.0)
        return (0.0, 0.0)

    def _separation(self, position: Point, reach: tuple[Cell, ...], span: float, owner: int) -> Point:
        """Push away from water somebody else carved, before the hard check has to fire.

        The push points away from the crowd and grows as the nearest of it closes in, reaching full
        strength on contact and nothing at all at the edge of what the course can sense. A course
        never pushes away from its own trail: its trail is right behind it, so that would only make
        it oscillate, and self-contact is what the hard check with its own memory is for.
        """
        return walk.push_away(position, self._others_near(position, reach, owner), span)

    def _others_near(self, position: Point, reach: tuple[Cell, ...], owner: int) -> Iterable[Cell]:
        cell = (int(position[0]), int(position[1]))
        return (
            spot
            for spot in carve.brush(cell, reach)
            if spot not in self.fork_mask and self.owner.get(spot, owner) != owner
        )
