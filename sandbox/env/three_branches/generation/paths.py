"""Footpaths. Where one goes is searched; how it gets there is walked.

A footpath is somebody's way to a door, so the search is asked one question only: what this route
has to join, and where it may cross water to get there. It prices a cell by how fast a body moves
over it and discounts a path already worn, which is what makes a new route meet an old one rather
than run beside it. Every road and bridge cell is a source at no cost, so a route is a spur off the
nearest way that already carries people rather than a line from one chosen point.

What comes back is a plan and not a shape. An ant walks it from the doorstep, pulled toward where it
is going, swinging across that line on a meander of its own, and nudged by a wobble every step. It
stops the moment it meets a way that already carries people, whatever the plan still had in mind,
which is what makes paths run together instead of side by side. Water it crosses only on the
straight run the search proved. When it runs out of room to turn, or out of steps, the rest of its
leg is painted along the plan, so a doorway the search could reach is always joined.

``joinable`` answers the other question this module is asked: whether somewhere could ever be joined
to the road at all, which is what lets a building site reject a doorway before it is painted.

One walk draws its gait per route and its wobble per step. Everything else here is deterministic,
and ties break on the cell.
"""

from __future__ import annotations

import heapq
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import inf, pi, sqrt

from ..grid import Cell, Point
from ..rules import FRAME, GROUND_BY_CODE
from . import carve, walk
from .config import Path, Retry
from .water import Water

# Ground a footpath may run over. Interiors, walls, and doorways are not routes: a path stops at the
# approach cell outside a door, and the door run itself is painted with the site.
_WALKABLE = frozenset({"r", "p", "b", "g", "f", "e"})
# Ground a footpath may paint itself over. Everything else already carries people.
_PAINTABLE = frozenset({"g", "f", "e"})
# Ground that already carries people, which is where every search starts and every walk ends.
_JOINABLE = frozenset({"r", "b", "p"})
# Steps a route may take, with what each costs in cell lengths. A footpath only repaints ground
# that already carries a body, so it may cut a corner; water is only ever crossed square on.
_STEPS = (
    (0, -1, 1.0),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, 1, 1.0),
    (-1, -1, sqrt(2.0)),
    (1, -1, sqrt(2.0)),
    (-1, 1, sqrt(2.0)),
    (1, 1, sqrt(2.0)),
)
# Recorded on a move that stepped rather than bridged.
_NO_CHANNEL = -1


@dataclass(frozen=True, slots=True)
class Crossing:
    """One channel a committed footpath bridged, and the water it bridged."""

    channel: int
    cells: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class Footpaths:
    """Everything the footpath stage committed."""

    crossings: tuple[Crossing, ...]


@dataclass(frozen=True, slots=True)
class _Leg:
    """One dry stretch of a planned route, and the water it ends at.

    Every leg but the last one ends at a bank, because a crossing is what splits the plan in two.
    """

    cells: tuple[Cell, ...]
    crossing: Crossing | None


def _sources(rows: list[list[str]]) -> tuple[Cell, ...]:
    """Every cell a footpath may leave from: the road, and any path already worn.

    A route joins the nearest way that already goes somewhere rather than always running its own
    line back to the road, which is what makes the footpaths a branching network instead of a set
    of private spurs.
    """
    return tuple((x, y) for y, row in enumerate(rows) for x, code in enumerate(row) if code in _JOINABLE)


def joinable(rows: list[list[str]], courses: Water, starts: Iterable[Cell], tuning: Path) -> frozenset[Cell]:
    """Return every cell a footpath could ever reach from the given ground.

    Settlement asks this once, before it stands anything, and then a candidate doorway only has to
    be a member. The road is not walked yet, so the question is asked of the road's band, which the
    road is going to span. Buildings placed afterwards only ever add obstacles, so this is the
    optimistic answer; the committed routing is what has the last word.
    """
    channels = {cell: index for index, channel in enumerate(courses.channels) for cell in channel}
    seen: set[Cell] = {cell for cell in starts if _inside(cell) and rows[cell[1]][cell[0]] in _WALKABLE}
    pending = list(seen)
    while pending:
        cell = pending.pop()
        for spot, _, _, _ in _moves(rows, courses, channels, cell, frozenset(), frozenset(), tuning):
            if spot not in seen:
                seen.add(spot)
                pending.append(spot)
    return frozenset(seen)


def lay_footpaths(
    stream: random.Random,
    rows: list[list[str]],
    courses: Water,
    targets: tuple[tuple[Cell, ...], ...],
    forbidden: frozenset[Cell],
    tuning: Path,
) -> Footpaths:
    """Plan and walk one footpath to each target group, the farthest from the road first.

    Each route is worn before the next is planned, so whichever goes first lays the way the rest
    join. Starting at the far end grows one path with spurs off it, where starting at the near end
    grows a bundle of separate lines that happen to run alongside each other. A channel one route
    bridged is closed to the rest, which is what holds every channel to at most one crossing.
    """
    crossings: list[Crossing] = []
    for group in _farthest_first(rows, courses, targets, forbidden, tuning):
        wanted = frozenset(group)
        if not wanted:
            continue
        target, _, came = _relax(
            rows,
            courses,
            _sources(rows),
            wanted.__contains__,
            forbidden,
            frozenset(crossing.channel for crossing in crossings),
            tuning,
        )
        if target is None:
            raise Retry("a footpath target could not be reached from the road")
        crossings.extend(_wear(stream, rows, _plan(rows, (target, came)), forbidden, tuning))
        # A route arrives at one cell of a doorway's approach. The rest of it is the same doorstep,
        # so it is worn too, and the whole run opens onto path rather than half of it.
        for cell in sorted(wanted):
            if rows[cell[1]][cell[0]] in _PAINTABLE:
                rows[cell[1]][cell[0]] = "p"
    return Footpaths(tuple(crossings))


# How each cell was reached: from where, over which water, and which channel that water was.
_Trail = dict[Cell, tuple[Cell, tuple[Cell, ...], int]]


def _relax(
    rows: list[list[str]],
    courses: Water,
    starts: tuple[Cell, ...],
    is_target: Callable[[Cell], bool],
    forbidden: frozenset[Cell],
    banned: frozenset[int],
    tuning: Path,
) -> tuple[Cell | None, dict[Cell, float], _Trail]:
    """Spread cost out from the ways that already carry people, stopping at the first target.

    Asked for a target it can never reach, it settles every cell instead, which is how the routes
    are put in order before any of them is worn.
    """
    channels = {cell: index for index, channel in enumerate(courses.channels) for cell in channel}
    best: dict[Cell, float] = {}
    came: _Trail = {}
    queue: list[tuple[float, Cell]] = []
    for cell in starts:
        if _inside(cell) and cell not in forbidden and cell not in best:
            best[cell] = 0.0
            heapq.heappush(queue, (0.0, cell))
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost > best.get(cell, cost):
            continue
        if is_target(cell):
            return cell, best, came
        for spot, water_cells, channel, price in _moves(
            rows, courses, channels, cell, forbidden, banned, tuning
        ):
            through = cost + price
            if through < best.get(spot, through + 1.0):
                best[spot] = through
                came[spot] = (cell, water_cells, channel)
                heapq.heappush(queue, (through, spot))
    return None, best, came


def _farthest_first(
    rows: list[list[str]],
    courses: Water,
    targets: tuple[tuple[Cell, ...], ...],
    forbidden: frozenset[Cell],
    tuning: Path,
) -> tuple[tuple[Cell, ...], ...]:
    """Put the target the road is furthest from first, and the one under its feet last."""
    _, costs, _ = _relax(rows, courses, _sources(rows), _unreachable, forbidden, frozenset(), tuning)
    return tuple(sorted(targets, key=lambda group: (-_nearest(costs, group), group)))


def _nearest(costs: dict[Cell, float], group: tuple[Cell, ...]) -> float:
    """What the cheapest way into a target group costs, or infinity when there is none."""
    return min((costs[cell] for cell in group if cell in costs), default=inf)


def _unreachable(cell: Cell) -> bool:
    """A target the search can never arrive at, which is what makes it settle the whole map."""
    return False


def _plan(rows: list[list[str]], found: tuple[Cell, _Trail]) -> tuple[_Leg, ...]:
    """Read the trail that reached a target as the legs of a route and the water between them.

    The trail is read from the target back to whichever way cell the search started from, which is
    the direction somebody leaving that door walks it.
    """
    target, came = found
    legs: list[_Leg] = []
    cells: list[Cell] = [target]
    cell = target
    while cell in came:
        previous, water_cells, channel = came[cell]
        if water_cells:
            legs.append(_Leg(tuple(cells), Crossing(channel, water_cells)))
            cells = []
        elif previous[0] != cell[0] and previous[1] != cell[1]:
            # A corner step touches only at the corner, which would leave the route a dotted line of
            # separate cells. Filling one side of the corner keeps it a way somebody could walk.
            corner = _corner(rows, previous, (cell[0] - previous[0], cell[1] - previous[1]), frozenset())
            if corner is not None:
                cells.append(corner)
        cells.append(previous)
        cell = previous
    legs.append(_Leg(tuple(cells), None))
    return tuple(legs)


def _wear(
    stream: random.Random,
    rows: list[list[str]],
    legs: tuple[_Leg, ...],
    forbidden: frozenset[Cell],
    tuning: Path,
) -> tuple[Crossing, ...]:
    """Walk a plan and paint what the walk wore, leg by leg.

    A walk that meets a way already carrying people has arrived, and the rest of the plan is not
    needed. A walk with nowhere left to turn, or one that has spent its steps, hands what is left
    back to the plan, which is what makes a reachable doorway always end up joined.
    """
    walker = tuning.walker
    # One walk has one gait, so a path bends its own way for the whole of its length.
    wavelength = float(stream.randint(*walker.meander_wavelength))
    phase = stream.uniform(0.0, 2.0 * pi)
    momentum = stream.uniform(*walker.momentum)
    crossings: list[Crossing] = []
    for index, leg in enumerate(legs):
        start = leg.cells[0]
        if rows[start[1]][start[0]] in _JOINABLE:
            break
        worn = _wander(stream, rows, start, leg.cells[-1], forbidden, (momentum, wavelength, phase), tuning)
        if worn is None:
            for rest in legs[index:]:
                _paint(rows, rest.cells, tuning)
                if rest.crossing is not None:
                    _paint(rows, rest.crossing.cells, tuning)
                    crossings.append(rest.crossing)
            break
        cells, joined = worn
        _paint(rows, cells, tuning)
        if joined:
            break
        if leg.crossing is not None:
            _paint(rows, leg.crossing.cells, tuning)
            crossings.append(leg.crossing)
    return tuple(crossings)


def _wander(
    stream: random.Random,
    rows: list[list[str]],
    start: Cell,
    goal: Cell,
    forbidden: frozenset[Cell],
    gait: tuple[float, float, float],
    tuning: Path,
) -> tuple[tuple[Cell, ...], bool] | None:
    """Walk one leg, reporting the cells it wore and whether it met a way already worn.

    Nothing when the walk ran out of room to turn or out of steps, which hands the leg back to the
    plan. The pull toward the goal counts as one, so the gait's weights are read against it.
    """
    walker = tuning.walker
    momentum, wavelength, phase = gait
    position = (start[0] + 0.5, start[1] + 0.5)
    aim = (goal[0] + 0.5, goal[1] + 0.5)
    heading = walk.unit((aim[0] - position[0], aim[1] - position[1]))
    worn = [start]
    cell = start
    for step in range(walker.step_budget):
        toward = walk.unit((aim[0] - position[0], aim[1] - position[1]), heading)
        heading = walk.steer(
            stream,
            ((momentum, heading), (1.0, toward)),
            toward,
            walker.meander,
            walk.sway(step, walker.step, wavelength, phase),
            walker.wobble,
            heading,
        )
        moved = walk.advance(
            position,
            heading,
            walker.step,
            walker.reroute_attempts,
            walker.reroute_degrees,
            lambda candidate, behind=cell: _free(rows, candidate, behind, forbidden, tuning),
        )
        if moved is None:
            return None
        position, heading = moved
        spot = (int(position[0]), int(position[1]))
        if spot == cell:
            continue
        if spot[0] != cell[0] and spot[1] != cell[1]:
            corner = _corner(rows, cell, (spot[0] - cell[0], spot[1] - cell[1]), forbidden)
            if corner is not None:
                worn.append(corner)
        code = rows[spot[1]][spot[0]]
        worn.append(spot)
        cell = spot
        if code in _JOINABLE:
            return tuple(worn), True
        if spot == goal:
            return tuple(worn), False
    return None


def _free(
    rows: list[list[str]], candidate: Point, behind: Cell, forbidden: frozenset[Cell], tuning: Path
) -> bool:
    """Whether a step lands where a body could walk without breaking the line behind it."""
    cell = (int(candidate[0]), int(candidate[1]))
    if not _inside(cell):
        return False
    if any(
        spot in forbidden or rows[spot[1]][spot[0]] not in _WALKABLE
        for spot in carve.stamp(cell, tuning.width)
    ):
        return False
    if cell[0] != behind[0] and cell[1] != behind[1]:
        return _corner(rows, behind, (cell[0] - behind[0], cell[1] - behind[1]), forbidden) is not None
    return True


def _paint(rows: list[list[str]], cells: Iterable[Cell], tuning: Path) -> None:
    """Wear a run of cells into path, and any water among them into bridge."""
    for spot in cells:
        for column, row in carve.stamp(spot, tuning.width):
            code = rows[row][column]
            if code == "w":
                rows[row][column] = "b"
            elif code in _PAINTABLE:
                rows[row][column] = "p"


def _moves(
    rows: list[list[str]],
    courses: Water,
    channels: dict[Cell, int],
    cell: Cell,
    forbidden: frozenset[Cell],
    banned: frozenset[int],
    tuning: Path,
) -> Iterable[tuple[Cell, tuple[Cell, ...], int, float]]:
    """Yield every move out of a cell: one step onto land, or one straight run over a channel."""
    x, y = cell
    for dx, dy, length in _STEPS:
        spot = (x + dx, y + dy)
        if not _inside(spot) or spot in forbidden:
            continue
        # A corner step needs one of its two sides open. Cutting between two blocked cells is a
        # move no body could make, and it would leave the painted path split at the corner.
        if length > 1.0 and _corner(rows, cell, (dx, dy), forbidden) is None:
            continue
        code = rows[spot[1]][spot[0]]
        if code in _WALKABLE:
            yield spot, (), _NO_CHANNEL, _price(code, tuning) * length
        elif code == "w" and length == 1.0:
            crossing = _crossing(rows, courses, channels, cell, (dx, dy), forbidden, banned, tuning)
            if crossing is not None:
                yield crossing


def _crossing(
    rows: list[list[str]],
    courses: Water,
    channels: dict[Cell, int],
    cell: Cell,
    step: tuple[int, int],
    forbidden: frozenset[Cell],
    banned: frozenset[int],
    tuning: Path,
) -> tuple[Cell, tuple[Cell, ...], int, float] | None:
    """Scan straight across water for a landing, keeping the run inside one unbanned channel."""
    x, y = cell
    dx, dy = step
    run: list[Cell] = []
    for reach in range(1, tuning.crossing_run + 1):
        over = (x + dx * reach, y + dy * reach)
        if not _inside(over) or over in forbidden:
            return None
        code = rows[over[1]][over[0]]
        if code == "w":
            # Sharing the trunk or the fork would put a bridge where the water is widest and where
            # the courses run together, so only a plain channel cell is bridgeable.
            if over in courses.trunk or over in courses.fork_mask or over not in channels:
                return None
            if channels[over] in banned or (run and channels[over] != channels[run[0]]):
                return None
            run.append(over)
            continue
        if code not in _WALKABLE or not run:
            return None
        # A crossing is priced by the water it spans, so the search bridges only where it must.
        price = len(run) * tuning.crossing_cost + _price(code, tuning)
        return over, tuple(run), channels[run[0]], price
    return None


def _corner(
    rows: list[list[str]], cell: Cell, step: tuple[int, int], forbidden: frozenset[Cell]
) -> Cell | None:
    """The cell a corner step is painted through, or nothing when both its sides are blocked."""
    x, y = cell
    dx, dy = step
    for spot in ((x + dx, y), (x, y + dy)):
        if _inside(spot) and spot not in forbidden and rows[spot[1]][spot[0]] in _WALKABLE:
            return spot
    return None


def _price(code: str, tuning: Path) -> float:
    """What entering a cell costs: how fast a body crosses it, less any path already worn."""
    cost = FRAME.cell_size / GROUND_BY_CODE[code].speed
    return cost * (1.0 - tuning.merge_discount) if code == "p" else cost


def _inside(cell: Cell) -> bool:
    return 0 <= cell[0] < FRAME.cells_x and 0 <= cell[1] < FRAME.cells_y
