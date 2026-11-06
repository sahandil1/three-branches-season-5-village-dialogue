"""Everything placed once the ground, the buildings, and the routes are committed.

Dressing changes no ground. A prop reserves its catalog rectangle turned to its facing, stands on
the class already painted under it, and banks one standing cell it can be used from. That witness is
found when the prop is placed and protected from everything placed after it, which is what makes the
guarantee hold without any stage having to look back.

Most spots come off the road. A station is a point along the road's centreline with the
direction of travel there, so a stall, a lantern, a shrine, or a pine can be set a fixed distance to
one side of it and turned to face the road. Stalls and shrines front whichever way is actually
nearest instead of the station's tangent: the nearest road cell when one is within reach, else the
nearest path cell, and a shrine only ever stands on the road's north side. The market is the stretch
of that centreline nearest the anchor.

Stalls, the board, benches, shrines, gardens, the interior props, the pump, and the bell are
mandatory: running out of candidates discards the layout. Lanterns and pines are not. They skip a
blocked spot and carry on, and what fits is kept in the shipped village.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, dist, hypot, pi

from ..catalog import BUILDING_BY_TOKEN, CATALOG, PROP_BY_TOKEN
from ..geometry import Circle, Point, Rect, circle_intersects_circle, circle_intersects_rect, nearest_point
from ..grid import Cell
from ..layout import PlacedProp, Scenery, footprint, footprint_cells, shape_for
from ..rules import FRAME, GROUND_BY_CODE, PROFILE
from .config import Accessories, Retry
from .network import Road
from .sites import Settlement, Site

# Ground an outdoor prop may stand on. Road, path, bridge, and doorway all carry people.
_STANDABLE = frozenset({"g", "f", "e"})
# Ground that carries people, whose corridors a unit circle must not pinch.
_WAYS = frozenset({"r", "p", "b"})
_FACINGS = ("north", "east", "south", "west")
# Props are published in catalog type order, which is the order the catalog lists them in.
_ORDER = {kind.token: index for index, kind in enumerate(CATALOG.props)}
# Stalls and shrines front the nearest way rather than the station they were derived from.
_ROADSIDE_FACED = frozenset({"stall", "shrine"})
# How close a cell of road has to be for it to count as the way a roadside prop fronts.
_WAY_REACH = 3


@dataclass(frozen=True, slots=True)
class Station:
    """A point on the road's centreline and the way the road is running there."""

    point: Point
    tangent: Point
    arc: float


@dataclass(frozen=True, slots=True)
class Dressing:
    """Everything the accessory stage placed."""

    props: tuple[PlacedProp, ...]
    scenery: tuple[Scenery, ...]
    witnesses: tuple[tuple[str, Cell], ...]


def stations(road: Road, spacing: float) -> tuple[Station, ...]:
    """Walk the road's centreline and stand a station every spacing of it."""
    found: list[Station] = []
    arc = 0.0
    next_at = 0.0
    for before, after in zip(road.centreline, road.centreline[1:], strict=False):
        length = dist(before, after)
        if length == 0.0:
            continue
        tangent = ((after[0] - before[0]) / length, (after[1] - before[1]) / length)
        while next_at <= arc + length:
            share = (next_at - arc) / length
            point = (before[0] + tangent[0] * length * share, before[1] + tangent[1] * length * share)
            found.append(Station(point, tangent, next_at))
            next_at += spacing
        arc += length
    return tuple(found)


def _posts(road: Road, spacing: float) -> tuple[Station, ...]:
    """Stations dressing may use: the road, less the stretches where it is up on a bridge."""
    return tuple(
        station
        for station in stations(road, spacing)
        if not any(
            crossing.span[0] - 1 <= station.point[0] <= crossing.span[1] + 1 for crossing in road.crossings
        )
    )


def shrine_spots(road: Road, tuning: Accessories) -> tuple[Station, ...]:
    """Find where the road turns hardest, keeping the shrines that stand there apart.

    Sharpness is read over a window either side rather than step to step, so it answers the shape of
    a bend instead of one step's wobble.
    """
    window = tuning.shrine.window
    line = road.centreline
    lengths = [dist(before, after) for before, after in zip(line, line[1:], strict=False)]
    arcs = [0.0]
    for length in lengths:
        arcs.append(arcs[-1] + length)
    turns: list[tuple[float, Station]] = []
    for index in range(window, len(line) - window):
        point = line[index]
        if any(crossing.span[0] <= point[0] <= crossing.span[1] for crossing in road.crossings):
            continue
        before = _unit((point[0] - line[index - window][0], point[1] - line[index - window][1]))
        after = _unit((line[index + window][0] - point[0], line[index + window][1] - point[1]))
        turn = abs(_wrap(atan2(after[1], after[0]) - atan2(before[1], before[0])))
        turns.append((turn, Station(point, after, arcs[index])))
    found: list[Station] = []
    for _, station in sorted(turns, key=lambda item: (-item[0], item[1].arc)):
        if len(found) == tuning.shrine.count:
            break
        if all(abs(station.arc - other.arc) >= tuning.shrine.separation for other in found):
            found.append(station)
    if len(found) < tuning.shrine.count:
        raise Retry("the road did not bend often enough to stand every shrine")
    return tuple(sorted(found, key=lambda station: station.arc))


def dress(
    stream: random.Random,
    rows: list[list[str]],
    settlement: Settlement,
    road: Road,
    shrines: tuple[Station, ...],
    tuning: Accessories,
    clearance: tuple[Point, float],
) -> Dressing:
    """Place every accessory in a fixed order, banking a witness for each interactive prop."""
    yard = _Yard(rows, settlement, tuning, clearance)
    market = _market_arc(road, settlement)
    _stalls(stream, yard, road, market, tuning)
    _crates(stream, yard, settlement, tuning)
    _board(stream, yard, road, market, tuning)
    _benches(stream, yard, settlement, tuning)
    _shrines(stream, yard, road, shrines, tuning)
    _gardens(yard, settlement)
    _interiors(yard, settlement)
    _pump(stream, yard, settlement, tuning)
    _bell(stream, yard, road, tuning)
    _lanterns(stream, yard, road, market, tuning)
    _pines(stream, yard, road, tuning)
    return yard.dressing()


class _Yard:
    """The village as dressing sees it: what is taken, what is placed, and what must stay usable."""

    def __init__(
        self,
        rows: list[list[str]],
        settlement: Settlement,
        tuning: Accessories,
        clearance: tuple[Point, float],
    ) -> None:
        self.rows = rows
        self.settlement = settlement
        self.tuning = tuning
        self.props: list[PlacedProp] = []
        self.scenery: list[Scenery] = []
        self.witnesses: dict[str, Cell] = {}
        self.taken: set[Cell] = set()
        self.solids: list[Rect | Circle] = []
        # Committed solids bucketed by the cells their boxes span, so overlap checks stay local.
        self.placed: dict[tuple[int, int], list[Rect | Circle]] = {}
        # Placed pine cells bucketed the same way, so the scatter gap check stays local too.
        self.pines: dict[tuple[int, int], list[Cell]] = {}
        # What no later solid may close off: the spawn, every doorway approach, and each witness
        # as it is banked.
        self.protected: list[tuple[Point, float]] = [clearance]
        for site in settlement.sites:
            self.protected.extend((_centre(cell), PROFILE.body_radius) for cell in site.approaches)

    def place(self, token: str, cell: Cell, facing: str, *, interior: bool = False) -> bool:
        """Stand one prop if its footprint is free and it can be reached, and bank its witness."""
        held = sum(1 for item in self.props if item.type == token)
        item = PlacedProp(f"{token}_{held}", token, cell, facing)
        if not self._footprint_free(item, interior=interior):
            return False
        shape = shape_for(item)
        if not self._keeps_promises(item, shape):
            return False
        witness = self._witness(item, shape)
        if witness is None:
            return False
        self.props.append(item)
        self.witnesses[item.id] = witness
        self.taken.update(footprint_cells(item))
        self.solids.append(shape)
        self._bank(shape)
        self.protected.append((_centre(witness), PROFILE.body_radius))
        return True

    def stand(self, token: str, cell: Cell, *, scale: float = 1.0) -> bool:
        """Stand one piece of scenery, which is solid but is never used and banks no witness.

        A circular placement is planted at its drawn size; when that solid would overlap a solid
        already standing, close off a protected point, or pinch a passable way, it shrinks back to
        the base size before leaving the spot empty, since a bare one-cell tree is always legal
        where its cell is.
        """
        if not self._footprint_free(Scenery(token, cell)):
            return False
        for candidate in (scale, 1.0):
            item = Scenery(token, cell, candidate)
            shape = shape_for(item)
            if not self._keeps_promises(item, shape):
                continue
            self.scenery.append(item)
            self.taken.update(footprint_cells(item))
            self.solids.append(shape)
            self._bank(shape)
            if token == "pine":
                self.pines.setdefault(item.cell, []).append(item.cell)
            return True
        return False

    def dressing(self) -> Dressing:
        ordered = tuple(sorted(self.props, key=lambda item: (_ORDER[item.type], _number(item.id))))
        return Dressing(ordered, tuple(self.scenery), tuple(sorted(self.witnesses.items())))

    def _footprint_free(self, item: PlacedProp | Scenery, *, interior: bool = False) -> bool:
        """An interior prop stands on floor. Everything else stands on ground nobody walks over."""
        allowed = frozenset({"i"}) if interior else _STANDABLE
        for cell in footprint_cells(item):
            if not _inside(cell) or cell in self.taken:
                return False
            if self.rows[cell[1]][cell[0]] not in allowed:
                return False
        return True

    def _keeps_promises(self, item: PlacedProp | Scenery, shape: Rect | Circle) -> bool:
        """Refuse a solid that would close off the spawn, a doorway, or a banked witness, overlap a
        solid already standing, or pinch a passable way, so cell exclusivity stays backed by solid
        exclusivity and the village stays one body-clear region."""
        if any(_touches(shape, point, radius) for point, radius in self.protected):
            return False
        if not self._wayside_clear(item, shape):
            return False
        return not any(_overlaps(shape, other) for other in self._near(shape))

    def _bank(self, shape: Rect | Circle) -> None:
        """Record a committed solid in the cell buckets the overlap checks search."""
        for cell in _span(shape):
            self.placed.setdefault(cell, []).append(shape)

    def _near(self, shape: Rect | Circle) -> tuple[Rect | Circle, ...]:
        """The solids committed in the cells a candidate's box reaches, deduplicated."""
        seen: set[int] = set()
        found: list[Rect | Circle] = []
        for cell in _span(shape):
            for other in self.placed.get(cell, ()):
                if id(other) in seen:
                    continue
                seen.add(id(other))
                found.append(other)
        return tuple(found)

    def _wayside_clear(self, item: PlacedProp | Scenery, shape: Rect | Circle) -> bool:
        """A unit circle may not pinch a passable way, whatever size it draws at.

        Only 1×1 circular placements can wedge between travellers and a way, so only they carry
        the check: the solid must stay body-clear of every road, path, or bridge cell centre it
        reaches. A scaled pine is just a larger instance of the same family, so the guard follows
        the geometry instead of a token list.
        """
        width, height = footprint(item)
        if not isinstance(shape, Circle) or min(width, height) > 1:
            return True
        ring = shape.radius + PROFILE.body_radius
        for row in range(int(shape.y - ring), int(shape.y + ring) + 1):
            for column in range(int(shape.x - ring), int(shape.x + ring) + 1):
                if not _inside((column, row)):
                    continue
                if self.rows[row][column] in _WAYS and _touches(
                    shape, _centre((column, row)), PROFILE.body_radius
                ):
                    return False
        return True

    def _witness(self, item: PlacedProp, shape: Rect | Circle) -> Cell | None:
        """Find the nearest standing cell in reach of a prop, with a clear line to it."""
        reach = int(PROFILE.prop_reach) + 1
        width, height = footprint(item)
        x, y = item.cell
        ring = [
            (column, row)
            for row in range(y - reach, y + height + reach)
            for column in range(x - reach, x + width + reach)
            if not (x <= column < x + width and y <= row < y + height)
        ]
        middle = (x + width / 2, y + height / 2)
        for cell in sorted(ring, key=lambda spot: (dist(_centre(spot), middle), spot)):
            if not _inside(cell) or cell in self.taken:
                continue
            if not GROUND_BY_CODE[self.rows[cell[1]][cell[0]]].passable:
                continue
            centre = _centre(cell)
            if _touches(shape, centre, PROFILE.body_radius):
                continue
            if any(_touches(other, centre, PROFILE.body_radius) for other in self.solids):
                continue
            spot = nearest_point(centre, shape)
            if dist(centre, spot) > PROFILE.prop_reach or not self._sees(centre, spot):
                continue
            return cell
        return None

    def _sees(self, start: Point, end: Point) -> bool:
        """Walk the short line to a prop, refusing anything that blocks sight along it."""
        span = dist(start, end)
        steps = max(int(span / 0.1), 1)
        for index in range(steps + 1):
            share = index / steps
            cell = (
                int(start[0] + (end[0] - start[0]) * share),
                int(start[1] + (end[1] - start[1]) * share),
            )
            if not _inside(cell) or GROUND_BY_CODE[self.rows[cell[1]][cell[0]]].blocks_sight:
                return False
        return True


def _stalls(stream: random.Random, yard: _Yard, road: Road, market: float, tuning: Accessories) -> None:
    """Scatter the stalls along both sides of the road through the market."""
    stall = tuning.stall
    posts = _near(_posts(road, float(stall.spacing)), market, float(stall.span))
    side = 1 if stream.random() < 0.5 else -1
    for index in range(stall.count):
        turn = side if index % 2 == 0 else -side
        if not _try(stream, yard, "stall", posts, turn, tuning, stall.budget):
            raise Retry("a market stall found nowhere to stand beside the road")


def _crates(stream: random.Random, yard: _Yard, settlement: Settlement, tuning: Accessories) -> None:
    """Litter the market with loose crates, skipping the spots that cannot take one."""
    crate = tuning.crate
    x, y = settlement.anchors[1]
    for _ in range(crate.count):
        cell = (
            x + stream.randint(-crate.span, crate.span),
            y + stream.randint(-crate.span, crate.span),
        )
        yard.stand("crate", cell)


def _board(stream: random.Random, yard: _Yard, road: Road, market: float, tuning: Accessories) -> None:
    posts = _near(_posts(road, 1.0), market, float(tuning.stall.span))
    if not _try(stream, yard, "board", posts, 1, tuning, tuning.board.budget):
        raise Retry("the notice board found nowhere to stand in the market")


def _benches(stream: random.Random, yard: _Yard, settlement: Settlement, tuning: Accessories) -> None:
    """Split the benches between the plaza, the market, and the front of the inn."""
    bench = tuning.bench
    wanted = (
        (settlement.plaza, bench.plaza),
        (settlement.anchors[1], bench.market),
        (settlement.site("inn").approaches[0], bench.inn),
    )
    for spot, count in wanted:
        for _ in range(count):
            if not _around(stream, yard, "bench", spot, bench.budget):
                raise Retry("a bench found nowhere to sit")


def _shrines(
    stream: random.Random, yard: _Yard, road: Road, shrines: tuple[Station, ...], tuning: Accessories
) -> None:
    """Stand each shrine at its turn on the road's north side, letting it slide to find room."""
    for station in shrines:
        posts = _near(_posts(road, 1.0), station.arc, float(tuning.shrine.window))
        if not _try(
            stream, yard, "shrine", posts or (station,), 1, tuning, tuning.shrine.budget, north_only=True
        ):
            raise Retry("a shrine found nowhere to stand at the turn it belongs to")


def _gardens(yard: _Yard, settlement: Settlement) -> None:
    """Stand each home's garden on the cells its site already set aside behind the wall."""
    for site in settlement.sites:
        if not site.garden:
            continue
        origin = (min(x for x, _ in site.garden), min(y for _, y in site.garden))
        if not yard.place("plot", origin, site.building.facing):
            raise Retry("a garden would not fit beside the wall its home left for it")


def _interiors(yard: _Yard, settlement: Settlement) -> None:
    """Stand the hearth and the repair bench against the wall opposite their doorway."""
    for building_id, token in (("inn", "hearth"), ("shed", "repair_bench")):
        site = settlement.site(building_id)
        cell = _interior_spot(site, token)
        if not yard.place(token, cell, site.building.facing, interior=True):
            raise Retry(f"the {token} would not stand inside the {building_id}")


def _pump(stream: random.Random, yard: _Yard, settlement: Settlement, tuning: Accessories) -> None:
    if not _around(stream, yard, "pump", settlement.plaza, tuning.pump.budget):
        raise Retry("the well pump found nowhere to stand in the plaza")


def _bell(stream: random.Random, yard: _Yard, road: Road, tuning: Accessories) -> None:
    """Hang the bell beside the west stretch of the road."""
    posts = tuple(station for station in _posts(road, 2.0) if station.point[0] < FRAME.cells_x / 3)
    if not _try(stream, yard, "bell", posts, 1, tuning, tuning.bell.budget):
        raise Retry("the beacon bell found nowhere to hang beside the west road")


def _lanterns(stream: random.Random, yard: _Yard, road: Road, market: float, tuning: Accessories) -> None:
    """Light the road, closer together at the market. A blocked station is simply skipped."""
    lantern = tuning.lantern
    posts = sorted(
        {
            *_posts(road, float(lantern.spacing)),
            *_near(_posts(road, float(lantern.market_spacing)), market, float(tuning.stall.span)),
        },
        key=lambda station: station.arc,
    )
    _wayside_line(stream, yard, tuple(posts), "lantern", tuning)


def _pines(stream: random.Random, yard: _Yard, road: Road, tuning: Accessories) -> None:
    """Plant the pines last, along the road and scattered over open ground, with companions."""
    pine = tuning.pine
    low, high = pine.size
    _wayside_line(
        stream,
        yard,
        _posts(road, float(pine.spacing)),
        "pine",
        tuning,
        scale=lambda stream: _pine_scale(stream, low, high),
    )
    for _ in range(pine.scatter):
        cell = (stream.randrange(FRAME.cells_x), stream.randrange(FRAME.cells_y))
        if not _spaced(yard, cell, pine.gap):
            continue
        if not yard.stand("pine", cell, scale=_pine_scale(stream, low, high)):
            continue
        if stream.random() >= pine.companion_chance:
            continue
        for _ in range(pine.companions):
            near = (
                cell[0] + stream.randint(-pine.gap, pine.gap),
                cell[1] + stream.randint(-pine.gap, pine.gap),
            )
            if _spaced(yard, near, pine.gap - 1):
                yard.stand("pine", near, scale=_pine_scale(stream, low, high))


def _wayside_line(
    stream: random.Random,
    yard: _Yard,
    posts: tuple[Station, ...],
    token: str,
    tuning: Accessories,
    *,
    scale: Callable[[random.Random], float] | None = None,
) -> None:
    """Plant optional unit circles along the road, alternating sides and skipping blocked spots.

    Lanterns and pines are the same placement: walk the stations in their centreline order, stand
    one on the drawn side or the other, and carry on when neither takes. A ``scale`` callable lets
    each station draw its own size, where one kind varies.
    """
    side = 1 if stream.random() < 0.5 else -1
    for index, station in enumerate(posts):
        turn = side if index % 2 == 0 else -side
        for attempt in (turn, -turn):
            if _stand_beside(
                yard, token, station, attempt, tuning, scale=scale(stream) if scale is not None else 1.0
            ):
                break


def _pine_scale(stream: random.Random, low: float, high: float) -> float:
    """Draw one pine's size. Being drawn at placement, the solid it makes keeps its true radius."""
    return round(stream.uniform(low, high), 3)


def _try(
    stream: random.Random,
    yard: _Yard,
    token: str,
    posts: tuple[Station, ...],
    side: int,
    tuning: Accessories,
    budget: int,
    *,
    north_only: bool = False,
) -> bool:
    """Sweep the stations from a drawn start until one takes the prop, on its side or the other.

    ``north_only`` confines a shrine to the side that fronts the road's north bank, which is
    whichever side lifts the prop above the station on the centreline.
    """
    if not posts:
        return False
    start = stream.randrange(len(posts))
    for offset in range(min(budget, len(posts))):
        station = posts[(start + offset) % len(posts)]
        turns = (_north_turn(station),) if north_only else (side, -side)
        for turn in turns:
            if _stand_beside(yard, token, station, turn, tuning):
                return True
    return False


def _north_turn(station: Station) -> int:
    """The side whose normal lifts the prop north so a shrine fronts the road's top bank."""
    return 1 if station.tangent[0] >= 0 else -1


def _stand_beside(
    yard: _Yard, token: str, station: Station, side: int, tuning: Accessories, *, scale: float = 1.0
) -> bool:
    """Set a prop one setback off the road at a station, turned to face the way it serves.

    Stalls and shrines front the nearest way instead of the station's tangent: the nearest road
    cell when one is within reach, else the nearest path cell. ``scale`` is a circular scenery's
    drawn size; only the scenery branch consumes it, so a prop caller passes the default and it is
    ignored.
    """
    kind = PROP_BY_TOKEN.get(token)
    normal = (-station.tangent[1] * side, station.tangent[0] * side)
    facing = _facing_of((-normal[0], -normal[1]))
    width, height = (kind.width, kind.height) if kind is not None else (1, 1)
    if kind is not None and facing in {"east", "west"}:
        width, height = height, width
    away = tuning.setback + max(width, height) / 2 + 1.5
    centre = (station.point[0] + normal[0] * away, station.point[1] + normal[1] * away)
    cell = (int(centre[0] - width / 2), int(centre[1] - height / 2))
    if token in _ROADSIDE_FACED:
        facing = _nearest_way_facing(yard.rows, cell, width, height, facing)
    return yard.place(token, cell, facing) if kind is not None else yard.stand(token, cell, scale=scale)


def _nearest_way_facing(rows: list[list[str]], cell: Cell, width: int, height: int, fallback: str) -> str:
    """The compass way to the nearest road from a placement, or the nearest path when no road is close.

    A stall or shrine fronts the road it can be reached from: the nearest road cell centre within
    reach when there is one, else the nearest path cell centre. Nothing within reach keeps the
    station's tangent facing.
    """
    centre = (cell[0] + width / 2, cell[1] + height / 2)
    spot = _nearest_ground(rows, centre, "r")
    if spot is None:
        spot = _nearest_ground(rows, centre, "p")
    if spot is None:
        return fallback
    return _facing_of((spot[0] - centre[0], spot[1] - centre[1]))


def _nearest_ground(rows: list[list[str]], centre: Point, code: str) -> Point | None:
    """The cell centre of the nearest ``code`` cell within reach of a point, or None."""
    best: tuple[float, Point] | None = None
    for row in range(int(centre[1]) - _WAY_REACH, int(centre[1]) + _WAY_REACH + 1):
        for column in range(int(centre[0]) - _WAY_REACH, int(centre[0]) + _WAY_REACH + 1):
            if not _inside((column, row)) or rows[row][column] != code:
                continue
            spot = (column + 0.5, row + 0.5)
            distance = hypot(spot[0] - centre[0], spot[1] - centre[1])
            if best is None or distance < best[0]:
                best = (distance, spot)
    return None if best is None else best[1]


def _around(stream: random.Random, yard: _Yard, token: str, centre: Cell, budget: int) -> bool:
    """Draw cells around a district centre until one of them takes the prop."""
    reach = 4
    for _ in range(budget):
        cell = (centre[0] + stream.randint(-reach, reach), centre[1] + stream.randint(-reach, reach))
        if yard.place(token, cell, _FACINGS[stream.randrange(len(_FACINGS))]):
            return True
    return False


def _spaced(yard: _Yard, cell: Cell, gap: int) -> bool:
    """Keep pines apart, windowed over only nearby trees, so a stand reads as trees rather than
    as a hedge. A pine closer than ``gap`` is never outside the box window around a candidate."""
    if not _inside(cell):
        return False
    for row in range(cell[1] - gap, cell[1] + gap + 1):
        for column in range(cell[0] - gap, cell[0] + gap + 1):
            for pine in yard.pines.get((column, row), ()):
                if dist(cell, pine) < gap:
                    return False
    return True


def _span(shape: Rect | Circle) -> tuple[tuple[int, int], ...]:
    """The unit cells a solid's bounding box can reach."""
    if isinstance(shape, Rect):
        left, right = int(shape.x), int(shape.right)
        bottom, top = int(shape.y), int(shape.top)
    else:
        left, right = int(shape.x - shape.radius), int(shape.x + shape.radius)
        bottom, top = int(shape.y - shape.radius), int(shape.y + shape.radius)
    return tuple((column, row) for row in range(bottom, top + 1) for column in range(left, right + 1))


def _market_arc(road: Road, settlement: Settlement) -> float:
    """Where along the road the market is, measured in the road's own arc length."""
    anchor = _centre(settlement.anchors[1])
    posts = stations(road, 1.0)
    return min(posts, key=lambda station: dist(station.point, anchor)).arc


def _near(posts: tuple[Station, ...], arc: float, span: float) -> tuple[Station, ...]:
    return tuple(station for station in posts if abs(station.arc - arc) <= span)


def _interior_spot(site: Site, token: str) -> Cell:
    """Centre an interior prop against the inside of the wall opposite the doorway."""
    kind = BUILDING_BY_TOKEN[site.building.type]
    facing = site.building.facing
    width, height = footprint(PlacedProp("probe", token, (0, 0), facing))
    x, y = site.building.cell
    floor = (x + 1, y + 1, kind.width - 2, kind.height - 2)
    if facing == "north":
        return (floor[0] + (floor[2] - width) // 2, floor[1])
    if facing == "south":
        return (floor[0] + (floor[2] - width) // 2, floor[1] + floor[3] - height)
    if facing == "east":
        return (floor[0], floor[1] + (floor[3] - height) // 2)
    return (floor[0] + floor[2] - width, floor[1] + (floor[3] - height) // 2)


def _touches(shape: Rect | Circle, point: Point, radius: float) -> bool:
    if isinstance(shape, Rect):
        return circle_intersects_rect(point, radius, shape)
    return circle_intersects_circle(point, radius, shape)


def _overlaps(first: Rect | Circle, second: Rect | Circle) -> bool:
    """Whether two placed solids share any interior, using the same strict tests movement uses."""
    if isinstance(first, Circle) and isinstance(second, Circle):
        return circle_intersects_circle((first.x, first.y), first.radius, second)
    if isinstance(first, Circle) and isinstance(second, Rect):
        return circle_intersects_rect((first.x, first.y), first.radius, second)
    if isinstance(first, Rect) and isinstance(second, Circle):
        return circle_intersects_rect((second.x, second.y), second.radius, first)
    if isinstance(first, Rect) and isinstance(second, Rect):
        return (
            first.x < second.right
            and second.x < first.right
            and first.y < second.top
            and second.y < first.top
        )
    return False


def _facing_of(direction: Point) -> str:
    if abs(direction[0]) >= abs(direction[1]):
        return "east" if direction[0] >= 0 else "west"
    return "north" if direction[1] >= 0 else "south"


def _centre(cell: Cell) -> Point:
    return (cell[0] + 0.5, cell[1] + 0.5)


def _number(prop_id: str) -> int:
    return int(prop_id.rsplit("_", 1)[1])


def _wrap(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def _unit(vector: Point) -> Point:
    length = hypot(*vector)
    return (1.0, 0.0) if length == 0.0 else (vector[0] / length, vector[1] / length)


def _inside(cell: Cell) -> bool:
    return 0 <= cell[0] < FRAME.cells_x and 0 <= cell[1] < FRAME.cells_y
