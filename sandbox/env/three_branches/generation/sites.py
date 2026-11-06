"""District anchors and the building sites placed against them.

Anchors come first and are chosen on the terrain alone, because the road has to know where it is
going before it can be walked. Each is a point inside the road band: the shed and bell to the west,
the market in the middle, the inn to the east. The well plaza is not a road anchor; it sits in the
crook below the fork and is reached by a footpath.

The inn and the shed then stand beside their anchors. A home stands wherever the ground suits it
best: bank side among the channels, level, dry, and clear of the homes already up. Nothing stands
inside the band. Keeping every site and its margin out of it is what leaves the road a clear run
across the frame, so the road walker never has to squeeze past a wall it cannot move.

A site is accepted only once its rectangle, its margin, its garden, and a doorway with a route to
the band all hold together, and a door never opens away from the road. Running out of candidates
discards the whole layout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import dist, hypot

from ..catalog import BUILDING_BY_TOKEN, PROP_BY_TOKEN, BuildingType
from ..grid import Cell
from ..layout import Building, doorway_cells, paint_site
from ..rules import FRAME
from . import grounds, paths
from .config import Network, Retry, Sites
from .water import Water

# Ground a site may not be dropped on. Everything else is land the village may build over.
_KEEP_CLEAR = frozenset({"w"})
_FACINGS = ("north", "east", "south", "west")
# Buildings in layout order. Inn and shed are placed first, since they anchor their districts.
_PLACEMENT = ("inn", "shed", "home_0", "home_1", "home_2", "home_3", "home_4")
_LAYOUT = ("home_0", "home_1", "home_2", "home_3", "home_4", "inn", "shed")


@dataclass(frozen=True, slots=True)
class Site:
    """One placed building and everything placement promised about it."""

    building: Building
    # The rectangle plus its margin, which is what later stages must stay out of.
    reserved: frozenset[Cell]
    # The garden plot's cells, empty for the inn and the shed. Routes and dressing keep off them.
    garden: tuple[Cell, ...]
    # The cells immediately outside the door run, which a footpath has to reach.
    approaches: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class Settlement:
    """Everything the settlement stage committed."""

    plaza: Cell
    # West, middle, and east road targets: the shed, the market, and the inn.
    anchors: tuple[Cell, Cell, Cell]
    sites: tuple[Site, ...]
    reserved: frozenset[Cell]
    keep_clear: frozenset[Cell]

    def site(self, building_id: str) -> Site:
        return next(site for site in self.sites if site.building.id == building_id)

    @property
    def buildings(self) -> tuple[Building, ...]:
        return tuple(site.building for site in self.sites)


def settle(
    stream: random.Random,
    rows: list[list[str]],
    elevation: list[list[float]],
    moisture: list[list[float]],
    courses: Water,
    tuning: Sites,
    network: Network,
) -> Settlement:
    """Choose the districts and stand every building, painting each site as it is accepted."""
    distance = grounds.water_distance(rows)
    band = network.road.band
    # Everywhere a footpath could ever reach the road from, answered once rather than per candidate.
    # Somewhere the road can never be joined to is somewhere the village does not build.
    joinable = paths.joinable(
        rows,
        courses,
        ((x, y) for y in range(band[0], band[1] + 1) for x in range(FRAME.cells_x)),
        network.path,
    )
    plaza = _plaza(stream, rows, elevation, moisture, courses, tuning, band, joinable)
    reserved = set(_disc(plaza, tuning.plaza_radius))
    anchors = _anchors(stream, rows, elevation, moisture, tuning, inner_band(network))
    anchored = {"inn": anchors[2], "shed": anchors[0]}
    sites: dict[str, Site] = {}
    # Where the homes already up are, which is what keeps the next one from crowding them.
    homes: list[Cell] = []
    for building_id in _PLACEMENT:
        kind = BUILDING_BY_TOKEN[_type_of(building_id)]
        if building_id in anchored:
            site = _stand(
                stream, rows, building_id, kind, anchored[building_id], reserved, joinable, tuning, band
            )
        else:
            site = _homestead(
                stream,
                rows,
                building_id,
                kind,
                elevation,
                moisture,
                distance,
                courses,
                reserved,
                joinable,
                tuning,
                band,
                homes,
            )
        paint_site(rows, site.building)
        reserved.update(site.reserved)
        sites[building_id] = site
    placed = tuple(sites[building_id] for building_id in _LAYOUT)
    gardens = frozenset(cell for site in placed for cell in site.garden)
    return Settlement(plaza, anchors, placed, frozenset(reserved), gardens)


def inner_band(network: Network) -> tuple[int, int]:
    """The rows the road actually uses.

    The band's outer rows are the wall the road is pushed off, so an anchor placed on one is an
    anchor the road can never come within reach of.
    """
    road = network.road
    return (road.band[0] + road.width, road.band[1] - road.width)


def _plaza(
    stream: random.Random,
    rows: list[list[str]],
    elevation: list[list[float]],
    moisture: list[list[float]],
    courses: Water,
    tuning: Sites,
    band: tuple[int, int],
    joinable: frozenset[Cell],
) -> Cell:
    """Find the well plaza in the crook below the fork: open, dry, and flat enough to gather on."""
    fork_x, fork_y = courses.fork
    best: tuple[float, Cell] | None = None
    for _ in range(tuning.budget):
        candidate = (
            fork_x + stream.randint(-tuning.reach, tuning.reach),
            fork_y - stream.randint(0, tuning.reach),
        )
        if candidate[1] <= band[1] or candidate not in joinable:
            continue
        if not _open(rows, _disc(candidate, tuning.plaza_radius)):
            continue
        score = (
            _dry(moisture, candidate) * tuning.scores.dry + _flat(elevation, candidate) * tuning.scores.flat
        )
        # The crook is the ground the fork closes around, so nearness to it settles ties.
        score -= dist(candidate, courses.fork) / FRAME.cells_y
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None:
        raise Retry("no room for a well plaza below the fork")
    return best[1]


def _anchors(
    stream: random.Random,
    rows: list[list[str]],
    elevation: list[list[float]],
    moisture: list[list[float]],
    tuning: Sites,
    band: tuple[int, int],
) -> tuple[Cell, Cell, Cell]:
    """Pick the west, middle, and east road targets on dry, flat ground inside the band."""
    third = FRAME.cells_x // 3
    spans = ((1, third), (third, 2 * third), (2 * third, FRAME.cells_x - 1))
    found: list[Cell] = []
    for low, high in spans:
        best: tuple[float, Cell] | None = None
        for _ in range(tuning.budget):
            candidate = (stream.randrange(low, high), stream.randint(*band))
            if rows[candidate[1]][candidate[0]] in _KEEP_CLEAR:
                continue
            score = (
                _dry(moisture, candidate) * tuning.scores.dry
                + _flat(elevation, candidate) * tuning.scores.flat
            )
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            raise Retry("a district anchor found no dry ground inside the road band")
        found.append(best[1])
    return (found[0], found[1], found[2])


def _homestead(
    stream: random.Random,
    rows: list[list[str]],
    building_id: str,
    kind: BuildingType,
    elevation: list[list[float]],
    moisture: list[list[float]],
    distance: list[list[int]],
    courses: Water,
    reserved: set[Cell],
    joinable: frozenset[Cell],
    tuning: Sites,
    band: tuple[int, int],
    others: list[Cell],
) -> Site:
    """Stand one home on the best ground it can find, and remember where it went.

    Homes live among the channels, south of the fork. Above it the trunk is too wide to bridge, so
    ground up there is somewhere a footpath could never carry anyone home from. Every candidate is
    a whole site rather than a point, so the ground a home is scored on is ground it really stands
    on, and the best of them wins rather than the first.
    """
    # As near the water as this home and its margin can stand, which is what bank side means to it.
    reach = max(kind.width, kind.height) // 2 + tuning.margin
    best: tuple[float, Site, Cell] | None = None
    for _ in range(tuning.budget):
        origin = (stream.randrange(1, FRAME.cells_x - 1), stream.randrange(1, courses.fork[1]))
        site = _candidate(stream, rows, building_id, kind, origin, reserved, joinable, tuning, band)
        if site is None:
            continue
        centre = (origin[0] + kind.width // 2, origin[1] + kind.height // 2)
        apart = min((dist(centre, other) for other in others), default=float(FRAME.cells_x))
        score = (
            _bank(distance, centre, reach) * tuning.scores.bank
            + _flat(elevation, centre) * tuning.scores.flat
            + _dry(moisture, centre) * tuning.scores.dry
            + min(apart / FRAME.cells_x, 1.0) * tuning.scores.apart
        )
        if best is None or score > best[0]:
            best = (score, site, centre)
    if best is None:
        raise Retry(f"nowhere to stand {building_id}")
    others.append(best[2])
    return best[1]


def _stand(
    stream: random.Random,
    rows: list[list[str]],
    building_id: str,
    kind: BuildingType,
    wanted: Cell,
    reserved: set[Cell],
    joinable: frozenset[Cell],
    tuning: Sites,
    band: tuple[int, int],
) -> Site:
    """Stand one building near where it belongs, taking the first candidate that holds together."""
    spread = tuning.reach + max(kind.width, kind.height)
    for _ in range(tuning.budget):
        origin = (
            wanted[0] + stream.randint(-spread, spread),
            wanted[1] + stream.randint(-spread, spread),
        )
        site = _candidate(stream, rows, building_id, kind, origin, reserved, joinable, tuning, band)
        if site is not None:
            return site
    raise Retry(f"nowhere to stand {building_id}")


def _candidate(
    stream: random.Random,
    rows: list[list[str]],
    building_id: str,
    kind: BuildingType,
    origin: Cell,
    reserved: set[Cell],
    joinable: frozenset[Cell],
    tuning: Sites,
    band: tuple[int, int],
) -> Site | None:
    """Try the ways a door may face from here, and take the first site that holds together."""
    facings = _facings(origin[1], band)
    start = stream.randrange(len(facings))
    for turn in range(len(facings)):
        site = _site(stream, building_id, kind, origin, facings[(start + turn) % len(facings)], tuning)
        if site is None or not _clear(rows, site, reserved, band):
            continue
        # The doorway has to open toward ground a footpath can carry back to the road.
        if any(cell in joinable for cell in site.approaches):
            return site
    return None


def _facings(row: int, band: tuple[int, int]) -> tuple[str, ...]:
    """The ways a door may face from a row. A door never opens away from the road.

    A site and its margin stand clear of the band, so a building is either north of the road or
    south of it, and the way that turns its back on the road is the one it may not face. A
    candidate straddling the band keeps all four, and is turned down for standing there at all.
    """
    if row > band[1]:
        return tuple(facing for facing in _FACINGS if facing != "north")
    if row < band[0]:
        return tuple(facing for facing in _FACINGS if facing != "south")
    return _FACINGS


def _site(
    stream: random.Random, building_id: str, kind: BuildingType, origin: Cell, facing: str, tuning: Sites
) -> Site | None:
    """Assemble one candidate site, or report that its geometry leaves the frame."""
    x, y = origin
    if x < 1 or y < 1 or x + kind.width >= FRAME.cells_x or y + kind.height >= FRAME.cells_y:
        return None
    building = Building(building_id, _type_of(building_id), origin, facing)
    garden = _garden(stream, kind, origin, facing, tuning) if building.type == "home" else ()
    if any(not _inside(cell) for cell in garden):
        return None
    # The margin is the site's clearance, and the garden, which may reach beyond it, is reserved
    # too, so a later site, the road, or a footpath never crosses either.
    reserved = set(
        (column, row)
        for row in range(y - tuning.margin, y + kind.height + tuning.margin)
        for column in range(x - tuning.margin, x + kind.width + tuning.margin)
        if _inside((column, row))
    )
    reserved.update(garden)
    return Site(building, frozenset(reserved), garden, _approaches(building, kind))


def _garden(
    stream: random.Random, kind: BuildingType, origin: Cell, facing: str, tuning: Sites
) -> tuple[Cell, ...]:
    """Place the garden on the wall opposite the doorway, drawn a gap away and slid along it.

    The gap is cells of open ground between the home wall and the plot's near edge, from zero up to
    the tuning. The slide moves the plot along the wall, at most a ``garden_slide`` past either
    end, so no two homes yield the same plot. A candidate whose garden does not hold together is a
    candidate the site rejects.
    """
    plot = PROP_BY_TOKEN["plot"]
    width, height = (plot.height, plot.width) if facing in {"east", "west"} else (plot.width, plot.height)
    gap = stream.randint(0, tuning.garden_gap)
    x, y = origin
    if facing in {"north", "south"}:
        slide = stream.randint(-tuning.garden_slide, kind.width - width + tuning.garden_slide)
        spot = (
            x + slide,
            y - height - gap if facing == "north" else y + kind.height + gap,
        )
    else:
        slide = stream.randint(-tuning.garden_slide, kind.height - height + tuning.garden_slide)
        spot = (
            x - width - gap if facing == "east" else x + kind.width + gap,
            y + slide,
        )
    return tuple((spot[0] + column, spot[1] + row) for row in range(height) for column in range(width))


def _approaches(building: Building, kind: BuildingType) -> tuple[Cell, ...]:
    """The cells a visitor stands on to use the door, one step outside the run."""
    x, y = building.cell
    door = doorway_cells(building)
    if building.facing == "north":
        return tuple((column, y + kind.height) for column, _ in door)
    if building.facing == "south":
        return tuple((column, y - 1) for column, _ in door)
    if building.facing == "east":
        return tuple((x + kind.width, row) for _, row in door)
    return tuple((x - 1, row) for _, row in door)


def _clear(rows: list[list[str]], site: Site, reserved: set[Cell], band: tuple[int, int]) -> bool:
    """Report whether a candidate stands on free land, out of the band and off every other site."""
    for cell in site.reserved:
        if cell in reserved or rows[cell[1]][cell[0]] in _KEEP_CLEAR:
            return False
        if band[0] <= cell[1] <= band[1]:
            return False
    return _open(rows, site.approaches)


def _open(rows: list[list[str]], cells: tuple[Cell, ...]) -> bool:
    return all(_inside(cell) and rows[cell[1]][cell[0]] not in _KEEP_CLEAR for cell in cells)


def _disc(centre: Cell, radius: float) -> tuple[Cell, ...]:
    reach = int(radius) + 1
    return tuple(
        (centre[0] + dx, centre[1] + dy)
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
        if hypot(dx, dy) <= radius
    )


def _flat(elevation: list[list[float]], cell: Cell) -> float:
    """How level the ground is, as a share of the unit range. Level ground scores one."""
    x, y = cell
    if not (0 < x < FRAME.cells_x - 1 and 0 < y < FRAME.cells_y - 1):
        return 0.0
    across = abs(elevation[y][x + 1] - elevation[y][x - 1]) / 2
    along = abs(elevation[y + 1][x] - elevation[y - 1][x]) / 2
    return max(0.0, 1.0 - max(across, along) * FRAME.cells_x)


def _dry(moisture: list[list[float]], cell: Cell) -> float:
    return 1.0 - moisture[cell[1]][cell[0]]


def _bank(distance: list[list[int]], cell: Cell, reach: int) -> float:
    """How much a cell reads as bank side: as near the water as a home and its margin can stand.

    Ground closer than that reach is bank a home would have its feet in, so it scores nothing at
    all rather than scoring best, which is the mistake that leaves five homes with nowhere to go.
    """
    span = distance[cell[1]][cell[0]]
    return 0.0 if span <= reach else 1.0 / (1.0 + span - reach)


def _type_of(building_id: str) -> str:
    return "home" if building_id.startswith("home") else building_id


def _inside(cell: Cell) -> bool:
    return 0 <= cell[0] < FRAME.cells_x and 0 <= cell[1] < FRAME.cells_y
