"""What the finished village is checked against before it is handed over.

Each construction stage already checks its own work as it commits, so this is the assembly pass:
the things no single stage could see. It reads the layout the way the engine will, through
``body_clear`` and ``line_clear``, so a village that passes here is one a day can actually be
played in. Every resident can stand where the day starts them, and each interactive prop's banked
standing cell is checked to be body-clear, in reach, and in line of sight of its prop. That the
banked cell is reachable from the spawn is left to the guarantee suite, which independently finds
each prop a standing cell reachable from the spawn over the pinned batch.
"""

from __future__ import annotations

from ..catalog import BUILDING_BY_TOKEN
from ..geometry import distance, nearest_point
from ..grid import Cell
from ..layout import Layout, doorway_cells, footprint_cells
from ..rules import FRAME, GROUND_BY_CODE, PROFILE
from .config import Generation, Retry
from .network import Road
from .paths import Footpaths
from .sites import Settlement, Site

_HOMES = ("home_0", "home_1", "home_2", "home_3", "home_4")
# One of each of these stands in every village, whatever the seed.
_STABLE = ("pump", "board", "hearth", "repair_bench", "bell")


def check(
    layout: Layout,
    settlement: Settlement,
    road: Road,
    footpaths: Footpaths,
    witnesses: tuple[tuple[str, Cell], ...],
    tuning: Generation,
) -> None:
    """Check the assembled village, raising ``Retry`` naming whatever did not hold."""
    _features(layout)
    _poses(layout)
    _ledger(layout)
    _sites(layout, settlement)
    _road(layout, road, tuning)
    _paths(layout, settlement, footpaths, tuning)
    _witnesses(layout, witnesses)


def _poses(layout: Layout) -> None:
    """Every resident can stand where the day starts them."""
    for building in layout.buildings:
        if building.type != "home":
            continue
        for resident in (0, 1):
            pose = layout.residence_pose(building.id, resident)
            if not layout.body_clear(pose.position):
                raise Retry(f"a resident of {building.id} cannot stand where the day starts them")


def _features(layout: Layout) -> None:
    """Every village holds one of each stable feature, five homes, and contiguous prop ids."""
    buildings = {building.id for building in layout.buildings}
    if buildings != {*_HOMES, "inn", "shed"}:
        raise Retry("the village did not come out with five homes, an inn, and a shed")
    for token in _STABLE:
        if sum(1 for item in layout.props if item.type == token) != 1:
            raise Retry(f"the village needs exactly one {token}")
    held: dict[str, list[int]] = {}
    for item in layout.props:
        held.setdefault(item.type, []).append(int(item.id.rsplit("_", 1)[1]))
    for token, numbers in held.items():
        if sorted(numbers) != list(range(len(numbers))):
            raise Retry(f"{token} ids came out with a gap in them")


def _ledger(layout: Layout) -> None:
    """No two placements share a cell, and none of them leave the frame."""
    taken: set[Cell] = set()
    for item in (*layout.props, *layout.scenery):
        cells = footprint_cells(item)
        if any(not layout.grid.in_bounds(cell) for cell in cells):
            raise Retry("a placement ran off the frame")
        if taken.intersection(cells):
            raise Retry("two placements were given the same cell")
        taken.update(cells)


def _sites(layout: Layout, settlement: Settlement) -> None:
    """Sites keep their margin clear of water, road, and each other, and are painted right.

    The margin itself came from the tuning when the site was reserved, so what is checked here is
    the ground that ended up inside it.
    """
    rectangles = {site.building.id: _rectangle(site) for site in settlement.sites}
    for site in settlement.sites:
        building = site.building
        doors = frozenset(doorway_cells(building))
        for cell in rectangles[building.id]:
            code = layout.grid.value_at(cell)
            wall = _on_edge(cell, rectangles[building.id])
            if code != ("d" if cell in doors else "x" if wall else "i"):
                raise Retry(f"the site painted for {building.id} does not match its catalog rectangle")
        for cell in site.reserved:
            if any(cell in other for name, other in rectangles.items() if name != building.id):
                raise Retry(f"the margin around {building.id} runs into another site")
            if layout.grid.value_at(cell) in {"w", "r", "b"}:
                raise Retry(f"the margin around {building.id} is not clear ground")


def _road(layout: Layout, road: Road, tuning: Generation) -> None:
    """The road spans the frame, bridges each channel once, and leaves the spawn standing."""
    network = tuning.network
    cells = {cell for cell in road.cells if layout.grid.value_at(cell) in {"r", "b"}}
    if not any(x == 0 for x, _ in cells) or not any(x == FRAME.cells_x - 1 for x, _ in cells):
        raise Retry("the road did not reach both frame edges")
    if len(_pieces(cells)) != 1:
        raise Retry("the road came out in more than one piece")
    if len(road.crossings) != 3:
        raise Retry("the road did not bridge all three channels")
    for crossing in road.crossings:
        if crossing.span[1] - crossing.span[0] + 1 > network.road.crossing_run:
            raise Retry("a bridge reached further than the configured crossing run")
        if any(layout.grid.value_at(cell) != "b" for cell in crossing.deck):
            raise Retry("a bridge deck did not come out as bridge ground")
    spawn = layout.grid.cell_at(layout.spawn)
    if spawn is None or layout.grid.value_at(spawn) != "r":
        raise Retry("the spawn is not standing on the road")
    if spawn[0] != network.spawn.edge_inset:
        raise Retry("the spawn is not the configured inset from the west edge")
    if not layout.body_clear(layout.spawn, network.spawn.clearance):
        raise Retry("the spawn does not have its configured clearance")


def _paths(layout: Layout, settlement: Settlement, footpaths: Footpaths, tuning: Generation) -> None:
    """Every doorway opens onto ground that carries people, and a channel is bridged once."""
    for site in settlement.sites:
        if not any(GROUND_BY_CODE[layout.grid.value_at(cell)].passable for cell in site.approaches):
            raise Retry(f"the doorway of {site.building.id} does not open onto walkable ground")
    crossed: dict[int, int] = {}
    for crossing in footpaths.crossings:
        crossed[crossing.channel] = crossed.get(crossing.channel, 0) + 1
        if len(crossing.cells) > tuning.network.path.crossing_run:
            raise Retry("a footpath crossing reached further than the configured run")
    if any(count > 1 for count in crossed.values()):
        raise Retry("a channel came out with more than one footpath crossing")


def _witnesses(layout: Layout, witnesses: tuple[tuple[str, Cell], ...]) -> None:
    """Every interactive prop still has the standing cell that was banked for it."""
    banked = dict(witnesses)
    for item in layout.props:
        cell = banked.get(item.id)
        if cell is None:
            raise Retry(f"{item.id} banked no standing cell")
        centre = layout.grid.center(cell)
        spot = nearest_point(centre, layout.shape_for(item))
        if not layout.body_clear(centre) or distance(centre, spot) > PROFILE.prop_reach:
            raise Retry(f"the standing cell banked for {item.id} no longer reaches it")
        if not layout.line_clear(centre, spot):
            raise Retry(f"the standing cell banked for {item.id} lost sight of it")


def _rectangle(site: Site) -> frozenset[Cell]:
    kind = BUILDING_BY_TOKEN[site.building.type]
    x, y = site.building.cell
    return frozenset(
        (column, row) for row in range(y, y + kind.height) for column in range(x, x + kind.width)
    )


def _on_edge(cell: Cell, rectangle: frozenset[Cell]) -> bool:
    x, y = cell
    return any(spot not in rectangle for spot in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))


def _pieces(cells: set[Cell]) -> list[frozenset[Cell]]:
    remaining = set(cells)
    found: list[frozenset[Cell]] = []
    while remaining:
        start = min(remaining)
        group = {start}
        pending = [start]
        remaining.discard(start)
        while pending:
            x, y = pending.pop()
            for spot in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
                if spot in remaining:
                    remaining.discard(spot)
                    group.add(spot)
                    pending.append(spot)
        found.append(frozenset(group))
    return found
