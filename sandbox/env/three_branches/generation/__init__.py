"""Seeded village generation for Days at Three Branches.

``build_village`` is the seam the environment resets through. ``generate`` is the same work plus an
internal report that only the guarantee suite and the batch summary read. Both run one labelled
stream, ``random.Random(f"{seed}:village")``, which is separate from anything a match plays with, so
the same build and the same seed always produce the same village.

Construction runs terrain fields, water, ground classes, the district anchors and their building
sites, the road with its bridges and the spawn, the footpaths, and then the dressing. Each stage
reads what the ones before it committed, and checks its own work as it goes. A mandatory stage that
runs out of candidates raises ``Retry``, which discards the partial village and draws the whole
layout again on the same stream.

Assembly is the last word. It checks the finished village with every placement still in place;
the connectivity of the published layout is established by the guarantee suite over the pinned
batch rather than demanded here, so a village is never stripped of its optional dressing to force
the flood to pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter

from ..grid import Cell, Grid
from ..layout import Layout
from ..rules import FRAME, RULES
from . import accessories, fields, grounds, network, paths, sites, validate, water
from .config import GENERATION, Generation, Retry

__all__ = ["Report", "build_village", "generate"]


@dataclass(frozen=True, slots=True)
class Report:
    """Generator internals. Only the guarantee suite and the batch summary read these."""

    seed: int
    redraws: int
    water: water.Water
    # The standing cell banked for each interactive prop, by prop id.
    witnesses: tuple[tuple[str, Cell], ...]
    # Generation and validation together, which is what a reset spends. Reported, never asserted.
    reset_seconds: float


def build_village(seed: int | None = None) -> Layout:
    """Build the village for a match seed. A reset without a seed builds the default village."""
    return generate(0 if seed is None else seed)[0]


def generate(seed: int, tuning: Generation = GENERATION) -> tuple[Layout, Report]:
    """Build the village for a seed and report how it went."""
    stream = random.Random(f"{seed}:village")
    started = perf_counter()
    for redraws in range(tuning.redraw_cap):
        try:
            layout, courses, witnesses = _draw(stream, tuning)
        except Retry:
            continue
        return layout, Report(seed, redraws, courses, witnesses, perf_counter() - started)
    raise RuntimeError(f"village generation for seed {seed} exceeded the redraw cap of {tuning.redraw_cap}")


def _draw(
    stream: random.Random, tuning: Generation
) -> tuple[Layout, water.Water, tuple[tuple[str, Cell], ...]]:
    """Draw one whole village. Any mandatory stage that runs out of room raises ``Retry``."""
    rows = [[RULES.fill] * FRAME.cells_x for _ in range(FRAME.cells_y)]
    elevation, moisture = fields.build_fields(stream, tuning.fields)
    courses = water.carve_water(stream, rows, elevation, tuning.water)
    grounds.paint_grounds(rows, elevation, moisture, grounds.water_distance(rows), tuning.grounds)
    settlement = sites.settle(stream, rows, elevation, moisture, courses, tuning.sites, tuning.network)
    road = network.lay_road(stream, rows, moisture, courses, settlement, tuning.network)
    shrines = accessories.shrine_spots(road, tuning.accessories)
    footpaths = paths.lay_footpaths(
        stream, rows, courses, _targets(settlement, shrines), settlement.keep_clear, tuning.network.path
    )
    dressing = accessories.dress(
        stream,
        rows,
        settlement,
        road,
        shrines,
        tuning.accessories,
        (road.spawn, tuning.network.spawn.clearance),
    )
    return _assemble(rows, courses, settlement, road, footpaths, dressing, tuning)


def _targets(
    settlement: sites.Settlement, shrines: tuple[accessories.Station, ...]
) -> tuple[tuple[Cell, ...], ...]:
    """Everywhere a footpath has to reach, in the order the routes are committed."""
    return (
        (settlement.plaza,),
        *(site.approaches for site in settlement.sites),
        *(((int(station.point[0]), int(station.point[1])),) for station in shrines),
    )


def _assemble(
    rows: list[list[str]],
    courses: water.Water,
    settlement: sites.Settlement,
    road: network.Road,
    footpaths: paths.Footpaths,
    dressing: accessories.Dressing,
    tuning: Generation,
) -> tuple[Layout, water.Water, tuple[tuple[str, Cell], ...]]:
    """Assemble the village as dressed and check it before handing it over."""
    grid = Grid(FRAME, rows)
    layout = Layout(grid, settlement.buildings, dressing.props, dressing.scenery, road.spawn)
    validate.check(layout, settlement, road, footpaths, dressing.witnesses, tuning)
    return layout, courses, dressing.witnesses
