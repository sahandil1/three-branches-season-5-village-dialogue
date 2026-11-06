"""The deliberately plain mechanics fixture used until seeded generation lands."""

from __future__ import annotations

from .grid import Grid
from .layout import Building, Layout, PlacedProp, Scenery, paint_site
from .rules import FRAME


def _path(rows: list[list[str]], start: tuple[int, int], end: tuple[int, int]) -> None:
    """Paint an L-shaped footpath, keeping already-open doors intact."""
    x, y = start
    target_x, target_y = end
    while x != target_x:
        if rows[y][x] not in {"d", "b"}:
            rows[y][x] = "p"
        x += 1 if target_x > x else -1
    while y != target_y:
        if rows[y][x] not in {"d", "b"}:
            rows[y][x] = "p"
        y += 1 if target_y > y else -1
    if rows[y][x] not in {"d", "b"}:
        rows[y][x] = "p"


def build_fixture() -> Layout:
    """Return a new complete village fixture with every static catalog type represented."""
    rows = [["g"] * FRAME.cells_x for _ in range(FRAME.cells_y)]
    # One north trunk becomes the three named south channels. The road is deliberately
    # below the fork, so it crosses each channel once and never the trunk.
    for y in range(66, FRAME.cells_y):
        rows[y][50] = "w"
    for channel in (25, 50, 75):
        for y in range(66):
            rows[y][channel] = "w"
            if channel > 0:
                rows[y][channel - 1] = "e"
            if channel < FRAME.cells_x - 1:
                rows[y][channel + 1] = "e"
    for x in range(FRAME.cells_x):
        for y in range(49, 52):
            rows[y][x] = "r"
    for channel in (25, 50, 75):
        for y in range(49, 52):
            rows[y][channel] = "b"

    buildings = (
        Building("home_0", "home", (5, 62), "south"),
        Building("home_1", "home", (5, 78), "south"),
        Building("home_2", "home", (63, 62), "south"),
        Building("home_3", "home", (84, 70), "south"),
        Building("home_4", "home", (7, 35), "north"),
        Building("inn", "inn", (62, 20), "north"),
        Building("shed", "shed", (8, 15), "north"),
    )
    for building in buildings:
        paint_site(rows, building)
    # Footpaths terminate on a doorway or its approach, and originate from the road or a path
    # already painted. Each leg keeps clear of the building rectangles, which paint first.
    for start, end in (
        ((12, 51), (8, 61)),
        ((8, 61), (4, 77)),
        ((4, 77), (8, 77)),
        ((68, 51), (66, 61)),
        ((88, 51), (87, 69)),
        ((10, 51), (10, 42)),
        ((68, 51), (67, 30)),
        ((10, 51), (5, 30)),
        ((5, 30), (11, 22)),
        ((43, 51), (45, 60)),
    ):
        _path(rows, start, end)

    props = (
        PlacedProp("stall_0", "stall", (32, 45), "north"),
        PlacedProp("stall_1", "stall", (36, 53), "south"),
        PlacedProp("stall_2", "stall", (40, 45), "north"),
        PlacedProp("stall_3", "stall", (44, 53), "south"),
        PlacedProp("stall_4", "stall", (55, 45), "north"),
        PlacedProp("lantern_0", "lantern", (3, 53)),
        PlacedProp("lantern_1", "lantern", (20, 46)),
        PlacedProp("lantern_2", "lantern", (34, 53)),
        PlacedProp("lantern_3", "lantern", (48, 46)),
        PlacedProp("lantern_4", "lantern", (60, 53)),
        PlacedProp("lantern_5", "lantern", (80, 46)),
        PlacedProp("lantern_6", "lantern", (96, 53)),
        PlacedProp("bench_0", "bench", (43, 62)),
        PlacedProp("bench_1", "bench", (38, 42)),
        PlacedProp("bench_2", "bench", (76, 42)),
        PlacedProp("bench_3", "bench", (16, 55)),
        PlacedProp("bench_4", "bench", (82, 55)),
        PlacedProp("shrine_0", "shrine", (20, 57)),
        PlacedProp("shrine_1", "shrine", (80, 57)),
        PlacedProp("board_0", "board", (46, 44)),
        PlacedProp("plot_0", "plot", (7, 69)),
        PlacedProp("plot_1", "plot", (7, 85)),
        PlacedProp("plot_2", "plot", (65, 69)),
        PlacedProp("plot_3", "plot", (86, 77)),
        PlacedProp("plot_4", "plot", (9, 33)),
        PlacedProp("hearth_0", "hearth", (70, 21)),
        PlacedProp("repair_bench_0", "repair_bench", (10, 16)),
        PlacedProp("pump_0", "pump", (45, 60)),
        PlacedProp("bell", "bell", (4, 56)),
    )
    scenery = (
        Scenery("pine", (2, 60)),
        Scenery("pine", (18, 65)),
        Scenery("pine", (30, 60)),
        Scenery("pine", (58, 60)),
        Scenery("pine", (92, 60)),
        Scenery("crate", (30, 45)),
        Scenery("crate", (38, 45)),
        Scenery("crate", (43, 45)),
        Scenery("crate", (53, 45)),
    )
    return Layout(Grid(FRAME, rows), buildings, props, scenery, (4.5, 50.5))
