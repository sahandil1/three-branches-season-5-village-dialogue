"""Read static village ground and test routes against static collision geometry."""

# pyright: reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

from collections.abc import Mapping
from math import hypot

from ._model import (
    BUILDING_BY_TYPE,
    GROUND_BY_CODE,
    Model,
    cell,
    center,
    ground,
    line_clear,
    model,
    segment_clear,
)
from .geometry import BODY_RADIUS

SPEED_LIMITS = {item["name"]: float(item["speed"]) for item in GROUND_BY_CODE.values()}
BUILDING_SIZES = {
    token: (int(item["width"]), int(item["height"])) for token, item in BUILDING_BY_TYPE.items()
}


def frame(observation: Mapping[str, object]):
    """Return the village grid dimensions and cell scale, as the ``cells_x``, ``cells_y``, and
    ``cell_size`` mapping from ``observation["village"]["size"]``. ``cell_size`` arrives as a NumPy
    ``float32`` scalar."""
    return observation["village"]["size"]


def cell_at(observation: Mapping[str, object], position: Mapping[str, object]):
    """Return the zero-based grid cell a position falls in, as an ``{"x": int, "y": int}``
    mapping, or ``None`` when the position is outside the village."""
    found = cell(model(observation), position)
    return None if found is None else {"x": found[0], "y": found[1]}


def cell_center(observation: Mapping[str, object], cell_value: Mapping[str, object]) -> dict[str, float]:
    """Return the position at the middle of a cell, such as a prop's or building's ``cell``."""
    size = model(observation).cell_size
    return {"x": (int(cell_value["x"]) + 0.5) * size, "y": (int(cell_value["y"]) + 0.5) * size}


def nearest_walkable(observation: Mapping[str, object], position: Mapping[str, object]):
    """Return the walkable cell whose center is closest to a position, as an ``{"x": int, "y": int}``
    mapping, or ``None`` when no cell is walkable. Useful as the destination for a prop or a
    building center, which a body cannot stand on."""
    village_model = model(observation)
    size = village_model.cell_size
    px, py = float(position["x"]), float(position["y"])
    x = min(max(int(px // size), 0), village_model.cells_x - 1)
    y = min(max(int(py // size), 0), village_model.cells_y - 1)
    best, best_distance = None, float("inf")
    # Search square rings outward. No cell in ring r can be closer than r - 1 cells, so the search
    # stops once that bound passes the best distance found.
    for r in range(max(village_model.cells_x, village_model.cells_y)):
        if (r - 1) * size > best_distance:
            break
        for cx in range(x - r, x + r + 1):
            step = 1 if abs(cx - x) == r else 2 * r or 1
            for cy in range(y - r, y + r + 1, step):
                if (cx, cy) not in village_model.walkable_cells:
                    continue
                distance = hypot((cx + 0.5) * size - px, (cy + 0.5) * size - py)
                if distance < best_distance:
                    best, best_distance = (cx, cy), distance
    return None if best is None else {"x": best[0], "y": best[1]}


def ground_at(observation: Mapping[str, object], cell_value: Mapping[str, object]) -> str | None:
    """Return the ground name under one cell, such as ``"ground"``, ``"road"``, ``"water"``, or
    ``"interior"``. ``None`` for a cell outside the village."""
    item = ground(model(observation), cell_value)
    return None if item is None else str(item["name"])


def walkable(observation: Mapping[str, object], cell_value: Mapping[str, object]) -> bool:
    """Return whether a character can stand on a cell: its ground is passable and a body the size
    of a villager clears it (no wall, water, or blocking prop)."""
    return _walkable(model(observation), cell_value)


def _walkable(village_model: Model, cell_value: Mapping[str, object]) -> bool:
    return (int(cell_value["x"]), int(cell_value["y"])) in village_model.walkable_cells


def can_step(
    observation: Mapping[str, object], start_cell: Mapping[str, object], end_cell: Mapping[str, object]
) -> bool:
    """Return whether a character can legally move from one cardinally adjacent cell to the next:
    both cells are walkable and the body clears the path between their centres."""
    village_model = model(observation)
    start, end = center(village_model, start_cell), center(village_model, end_cell)
    if start is None or end is None:
        return False
    start_x, start_y = int(start_cell["x"]), int(start_cell["y"])
    end_x, end_y = int(end_cell["x"]), int(end_cell["y"])
    if abs(start_x - end_x) + abs(start_y - end_y) != 1:
        return False
    return (
        _walkable(village_model, start_cell)
        and _walkable(village_model, end_cell)
        and segment_clear(village_model, start, end, BODY_RADIUS)
    )


def line_of_sight(
    observation: Mapping[str, object], start_pos: Mapping[str, object], end_pos: Mapping[str, object]
) -> bool:
    """Return whether the straight line between two positions is clear, meaning no sight-blocking
    ground such as a wall lies across it. Props are not tested and doorways do not block, so this
    is "could the two points see each other, ignoring the vision cone". A position outside the
    village is never clear."""
    return line_clear(model(observation), start_pos, end_pos)


def buildings(observation: Mapping[str, object]):
    """Return every building placement in the village, each with an ``id``, ``type``, and
    ``cell``."""
    return observation["village"]["buildings"]


def building(observation: Mapping[str, object], building_id: str):
    """Return the building placement with the given id, or ``None`` when there is no such
    building."""
    return next((item for item in buildings(observation) if item["id"] == building_id), None)


def building_center(observation: Mapping[str, object], building_id: str) -> dict[str, float] | None:
    """Return the center of a building's footprint, or ``None`` when there is no such building."""
    item = building(observation, building_id)
    if item is None:
        return None
    width, height = BUILDING_SIZES[str(item["type"])]
    size = model(observation).cell_size
    return {
        "x": (float(item["cell"]["x"]) + width / 2) * size,
        "y": (float(item["cell"]["y"]) + height / 2) * size,
    }


def building_at(observation: Mapping[str, object], position: Mapping[str, object]) -> str | None:
    """Return the id of the building whose floor holds a position, or ``None`` outdoors. A building's
    walls take up the outermost ring of its footprint, so a position in a wall or doorway is not
    on the floor."""
    found = cell_at(observation, position)
    if found is None:
        return None
    for item in buildings(observation):
        width, height = BUILDING_SIZES[str(item["type"])]
        x, y = int(item["cell"]["x"]), int(item["cell"]["y"])
        if x < found["x"] < x + width - 1 and y < found["y"] < y + height - 1:
            return str(item["id"])
    return None


def doorway(observation: Mapping[str, object], building_id: str) -> dict[str, float] | None:
    """Return the center of the doorway run nearest one building's anchor."""
    item = building(observation, building_id)
    if item is None:
        return None
    village_model = model(observation)
    if not village_model.doorways:
        return None

    anchor = (
        (float(item["cell"]["x"]) + 0.5) * village_model.cell_size,
        (float(item["cell"]["y"]) + 0.5) * village_model.cell_size,
    )

    def run_key(run: tuple[tuple[int, int], ...]):
        point = _run_center(run, village_model.cell_size)
        earliest = min(run, key=lambda value: (value[1], value[0]))
        return (point[0] - anchor[0]) ** 2 + (point[1] - anchor[1]) ** 2, earliest[1], earliest[0]

    chosen = min(village_model.doorways, key=run_key)
    x, y = _run_center(chosen, village_model.cell_size)
    return {"x": x, "y": y}


def _run_center(run: tuple[tuple[int, int], ...], cell_size: float) -> tuple[float, float]:
    return (
        sum((x + 0.5) * cell_size for x, _ in run) / len(run),
        sum((y + 0.5) * cell_size for _, y in run) / len(run),
    )


def spawn(observation: Mapping[str, object]):
    """Return the village spawn position as an ``{"x": float, "y": float}`` mapping, in metres
    from the village southwest corner. Both values arrive as NumPy ``float32`` scalars."""
    return observation["village"]["spawn"]
