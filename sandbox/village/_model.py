"""Private, stdlib-only static village calculations for the public helpers."""

# pyright: reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from math import floor, hypot
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

_ROOT = Path(__file__).parent
RULES: dict[str, Any] = json.loads((_ROOT / "rules.json").read_text(encoding="utf-8"))
CATALOG: dict[str, Any] = json.loads((_ROOT / "catalog.json").read_text(encoding="utf-8"))
GROUND_BY_CODE = {item["code"]: item for item in RULES["grounds"]}
DOORWAY_CODE = next(item["code"] for item in RULES["grounds"] if item["name"] == "doorway")
PROP_BY_TYPE = {item["token"]: item for item in CATALOG["props"]}
SCENERY_BY_TYPE = {item["token"]: item for item in CATALOG["scenery"]}
BUILDING_BY_TYPE = {item["token"]: item for item in CATALOG["buildings"]}

# geometry.BODY_RADIUS cannot be imported here: geometry imports RULES from this module, so the
# reverse import would cycle. It is the same static profile constant either way.
_BODY_RADIUS = float(RULES["profile"]["body_radius"])


@dataclass(frozen=True, slots=True)
class Shape:
    kind: str
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def radius(self) -> float:
        return min(self.width, self.height) / 2


@dataclass(frozen=True, slots=True)
class Model:
    cells_x: int
    cells_y: int
    cell_size: float
    ground: tuple[str, ...]
    doorways: tuple[tuple[tuple[int, int], ...], ...]
    blocked: tuple[Shape, ...]
    prop_shapes: tuple[Shape, ...]
    scenery_shapes: tuple[Shape, ...]
    collision_shapes: tuple[Shape, ...]
    collision_buckets: Mapping[tuple[int, int], tuple[int, ...]]
    walkable_cells: frozenset[tuple[int, int]]


PlacementFingerprint = tuple[str, float, float, bool, float]
VillageFingerprint = tuple[
    int,
    int,
    float,
    tuple[str, ...],
    tuple[PlacementFingerprint, ...],
    tuple[PlacementFingerprint, ...],
]


@dataclass(frozen=True, slots=True)
class _IdentityEntry:
    village: Mapping[str, object]
    size: Mapping[str, object]
    ground: str | tuple[str, ...]
    props: tuple[Mapping[str, object], ...]
    scenery: tuple[Mapping[str, object], ...]
    model: Model


_IDENTITY_CACHE_LIMIT = 8
_IDENTITY_CACHE: dict[int, _IdentityEntry] = {}


def model(observation: Mapping[str, object]) -> Model:
    """Return the cached immutable calculation model for one observation's village.

    A same-observation query reuses the stored model by object identity. The village
    sub-mappings are therefore expected to be replaced wholesale, never mutated in place;
    in-place leaf mutation followed by a same-object re-query would return a stale model.
    """
    village = observation["village"]
    assert isinstance(village, Mapping)
    entry = _IDENTITY_CACHE.get(id(village))
    if (
        entry is not None
        and entry.village is village
        and entry.size is village["size"]
        and entry.ground is village["ground"]
        and entry.props is village["props"]
        and entry.scenery is village["scenery"]
    ):
        return entry.model
    size, ground, props, scenery = village["size"], village["ground"], village["props"], village["scenery"]
    assert isinstance(size, Mapping)
    result = _model(
        _fingerprint(
            size,
            cast("str | tuple[str, ...]", ground),
            cast("tuple[Mapping[str, object], ...]", props),
            cast("tuple[Mapping[str, object], ...]", scenery),
        )
    )
    _IDENTITY_CACHE[id(village)] = _IdentityEntry(village, size, ground, props, scenery, result)
    if len(_IDENTITY_CACHE) > _IDENTITY_CACHE_LIMIT:
        _IDENTITY_CACHE.pop(next(iter(_IDENTITY_CACHE)), None)
    return result


def _fingerprint(
    size: Mapping[str, object],
    ground: str | tuple[str, ...],
    props: tuple[Mapping[str, object], ...],
    scenery: tuple[Mapping[str, object], ...],
) -> VillageFingerprint:
    return (
        int(size["cells_x"]),
        int(size["cells_y"]),
        float(size["cell_size"]),
        ground if isinstance(ground, tuple) else tuple(str(row) for row in ground),
        tuple(_placement_fingerprint(item) for item in props),
        tuple(_placement_fingerprint(item) for item in scenery),
    )


def _placement_fingerprint(item: object) -> PlacementFingerprint:
    assert isinstance(item, Mapping)
    cell = item["cell"]
    assert isinstance(cell, Mapping)
    return (
        str(item["type"]),
        float(cell["x"]),
        float(cell["y"]),
        item.get("facing") in {"east", "west"},
        float(item.get("scale", 1.0)),
    )


@lru_cache(maxsize=32)
def _model(fingerprint: VillageFingerprint) -> Model:
    cells_x, cells_y, cell_size, ground, props, scenery = fingerprint
    blocked_cells = {
        (x, y)
        for y, row in enumerate(ground)
        for x, code in enumerate(row)
        if not GROUND_BY_CODE[code]["passable"]
    }
    doorway_cells = {
        (x, y) for y, row in enumerate(ground) for x, code in enumerate(row) if code == DOORWAY_CODE
    }
    blocked = _rectangles(blocked_cells, cell_size)
    prop_shapes = tuple(_shape(item, PROP_BY_TYPE[item[0]], cell_size) for item in props)
    scenery_shapes = tuple(_shape(item, SCENERY_BY_TYPE[item[0]], cell_size) for item in scenery)
    collision_shapes = (*blocked, *prop_shapes, *scenery_shapes)
    partial = Model(
        cells_x,
        cells_y,
        cell_size,
        ground,
        _components(doorway_cells),
        blocked,
        prop_shapes,
        scenery_shapes,
        collision_shapes,
        _collision_buckets(collision_shapes, cells_x, cells_y, cell_size),
        frozenset(),
    )
    return replace(partial, walkable_cells=_walkable_cells(partial))


def _walkable_cells(partial: Model) -> frozenset[tuple[int, int]]:
    """Return every cell whose ground is passable and whose centre a body clears, matching the
    semantics of the public ``layout.walkable`` helper."""
    return frozenset(
        (x, y)
        for y in range(partial.cells_y)
        for x in range(partial.cells_x)
        if GROUND_BY_CODE[partial.ground[y][x]]["passable"]
        and body_clear(partial, ((x + 0.5) * partial.cell_size, (y + 0.5) * partial.cell_size), _BODY_RADIUS)
    )


def _components(cells: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Group cells into deterministic orthogonally connected components."""
    remaining = set(cells)
    components = []
    while remaining:
        pending = [min(remaining, key=lambda value: (value[1], value[0]))]
        remaining.remove(pending[0])
        found = []
        while pending:
            current = pending.pop()
            found.append(current)
            x, y = current
            for neighbor in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        components.append(tuple(found))
    return tuple(components)


def _rectangles(cells: set[tuple[int, int]], cell_size: float) -> tuple[Shape, ...]:
    """Coalesce blocked cells into deterministic non-overlapping rectangles."""
    remaining = set(cells)
    rectangles: list[Shape] = []
    while remaining:
        x, y = min(remaining, key=lambda cell: (cell[1], cell[0]))
        width = 1
        while (x + width, y) in remaining:
            width += 1
        height = 1
        while all((column, y + height) in remaining for column in range(x, x + width)):
            height += 1
        for row in range(y, y + height):
            for column in range(x, x + width):
                remaining.remove((column, row))
        rectangles.append(Shape("box", x * cell_size, y * cell_size, width * cell_size, height * cell_size))
    return tuple(rectangles)


def _shape(item: PlacementFingerprint, kind: Mapping[str, object], cell_size: float) -> Shape:
    width, height = float(kind["width"]), float(kind["height"])
    if item[3]:
        width, height = height, width
    x, y = item[1] * cell_size, item[2] * cell_size
    if kind["shape"] != "circle":
        return Shape("box", x, y, width, height)
    diameter = min(width, height) * float(kind.get("collision_scale", 1.0)) * item[4]
    return Shape("circle", x + (width - diameter) / 2, y + (height - diameter) / 2, diameter, diameter)


def _collision_buckets(
    shapes: tuple[Shape, ...], cells_x: int, cells_y: int, cell_size: float
) -> Mapping[tuple[int, int], tuple[int, ...]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, shape in enumerate(shapes):
        low_x, low_y, high_x, high_y = _shape_bounds(shape)
        for y in _bucket_range(low_y, high_y, cell_size, cells_y):
            for x in _bucket_range(low_x, high_x, cell_size, cells_x):
                buckets.setdefault((x, y), []).append(index)
    return MappingProxyType({cell: tuple(indices) for cell, indices in buckets.items()})


def _shape_bounds(shape: Shape) -> tuple[float, float, float, float]:
    if shape.kind == "circle":
        x, y = shape.center
        return x - shape.radius, y - shape.radius, x + shape.radius, y + shape.radius
    return shape.x, shape.y, shape.x + shape.width, shape.y + shape.height


def _bucket_range(low: float, high: float, cell_size: float, limit: int) -> range:
    first = max(0, floor(low / cell_size))
    last = min(limit - 1, floor(high / cell_size))
    return range(first, last + 1) if first <= last else range(0)


def _collision_candidates(
    model: Model, low_x: float, low_y: float, high_x: float, high_y: float
) -> tuple[Shape, ...]:
    indices: set[int] = set()
    for y in _bucket_range(low_y, high_y, model.cell_size, model.cells_y):
        for x in _bucket_range(low_x, high_x, model.cell_size, model.cells_x):
            indices.update(model.collision_buckets.get((x, y), ()))
    return tuple(model.collision_shapes[index] for index in indices)


def cell(model: Model, point: Mapping[str, object]) -> tuple[int, int] | None:
    x, y = float(point["x"]), float(point["y"])
    if not (0 <= x < model.cells_x * model.cell_size and 0 <= y < model.cells_y * model.cell_size):
        return None
    return int(x // model.cell_size), int(y // model.cell_size)


def center(model: Model, cell: Mapping[str, object] | tuple[int, int]) -> tuple[float, float] | None:
    x, y = (int(cell["x"]), int(cell["y"])) if isinstance(cell, Mapping) else cell
    if not (0 <= x < model.cells_x and 0 <= y < model.cells_y):
        return None
    return (x + 0.5) * model.cell_size, (y + 0.5) * model.cell_size


def ground(model: Model, cell: Mapping[str, object] | tuple[int, int]) -> Mapping[str, object] | None:
    point = center(model, cell)
    if point is None:
        return None
    x, y = (int(cell["x"]), int(cell["y"])) if isinstance(cell, Mapping) else cell
    return GROUND_BY_CODE[model.ground[y][x]]


def nearest(point: tuple[float, float], shape: Shape) -> tuple[float, float]:
    if shape.kind == "box":
        return min(max(point[0], shape.x), shape.x + shape.width), min(
            max(point[1], shape.y), shape.y + shape.height
        )
    x, y = shape.center
    dx, dy = point[0] - x, point[1] - y
    length = hypot(dx, dy)
    if length == 0:
        return x + shape.radius, y
    return x + shape.radius * dx / length, y + shape.radius * dy / length


def distance_to_shape(point: tuple[float, float], shape: Shape) -> float:
    nearest_point = nearest(point, shape)
    return hypot(point[0] - nearest_point[0], point[1] - nearest_point[1])


def body_clear(model: Model, point: tuple[float, float], radius: float) -> bool:
    width, height = model.cells_x * model.cell_size, model.cells_y * model.cell_size
    if not (radius <= point[0] <= width - radius and radius <= point[1] <= height - radius):
        return False
    return not any(
        _circle_hits_shape(point, radius, shape)
        for shape in _collision_candidates(
            model, point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius
        )
    )


def _circle_hits_shape(point: tuple[float, float], radius: float, shape: Shape) -> bool:
    if shape.kind == "circle":
        return hypot(point[0] - shape.center[0], point[1] - shape.center[1]) < radius + shape.radius
    return distance_to_shape(point, shape) < radius


def segment_clear(model: Model, start: tuple[float, float], end: tuple[float, float], radius: float) -> bool:
    """Check a center path against the same static body geometry as ``walkable``."""
    if not (body_clear(model, start, radius) and body_clear(model, end, radius)):
        return False
    return not any(
        _segment_hits_shape(start, end, radius, shape)
        for shape in _collision_candidates(
            model,
            min(start[0], end[0]) - radius,
            min(start[1], end[1]) - radius,
            max(start[0], end[0]) + radius,
            max(start[1], end[1]) + radius,
        )
    )


def _segment_hits_shape(
    start: tuple[float, float], end: tuple[float, float], radius: float, shape: Shape
) -> bool:
    if shape.kind == "circle":
        return _point_segment_distance(shape.center, start, end) < radius + shape.radius
    if _segment_intersects_box(start, end, shape):
        return True
    corners = (
        (shape.x, shape.y),
        (shape.x + shape.width, shape.y),
        (shape.x, shape.y + shape.height),
        (shape.x + shape.width, shape.y + shape.height),
    )
    return (
        min(
            distance_to_shape(start, shape),
            distance_to_shape(end, shape),
            *(_point_segment_distance(corner, start, end) for corner in corners),
        )
        < radius
    )


def _segment_intersects_box(start: tuple[float, float], end: tuple[float, float], shape: Shape) -> bool:
    """Return whether the center line crosses a box before radius expansion."""
    first, last = 0.0, 1.0
    for origin, delta, low, high in (
        (start[0], end[0] - start[0], shape.x, shape.x + shape.width),
        (start[1], end[1] - start[1], shape.y, shape.y + shape.height),
    ):
        if delta == 0:
            if not low <= origin <= high:
                return False
            continue
        left, right = sorted(((low - origin) / delta, (high - origin) / delta))
        first, last = max(first, left), min(last, right)
        if first > last:
            return False
    return True


def _point_segment_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    progress = min(1.0, max(0.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    return hypot(point[0] - start[0] - progress * dx, point[1] - start[1] - progress * dy)


def line_clear(model: Model, start: Mapping[str, object], end: Mapping[str, object]) -> bool:
    first, last = cell(model, start), cell(model, end)
    if first is None or last is None:
        return False
    return all(
        not GROUND_BY_CODE[model.ground[y][x]]["blocks_sight"] for x, y in supercover(model, start, end)
    )


def supercover(
    model: Model, start: Mapping[str, object], end: Mapping[str, object]
) -> tuple[tuple[int, int], ...]:
    """Return all touched grid cells, including both side cells at a corner crossing."""
    first, last = cell(model, start), cell(model, end)
    if first is None or last is None:
        return ()
    if first == last:
        return (first,)
    x0, y0 = float(start["x"]) / model.cell_size, float(start["y"]) / model.cell_size
    x1, y1 = float(end["x"]) / model.cell_size, float(end["y"]) / model.cell_size
    cell_x, cell_y = first
    target_x, target_y = last
    dx, dy = x1 - x0, y1 - y0
    step_x, step_y = (1 if dx > 0 else -1 if dx < 0 else 0), (1 if dy > 0 else -1 if dy < 0 else 0)
    delta_x, delta_y = (abs(1 / dx) if dx else float("inf")), (abs(1 / dy) if dy else float("inf"))
    max_x = ((cell_x + (step_x > 0)) - x0) / dx if dx else float("inf")
    max_y = ((cell_y + (step_y > 0)) - y0) / dy if dy else float("inf")
    cells = [(cell_x, cell_y)]
    limit = 2 * (abs(target_x - cell_x) + abs(target_y - cell_y) + 1)
    for _ in range(limit):
        if (cell_x, cell_y) == (target_x, target_y):
            return tuple(cells)
        if min(max_x, max_y) >= 1 - 1e-12:
            if last not in cells:
                cells.append(last)
            return tuple(cells)
        if abs(max_x - max_y) < 1e-12:
            for candidate in ((cell_x + step_x, cell_y), (cell_x, cell_y + step_y)):
                if (
                    0 <= candidate[0] < model.cells_x
                    and 0 <= candidate[1] < model.cells_y
                    and candidate not in cells
                ):
                    cells.append(candidate)
            if last in cells:
                return tuple(cells)
            cell_x, cell_y = cell_x + step_x, cell_y + step_y
            max_x, max_y = max_x + delta_x, max_y + delta_y
        elif max_x < max_y:
            cell_x, max_x = cell_x + step_x, max_x + delta_x
        else:
            cell_y, max_y = cell_y + step_y, max_y + delta_y
        if (cell_x, cell_y) not in cells:
            cells.append((cell_x, cell_y))
    raise ValueError("supercover did not reach its target")
