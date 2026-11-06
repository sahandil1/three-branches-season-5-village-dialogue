"""Reed flats and terraced fields, painted over the open ground the water left behind.

Reeds collect on wet banks and at every channel mouth. Fields terrace where the land is low, dry, and
flat, which the southward elevation bias puts toward the south edge. A few passes of majority
smoothing then clear the single-cell speckle that per-cell thresholds always produce. Nothing here
draws from the stream, so the ground classes follow from the fields and the water alone.
"""

from __future__ import annotations

from collections import deque

from ..rules import FRAME
from .config import Grounds

_CLASSES = ("g", "e", "f")


def water_distance(rows: list[list[str]]) -> list[list[int]]:
    """Cell distance to the nearest water, counted in four-way steps."""
    far = FRAME.cells_x + FRAME.cells_y
    distance = [[far] * FRAME.cells_x for _ in range(FRAME.cells_y)]
    pending: deque[tuple[int, int]] = deque()
    for y, row in enumerate(rows):
        for x, code in enumerate(row):
            if code == "w":
                distance[y][x] = 0
                pending.append((x, y))
    while pending:
        x, y = pending.popleft()
        step = distance[y][x] + 1
        for nx, ny in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
            if 0 <= nx < FRAME.cells_x and 0 <= ny < FRAME.cells_y and distance[ny][nx] > step:
                distance[ny][nx] = step
                pending.append((nx, ny))
    return distance


def paint_grounds(
    rows: list[list[str]],
    elevation: list[list[float]],
    moisture: list[list[float]],
    distance: list[list[int]],
    tuning: Grounds,
) -> None:
    """Paint reeds and fields in place, leaving water alone."""
    for y in range(FRAME.cells_y):
        row = rows[y]
        for x in range(FRAME.cells_x):
            if row[x] == "w":
                continue
            near_water = distance[y][x] <= tuning.reed_distance
            if near_water and (moisture[y][x] >= tuning.reed_moisture or y < tuning.mouth_reed_depth):
                row[x] = "e"
            elif (
                not near_water
                and elevation[y][x] <= tuning.field_elevation
                and moisture[y][x] <= tuning.field_moisture
                and _slope(elevation, x, y) <= tuning.field_slope
            ):
                row[x] = "f"
    for _ in range(tuning.smoothing_passes):
        _smooth(rows)


def _slope(elevation: list[list[float]], x: int, y: int) -> float:
    left = elevation[y][max(x - 1, 0)]
    right = elevation[y][min(x + 1, FRAME.cells_x - 1)]
    below = elevation[max(y - 1, 0)][x]
    above = elevation[min(y + 1, FRAME.cells_y - 1)][x]
    return max(abs(right - left), abs(above - below)) / 2.0


def _smooth(rows: list[list[str]]) -> None:
    """Give every land cell the majority class of its neighbourhood, keeping ties as they are."""
    before = [row[:] for row in rows]
    for y in range(FRAME.cells_y):
        lines = before[max(y - 1, 0) : y + 2]
        for x in range(FRAME.cells_x):
            current = before[y][x]
            if current not in _CLASSES:
                continue
            near = [code for line in lines for code in line[max(x - 1, 0) : x + 2]]
            counts = tuple(near.count(code) for code in _CLASSES)
            best = max(counts)
            if counts[_CLASSES.index(current)] < best:
                rows[y][x] = _CLASSES[counts.index(best)]
