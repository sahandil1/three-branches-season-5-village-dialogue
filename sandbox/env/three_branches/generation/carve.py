"""Brushes that paint ground cells. Nothing here draws from the stream."""

from __future__ import annotations

from collections.abc import Iterable
from math import hypot

from ..grid import Cell
from ..rules import FRAME


def disc_offsets(radius: float) -> tuple[Cell, ...]:
    """Offsets of a round brush, in a fixed order so painting never depends on set iteration."""
    reach = int(radius) + 1
    return tuple(
        (dx, dy)
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
        if hypot(dx, dy) <= radius
    )


def brush(center: Cell, offsets: Iterable[Cell]) -> tuple[Cell, ...]:
    """Place a brush on a cell and drop whatever falls outside the frame."""
    x, y = center
    return tuple(
        (x + dx, y + dy) for dx, dy in offsets if 0 <= x + dx < FRAME.cells_x and 0 <= y + dy < FRAME.cells_y
    )


def stamp(center: Cell, width: int) -> tuple[Cell, ...]:
    """Cells a square brush of this width covers, centred on a cell.

    An even width has no true centre, so the extra cell goes to the lower index, matching the
    centring rule the rest of the village uses.
    """
    low = -(width // 2)
    span = range(low, low + width)
    x, y = center
    return tuple(
        (x + dx, y + dy)
        for dy in span
        for dx in span
        if 0 <= x + dx < FRAME.cells_x and 0 <= y + dy < FRAME.cells_y
    )
