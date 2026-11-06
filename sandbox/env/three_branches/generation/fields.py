"""Terrain noise, sampled once per cell before anything is painted.

Every field is pure-Python fractal value noise: a lattice of stream draws covering the frame plus
one ring of nodes, read back through smoothstep-faded bilinear interpolation and normalised to the
unit range. Elevation then takes the southward slope bias, so water has somewhere to flow.

Elevation shapes the land and moisture shapes what grows on it.

The fields are generation-only artifacts and never reach the layout.
"""

from __future__ import annotations

import random

from ..rules import FRAME
from .config import Fields, Octave

Field = list[list[float]]


def build_fields(stream: random.Random, tuning: Fields) -> tuple[Field, Field]:
    """Return elevation and moisture as ``[y][x]`` rows in the unit range."""
    elevation = _normalise(_noise(stream, tuning.elevation_octaves))
    moisture = _normalise(_noise(stream, tuning.moisture_octaves))
    bias = tuning.south_bias
    if bias > 0.0:
        last = FRAME.cells_y - 1
        for y, row in enumerate(elevation):
            lift = bias * y / last
            for x, value in enumerate(row):
                row[x] = value * (1.0 - bias) + lift
    return elevation, moisture


def _noise(stream: random.Random, octaves: tuple[Octave, ...]) -> Field:
    total: Field = [[0.0] * FRAME.cells_x for _ in range(FRAME.cells_y)]
    for octave in octaves:
        spacing = octave.spacing
        amplitude = octave.amplitude
        # Enough nodes to cover the frame plus a ring beyond it, so the last cell of a row
        # still has a node on either side to read between.
        lattice = [
            [stream.random() for _ in range(FRAME.cells_x // spacing + 3)]
            for _ in range(FRAME.cells_y // spacing + 3)
        ]
        columns = [(x // spacing, _smoothstep(x % spacing / spacing)) for x in range(FRAME.cells_x)]
        for y in range(FRAME.cells_y):
            south = lattice[y // spacing]
            north = lattice[y // spacing + 1]
            fade_y = _smoothstep(y % spacing / spacing)
            row = total[y]
            for x, (index, fade_x) in enumerate(columns):
                low = south[index] + (south[index + 1] - south[index]) * fade_x
                high = north[index] + (north[index + 1] - north[index]) * fade_x
                row[x] += amplitude * (low + (high - low) * fade_y)
    return total


def _normalise(field: Field) -> Field:
    lowest = min(min(row) for row in field)
    highest = max(max(row) for row in field)
    if highest <= lowest:
        return [[0.5] * FRAME.cells_x for _ in range(FRAME.cells_y)]
    span = highest - lowest
    return [[(value - lowest) / span for value in row] for row in field]


def _smoothstep(fraction: float) -> float:
    return fraction * fraction * (3.0 - 2.0 * fraction)
