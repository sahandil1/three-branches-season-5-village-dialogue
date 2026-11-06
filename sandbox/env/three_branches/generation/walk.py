"""One walker, shared by everything that draws a line across the map.

A water course, the road, and a footpath are the same idea carrying different weights. A brush sits
on a point, blends what it wants into a single heading, steps along it, and paints. What tells the
three apart is the urges they blend and the ground they refuse, not how they move, so how they move
lives here.

Nothing in this module draws from the stream except ``steer``, which draws the one wobble angle a
step makes.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from math import cos, hypot, pi, radians, sin

from ..grid import Cell, Point
from ..rules import FRAME

# Where a walker faces when every urge it has cancels out. East suits anything crossing the map; a
# water course runs down it and says so.
EAST: Point = (1.0, 0.0)


def unit(vector: Point, fallback: Point = EAST) -> Point:
    """The direction a vector points, or the fallback when it points nowhere at all."""
    length = hypot(*vector)
    if length == 0.0:
        return fallback
    return vector[0] / length, vector[1] / length


def rotate(vector: Point, degrees: float) -> Point:
    """Turn a vector by an angle."""
    if degrees == 0.0:
        return vector
    angle = radians(degrees)
    turn_cos, turn_sin = cos(angle), sin(angle)
    return (
        vector[0] * turn_cos - vector[1] * turn_sin,
        vector[0] * turn_sin + vector[1] * turn_cos,
    )


def notch(attempt: int) -> int:
    """Reroute alternately to each side, one notch further out every second try."""
    return ((attempt + 1) // 2) * (1 if attempt % 2 else -1)


def sway(step: int, distance: float, wavelength: float, phase: float) -> float:
    """Where a meander stands in its swing, after so many steps of so many cells each."""
    return sin(step * distance / wavelength * 2.0 * pi + phase)


def steer(
    stream: random.Random,
    urges: Iterable[tuple[float, Point]],
    toward: Point,
    meander: float,
    swing: float,
    wobble: float,
    fallback: Point = EAST,
) -> Point:
    """Blend weighted urges, a meander, and a wobble into one heading.

    The meander swings across the line to whatever the walker is heading for, so it bends the line
    without ever turning it around. The wobble is the one draw a step makes, and it comes last.
    """
    x = y = 0.0
    for weight, urge in urges:
        x += weight * urge[0]
        y += weight * urge[1]
    x += -toward[1] * swing * meander
    y += toward[0] * swing * meander
    angle = stream.uniform(0.0, 2.0 * pi)
    return unit((x + wobble * cos(angle), y + wobble * sin(angle)), fallback)


def advance(
    position: Point,
    heading: Point,
    step: float,
    attempts: int,
    degrees: float,
    free: Callable[[Point], bool],
) -> tuple[Point, Point] | None:
    """Step along a heading, turning aside a notch at a time until the way ahead is free.

    Nothing when every turn is blocked. What that means is the caller's to say: a water course gives
    up on the layout, and a footpath falls back on the route its search already proved.
    """
    for attempt in range(attempts + 1):
        turned = rotate(heading, notch(attempt) * degrees)
        candidate = (position[0] + turned[0] * step, position[1] + turned[1] * step)
        if free(candidate):
            return candidate, turned
    return None


def downhill(field: list[list[float]], position: Point, fallback: Point = EAST) -> Point:
    """The way the steepest slope of a field runs down from a point."""
    x = min(max(int(position[0]), 1), FRAME.cells_x - 2)
    y = min(max(int(position[1]), 1), FRAME.cells_y - 2)
    rise_x = field[y][x + 1] - field[y][x - 1]
    rise_y = field[y + 1][x] - field[y - 1][x]
    return unit((-rise_x, -rise_y), fallback)


def push_away(position: Point, crowd: Iterable[Cell], span: float) -> Point:
    """Push away from a crowd of cells, before whatever hard check guards them has to fire.

    The push points away from the crowd and grows as the nearest of it closes in, reaching full
    strength on contact and nothing at all at the edge of what the walker can sense.
    """
    away = (0.0, 0.0)
    closest = span
    for spot in crowd:
        gap = (position[0] - (spot[0] + 0.5), position[1] - (spot[1] + 0.5))
        length = hypot(*gap)
        if 0.0 < length < span:
            weight = 1.0 - length / span
            away = (away[0] + weight * gap[0] / length, away[1] + weight * gap[1] / length)
            closest = min(closest, length)
    if away == (0.0, 0.0):
        return away
    strength = 1.0 - closest / span
    direction = unit(away)
    return direction[0] * strength, direction[1] * strength
