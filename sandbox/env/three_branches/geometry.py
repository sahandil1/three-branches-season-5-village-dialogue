"""Continuous geometry shared by movement, props, and perception."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class Circle:
    x: float
    y: float
    radius: float


def wrap(heading: float) -> float:
    return heading % 360.0


def heading_vector(heading: float) -> Point:
    angle = radians(wrap(heading))
    return cos(angle), sin(angle)


def distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def nearest_point_rect(point: Point, rect: Rect) -> Point:
    return min(max(point[0], rect.x), rect.right), min(max(point[1], rect.y), rect.top)


def nearest_point_circle(point: Point, circle: Circle) -> Point:
    dx, dy = point[0] - circle.x, point[1] - circle.y
    length = hypot(dx, dy)
    if length == 0:
        return circle.x + circle.radius, circle.y
    return circle.x + circle.radius * dx / length, circle.y + circle.radius * dy / length


def nearest_point(point: Point, shape: Rect | Circle) -> Point:
    if isinstance(shape, Rect):
        return nearest_point_rect(point, shape)
    return nearest_point_circle(point, shape)


def point_in_cone(origin: Point, heading: float, point: Point, degrees_wide: float, reach: float) -> bool:
    """Test an inclusive cone edge so perception has no flickering boundary."""
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    length = hypot(dx, dy)
    if length > reach:
        return False
    if length == 0:
        return True
    forward = heading_vector(heading)
    return forward[0] * dx / length + forward[1] * dy / length >= cos(radians(degrees_wide / 2)) - 1e-12


def circle_intersects_rect(center: Point, radius: float, rect: Rect) -> bool:
    return distance(center, nearest_point_rect(center, rect)) < radius


def circle_intersects_circle(center: Point, radius: float, circle: Circle) -> bool:
    return distance(center, (circle.x, circle.y)) < radius + circle.radius
