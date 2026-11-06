"""Pure village geometry and the fixed perception distances."""

from __future__ import annotations

from math import atan2, cos, degrees, hypot, radians, sin

from ._model import RULES

BODY_RADIUS = float(RULES["profile"]["body_radius"])
VISION_DEGREES = float(RULES["profile"]["vision_degrees"])
VISION_RANGE = float(RULES["profile"]["vision_range"])
HEARING_RANGE = float(RULES["profile"]["hearing_range"])
PROP_REACH = float(RULES["profile"]["prop_reach"])


def distance(first, second) -> float:
    """Return the straight-line distance between two positions in metres. Each position is an
    ``{"x": float, "y": float}`` mapping from ``me.position`` or a ``"position"`` field."""
    return hypot(float(first["x"]) - float(second["x"]), float(first["y"]) - float(second["y"]))


def heading_to(start, end) -> float:
    """Return the heading in degrees (``0.0`` east, counter-clockwise) that points from ``start``
    toward ``end``. Shares the convention of ``me.heading``: face this way to walk to ``end``."""
    return wrap(degrees(atan2(float(end["y"]) - float(start["y"]), float(end["x"]) - float(start["x"]))))


def wrap(heading: float) -> float:
    """Wrap a heading into the ``[0.0, 360.0)`` range, the range ``me.heading`` and
    ``action.walk`` use."""
    return float(heading) % 360.0


def in_cone(
    origin, heading: float, point, degrees_wide: float = VISION_DEGREES, reach: float = VISION_RANGE
) -> bool:
    """Return whether ``point`` lies inside a cone of ``degrees_wide`` centred on ``heading``
    from ``origin``, out to ``reach`` metres. The default cone is the vision cone, so this is
    "can I see the point"."""
    dx, dy = float(point["x"]) - float(origin["x"]), float(point["y"]) - float(origin["y"])
    length = hypot(dx, dy)
    if length > reach:
        return False
    if length == 0:
        return True
    angle = radians(wrap(heading))
    return (cos(angle) * dx + sin(angle) * dy) / length >= cos(radians(degrees_wide / 2)) - 1e-12
