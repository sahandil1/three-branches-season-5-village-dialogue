"""Pymunk-backed movement for the Three Branches day engine."""

from __future__ import annotations

from math import cos, inf, pi, sin
from typing import TYPE_CHECKING

import pymunk

from .geometry import Rect, distance
from .rules import PROFILE, RULES

if TYPE_CHECKING:
    from .layout import Layout

_BODY_MASS = 1.0
_BODY_MOMENT = inf


class Physics:
    """Keep one Pymunk space for a day and move its characters together."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        # Pymunk applies collision bias over one simulated second, independent of the substep count.
        # Its standard rate prevents a commanded body from accumulating penetration across ticks.
        # Zero slop is stable here because the village has no gravity, stacking, or resting contacts.
        self.space.collision_slop = 0.0
        self._bodies: dict[str, pymunk.Body] = {}
        shapes: list[pymunk.Shape] = []
        for rect in layout.blocked:
            shapes.append(_rect(self.space.static_body, rect))
        for shape in layout.solids:
            if isinstance(shape, Rect):
                shapes.append(_rect(self.space.static_body, shape))
            else:
                shapes.append(pymunk.Circle(self.space.static_body, shape.radius, (shape.x, shape.y)))
        width, height = layout.grid.frame.width, layout.grid.frame.height
        shapes.extend(
            (
                pymunk.Segment(self.space.static_body, (0, 0), (width, 0), 0),
                pymunk.Segment(self.space.static_body, (width, 0), (width, height), 0),
                pymunk.Segment(self.space.static_body, (width, height), (0, height), 0),
                pymunk.Segment(self.space.static_body, (0, height), (0, 0), 0),
            )
        )
        for shape in shapes:
            shape.friction = 0
            shape.elasticity = 0
        self.space.add(*shapes)

    def add(self, character_id: str, position: tuple[float, float]) -> None:
        """Add one configured character body at a north-up world position."""
        body = pymunk.Body(_BODY_MASS, _BODY_MOMENT)
        body.position = position
        shape = pymunk.Circle(body, PROFILE.body_radius)
        shape.friction = 0
        shape.elasticity = 0
        self.space.add(body, shape)
        self._bodies[character_id] = body

    def position(self, character_id: str) -> tuple[float, float]:
        """Return one character body's current north-up world position."""
        point = self._bodies[character_id].position
        return float(point.x), float(point.y)

    def place(self, character_id: str, position: tuple[float, float]) -> None:
        """Set a body outright, outside the tick's solved movement."""
        self._bodies[character_id].position = position

    def move(self, speeds: dict[str, float], headings: dict[str, float]) -> dict[str, float]:
        """Advance every body by one rules tick and report actual distances."""
        before = {character_id: self.position(character_id) for character_id in self._bodies}
        for character_id, body in self._bodies.items():
            speed = speeds[character_id]
            if speed == 0:
                body.body_type = pymunk.Body.KINEMATIC
                body.velocity = (0, 0)
            else:
                if body.body_type != pymunk.Body.DYNAMIC:
                    body.body_type = pymunk.Body.DYNAMIC
                    # Pymunk clears mass and moment when a kinematic body becomes dynamic. Restore
                    # the character body before solving contacts, since zero mass produces NaNs.
                    body.mass = _BODY_MASS
                    body.moment = _BODY_MOMENT
                angle = headings[character_id] * pi / 180
                body.velocity = (speed * cos(angle), speed * sin(angle))
        for _ in range(RULES.physics_substeps):
            self.space.step(1 / RULES.physics_substeps)
        moved: dict[str, float] = {}
        for character_id, body in self._bodies.items():
            # Pymunk resolves contacts during substeps. Clamp the frame as a final invariant so
            # a body that began exactly on an edge cannot leave through numerical overlap.
            body.position = (
                min(
                    max(body.position.x, PROFILE.body_radius),
                    self.layout.grid.frame.width - PROFILE.body_radius,
                ),
                min(
                    max(body.position.y, PROFILE.body_radius),
                    self.layout.grid.frame.height - PROFILE.body_radius,
                ),
            )
            position = self.position(character_id)
            # A tick simulates one second, so a commanded body covers at most its full speed of 1.
            # Contact separation can add to that, and the reported distance is a rules value with a
            # declared range, so hold it inside the range the observation promises.
            moved[character_id] = min(distance(before[character_id], position), 1.0)
            body.velocity = (0, 0)
        return moved


def _rect(body: pymunk.Body, rect: Rect) -> pymunk.Poly:
    return pymunk.Poly(
        body, ((rect.x, rect.y), (rect.right, rect.y), (rect.right, rect.top), (rect.x, rect.top))
    )
