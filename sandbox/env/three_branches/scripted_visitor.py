"""Short, purposeful outings with probabilistic choices and local steering."""

from __future__ import annotations

import json
import math
import random
import re
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

_RULES = json.loads(Path(__file__).with_name("rules.json").read_text(encoding="utf-8"))
_CATALOG = json.loads(Path(__file__).with_name("catalog.json").read_text(encoding="utf-8"))
_MODES = ("activities", "sightseeing", "social")
# Activities use these props. Sightseeing, and social outings without a villager, only look.
_ACTIVITY_PROPS = frozenset({"board", "bench", "pump", "shrine"})
_SIGHTSEEING = frozenset({"stall", "pump", "shrine", "bell", "inn", "shed", "board", "bench"})
_PROP_START = {kind["token"]: kind["start"] for kind in _CATALOG["props"]}
# Tests replace this to control the visitor's entropy without touching the random module.
_new_rng = random.Random
_NPC_PLAYER_ID = re.compile(r"player_[1-9][0-9]*\Z")
_DIRECTIONS = tuple((math.cos(i * math.tau / 16), math.sin(i * math.tau / 16)) for i in range(16))
Point = tuple[float, float]


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(cast(Any, value))
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _record(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _records(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _position(record: Mapping[str, Any], key: str = "position") -> Point:
    point = _record(record.get(key))
    return _number(point.get("x")), _number(point.get("y"))


def _heading_to(start: Point, end: Point) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360.0


def _segment_distance2(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    t = (
        max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2))
        if length2
        else 0.0
    )
    return (point[0] - start[0] - t * dx) ** 2 + (point[1] - start[1] - t * dy) ** 2


def _crosses_box(start: Point, end: Point, left: float, bottom: float, right: float, top: float) -> bool:
    low, high = 0.0, 1.0
    for origin, delta, minimum, maximum in (
        (start[0], end[0] - start[0], left, right),
        (start[1], end[1] - start[1], bottom, top),
    ):
        if abs(delta) < 1e-12:
            if not minimum <= origin <= maximum:
                return False
        else:
            a, b = (minimum - origin) / delta, (maximum - origin) / delta
            low, high = max(low, min(a, b)), min(high, max(a, b))
            if low > high:
                return False
    return True


class _Solid(NamedTuple):
    x: float
    y: float
    half_width: float
    half_height: float
    radius: float

    def nearest(self, point: Point) -> Point:
        if self.radius:
            length = math.dist(point, (self.x, self.y)) or 1.0
            return (
                self.x + (point[0] - self.x) * self.radius / length,
                self.y + (point[1] - self.y) * self.radius / length,
            )
        return (
            min(max(point[0], self.x - self.half_width), self.x + self.half_width),
            min(max(point[1], self.y - self.half_height), self.y + self.half_height),
        )

    def blocks(self, start: Point, end: Point, body: float) -> bool:
        if self.radius:
            hit = _segment_distance2((self.x, self.y), start, end) < (self.radius + body) ** 2 - 1e-12
            return hit and not self._leaving_contact(start, end, body)
        left, right = self.x - self.half_width, self.x + self.half_width
        bottom, top = self.y - self.half_height, self.y + self.half_height
        if not _crosses_box(start, end, left - body, bottom - body, right + body, top + body):
            return False
        if _crosses_box(start, end, left, bottom, right, top):
            return True
        hit = min(
            math.dist(start, self.nearest(start)), math.dist(end, self.nearest(end))
        ) < body - 1e-12 or any(
            _segment_distance2(corner, start, end) < body * body - 1e-12
            for corner in ((left, bottom), (left, top), (right, bottom), (right, top))
        )
        return hit and not self._leaving_contact(start, end, body)

    def _leaving_contact(self, start: Point, end: Point, body: float) -> bool:
        # Float32 observations or a physics contact can put the body slightly into a solid.
        # Allow an outward step that only reduces this existing overlap.
        if self.radius and math.dist(start, (self.x, self.y)) < self.radius:
            return False
        surface = self.nearest(start)
        nx, ny = start[0] - surface[0], start[1] - surface[1]
        return 0 < nx * nx + ny * ny < body * body and nx * (end[0] - start[0]) + ny * (end[1] - start[1]) > 0


class _Prop(NamedTuple):
    id: str
    type: str
    shape: _Solid


class _Goal(NamedTuple):
    key: str
    category: str
    point: Point
    look: Point
    prop: str = ""
    person: str = ""


class _Village:
    """Static geometry only. The index answers short swept-body queries, never routes."""

    def __init__(self, observation: object) -> None:
        village = _record(_record(observation).get("village"))
        self.ground = tuple(row for row in village.get("ground", ()) if isinstance(row, str))
        size = _record(village.get("size"))
        self.cell = _number(size.get("cell_size"), _RULES["frame"]["cell_size"])
        self.width = min(map(len, self.ground), default=0) * self.cell
        self.height = len(self.ground) * self.cell
        self.body = float(_RULES["profile"]["body_radius"])
        self.reach = float(_RULES["profile"]["prop_reach"])
        self.speeds = {item["code"]: item["speed"] for item in _RULES["grounds"]}
        self.passable = {item["code"] for item in _RULES["grounds"] if item["passable"]}
        self.outdoor = self.passable - {"i", "d"}
        self.index: dict[tuple[int, int], list[_Solid]] = {}
        self.props: list[_Prop] = []
        for section in ("props", "scenery"):
            kinds = {kind["token"]: kind for kind in _CATALOG[section]}
            for item in _records(village.get(section)):
                kind = kinds.get(item.get("type"))
                if kind is None:
                    continue
                width, height = kind["width"] * self.cell, kind["height"] * self.cell
                if item.get("facing") in {"east", "west"}:
                    width, height = height, width
                x, y = _position(item, "cell")
                radius = (
                    min(width, height) / 2 * kind.get("collision_scale", 1.0) * _number(item.get("scale"), 1)
                )
                shape = _Solid(
                    x * self.cell + width / 2,
                    y * self.cell + height / 2,
                    width / 2,
                    height / 2,
                    radius if kind["shape"] == "circle" else 0,
                )
                extent_x, extent_y = (shape.radius, shape.radius) if shape.radius else (width / 2, height / 2)
                for bx in range(
                    math.floor((shape.x - extent_x) / 4), math.floor((shape.x + extent_x) / 4) + 1
                ):
                    for by in range(
                        math.floor((shape.y - extent_y) / 4), math.floor((shape.y + extent_y) / 4) + 1
                    ):
                        self.index.setdefault((bx, by), []).append(shape)
                if section == "props":
                    self.props.append(_Prop(str(item["id"]), str(item["type"]), shape))
        self.places: list[_Goal] = []
        for prop in self.props:
            if prop.type not in _ACTIVITY_PROPS | _SIGHTSEEING:
                continue
            center = (prop.shape.x, prop.shape.y)
            for dx, dy in _DIRECTIONS[::2]:
                extent = prop.shape.radius or min(
                    prop.shape.half_width / abs(dx) if abs(dx) > 1e-9 else math.inf,
                    prop.shape.half_height / abs(dy) if abs(dy) > 1e-9 else math.inf,
                )
                point = (center[0] + dx * (extent + 0.85), center[1] + dy * (extent + 0.85))
                if self.clear(point, point):
                    self.places.append(_Goal(prop.id, prop.type, point, center, prop.id))
        building_kinds = {kind["token"]: kind for kind in _CATALOG["buildings"]}
        for building in _records(village.get("buildings")):
            if building.get("type") not in _SIGHTSEEING:
                continue
            kind = building_kinds[building["type"]]
            x, y = _position(building, "cell")
            for cx in range(int(x), int(x) + kind["width"]):
                for cy in range(int(y), int(y) + kind["height"]):
                    center = ((cx + 0.5) * self.cell, (cy + 0.5) * self.cell)
                    if self.code(center) != "d":
                        continue
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        point = (center[0] + dx * self.cell, center[1] + dy * self.cell)
                        if self.clear(point, point):
                            self.places.append(
                                _Goal(str(building["id"]), str(building["type"]), point, center)
                            )

    def code(self, point: Point) -> str:
        x, y = math.floor(point[0] / self.cell), math.floor(point[1] / self.cell)
        if 0 <= y < len(self.ground) and 0 <= x < len(self.ground[y]):
            return self.ground[y][x]
        return "w"

    def clear(self, start: Point, end: Point, indoors: bool = False) -> bool:
        """Check a swept body against static solids, treating doorways and interiors as solid
        unless the body is already indoors and needs to cross them to get out."""
        body = self.body
        grounds = self.passable if indoors else self.outdoor
        if not all(
            body <= p[0] <= self.width - body and body <= p[1] <= self.height - body for p in (start, end)
        ):
            return False
        left, right = min(start[0], end[0]) - body, max(start[0], end[0]) + body
        bottom, top = min(start[1], end[1]) - body, max(start[1], end[1]) + body
        for x in range(math.floor(left / self.cell), math.floor(right / self.cell) + 1):
            for y in range(math.floor(bottom / self.cell), math.floor(top / self.cell) + 1):
                if self.code(((x + 0.5) * self.cell, (y + 0.5) * self.cell)) not in grounds:
                    shape = _Solid(
                        (x + 0.5) * self.cell, (y + 0.5) * self.cell, self.cell / 2, self.cell / 2, 0
                    )
                    if shape.blocks(start, end, body):
                        return False
        checked: set[_Solid] = set()
        for bx in range(math.floor(left / 4), math.floor(right / 4) + 1):
            for by in range(math.floor(bottom / 4), math.floor(top / 4) + 1):
                for solid in self.index.get((bx, by), ()):
                    if solid not in checked:
                        if solid.blocks(start, end, body):
                            return False
                        checked.add(solid)
        return True

    def line_clear(self, start: Point, end: Point) -> bool:
        # Match the engine's supercover, including its endpoint cell and corner rules.
        x0, y0 = start[0] / self.cell, start[1] / self.cell
        x1, y1 = end[0] / self.cell, end[1] / self.cell
        x, y = math.floor(x0), math.floor(y0)
        target = (math.floor(x1), math.floor(y1))
        dx, dy = x1 - x0, y1 - y0
        sx, sy = (1 if dx > 0 else -1 if dx < 0 else 0), (1 if dy > 0 else -1 if dy < 0 else 0)
        delta_x, delta_y = (abs(1 / dx) if dx else math.inf), (abs(1 / dy) if dy else math.inf)
        cross_x = ((x + (sx > 0)) - x0) / dx if dx else math.inf
        cross_y = ((y + (sy > 0)) - y0) / dy if dy else math.inf
        for _ in range(2 * (abs(target[0] - x) + abs(target[1] - y) + 1)):
            if self.code(((x + 0.5) * self.cell, (y + 0.5) * self.cell)) == "x":
                return False
            if (x, y) == target:
                return True
            if min(cross_x, cross_y) >= 1 - 1e-12:
                return self.code(end) != "x"
            if abs(cross_x - cross_y) < 1e-12:
                sides = ((x + sx, y), (x, y + sy))
                if any(
                    self.code(((cx + 0.5) * self.cell, (cy + 0.5) * self.cell)) == "x" for cx, cy in sides
                ):
                    return False
                if target in sides:
                    return True
                x, y = x + sx, y + sy
                cross_x, cross_y = cross_x + delta_x, cross_y + delta_y
            elif cross_x < cross_y:
                x, cross_x = x + sx, cross_x + delta_x
            else:
                y, cross_y = y + sy, cross_y + delta_y
        return False

    def selected_prop(self, point: Point) -> str:
        nearest, selected = self.reach + 1e-12, ""
        for prop in self.props:
            surface = prop.shape.nearest(point)
            distance = math.dist(point, surface)
            if distance < nearest and self.line_clear(point, surface):
                nearest, selected = distance, prop.id
        return selected


class Agent:
    """Visit public places, try quiet activities, and meet perceived villagers."""

    def reset(self, seed: object, observation: object) -> None:
        """Start with fresh entropy; the session seed belongs to the village."""
        del seed
        self._rng = _new_rng()
        self._village = _Village(observation)
        self._mode = self._rng.choice(_MODES)
        self._state = "travel"
        self._tick = 0
        self._goal: _Goal | None = None
        self._heading = _number(_record(_record(observation).get("self")).get("heading")) % 360
        self._deadline = 0
        self._retry_at = 0
        self._visit_until = 0
        self._use_until = 0
        self._use_pending = False
        self._target: str | None = None
        self._greeting_due = False
        self._greeted = False
        self._replied = False
        self._hearing: set[str] = set()
        self._last_seen = 0
        self._recent: dict[str, int] = {}
        self._failed: dict[str, int] = {}
        self._history: deque[tuple[int, int]] = deque(maxlen=16)
        self._best_distance = math.inf
        self._progress_at = 0
        self._stalls = 0
        self._expected = 0.0
        self._previous: Point | None = None
        self._escape_until = 0
        self._escape_side = 1
        self._steer_until = 0

    def _finish(self, failed: bool = False) -> None:
        if self._goal is not None:
            if failed:
                self._failed[self._goal.key] = self._tick + 60
            elif self._state == "visit":
                self._recent[self._goal.key] = self._tick
        self._goal = None
        self._target = None
        self._state = "travel"
        self._greeting_due = self._greeted = self._replied = self._use_pending = False
        self._retry_at = self._escape_until = self._steer_until = self._stalls = 0
        self._expected = 0.0

    def _choose(self, position: Point, people: list[Mapping[str, Any]]) -> None:
        candidates: list[_Goal] = []
        if self._mode == "social":
            candidates = [
                _Goal(str(p["id"]), "person", _position(p), _position(p), person=str(p["id"]))
                for p in people
                if self._tick - self._recent.get(str(p["id"]), -1000) >= 120
                and self._failed.get(str(p["id"]), 0) <= self._tick
            ]
            heard = [goal for goal in candidates if goal.person in self._hearing]
            candidates = heard or candidates
        if not candidates:
            allowed = _ACTIVITY_PROPS if self._mode == "activities" else _SIGHTSEEING
            places: dict[str, _Goal] = {}
            for goal in self._village.places:
                if goal.category not in allowed or self._failed.get(goal.key, 0) > self._tick:
                    continue
                if self._mode == "activities" and self._village.selected_prop(goal.point) != goal.prop:
                    continue
                if goal.key not in places or math.dist(position, goal.point) < math.dist(
                    position, places[goal.key].point
                ):
                    places[goal.key] = goal
            categories = sorted({goal.category for goal in places.values()})
            if categories:
                category = self._rng.choice(categories)
                candidates = [goal for goal in places.values() if goal.category == category]
        if candidates:
            weights = [
                (0.2 if self._tick - self._recent.get(g.key, -1000) < 180 else 1.0)
                / (1 + math.dist(position, g.point) / 20)
                for g in candidates
            ]
            self._goal = self._rng.choices(candidates, weights=weights, k=1)[0]
        else:
            points = [(position[0] + dx * 8, position[1] + dy * 8) for dx, dy in _DIRECTIONS]
            points = [point for point in points if self._village.clear(point, point)]
            if points:
                point = self._rng.choice(points)
                self._goal = _Goal("stroll", "stroll", point, point)
        if self._goal:
            self._deadline = self._tick + self._rng.randint(100, 180)
        else:
            self._retry_at = self._tick + 20
        self._best_distance = math.inf
        self._progress_at = self._last_seen = self._tick
        self._target = (self._goal.person or None) if self._goal else None

    def act(self, observation: object) -> dict[str, float | int]:
        obs = _record(observation)
        me = _record(obs.get("self"))
        position = _position(me)
        self._tick += 1
        self._hearing = {str(p["id"]) for p in _records(obs.get("nearby")) if "id" in p}
        people = {
            str(p["id"]): p
            for p in (*_records(obs.get("seen")), *_records(obs.get("nearby")))
            if _NPC_PLAYER_ID.fullmatch(str(p.get("id", "")))
        }
        cell = (math.floor(position[0] / 2), math.floor(position[1] / 2))
        if not self._history or cell != self._history[-1]:
            self._history.append(cell)
        if self._previous is not None and self._expected > 0.1:
            self._stalls = self._stalls + 1 if math.dist(position, self._previous) < 0.05 else 0
        self._previous, self._expected = position, 0.0
        if self._stalls >= 3:
            self._escape_until = self._tick + self._rng.randint(6, 12)
            self._escape_side = self._rng.choice((-1, 1))
            self._steer_until = self._stalls = 0
        if (
            self._tick % 20 == 0
            and not (self._state == "visit" and self._target)
            and self._rng.random() < 0.15
        ):
            self._finish()
            self._mode = self._rng.choice([mode for mode in _MODES if mode != self._mode])
        # The deadline bounds travel only. A visit already has its own end in `_visit_until`.
        if self._goal is not None and self._state == "travel" and self._tick >= self._deadline:
            self._finish(failed=True)
        if self._goal is None:
            if self._tick >= self._retry_at:
                self._choose(position, list(people.values()))
            if self._goal is None:
                self._heading = (self._heading + 22.5) % 360
                return self._action()
        goal = self._goal
        if goal is None:
            return self._action()
        if goal.person:
            person = people.get(goal.person)
            if person is not None:
                self._last_seen = self._tick
                goal = self._goal = goal._replace(point=_position(person), look=_position(person))
            if self._tick - self._last_seen > 8:
                self._finish(failed=True)
                return self._action()
        if self._state == "visit":
            return self._visit(obs, position)
        distance = math.dist(position, goal.point)
        if distance < self._best_distance - 0.5:
            self._best_distance, self._progress_at = distance, self._tick
        elif self._tick - self._progress_at >= 40:
            self._finish(failed=True)
            return self._action()
        arrival = 1.7 if goal.person else 0.25
        if distance <= arrival and (not goal.person or goal.person in self._hearing):
            self._state = "visit"
            self._heading = _heading_to(position, goal.look)
            self._visit_until = (
                self._tick + self._rng.randint(4, 7) if goal.person else self._tick + self._rng.randint(8, 24)
            )
            self._use_until = self._tick + self._rng.randint(1, 3)
            if goal.person:
                self._greeting_due = True
                return self._action(expression=2)
            return self._action()
        return self._move(position, max(0.0, distance - arrival * 0.5))

    def _visit(self, observation: Mapping[str, Any], position: Point) -> dict[str, float | int]:
        goal = self._goal
        if goal is None:
            return self._action()
        if self._tick >= self._visit_until or (goal.person and goal.person not in self._hearing):
            self._finish()
            return self._action()
        if goal.person or self._mode == "activities":
            self._heading = _heading_to(position, goal.look)
        if self._mode == "activities" and goal.prop and goal.category in _ACTIVITY_PROPS:
            expression = _record(_record(observation.get("self")).get("expression"))
            own_use = expression.get("type") == "use" and expression.get("target") == goal.prop
            if self._use_pending and not own_use:
                self._finish(failed=True)
                return self._action()
            hold = goal.category in {"board", "bench"} or self._tick <= self._use_until
            if hold:
                states = {p.get("prop"): p.get("state") for p in _records(observation.get("props"))}
                available = states.get(goal.prop) == _PROP_START[goal.category]
                if (available or own_use) and self._village.selected_prop(position) == goal.prop:
                    self._use_pending = True
                    return self._action(expression=1)
                if self._tick > self._visit_until - 6:
                    self._finish(failed=True)
                return self._action()
            self._use_pending = False
        elif not goal.person and self._tick % 4 == 0:
            self._heading = (self._heading + self._rng.choice((-60, -30, 30, 60))) % 360
        return self._action()

    def _move(self, position: Point, remaining: float) -> dict[str, float | int]:
        goal = self._goal
        if goal is None:
            return self._action()
        code = self._village.code(position)
        # A push can leave the body on doorway or interior ground, which it never chooses to enter.
        # It may then cross any passable ground until it is outdoors again.
        indoors = code not in self._village.outdoor
        ground_speed = self._village.speeds.get(code, 0.0)
        speed = min(0.7, remaining / ground_speed) if ground_speed else 0.0
        step = speed * ground_speed
        forward = (math.cos(math.radians(self._heading)), math.sin(math.radians(self._heading)))
        endpoint = (position[0] + forward[0] * step, position[1] + forward[1] * step)
        if self._tick >= self._steer_until or not self._village.clear(position, endpoint, indoors):
            direction = _heading_to(position, goal.point)
            desired = (math.cos(math.radians(direction)), math.sin(math.radians(direction)))
            escaping = self._tick < self._escape_until
            candidates: list[tuple[float, float]] = []
            for dx, dy in _DIRECTIONS:
                # A short escape step can fit a tight corner that rejects the normal preview.
                preview = min(step if escaping else 2.0, remaining)
                end = (position[0] + dx * preview, position[1] + dy * preview)
                if not self._village.clear(position, end, indoors):
                    continue
                coarse = (math.floor(end[0] / 2), math.floor(end[1] / 2))
                novelty = -2.0 if coarse in self._history else 0.0
                continuity = dx * forward[0] + dy * forward[1]
                alignment = 0.0 if escaping else 3 * (dx * desired[0] + dy * desired[1])
                turn = self._escape_side * (forward[0] * dy - forward[1] * dx) if escaping else 0.0
                road = 1.0 if self._village.code(end) in {"r", "p", "b"} else 0.0
                candidates.append(
                    (alignment + continuity + road + novelty + turn, math.degrees(math.atan2(dy, dx)) % 360)
                )
            if not candidates:
                self._stalls += 1
                return self._action()
            if self._rng.random() < 0.8:
                _, self._heading = max(candidates)
            else:
                self._heading = self._rng.choices(
                    [angle for _, angle in candidates],
                    weights=[math.exp(score) for score, _ in candidates],
                    k=1,
                )[0]
            self._steer_until = self._tick + 3
        self._expected = step
        return self._action(speed=speed)

    def _action(self, speed: float = 0.0, expression: int = 0) -> dict[str, float | int]:
        return {"heading": self._heading, "speed": speed, "action": expression}

    def chat(self, inbox: object) -> list[dict[str, str]]:
        """Greet once, then answer once while the same conversation is in hearing range."""
        if self._state != "visit" or not self._target or self._target not in self._hearing:
            return []
        if self._greeting_due:
            self._greeting_due = False
            self._greeted = True
            return [{"to": self._target, "text": "Hello. How is your day going?"}]
        if (
            self._greeted
            and not self._replied
            and any(m.get("from") == self._target for m in _records(inbox))
        ):
            self._replied = True
            return [{"to": self._target, "text": "Thank you. I am glad to be here."}]
        return []
