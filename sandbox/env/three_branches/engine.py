"""The small, simultaneous Three Branches rules engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .catalog import PROP_BY_TOKEN
from .geometry import wrap
from .perception import observe
from .physics import Physics
from .prop_use import apply, select
from .rules import EMOTES, RULES

if TYPE_CHECKING:
    from .layout import Layout


@dataclass(frozen=True, slots=True)
class Order:
    heading: float
    speed: float
    action: int


@dataclass(slots=True)
class Character:
    id: str
    position: tuple[float, float]
    heading: float
    moved: float = 0.0
    expression_type: str = "none"
    expression_target: str = "none"


class Day:
    """All mutable state for one deterministic village day."""

    def __init__(self, layout: Layout, cast_size: int, daynight: bool) -> None:
        self.layout = layout
        self.cast_size = cast_size
        self.daynight = daynight
        self.tick = 0
        self.characters: dict[str, Character] = {}
        visitor = Character("player_0", layout.spawn, 0.0)
        self.characters[visitor.id] = visitor
        for player_index in range(1, cast_size + 1):
            resident_index = player_index - 1
            home = f"home_{resident_index % 5}"
            pose = layout.residence_pose(home, resident_index // 5)
            character = Character(f"player_{player_index}", pose.position, pose.heading)
            self.characters[character.id] = character
        self.prop_states = {prop.id: PROP_BY_TOKEN[prop.type].start for prop in layout.props}
        self.holders: dict[str, str | None] = {prop.id: None for prop in layout.props}
        self.last_held = {prop.id: -10_000 for prop in layout.props}
        self.physics = Physics(layout)
        for character in self.characters.values():
            self.physics.add(character.id, character.position)

    def place(self, character_id: str, position: tuple[float, float]) -> None:
        """Move a character outright, keeping its record and its physics body together."""
        self.characters[character_id].position = position
        self.physics.place(character_id, position)

    @property
    def phase(self) -> str:
        return phase_at(self.tick, self.daynight)

    @property
    def terminal(self) -> bool:
        return self.tick >= RULES.day_ticks


def phase_at(tick: int, daynight: bool) -> str:
    """Name a displayed tick. Tick zero is the opening dawn state for the renderer."""
    if not daynight:
        return RULES.off_phase
    tick = max(1, tick)
    return next(phase.name for phase in RULES.phases if phase.start <= tick <= phase.end)


def normalise(action: Mapping[str, object], heading: float) -> Order:
    """Degrade a mapping already admitted by the action space into a safe order."""
    raw_heading = action.get("heading", heading)
    raw_speed = action.get("speed", 0.0)
    raw_action = action.get("action", 0)
    turn = wrap(float(raw_heading)) if isinstance(raw_heading, int | float) else heading
    speed = float(raw_speed) if isinstance(raw_speed, int | float) else 0.0
    expression = int(raw_action) if isinstance(raw_action, int | float) else 0
    return Order(turn, min(1.0, max(0.0, speed)), expression if 0 <= expression <= len(EMOTES) + 1 else 0)


def _expression(day: Day, character_id: str, order: Order) -> tuple[str, str]:
    """Name the expression an order produces, as a type and the prop it names."""
    if order.action == 0:
        return "none", "none"
    if order.action > 1:
        return EMOTES[order.action - 2], "none"
    # Emotes accompany movement, but using a prop represents stopping to interact with it.
    if order.speed != 0:
        return "none", "none"
    target = select(day.layout, day.characters[character_id].position)
    if target is None:
        return "none", "none"
    # A prop stays with the character who held it on the previous tick.
    held_by = day.holders[target.id]
    if held_by is not None and held_by != character_id:
        return "none", "none"
    return "use", target.id


def step(day: Day, actions: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Resolve one complete parallel tick and return post-tick perceptions."""
    if day.terminal:
        return {character_id: observe(day, character_id) for character_id in day.characters}
    orders = {
        # A missing action is the ruleset's late-action default: heading unchanged, speed 0,
        # expression none. The environment always supplies every action, so this serves the
        # engine's own callers.
        character_id: normalise(actions.get(character_id, {}), character.heading)
        for character_id, character in day.characters.items()
    }
    # A prop takes one user per tick, and `users` is the only record of who won it. Selection runs
    # before physics so every contender reads the same pre-tick state, and roster order puts the
    # player_0 first. A contender who loses simply expresses nothing.
    users: dict[str, str] = {}
    for character_id, order in orders.items():
        character = day.characters[character_id]
        character.heading = order.heading
        kind, target = _expression(day, character_id, order)
        if kind == "use":
            if target in users:
                kind, target = "none", "none"
            else:
                users[target] = character_id
        character.expression_type = kind
        character.expression_target = target
    speeds = {}
    for character_id, order in orders.items():
        ground = day.layout.ground_at(day.characters[character_id].position)
        speeds[character_id] = order.speed * (ground.speed if ground is not None and ground.passable else 0.0)
    moved = day.physics.move(speeds, {character_id: order.heading for character_id, order in orders.items()})
    for character_id, character in day.characters.items():
        character.position = day.physics.position(character_id)
        character.moved = moved[character_id]
    apply(day.prop_states, day.holders, day.last_held, day.layout, users, day.tick + 1)
    day.tick += 1
    return {character_id: observe(day, character_id) for character_id in day.characters}
