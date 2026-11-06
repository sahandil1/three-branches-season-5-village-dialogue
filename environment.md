# Days at Three Branches

Days at Three Branches is a village simulation. Your program controls one NPC, and the platform starts a separate instance for every villager. Every instance has its own memory and sees only its own observation. See [Agent instances and state](https://vox-deorum.github.io/game-sandbox/students/agent-interface/#agent-instances-and-state) for how instances and state work. Villagers can coordinate by moving, noticing one another, using props, and speaking.

Start with the [template](https://github.com/Minor-Solutions/game-sandbox/tree/main/environments/three_branches/template). After you complete [Getting Started](https://vox-deorum.github.io/game-sandbox/students/getting-started/), open `agent.py` and run `python -m sandbox watch` from that folder to watch a day beside the scripted visitor. Then follow [Improve the starter](#improve-the-starter).

## Make an action

Import the small helper namespaces rather than assembling raw action dictionaries:

```python
from sandbox.village import action, me, people


class Agent:
    def reset(self, seed, observation):
        # Called once before each day. Build durable state here, not in act.
        pass

    def act(self, observation):
        heading = me.heading(observation)
        if people.seen(observation):
            return action.walk(heading, 0.5, "wave")
        return action.stand(heading, "none")
```

`action.walk(heading, speed, expression)` wraps headings into `[0.0, 360.0)` and clamps speed to `0.0` through `1.0`. `action.stand(heading, expression)` is the still version. Expressions are `"none"`, `"use"`, or one of `action.EMOTES`.

## Improve the starter

The starter walks toward its home's doorway while it is on interior ground. Outside, it heads toward the well-plaza pump. It waves when it sees somebody and sits only when `props.usable(observation)` previews a bench.

It does not remember a route or avoid walls. A useful first improvement is to choose a destination from the static village, then check nearby points with `layout.walkable` and `layout.can_step` before walking.

## A village day

One submission controls the cast seat: `player_1` through `player_5` in `cast_5`, or through `player_10` in `cast_10`. The visitor is `player_0`, human-controlled in live play or run by the scripted visitor in automated runs. `reset(seed, observation)` runs once before a day, so build any route graph there instead of in every `act` call.

One episode is one day of village life, with 1200 simultaneous ticks. Your action is chosen from the current observation, then every character moves, then prop use resolves in character order. Every healthy completed character scores 100. The automated board uses compute time as its tiebreaker, while people judge whether the village feels alive.

Ground, walls, doorways, buildings, props, and scenery are standing knowledge in `observation["village"]`. Water and walls stop movement. Walls block sight and hearing, but doorways do not. Interactive props have collision shapes, so `layout.walkable` and `layout.can_step` are safer than assuming a prop's cell is empty.

## Helpers

Import the helpers you need at the top of `agent.py`:

```python
from sandbox.village import action, day, geometry, layout, me, people, props
```

Functions that inspect the village take `observation` first. `action` builds your next order and `geometry` does calculations, so neither needs an observation. Positions are `{"x": float, "y": float}` mappings in metres from the village southwest corner.

`people.seen`, `people.nearby`, and prop states only describe what your villager can currently perceive. The roster and village layout are available in every observation and do not change during the day. Use `layout.walkable` and `layout.can_step` to test a route because water, walls, props, and scenery can block movement.

| Group | Helper | Result |
| --- | --- | --- |
| Orders | `action.walk(heading, speed=1.0, expression="none")` | Builds a walking order. It wraps heading into `0.0` through `< 360.0`, clamps speed to `0.0` through `1.0`, and accepts `"none"`, `"use"`, or an emote. |
| Orders | `action.stand(heading, expression="none")` | Builds a zero-speed order, including an order to emote or use a prop. |
| Orders | `action.EMOTES` | The available emote names: use one as the `expression` argument. |
| Self | `me.player_id(observation)` | Your stable player id, such as `"player_1"`. Use it for chat recipients. |
| Self | `me.position(observation)`, `me.heading(observation)` | Your current position and facing. Heading is degrees, with `0.0` east and `90.0` north. |
| Self | `me.moved(observation)` | How many metres you moved on the previous tick, or `0.0` when you stood still. |
| Self | `me.expression(observation)` | Your current `{"type", "target"}` expression. `target` names the prop in use, or is `"none"`. |
| Self | `me.home(observation)` | Your home building id from the roster. |
| Self | `me.rng(observation, session_seed)` | A stable, private `random.Random` stream for this player and session seed. |
| Other villagers | `people.seen(observation)` | Villagers in your vision cone with no wall between you, including id, position, heading, previous movement, and expression. |
| Other villagers | `people.nearby(observation)` | Villagers within hearing range and a clear line, with id and position only. |
| Other villagers | `people.find(observation, player_id)` | The character with this id, taken from those you see first, then those you hear. `None` when you can neither see nor hear them. |
| Other villagers | `people.roster(observation)` | Every villager's stable id and home. It does not show where they are now. |
| Other villagers | `people.is_visitor(player_id)`, `people.is_npc(player_id)` | Whether an id is the visitor (`"player_0"`) or has the normal NPC id form. |
| Props | `props.all(observation)` | Every static prop placement, in layout order, including id, type, cell, and facing. |
| Props | `props.find(observation, prop_id)` | The static prop placement with this id, or `None` when there is no such prop. |
| Props | `props.seen(observation)` | Visible prop records with a prop id and current state. A prop outside your vision cone or behind a wall is absent. |
| Props | `props.state(observation, prop_id)` | The visible state string for one prop, or `None` when that prop is not visible. |
| Props | `props.in_reach(observation)` | Static props close enough to use by distance alone. It does not test whether a wall is in the way. |
| Props | `props.usable(observation)` | The nearest prop a use action would select, or `None`. It checks distance and walls, but not whether another villager holds the prop. |
| Props | `props.TYPES` | Every available prop type token. |
| Village | `layout.frame(observation)` | Grid dimensions and cell size as `cells_x`, `cells_y`, and `cell_size`. |
| Village | `layout.cell_at(observation, position)`, `layout.ground_at(observation, cell)` | The grid cell containing a position, or `None` outside the village, and that cell's ground name, also `None` outside. |
| Village | `layout.cell_center(observation, cell)` | The position at the middle of a cell, such as a prop's or building's `cell`. |
| Village | `layout.nearest_walkable(observation, position)` | The walkable cell whose center is closest to a position, or `None` when no cell is walkable. Useful as a destination for a prop or a building center, which a body cannot stand on. |
| Village | `layout.walkable(observation, cell)` | Whether a villager can stand in a cell, including ground, walls, blocking props, and scenery. |
| Village | `layout.can_step(observation, start_cell, end_cell)` | Whether a cardinal one-cell move is clear and both cells are walkable. |
| Village | `layout.line_of_sight(observation, start_pos, end_pos)` | Whether a straight line crosses no sight-blocking ground. It ignores props and vision-cone range, and returns `False` outside the village. |
| Village | `layout.buildings(observation)`, `layout.building(observation, building_id)` | Every static building, or one building by id and `None` when it does not exist. |
| Village | `layout.building_center(observation, building_id)` | The center of a building's footprint, or `None` when there is no such building. |
| Village | `layout.building_at(observation, position)` | The id of the building whose floor holds a position, or `None` outdoors. Walls around the edge of a footprint are not floor. |
| Village | `layout.doorway(observation, building_id)` | The nearest doorway position for a building, or `None` when the building or a doorway is absent. |
| Village | `layout.spawn(observation)` | The village spawn position. |
| Village | `layout.SPEED_LIMITS` | The maximum movement speed for each ground name. |
| Village | `layout.BUILDING_SIZES` | Each building type's `(width, height)` footprint in cells, counted from the building's `cell` corner and including its walls. |
| Geometry | `geometry.distance(first, second)` | Straight-line distance between two positions, in metres. |
| Geometry | `geometry.heading_to(start, end)`, `geometry.wrap(heading)` | The heading toward a position, or a heading wrapped into `0.0` through `< 360.0`. |
| Geometry | `geometry.in_cone(origin, heading, point, degrees_wide=geometry.VISION_DEGREES, reach=geometry.VISION_RANGE)` | Whether a point lies in a cone. The defaults match your vision cone but do not test walls. |
| Geometry | `geometry.BODY_RADIUS`, `geometry.VISION_DEGREES`, `geometry.VISION_RANGE`, `geometry.HEARING_RANGE`, `geometry.PROP_REACH` | Fixed distances and angle limits in metres or degrees. |
| Day | `day.tick(observation)`, `day.phase(observation)` | The tick, starting at `1`, and the current phase. A season without day and night reports `"day"`. |
| Day | `day.bell_ringing(observation)` | Whether the village bell is ringing now. |
| Day | `day.parameters(observation)` | The resolved settings for this day, such as the seat plan and whether day and night is on. |

### Using props

Call `props.state` with a static prop's id to get its visible state. `None` means your villager cannot currently see that prop, not that the prop has a state named `None`.

```python
bench = next((prop for prop in props.all(observation) if prop["type"] == "bench"), None)
if bench is not None:
    bench_state = props.state(observation, bench["id"])
    if bench_state is None:
        print("The bench is not visible.")
    elif bench_state == "empty":
        print("The bench is empty.")
```

To use a prop, return `action.stand(heading, "use")`. Use requires speed `0`, no wall between you and the prop, and being within reach of its collision shape. You never name the prop: `props.usable(observation)` previews the nearest prop the engine would select. Check that preview before using when you need a particular prop. Use can still fail when another villager already holds that prop. A use lasts one tick, so keep returning `use` while you intend to hold the prop.

## Season settings and limits

The first season runs with day and night off; later seasons can use day phases and the optional LLM API. `day.parameters(observation)` shows the resolved settings, and `day.phase(observation)` is `"day"` when day and night is off.

Each `act` call has a 0.25 second limit. A whole day has a 120 second limit. The cast decisions run sequentially, so quick code matters more in `cast_10`. Build durable data in `reset` and keep per-tick work small. See [Time limits](https://vox-deorum.github.io/game-sandbox/students/agent-interface/#time-limits) for how limits are measured and enforced.

## Run locally

```console
python -m sandbox watch  # watch your villagers and the scripted visitor
python -m sandbox test   # run supplied checks
python -m sandbox eval   # run repeatable automated days
```

See the shared [agent interface](https://vox-deorum.github.io/game-sandbox/students/agent-interface/) for `reset`, optional learning, chat, and submission details.

## Chat with other agents

Chat is optional, and it uses the same canonical player IDs that appear in `me.player_id(observation)`, `people.seen(observation)`, and the roster. For example, a villager might receive a message from the visitor as `"player_0"` and reply directly to `"player_0"`. There is no chat helper namespace: return ordinary dictionaries from an optional `chat` method.

```python
def chat(self, inbox):
    for message in inbox:
        if message["from"] == "player_0":
            return [{"to": "player_0", "text": "Hello from the village."}]
    return []
```

A message is a plain dictionary with a sender, recipient, text, and the tick it was sent on, such as `{"from": "player_0", "to": "player_2", "text": "Meet at the pump.", "tick": 3}`. A broadcast with `None` as the recipient reaches every player who can hear the sender. See the [agent interface](https://vox-deorum.github.io/game-sandbox/students/agent-interface/#chatinbox) for the full message shape, delivery timing, and replay visibility.

## Optional raw reference

This optional raw reference shows the exact shapes for agents that need them. `act` receives a plain dictionary. You can use it directly, but the helpers above keep common work readable.

| Field | Contents |
| --- | --- |
| `self` | id, position, heading, `moved` (movement from the previous tick), expression |
| `seen` | visible characters: id, position, heading, movement, expression |
| `nearby` | characters within hearing range and line, with id and position only |
| `props` | visible interactive prop ids and states |
| `bell`, `tick`, `phase` | global bell state, current tick, and day phase |
| `village` | static size, ground rows, buildings, prop placements, scenery, and spawn |
| `roster`, `parameters` | character homes and resolved season settings |

The raw action space is:

```text
Dict{
  "heading": [0.0, 360.0),
  "speed":   Box(0.0, 1.0, shape=(), float32),
  "action":  Discrete(11)
}
```

Action `0` is none and `1` is use. The remaining ids are: `2` wave, `3` nod, `4` shake_head, `5` point, `6` laugh, `7` shrug, `8` startle, `9` sleep, and `10` sweep.
