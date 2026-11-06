"""Pins for the public ``sandbox.village`` helpers."""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest
from sandbox.env import META, make_env
from sandbox.harness.environment import resolve_parameters
from sandbox.village import _model as village_model
from sandbox.village import action, day, geometry, layout, me, people, props


def _observations():
    env = make_env(resolve_parameters(META))
    observations, _ = env.reset(seed=7)
    return env, observations


def test_helpers_read_native_observation_values_and_build_normalized_actions():
    env, observations = _observations()
    observation = observations["player_1"]

    assert me.player_id(observation) == "player_1"
    assert me.position(observation) is observation["self"]["position"]
    assert people.seen(observation) is observation["seen"]
    assert people.nearby(observation) is observation["nearby"]
    assert people.roster(observation) is observation["roster"]
    assert props.all(observation) is observation["village"]["props"]
    assert props.seen(observation) is observation["props"]
    assert layout.frame(observation) is observation["village"]["size"]
    assert layout.buildings(observation) is observation["village"]["buildings"]
    assert layout.spawn(observation) is observation["village"]["spawn"]
    assert me.heading(observation) is observation["self"]["heading"]
    assert me.moved(observation) is observation["self"]["moved"]
    assert me.expression(observation) is observation["self"]["expression"]
    assert me.home(observation) == next(
        item["home"] for item in observation["roster"] if item["id"] == "player_1"
    )
    assert day.tick(observation) is observation["tick"]
    assert day.phase(observation) is observation["phase"]
    assert day.bell_ringing(observation) is bool(observation["bell"])
    assert day.parameters(observation) is observation["parameters"]
    assert action.walk(360.0, 4.0, "wave") == {"heading": 0.0, "speed": 1.0, "action": 2}
    assert action.stand(-90.0, "use") == {"heading": 270.0, "speed": 0.0, "action": 1}
    assert action.walk(-1.0, -4.0) == {"heading": 359.0, "speed": 0.0, "action": 0}
    with pytest.raises(ValueError, match="unknown expression"):
        action.walk(0.0, 1.0, "dance")
    assert env.action_space("player_1").contains(action.stand(me.heading(observation)))


def test_static_helpers_respect_the_boundary_and_do_not_cross_observations():
    _env, observations = _observations()
    first, second = observations["player_1"], observations["player_2"]
    here = me.position(first)
    assert layout.cell_at(first, {"x": -0.01, "y": 0.0}) is None
    assert layout.ground_at(first, {"x": -1, "y": 0}) is None
    assert not layout.walkable(first, {"x": -1, "y": 0})
    assert not layout.can_step(first, {"x": 0, "y": 0}, {"x": -1, "y": 0})
    assert not layout.line_of_sight(first, {"x": -1.0, "y": 0.0}, here)
    assert layout.line_of_sight(first, here, here)

    building = layout.buildings(first)[0]
    assert layout.building(first, building["id"]) is building
    assert layout.building(first, "not-a-building") is None
    assert layout.doorway(first, building["id"]) is not None
    assert layout.doorway(first, "not-a-building") is None
    assert layout.SPEED_LIMITS["road"] == 1.0
    assert layout.SPEED_LIMITS["water"] == 0.0
    assert layout.BUILDING_SIZES == {"home": (8, 7), "inn": (12, 10), "shed": (8, 8)}
    assert {item["type"] for item in layout.buildings(first)} <= set(layout.BUILDING_SIZES)

    # Build a cached model before mutating one player's isolated static mapping.
    assert layout.cell_at(first, here) is not None
    first_prop = props.all(first)[0]
    second_prop = props.all(second)[0]
    first_prop["cell"]["x"] = 99
    assert second_prop["cell"]["x"] != 99


def test_static_helpers_reuse_models_for_unchanged_or_equal_village_mappings():
    _env, observations = _observations()
    observation = copy.deepcopy(observations["player_1"])
    equal_observation = copy.deepcopy(observation)
    village_model._model.cache_clear()
    village_model._IDENTITY_CACHE.clear()

    first = village_model.model(observation)
    assert layout.cell_at(observation, me.position(observation)) is not None
    assert isinstance(layout.walkable(observation, {"x": 0, "y": 0}), bool)

    assert observation["village"] is not equal_observation["village"]
    assert village_model.model(equal_observation) is first

    previous_props = observation["village"]["props"]
    observation["village"]["props"] = tuple(prop for prop in previous_props)
    assert observation["village"]["props"] is not previous_props
    assert village_model.model(observation) is first


def test_identity_cache_reuses_one_model_and_rebuilds_on_wholesale_replacement():
    _env, observations = _observations()
    observation = copy.deepcopy(observations["player_1"])
    village_model._model.cache_clear()
    village_model._IDENTITY_CACHE.clear()

    first = village_model.model(observation)
    assert village_model.model(observation) is first

    previous_props = observation["village"]["props"]
    observation["village"]["props"] = tuple(
        {
            "id": prop["id"],
            "type": prop["type"],
            "cell": {"x": prop["cell"]["x"] + 1, "y": prop["cell"]["y"]},
            "facing": prop["facing"],
        }
        for prop in previous_props
    )
    second = village_model.model(observation)
    assert second is not first
    assert village_model.model(observation) is second


def test_content_cache_distinguishes_dropped_equal_and_variant_village_mappings():
    _env, observations = _observations()
    source = observations["player_1"]["village"]
    village_model._model.cache_clear()
    village_model._IDENTITY_CACHE.clear()
    expected: dict[bool, village_model.Model] = {}

    for index in range(200):
        village = copy.deepcopy(source)
        variant = index % 2 == 1
        if variant:
            village["props"][0]["cell"]["x"] += 1
        result = village_model.model({"village": village})
        if variant not in expected:
            expected[variant] = result
        assert result is expected[variant]
        del village


def test_collision_buckets_match_the_full_collision_shape_checks():
    _env, observations = _observations()
    village = village_model.model(observations["player_1"])
    radius = geometry.BODY_RADIUS
    width, height = village.cells_x * village.cell_size, village.cells_y * village.cell_size
    step_x, step_y = max(1, village.cells_x // 5), max(1, village.cells_y // 5)
    points = tuple(
        ((x + 0.5) * village.cell_size, (y + 0.5) * village.cell_size)
        for y in range(0, village.cells_y, step_y)
        for x in range(0, village.cells_x, step_x)
    )

    def brute_body(point):
        return (
            radius <= point[0] <= width - radius
            and radius <= point[1] <= height - radius
            and not any(
                village_model._circle_hits_shape(point, radius, shape) for shape in village.collision_shapes
            )
        )

    assert all(village_model.body_clear(village, point, radius) == brute_body(point) for point in points)
    assert all(
        village_model.segment_clear(village, start, end, radius)
        == (
            brute_body(start)
            and brute_body(end)
            and not any(
                village_model._segment_hits_shape(start, end, radius, shape)
                for shape in village.collision_shapes
            )
        )
        for start, end in zip(points, points[1:], strict=False)
    )


def test_player_id_predicates_and_geometry_are_stable():
    assert people.is_visitor("player_0")
    assert not people.is_visitor("visitor")
    assert people.is_npc("player_1")
    assert people.is_npc("player_10")
    assert not people.is_npc("player_0")
    assert not people.is_npc("player_01")
    assert not people.is_npc("npc_0")
    assert geometry.distance({"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 4.0}) == 5.0
    assert geometry.heading_to({"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.0}) == 0.0
    assert geometry.heading_to({"x": 0.0, "y": 0.0}, {"x": 0.0, "y": -1.0}) == 270.0
    assert geometry.wrap(-1.0) == 359.0
    assert geometry.in_cone({"x": 0.0, "y": 0.0}, 0.0, {"x": 1.0, "y": 0.0})
    assert not geometry.in_cone({"x": 0.0, "y": 0.0}, 0.0, {"x": 0.0, "y": 1.0}, 90.0)
    assert not geometry.in_cone({"x": 0.0, "y": 0.0}, 0.0, {"x": 2.0, "y": 0.0}, reach=1.0)


def test_rng_is_stable_per_player_and_seed():
    _env, observations = _observations()

    first = me.rng(observations["player_1"], 17).random()
    assert first == me.rng(observations["player_1"], 17).random()
    assert first != me.rng(observations["player_2"], 17).random()
    assert first != me.rng(observations["player_1"], 18).random()


def test_props_preview_real_shape_reach_and_preserve_observation_records():
    _env, observations = _observations()
    observation = copy.deepcopy(observations["player_1"])
    bench = next(prop for prop in props.all(observation) if prop["type"] == "bench")
    observation["self"]["position"] = {"x": bench["cell"]["x"] + 0.5, "y": bench["cell"]["y"] + 0.5}

    assert bench in props.in_reach(observation)
    assert props.usable(observation) is bench
    assert set(props.TYPES) == {prop["type"] for prop in props.all(observation)}


def test_props_state_returns_a_visible_prop_state_or_none():
    _env, observations = _observations()
    observation = observations["player_0"]
    visible_prop = props.seen(observation)[0]

    assert props.state(observation, visible_prop["prop"]) == visible_prop["state"]
    assert props.state(observation, "not-visible") is None


def test_find_helpers_return_records_by_id_and_prefer_sight_over_hearing():
    _env, observations = _observations()
    observation = copy.deepcopy(observations["player_1"])
    prop = props.all(observation)[0]
    assert props.find(observation, prop["id"]) is prop
    assert props.find(observation, "not-a-prop") is None

    seen = {"id": "player_3", "position": {"x": 1.0, "y": 1.0}}
    heard = {"id": "player_4", "position": {"x": 2.0, "y": 2.0}}
    observation["seen"] = (seen,)
    observation["nearby"] = (dict(seen), heard)
    assert people.find(observation, "player_3") is seen
    assert people.find(observation, "player_4") is heard
    assert people.find(observation, "player_9") is None


def test_center_and_floor_helpers_follow_cells_and_building_footprints():
    _env, observations = _observations()
    observation = observations["player_1"]
    assert layout.cell_center(observation, {"x": 3, "y": 7}) == {"x": 3.5, "y": 7.5}

    building = layout.buildings(observation)[0]
    width, height = layout.BUILDING_SIZES[building["type"]]
    x, y = int(building["cell"]["x"]), int(building["cell"]["y"])
    assert layout.building_center(observation, building["id"]) == {"x": x + width / 2, "y": y + height / 2}
    assert layout.building_center(observation, "not-a-building") is None
    assert layout.building_at(observation, {"x": x + 1.5, "y": y + 1.5}) == building["id"]
    assert layout.building_at(observation, {"x": x + 0.5, "y": y + 1.5}) is None  # the wall ring
    assert layout.building_at(observation, {"x": -1.0, "y": 0.0}) is None


def test_nearest_walkable_matches_a_full_scan_of_walkable_cells():
    _env, observations = _observations()
    observation = observations["player_1"]
    walkable = village_model.model(observation).walkable_cells
    blocked_prop = next(
        item for item in props.all(observation) if not layout.walkable(observation, item["cell"])
    )
    points = [
        layout.cell_center(observation, blocked_prop["cell"]),
        *(layout.building_center(observation, item["id"]) for item in layout.buildings(observation)),
        me.position(observation),
    ]
    for point in points:
        found = layout.nearest_walkable(observation, point)
        assert found is not None and layout.walkable(observation, found)
        best = min(geometry.distance({"x": cx + 0.5, "y": cy + 0.5}, point) for cx, cy in walkable)
        assert geometry.distance(layout.cell_center(observation, found), point) == pytest.approx(best)


def test_doorway_returns_none_or_the_nearest_deterministic_multi_cell_run():
    _env, observations = _observations()
    observation = copy.deepcopy(observations["player_1"])
    rows = [["g"] * 120 for _ in range(120)]
    for x, y in ((10, 12), (11, 12), (30, 30), (31, 30)):
        rows[y][x] = "d"
    observation["village"]["ground"] = tuple("".join(row) for row in rows)
    observation["village"]["buildings"] = (
        {"id": "near", "type": "home", "cell": {"x": 10, "y": 10}},
        {"id": "far", "type": "home", "cell": {"x": 30, "y": 30}},
        {"id": "tie", "type": "home", "cell": {"x": 20, "y": 20}},
    )

    assert layout.doorway(observation, "near") == {"x": 11.0, "y": 12.5}
    assert layout.doorway(observation, "far") == {"x": 31.0, "y": 30.5}
    assert layout.doorway(observation, "tie") == {"x": 11.0, "y": 12.5}

    observation["village"]["ground"] = tuple("g" * 120 for _ in range(120))
    assert layout.doorway(observation, "near") is None


def test_importing_helpers_does_not_load_the_engine_stack():
    code = (
        "import sys; from sandbox.village import action, day, geometry, layout, me, people, props; "
        "assert 'gymnasium' not in sys.modules; assert 'pettingzoo' not in sys.modules; "
        "assert 'pymunk' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
