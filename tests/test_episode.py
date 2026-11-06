"""Full-day smoke coverage for the composed Days at Three Branches starter."""

from __future__ import annotations

from time import perf_counter

import agent
import pytest
from sandbox.env import META, make_env
from sandbox.harness.environment import resolve_parameters


@pytest.mark.parametrize(("seat_plan", "player_count"), [("cast_5", 6), ("cast_10", 11)])
def test_starter_agents_complete_a_day_for_each_cast(
    seat_plan: str, player_count: int, record_property
) -> None:
    env = make_env(resolve_parameters(META, {"seat_plan": seat_plan, "daynight": False}))
    observations, _ = env.reset(seed=11)
    agents = {player_id: agent.Agent() for player_id in observations}
    for player_id, example in agents.items():
        example.reset(11, observations[player_id])

    decisions = 0
    decision_seconds = 0.0
    ticks = 0
    while env.agents:
        started = perf_counter()
        actions = {player_id: agents[player_id].act(observations[player_id]) for player_id in env.agents}
        decision_seconds += perf_counter() - started
        decisions += len(actions)
        observations, _, _, _, _ = env.step(actions)
        ticks += 1

    assert ticks == 1200
    assert decisions == 1200 * player_count
    record_property(f"{seat_plan}_decision_ms", round(decision_seconds * 1000, 3))
    assert decision_seconds / decisions < 0.25
