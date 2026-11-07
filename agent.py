"""Optimized lively Days at Three Branches villager.

Architecture
------------
* Finite-state machine for longer-term goals.
* A small priority/behaviour-tree style decision layer in ``act``.
* A static navigation graph built once in ``reset``.
* Cached BFS routes that are recomputed only when a destination changes.
* Temporary social conversations that interrupt, then resume, normal travel.
* Stable ID-based roles so ten villagers running the same code have distinct jobs.
* ``ROLE_CONFIGS`` keeps role parameters separate from shared behavior.
* Reusable behavioral blocks handle travel, socializing, prop use, and returning home.
* Season 5 conversation state plus role-based scripted knowledge.
* Lightweight random prop activities that temporarily interrupt normal work.\n* Season 5 preparation: passive scripted, observed, and remembered knowledge.
* Numbered scripted visitor dialogue through the optional ``chat`` interface.
* Keyword matching maps natural visitor questions onto the same scripted topics.
* Visitor action requests are detected and stored as structured intentions.
* Detected intentions are accepted or refused using the villager's real role/state.
* Accepted ``show_work`` requests temporarily reuse the existing cached navigation.
* Season 5-ready per-villager conversation state, without changing Season 3 behavior.

The main optimization is that expensive layout checks are not repeated on every
simulation tick. The village geometry is static, so walkability and legal
cardinal steps are cached once at the start of the day. Role configuration
changes destinations, pauses, and social parameters while the behavioral blocks
remain shared and lightweight.
"""

from collections import deque
from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props


CellKey = tuple[int, int]


# ---------------------------------------------------------------------------
# Small geometry / cell helpers
# ---------------------------------------------------------------------------


def _cell_centre_from_key(key: CellKey) -> dict[str, float]:
    """Return the centre point of a grid cell key."""

    return {"x": key[0] + 0.5, "y": key[1] + 0.5}


def _cell_key(cell: dict[str, int] | None) -> CellKey | None:
    if cell is None:
        return None
    return (cell["x"], cell["y"])


def _cell_from_key(key: CellKey) -> dict[str, int]:
    return {"x": key[0], "y": key[1]}


def _manhattan_keys(first: CellKey, second: CellKey) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


class Agent:
    """A fast villager that explores, uses props, and socializes."""

    LEAVE_HOME = "LEAVE_HOME"
    MORNING_SWEEP = "MORNING_SWEEP"
    VISIT_PUMP = "VISIT_PUMP"
    EXPLORE = "EXPLORE"
    REST = "REST"
    RETURN_HOME = "RETURN_HOME"
    NAP_GO_HOME = "NAP_GO_HOME"
    NAP_SLEEP = "NAP_SLEEP"

    # One randomly selected NPC takes a mid-day nap. The nap lasts exactly
    # 100 action ticks once the villager is inside its own house.
    NAP_DURATION = 100

    # Stable roles assigned from player ID. Every villager still runs this same
    # Agent class; the role changes parameters and the EXPLORE routine.
    ROLE_WELL_KEEPER = "well_keeper"
    ROLE_WORKER = "worker"
    ROLE_SWEEPER = "sweeper"
    ROLE_WANDERER = "wanderer"

    # Full NPC conversations are intentionally disabled while travelling to the
    # pump. This reduces early-day crowding; passing waves still happen there.
    SOCIALIZABLE_STATES = {EXPLORE, REST}
    VISITOR_SOCIALIZABLE_STATES = {MORNING_SWEEP, VISIT_PUMP, EXPLORE, REST}

    RETURN_HOME_TICK = 1000

    # ------------------------------------------------------------------
    # Role configuration
    # ------------------------------------------------------------------
    # The same behavioral blocks are shared by every role.  These values tune
    # how often and how long each role uses those blocks, keeping jobs visibly
    # different without duplicating travel/social/prop code.
    # ------------------------------------------------------------------
    # Season 5 scripted knowledge
    # ------------------------------------------------------------------
    # These are facts the villager is allowed to know even when it cannot
    # currently see the subject.  They are deliberately grounded in behavior
    # that already exists in this Season 3 agent.  Later conversation code can
    # retrieve these facts by topic instead of inventing answers.
    COMMON_SCRIPTED_KNOWLEDGE = {
        "village": (
            "People in the village follow different daily routines and jobs."
        ),
        "home": (
            "Each resident has a home and normally returns there toward the end of the day."
        ),
        "visitor": (
            "Visitors can meet residents while they are working, travelling, resting, or socializing."
        ),
    }

    ROLE_SCRIPTED_KNOWLEDGE = {
        ROLE_WELL_KEEPER: {
            "role": "I am the well keeper.",
            "work": "I look after the village pump and spend much of my work time near it.",
            "workplace": "My main workplace is the village pump.",
            "routine": "I make short trips around the pump area and regularly return to it.",
        },
        ROLE_WORKER: {
            "role": "I am a village worker.",
            "work": "I travel to a work area and spend long periods working there.",
            "workplace": "I have a regular work area away from my home.",
            "routine": "I move between my work area and home, with pauses between trips.",
        },
        ROLE_SWEEPER: {
            "role": "I help keep the village paths clean.",
            "work": "I patrol outdoor paths and sweep at different stops along my route.",
            "workplace": "My work takes me along several outdoor paths around the village.",
            "routine": "I travel between patrol points and stop to sweep before moving on.",
        },
        ROLE_WANDERER: {
            "role": "I spend much of my day moving around the village.",
            "work": "I do not follow one fixed workplace; I roam, pause, rest, and meet people.",
            "workplace": "I move between different outdoor parts of the village rather than one work site.",
            "routine": "I wander between places, pause for a while, and may rest on a bench.",
        },
    }

    # ------------------------------------------------------------------
    # Season 5 numbered scripted dialogue
    # ------------------------------------------------------------------
    # The visitor uses player_0 in the Three Branches chat interface. Dialogue
    # is deliberately small and deterministic at this stage. It reads the
    # scripted knowledge already attached to the villager but does not affect
    # movement, jobs, social emotes, navigation, or prop behavior.
    VISITOR_ID = "player_0"

    MAIN_DIALOGUE_MENU = (
        "Say 1 for my work; 2 for our day; 3 to ask who I am; 0 to end."
    )

    WORK_DIALOGUE_MENU = (
        "Work topic: say 1 for where I work; 2 for my routine; "
        "9 for the main menu; 0 to end."
    )

    DAY_DIALOGUE_MENU = (
        "Our day: say 1 about the village; 2 about going home; "
        "9 for the main menu; 0 to end."
    )

    ROLE_CONFIGS = {
        ROLE_WELL_KEEPER: {
            "prop_interval": (150, 240),
            "social_chance": 0.38,
            "passing_wave_chance": 0.40,
            "social_cooldown": (55, 85),
            "npc_middle_actions": (2, 4),
            "visitor_middle_actions": (2, 5),
            "keeper_pause": (12, 24),
            "morning_sweep": (14, 22),
        },
        ROLE_WORKER: {
            "prop_interval": (190, 300),
            "social_chance": 0.18,
            "passing_wave_chance": 0.14,
            "social_cooldown": (90, 130),
            "npc_middle_actions": (2, 3),
            "visitor_middle_actions": (2, 4),
            "worker_work_pause": (38, 62),
            "worker_home_pause": (12, 22),
            "morning_sweep": (14, 22),
        },
        ROLE_SWEEPER: {
            "prop_interval": (180, 280),
            "social_chance": 0.28,
            "passing_wave_chance": 0.34,
            "social_cooldown": (75, 115),
            "npc_middle_actions": (2, 4),
            "visitor_middle_actions": (2, 4),
            "sweeper_stop": (12, 20),
            "morning_sweep": (14, 22),
        },
        ROLE_WANDERER: {
            "prop_interval": (100, 180),
            "social_chance": 0.62,
            "passing_wave_chance": 0.42,
            "social_cooldown": (40, 70),
            "npc_middle_actions": (3, 6),
            "visitor_middle_actions": (3, 6),
            "wanderer_dwell": (8, 18),
            "morning_sweep": (14, 22),
        },
    }

    # Social interaction block configuration.  Every conversation begins and
    # ends with a wave.  Middle actions are randomized from conversational
    # emotes only: no ``sweep``, ``sleep`` or ``use``.  Six middle actions at
    # seven ticks each plus two four-tick waves gives an absolute 50-tick cap.
    SOCIAL_RANDOM_EMOTES = (
        "nod",
        "shake_head",
        "point",
        "laugh",
        "shrug",
        "startle",
    )
    SOCIAL_WAVE_TICKS = 4
    SOCIAL_ACTION_TICKS = (4, 7)
    MAX_SOCIAL_TICKS = 50

    # Bound visitor-request travel so a bad target can never trap a villager.
    MAX_INTENTION_TRAVEL_TICKS = 220

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Build durable state and static navigation data once per day."""

        self.rng = me.rng(observation, seed)
        self.player_id = me.player_id(observation)
        self.player_slot = self._player_number(self.player_id)
        self.role = self._role_for(self.player_id)
        self.config = self.ROLE_CONFIGS[self.role]

        # Pick exactly one NPC as the mid-day napper. Every agent receives the
        # same episode seed and roster, so they independently agree on the same
        # selected ID without communicating. A different seed can select a
        # different villager and nap start time on another day.
        npc_ids = sorted(
            person["id"]
            for person in people.roster(observation)
            if people.is_npc(person["id"])
        )
        selector = abs(int(seed)) + 7319
        self.napper_id = npc_ids[selector % len(npc_ids)] if npc_ids else None
        self.is_napper = self.player_id == self.napper_id
        self.nap_start_tick = 450 + ((abs(int(seed)) + 1973) % 201)
        self.nap_ticks = 0
        self.nap_done = False

        # Main finite-state machine.
        self.state = self.LEAVE_HOME
        self.state_started_tick = day.tick(observation)

        # Navigation memory. Route stores cell keys and uses popleft(), avoiding
        # the O(n) list.pop(0) operation used by the earlier implementation.
        self.destination_key: CellKey | None = None
        self.route: deque[CellKey] = deque()
        self.expected_to_move = False
        self.stuck_ticks = 0
        self.last_roam_key: CellKey | None = None
        self.visited_places: set[CellKey] = set()
        self.dwell_until = 0

        # Static village geometry.
        self.frame = layout.frame(observation)
        self.walkable_keys: set[CellKey] = set()
        self.ground_by_key: dict[CellKey, str | None] = {}
        self.nav_neighbors: dict[CellKey, tuple[CellKey, ...]] = {}
        self.component_of: dict[CellKey, int] = {}
        self.outdoor_by_component: dict[int, list[CellKey]] = {}
        self._build_navigation_graph(observation)
        self._build_components()

        # Home information is static, so calculate it once.
        self.home_id = me.home(observation)
        self.home_door = (
            layout.doorway(observation, self.home_id)
            if self.home_id != "none"
            else None
        )
        self.home_door_key = _cell_key(
            layout.cell_at(observation, self.home_door)
            if self.home_door is not None
            else None
        )
        self.home_inside_key = self._find_home_interior_key(self.home_door_key)

        # Props and their static approach cells. Keep all prop records because
        # villagers can occasionally choose one as a temporary activity.
        self.all_props = list(props.all(observation))

        self.pump = next((p for p in self.all_props if p["type"] == "pump"), None)
        self.benches = [p for p in self.all_props if p["type"] == "bench"]

        # Cache approach cells for every prop ONCE. Random prop use therefore
        # never needs repeated layout.walkable/can_step calls during act().
        self.prop_approaches: dict[str, tuple[CellKey, ...]] = {
            prop["id"]: self._prop_approach_keys(prop["cell"])
            for prop in self.all_props
        }

        self.pump_approaches: tuple[CellKey, ...] = ()
        if self.pump is not None:
            self.pump_approaches = self._prop_approach_keys(self.pump["cell"])

        self.bench_approaches: dict[str, tuple[CellKey, ...]] = {
            bench["id"]: self.prop_approaches.get(bench["id"], ())
            for bench in self.benches
        }

        # Pump goal state.
        self.pump_target_key: CellKey | None = None
        self.pump_use_ticks = 0
        self.pump_wait_ticks = 0
        self.pump_used = False

        # Bench/rest state.
        self.rest_bench: dict | None = None
        self.rest_target_key: CellKey | None = None
        self.rest_ticks = 0
        self.rest_attempt_ticks = 0
        self.rest_duration = self.rng.randint(18, 30)
        self.rest_failed_ids: set[str] = set()
        self.has_rested = False

        # Temporary random prop activity. This is a behavior-tree interrupt,
        # not a new long-term FSM state. The agent visits a reachable prop,
        # uses it briefly, then resumes its role routine.
        self.random_prop: dict | None = None
        self.random_prop_target_key: CellKey | None = None
        self.random_prop_use_ticks = 0
        self.random_prop_attempt_ticks = 0
        self.random_prop_use_duration = 0
        self.last_random_prop_id: str | None = None
        self.next_prop_activity_tick = day.tick(observation) + self._next_prop_interval()

        # Role-specific routine state. Sweepers all begin with a short visible
        # sweep, then continue a path-maintenance patrol during the day.
        self.is_morning_sweeper = self.role == self.ROLE_SWEEPER
        self.morning_sweep_done = False
        self.sweep_ticks = 0
        sweep_low, sweep_high = self.config.get("morning_sweep", (14, 22))
        self.sweep_duration = self.rng.randint(sweep_low, sweep_high)
        self.sweep_heading = me.heading(observation)

        self.role_phase = "start"
        self.role_pause_until = 0
        self.work_target_key: CellKey | None = None
        self.work_cycles = 0
        self.sweep_route_keys: list[CellKey] = []
        self.sweep_route_index = 0
        self.sweep_stop_until = 0
        self.keeper_anchor_key: CellKey | None = None
        self.keeper_patrol_keys: list[CellKey] = []
        self.keeper_next_key: CellKey | None = None
        self._initialize_role_behavior()

        # Social state. A conversation interrupts behavior but does not destroy
        # the route/destination the villager was following.
        self.social_partner: str | None = None
        self.social_kind: str | None = None
        self.social_point_key: CellKey | None = None
        self.social_sequence: tuple[tuple[str, int], ...] = ()
        self.social_stage = 0
        self.social_stage_tick = 0
        self.social_cooldowns: dict[str, int] = {}
        # Stagger early conversations so villagers first establish their jobs.
        self.social_block_until = self.rng.randint(20, 55)
        self.last_passing_wave_tick = -999

        # --------------------------------------------------------------
        # Season 5 conversation state
        # --------------------------------------------------------------
        # This state is intentionally separate from the Season 3 emote-based
        # social system above. It will later support visitor dialogue without
        # changing the working movement, role, or social behavior yet. Because
        # every villager has its own Agent instance, these values are private to
        # that villager.
        self.in_conversation: bool = False
        self.conversation_partner: str | None = None
        self.conversation_topic: str = "main"
        self.last_visitor_message: str = ""
        self.conversation_started_tick: int | None = None

        # Season 5 intention state. An intention represents something the visitor
        # is asking this villager to DO, rather than information the visitor is
        # asking to KNOW. This version detects, stores, and decides whether to
        # accept/refuse the request. Accepted requests can temporarily use the
        # existing travel block without replacing the normal role/FSM state.
        self.pending_intention: dict[str, object] | None = None
        self.last_intention_type: str | None = None

        # Temporary execution state for an accepted visitor request.
        self.intention_target_key: CellKey | None = None
        self.intention_started_tick: int | None = None
        self.intention_attempt_ticks = 0

        # Scripted knowledge is the first information source for Season 5
        # dialogue. Every villager receives a private dictionary so later memory
        # updates cannot accidentally modify another villager or the class-level
        # templates. Nothing reads this during act() yet, so gameplay is unchanged.
        self.scripted_knowledge: dict[str, str] = self._build_scripted_knowledge()

        # Observed knowledge is a CURRENT perception snapshot. It is kept
        # separate from scripted knowledge and from the memory system we will
        # add later. The values below are refreshed from self._seen each tick,
        # so the villager only claims to see what its normal perception reports.
        self.observed_knowledge: dict[str, object] = {
            "tick": day.tick(observation),
            "visible_people": (),
            "visible_npcs": (),
            "visible_visitors": (),
            "visible_positions": {},
            "visible_expressions": {},
        }

        # Season 5 simple memory. Unlike observed_knowledge, which describes only
        # what is visible RIGHT NOW, this dictionary keeps selected facts from
        # earlier observations. It is passive for now: no movement, social, or
        # role behavior reads it yet.
        #
        # people[person_id] stores:
        #   kind            -> "npc", "visitor", or "person"
        #   first_seen_tick -> first tick this villager saw the person
        #   last_seen_tick  -> most recent tick this villager saw the person
        #   last_position   -> last observed position
        #   last_expression -> last observed expression type
        #   times_seen      -> number of perception snapshots containing them
        self.memory: dict[str, object] = {
            "people": {},
            "ever_seen_npcs": set(),
            "ever_seen_visitors": set(),
        }

        # Per-tick cached perception values. They are refreshed at the start of
        # every act() and then reused by all behavior methods.
        self._tick = day.tick(observation)
        self._heading = me.heading(observation)
        self._here = me.position(observation)
        self._here_key = _cell_key(layout.cell_at(observation, self._here))
        self._seen: list[dict] = []
        self._nearby: list[dict] = []
        self._usable: dict | None = None
        self._expression: dict = {"type": "none", "target": "none"}

    # ------------------------------------------------------------------
    # Main decision loop: behaviour-tree style priority order
    # ------------------------------------------------------------------

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Choose one action from current perception plus durable state."""

        self._cache_tick(observation)

        # Update stuck detection from movement requested on the previous tick.
        if self.expected_to_move:
            if me.moved(observation) < 0.01:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0
        else:
            self.stuck_ticks = 0
        self.expected_to_move = False

        # Highest strategic priority: leave enough time to get home.
        if self._tick >= self.RETURN_HOME_TICK and self.state != self.RETURN_HOME:
            self._cancel_active_intention("It is time to return home.")
            self._finish_social(self._tick)
            self._cancel_random_prop_activity(schedule_next=False)
            self._set_state(self.RETURN_HOME, observation)

        # Mid-day nap interrupt. Exactly one selected NPC leaves its current
        # activity, returns to its own house, sleeps for 100 ticks, then resumes
        # its normal role. Strategic end-of-day return-home still outranks this.
        if (
            self.is_napper
            and not self.nap_done
            and self._tick >= self.nap_start_tick
            and self.state not in {self.NAP_GO_HOME, self.NAP_SLEEP, self.RETURN_HOME}
        ):
            self._cancel_active_intention("It is time for the scheduled nap.")
            self._finish_social(self._tick)
            self._cancel_random_prop_activity(schedule_next=False)
            self._prepare_role_after_nap()
            self._set_state(self.NAP_GO_HOME, observation)

        # Priority 1: execute an accepted visitor request. Strategic return-home
        # and nap transitions above still outrank it.
        intention_action = self._act_pending_intention(observation)
        if intention_action is not None:
            return intention_action

        # Priority 2: finish an existing emote conversation.
        if self.social_partner is not None:
            social_action = self._block_socialize(observation)
            if social_action is not None:
                return social_action

        # Priority 3: recover from repeated failed movement.
        if self.stuck_ticks >= 5:
            recovery = self._recover_from_stuck()
            if recovery is not None:
                return recovery

        # Priority 4: visitor interaction.
        if (
            self._tick >= self.social_block_until
            and self.state in self.VISITOR_SOCIALIZABLE_STATES
        ):
            visitor = self._choose_visitor_partner()
            if visitor is not None:
                self._start_social(visitor["id"], "visitor")
                social_action = self._block_socialize(observation)
                if social_action is not None:
                    return social_action

        # Priority 5: continue an already selected random prop activity. The
        # visitor can still interrupt it because visitor interaction is checked
        # first; once that conversation ends, the prop trip resumes.
        if self.random_prop is not None:
            prop_action = self._act_random_prop(observation)
            if prop_action is not None:
                return prop_action

        # Priority 6: NPC-to-NPC interaction. This is not allowed at the pump,
        # which avoids large early-day clusters of stationary villagers.
        if (
            self._tick >= self.social_block_until
            and self.state in self.SOCIALIZABLE_STATES
        ):
            partner = self._choose_social_partner()
            if partner is not None:
                self._start_social(partner["id"], "npc")
                social_action = self._block_socialize(observation)
                if social_action is not None:
                    return social_action

        # Priority 7: occasionally begin a reachable random prop activity.
        # This runs only in EXPLORE and only when the current role is at a safe
        # interruption point, so jobs remain visually recognizable.
        if self._maybe_start_random_prop_activity():
            prop_action = self._act_random_prop(observation)
            if prop_action is not None:
                return prop_action

        # Priority 8: execute the main FSM state.
        if self.state == self.NAP_GO_HOME:
            return self._act_nap_go_home(observation)
        if self.state == self.NAP_SLEEP:
            return self._act_nap_sleep(observation)
        if self.state == self.LEAVE_HOME:
            return self._act_leave_home(observation)
        if self.state == self.MORNING_SWEEP:
            return self._act_morning_sweep(observation)
        if self.state == self.VISIT_PUMP:
            return self._act_visit_pump(observation)
        if self.state == self.REST:
            return self._act_rest(observation)
        if self.state == self.RETURN_HOME:
            return self._act_return_home(observation)

        if self.state != self.EXPLORE:
            self._set_state(self.EXPLORE, observation)
        return self._act_explore(observation)

    def _cache_tick(self, observation: ThreeBranchesObservation) -> None:
        """Read commonly used observation values once per act() call."""

        self._tick = day.tick(observation)
        self._heading = me.heading(observation)
        self._here = me.position(observation)
        self._here_key = _cell_key(layout.cell_at(observation, self._here))
        self._seen = people.seen(observation)
        self._update_observed_knowledge()
        self._remember_current_observations()

        # Hearing/proximity data is only needed when a conversation can start
        # or is already in progress. Avoid requesting it on unrelated ticks.
        needs_nearby = (
            self.social_partner is not None
            or (
                self._tick >= self.social_block_until
                and (
                    self.state in self.SOCIALIZABLE_STATES
                    or self.state in self.VISITOR_SOCIALIZABLE_STATES
                )
            )
        )
        self._nearby = people.nearby(observation) if needs_nearby else []

        # Prop selection/expression state is only relevant while using the pump,
        # a bench, or a temporary random prop. Ordinary exploration stays cheap.
        if self.state in {self.VISIT_PUMP, self.REST} or self.random_prop is not None:
            self._usable = props.usable(observation)
            self._expression = me.expression(observation)
        else:
            self._usable = None
            self._expression = {"type": "none", "target": "none"}

    # ------------------------------------------------------------------
    # Shared behavioral blocks
    # ------------------------------------------------------------------

    def _block_use_prop(self, prop_id: str) -> ThreeBranchesAction | None:
        """Use one intended prop when the engine says it is the usable target."""

        if self._usable is not None and self._usable.get("id") == prop_id:
            return self._stand(self._heading, "use")
        return None

    def _block_return_home(self) -> tuple[ThreeBranchesAction | None, str]:
        """Shared return-home block.

        Returns ``(order, status)`` where status is ``"running"``, ``"arrived"``,
        or ``"failed"``.  Mid-day naps and end-of-day return both reuse this
        exact behavior.
        """

        if self._here_key is None:
            return self._stand(self._heading, "none"), "running"

        if self.ground_by_key.get(self._here_key) == "interior":
            return None, "arrived"

        if self.home_door is None or self.home_door_key is None:
            return self._stand(self._heading, "none"), "failed"

        if self._here_key != self.home_door_key:
            move = self._block_travel_to(self.home_door_key)
            if move is not None:
                return move, "running"

        if self.home_inside_key is not None and self._here_key != self.home_inside_key:
            move = self._block_travel_to(self.home_inside_key)
            if move is not None:
                return move, "running"

        if self.home_inside_key is not None:
            return (
                self._walk(
                    geometry.heading_to(
                        self._here,
                        _cell_centre_from_key(self.home_inside_key),
                    ),
                    0.55,
                    "none",
                ),
                "running",
            )

        return (
            self._walk(
                geometry.heading_to(self._here, self.home_door),
                0.45,
                "none",
            ),
            "running",
        )

    # ------------------------------------------------------------------
    # FSM states
    # ------------------------------------------------------------------

    def _act_leave_home(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        if self._here_key is None:
            return self._stand(self._heading, "none")

        if self.ground_by_key.get(self._here_key) != "interior":
            # Roles diverge as soon as villagers leave home. Only the well
            # keeper begins by tending the pump; sweepers start their visible
            # chore, while workers and wanderers begin their own routines.
            if self.role == self.ROLE_SWEEPER and not self.morning_sweep_done:
                self._set_state(self.MORNING_SWEEP, observation)
                return self._act_morning_sweep(observation)
            if self.role == self.ROLE_WELL_KEEPER:
                self._set_state(self.VISIT_PUMP, observation)
                return self._act_visit_pump(observation)
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        if self.home_door is None or self.home_door_key is None:
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        if self._here_key != self.home_door_key:
            move = self._block_travel_to(self.home_door_key)
            if move is not None:
                return move

        # At/near the doorway but still classified as interior: move directly
        # toward the doorway until the ground changes.
        return self._walk(
            geometry.heading_to(self._here, self.home_door),
            0.75,
            self._passing_expression(),
        )

    def _act_morning_sweep(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Let the designated villager sweep briefly after leaving home."""

        if not self.is_morning_sweeper or self.morning_sweep_done:
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        self.sweep_ticks += 1

        if self.sweep_ticks == 1:
            self.sweep_heading = self._heading
        elif self.sweep_ticks % 9 == 0:
            self.sweep_heading = geometry.wrap(
                self.sweep_heading + self.rng.choice((-70.0, -40.0, 40.0, 70.0))
            )

        if self.sweep_ticks >= self.sweep_duration:
            self.morning_sweep_done = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        return self._stand(self.sweep_heading, "sweep")

    def _act_visit_pump(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        if self.pump is None:
            self.pump_used = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        # Count confirmed use from the previous tick.
        if self._expression.get("target") == self.pump["id"]:
            self.pump_use_ticks += 1

        if self.pump_use_ticks >= 5:
            self.pump_used = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        use_action = self._block_use_prop(self.pump["id"])
        if use_action is not None:
            self.pump_wait_ticks = 0
            return use_action

        # Choose the approach cell ONCE instead of recalculating it every tick.
        if self.pump_target_key is None:
            self.pump_target_key = self._choose_approach_key(self.pump_approaches)
            if self.pump_target_key is not None:
                self._set_destination_key(self.pump_target_key)

        if self.pump_target_key is None:
            self.pump_wait_ticks += 1
            if self.pump_wait_ticks > 20:
                self.pump_used = True
                self._set_state(self.EXPLORE, observation)
                return self._act_explore(observation)
            return self._stand(self._heading, "none")

        if self._here_key == self.pump_target_key:
            # Edge toward the prop until it becomes usable. This does not count
            # as expected movement because the collision shape may stop us.
            self.expected_to_move = False
            self.pump_wait_ticks += 1
            if self.pump_wait_ticks > 55:
                self.pump_used = True
                self._set_state(self.EXPLORE, observation)
                return self._act_explore(observation)

            pump_key = _cell_key(self.pump["cell"])
            if pump_key is None:
                return self._stand(self._heading, "none")
            return action.walk(
                geometry.heading_to(self._here, _cell_centre_from_key(pump_key)),
                0.25,
                self._passing_expression(),
            )

        move = self._block_travel_to(self.pump_target_key)
        if move is not None:
            return move

        # Static route unexpectedly failed; choose another approach once.
        self.pump_target_key = self._choose_approach_key(
            self.pump_approaches,
            exclude=self.pump_target_key,
        )
        self._clear_navigation(keep_destination=False)
        self.pump_wait_ticks += 1
        if self.pump_wait_ticks > 30:
            self.pump_used = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)
        return self._stand(self._heading, "none")

    def _act_explore(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Run the visible daily routine associated with this villager's role."""

        if self.role == self.ROLE_WELL_KEEPER:
            return self._act_well_keeper(observation)
        if self.role == self.ROLE_WORKER:
            return self._act_worker(observation)
        if self.role == self.ROLE_SWEEPER:
            return self._act_sweeper(observation)
        return self._act_wanderer(observation)

    def _act_well_keeper(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Stay near the pump, make short local trips, and return quickly."""

        if not self.keeper_patrol_keys:
            return self._stand(self._heading, "none")

        if self._tick < self.role_pause_until:
            return self._stand(self._heading, "none")

        if self.destination_key is None:
            # Most trips return to the pump-side anchor. Other trips are only a
            # few cells away, making the role visually pump-centered.
            if self.rng.random() < 0.60 and self.keeper_anchor_key is not None:
                target = self.keeper_anchor_key
            else:
                target = self.rng.choice(self.keeper_patrol_keys)
            self._set_destination_key(target)

        if self._here_key == self.destination_key:
            self._clear_navigation(keep_destination=False)
            self.role_pause_until = self._tick + self.rng.randint(*self.config.get("keeper_pause", (12, 24)))
            return self._stand(self._heading, "none")

        move = self._block_travel_to(self.destination_key)
        if move is not None:
            return move

        self._clear_navigation(keep_destination=False)
        return self._stand(self._heading, "none")

    def _act_worker(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Make regular home/work round trips with long visible work periods."""

        if self.work_target_key is None:
            return self._act_wanderer(observation)

        if self.role_phase == "start":
            self.role_phase = "to_work"
            self._set_destination_key(self.work_target_key)

        if self.role_phase == "working":
            if self._tick < self.role_pause_until:
                # A small occasional point makes the long work period visible
                # without creating another costly state machine.
                expression = "point" if (self._tick - self.state_started_tick) % 24 == 0 else "none"
                return self._stand(self._heading, expression)
            self.role_phase = "to_home"
            if self.home_door_key is not None:
                self._set_destination_key(self.home_door_key)
            else:
                self.role_phase = "to_work"
                self._set_destination_key(self.work_target_key)

        if self.role_phase == "home_pause":
            if self._tick < self.role_pause_until:
                return self._stand(self._heading, "none")
            self.role_phase = "to_work"
            self._set_destination_key(self.work_target_key)

        if self.role_phase == "to_work" and self._here_key == self.work_target_key:
            self.work_cycles += 1
            self.role_phase = "working"
            self.role_pause_until = self._tick + self.rng.randint(*self.config.get("worker_work_pause", (38, 62)))
            self._clear_navigation(keep_destination=False)
            return self._stand(self._heading, "none")

        if (
            self.role_phase == "to_home"
            and self.home_door_key is not None
            and self._here_key == self.home_door_key
        ):
            self.role_phase = "home_pause"
            self.role_pause_until = self._tick + self.rng.randint(*self.config.get("worker_home_pause", (12, 22)))
            self._clear_navigation(keep_destination=False)
            return self._stand(self._heading, "none")

        target = self.work_target_key if self.role_phase == "to_work" else self.home_door_key
        if target is None:
            self.role_phase = "to_work"
            target = self.work_target_key

        move = self._block_travel_to(target)
        if move is not None:
            return move

        # If a route unexpectedly fails, choose a fresh work spot cheaply.
        self.work_target_key = self._choose_worker_spot()
        self.role_phase = "to_work"
        self._clear_navigation(keep_destination=False)
        return self._stand(self._heading, "shrug")

    def _act_sweeper(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Patrol a repeating path and sweep at each waypoint."""

        if not self.sweep_route_keys:
            return self._act_wanderer(observation)

        if self.sweep_stop_until > self._tick:
            # Turn slightly during a long sweep so the animation looks active.
            if self._tick % 8 == 0:
                self.sweep_heading = geometry.wrap(
                    self.sweep_heading + self.rng.choice((-45.0, 45.0))
                )
            return self._stand(self.sweep_heading, "sweep")

        target = self.sweep_route_keys[self.sweep_route_index]
        if self._here_key == target:
            self.sweep_heading = self._heading
            self.sweep_stop_until = self._tick + self.rng.randint(*self.config.get("sweeper_stop", (12, 20)))
            self.sweep_route_index = (self.sweep_route_index + 1) % len(self.sweep_route_keys)
            self._clear_navigation(keep_destination=False)
            return self._stand(self.sweep_heading, "sweep")

        if self.destination_key != target:
            self._set_destination_key(target)

        move = self._block_travel_to(target)
        if move is not None:
            return move

        # Skip a waypoint if something unusual prevents reaching it.
        self.sweep_route_index = (self.sweep_route_index + 1) % len(self.sweep_route_keys)
        self._clear_navigation(keep_destination=False)
        return self._stand(self._heading, "shrug")

    def _act_wanderer(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Roam widely, pause often, rest once, and socialize readily."""

        if (
            not self.has_rested
            and self.benches
            and len(self.visited_places) >= 2
            and self._tick < self.RETURN_HOME_TICK - 150
        ):
            self._set_state(self.REST, observation)
            return self._act_rest(observation)

        if self._tick < self.dwell_until:
            return self._stand(self._heading, "none")

        if self.destination_key is None:
            destination = self._choose_roam_destination()
            if destination is None:
                return self._stand(self._heading, "none")
            self._set_destination_key(destination)

        if self._here_key == self.destination_key:
            if self._here_key is not None:
                self.visited_places.add(self._here_key)
                self.last_roam_key = self._here_key
            self._clear_navigation(keep_destination=False)
            self.dwell_until = self._tick + self.rng.randint(*self.config.get("wanderer_dwell", (8, 18)))
            return self._stand(self._heading, "none")

        move = self._block_travel_to(self.destination_key)
        if move is not None:
            return move

        self._clear_navigation(keep_destination=False)
        return self._stand(self._heading, "shrug")

    def _act_rest(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        if not self.benches:
            self.has_rested = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        if self.rest_bench is None:
            self.rest_bench, self.rest_target_key = self._choose_bench_and_approach()
            self.rest_ticks = 0
            self.rest_attempt_ticks = 0
            self._clear_navigation(keep_destination=False)
            if self.rest_target_key is not None:
                self._set_destination_key(self.rest_target_key)

        if self.rest_bench is None or self.rest_target_key is None:
            self.has_rested = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        if self._expression.get("target") == self.rest_bench["id"]:
            self.rest_ticks += 1

        if self.rest_ticks >= self.rest_duration:
            self.has_rested = True
            self.rest_bench = None
            self.rest_target_key = None
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        use_action = self._block_use_prop(self.rest_bench["id"])
        if use_action is not None:
            self.rest_attempt_ticks += 1
            if self.rest_attempt_ticks > self.rest_duration + 45:
                return self._abandon_current_bench(observation)
            return use_action

        if self._here_key == self.rest_target_key:
            self.rest_attempt_ticks += 1
            if self.rest_attempt_ticks > 38:
                return self._abandon_current_bench(observation)

            bench_key = _cell_key(self.rest_bench["cell"])
            if bench_key is None:
                return self._abandon_current_bench(observation)

            self.expected_to_move = False
            return action.walk(
                geometry.heading_to(self._here, _cell_centre_from_key(bench_key)),
                0.20,
                "none",
            )

        move = self._block_travel_to(self.rest_target_key)
        if move is not None:
            return move

        return self._abandon_current_bench(observation)

    def _abandon_current_bench(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction:
        if self.rest_bench is not None:
            self.rest_failed_ids.add(self.rest_bench["id"])

        self.rest_bench = None
        self.rest_target_key = None
        self.rest_attempt_ticks = 0
        self._clear_navigation(keep_destination=False)

        # Let the next tick try another bench; if all benches have failed,
        # finish the rest goal and continue exploring.
        if len(self.rest_failed_ids) >= len(self.benches):
            self.has_rested = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        return self._stand(self._heading, "none")

    # ------------------------------------------------------------------
    # Temporary random prop activity
    # ------------------------------------------------------------------

    def _next_prop_interval(self) -> int:
        """Return the configured delay before the next shared prop-use block."""

        low, high = self.config.get("prop_interval", (160, 260))
        return self.rng.randint(low, high)

    def _can_start_random_prop_activity(self) -> bool:
        """Whether the current job can safely be interrupted by a prop visit."""

        if self.random_prop is not None:
            return False
        if self.state != self.EXPLORE:
            return False
        if self._tick < self.next_prop_activity_tick:
            return False
        if self._tick >= self.RETURN_HOME_TICK - 100:
            return False
        if not self.all_props or self._here_key is None:
            return False

        # Preserve clear role silhouettes: workers finish visible work stretches
        # and sweepers finish an active sweep before wandering off to a prop.
        if self.role == self.ROLE_WORKER and self.role_phase in {"working", "home_pause"}:
            return False
        if self.role == self.ROLE_SWEEPER and self.sweep_stop_until > self._tick:
            return False

        return True

    def _maybe_start_random_prop_activity(self) -> bool:
        """Choose one reachable prop using only cached connectivity data."""

        if not self._can_start_random_prop_activity():
            return False

        component = self.component_of.get(self._here_key)
        if component is None:
            self.next_prop_activity_tick = self._tick + self._next_prop_interval()
            return False

        candidates: list[tuple[dict, tuple[CellKey, ...]]] = []
        for prop in self.all_props:
            if prop["id"] == self.last_random_prop_id:
                continue
            approaches = tuple(
                key
                for key in self.prop_approaches.get(prop["id"], ())
                if self.component_of.get(key) == component
            )
            if approaches:
                candidates.append((prop, approaches))

        # If the village has only one reachable prop, allow using it again.
        if not candidates and self.last_random_prop_id is not None:
            for prop in self.all_props:
                approaches = tuple(
                    key
                    for key in self.prop_approaches.get(prop["id"], ())
                    if self.component_of.get(key) == component
                )
                if approaches:
                    candidates.append((prop, approaches))

        if not candidates:
            self.next_prop_activity_tick = self._tick + self._next_prop_interval()
            return False

        prop, approaches = self.rng.choice(candidates)

        # Prefer one of the nearest few approach cells, but randomize within the
        # shortlist so agents do not all approach the same side of a prop.
        ordered = sorted(
            approaches,
            key=lambda key: _manhattan_keys(self._here_key, key),
        )
        shortlist = ordered[: min(3, len(ordered))]
        target = self.rng.choice(shortlist)

        self.random_prop = prop
        self.random_prop_target_key = target
        self.random_prop_use_ticks = 0
        self.random_prop_attempt_ticks = 0
        self.random_prop_use_duration = self.rng.randint(4, 10)
        self._set_destination_key(target)
        return True

    def _act_random_prop(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction | None:
        """Travel to the selected prop, use it briefly, then resume the role."""

        if self.random_prop is None or self.random_prop_target_key is None:
            self._cancel_random_prop_activity()
            return None

        # A strategic state change cancels this lightweight side activity.
        if self.state != self.EXPLORE:
            self._cancel_random_prop_activity()
            return None

        prop_id = self.random_prop["id"]

        # Count confirmed use from the previous tick.
        if self._expression.get("target") == prop_id:
            self.random_prop_use_ticks += 1

        if self.random_prop_use_ticks >= self.random_prop_use_duration:
            self.last_random_prop_id = prop_id
            self._finish_random_prop_activity()
            return None

        # props.usable() previews the prop the engine will actually select. Only
        # issue use when it matches our intended target.
        use_action = self._block_use_prop(prop_id)
        if use_action is not None:
            self.random_prop_attempt_ticks += 1
            if self.random_prop_attempt_ticks > self.random_prop_use_duration + 30:
                self.last_random_prop_id = prop_id
                self._finish_random_prop_activity()
                return None
            return use_action

        # Walk to the cached approach cell using the normal cached-route system.
        if self._here_key != self.random_prop_target_key:
            move = self._block_travel_to(self.random_prop_target_key)
            if move is not None:
                return move

            # Route unexpectedly failed. Abandon rather than performing repeated
            # expensive searches or blocking the villager's normal routine.
            self.last_random_prop_id = prop_id
            self._finish_random_prop_activity()
            return None

        # At the approach cell but not quite within the prop collision reach.
        # Creep toward the prop for a bounded number of ticks.
        self.random_prop_attempt_ticks += 1
        if self.random_prop_attempt_ticks > 28:
            self.last_random_prop_id = prop_id
            self._finish_random_prop_activity()
            return None

        prop_key = _cell_key(self.random_prop["cell"])
        if prop_key is None:
            self._finish_random_prop_activity()
            return None

        self.expected_to_move = False
        return action.walk(
            geometry.heading_to(self._here, _cell_centre_from_key(prop_key)),
            0.20,
            "none",
        )

    def _finish_random_prop_activity(self) -> None:
        """Clear the temporary prop goal and schedule the next opportunity."""

        self.random_prop = None
        self.random_prop_target_key = None
        self.random_prop_use_ticks = 0
        self.random_prop_attempt_ticks = 0
        self.random_prop_use_duration = 0
        self._clear_navigation(keep_destination=False)
        self.next_prop_activity_tick = self._tick + self._next_prop_interval()

    def _cancel_random_prop_activity(self, schedule_next: bool = True) -> None:
        """Cancel a prop trip, normally because a strategic state took over."""

        self.random_prop = None
        self.random_prop_target_key = None
        self.random_prop_use_ticks = 0
        self.random_prop_attempt_ticks = 0
        self.random_prop_use_duration = 0
        self._clear_navigation(keep_destination=False)
        if schedule_next:
            self.next_prop_activity_tick = self._tick + self._next_prop_interval()

    def _act_nap_go_home(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction:
        """Use the shared return-home block, then begin the 100-tick nap."""

        order, status = self._block_return_home()

        if status == "arrived":
            self.nap_ticks = 0
            self._set_state(self.NAP_SLEEP, observation)
            return self._stand(self._heading, "sleep")

        if status == "failed":
            self.nap_done = True
            self._set_state(self.EXPLORE, observation)
            return self._act_explore(observation)

        return order if order is not None else self._stand(self._heading, "none")

    def _act_nap_sleep(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction:
        """Sleep inside the house for exactly NAP_DURATION ticks."""

        self.nap_ticks += 1

        if self.nap_ticks < self.NAP_DURATION:
            return self._stand(self._heading, "sleep")

        # Wake up and leave the house. _prepare_role_after_nap() already reset
        # any role-specific phase that needed it, so normal LEAVE_HOME logic can
        # send the villager back to the correct job.
        self.nap_done = True
        self.nap_ticks = 0
        self._set_state(self.LEAVE_HOME, observation)
        return self._act_leave_home(observation)

    def _prepare_role_after_nap(self) -> None:
        """Prepare role memory so the villager resumes sensible work after waking."""

        # Workers should commute back to their work site rather than resuming a
        # stale pause/home leg from before the nap.
        if self.role == self.ROLE_WORKER:
            self.role_phase = "to_work"
            self.role_pause_until = 0

        # Well keepers will naturally return to the pump through LEAVE_HOME.
        # Sweepers keep their patrol index, and wanderers simply choose another
        # destination when they come outside.
        self.dwell_until = 0
        self.role_pause_until = 0
        self.sweep_stop_until = 0
        self._clear_navigation(keep_destination=False)

    def _act_return_home(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Use the same return-home block as the mid-day nap."""

        order, status = self._block_return_home()

        if status == "arrived":
            return self._stand(self._heading, "sleep")

        return order if order is not None else self._stand(self._heading, "none")


    # ------------------------------------------------------------------
    # Social behavior
    # ------------------------------------------------------------------

    def _choose_visitor_partner(self) -> dict | None:
        visible_ids = {person["id"] for person in self._seen}
        candidates: list[tuple[float, dict]] = []

        for person in self._nearby:
            pid = person["id"]
            if not people.is_visitor(pid):
                continue
            if self._tick < self.social_cooldowns.get(pid, -1):
                continue
            if pid not in visible_ids:
                continue
            candidates.append((geometry.distance(self._here, person["position"]), person))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _choose_social_partner(self) -> dict | None:
        if not self._nearby:
            return None

        visible_by_id = {person["id"]: person for person in self._seen}
        candidates: list[tuple[int, float, str, dict]] = []

        for person in self._nearby:
            pid = person["id"]
            if not people.is_npc(pid) or pid == self.player_id:
                continue
            if self._tick < self.social_cooldowns.get(pid, -1):
                continue

            visible = visible_by_id.get(pid)
            if visible is None:
                continue

            expression = visible.get("expression", {})
            expression_type = expression.get("type", "none")
            responding = 0 if expression_type in {"wave", "nod", "shrug", "laugh"} else 1
            distance = geometry.distance(self._here, person["position"])
            candidates.append((responding, distance, pid, person))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best = candidates[0]

        # Role configuration controls sociability while the shared social
        # behavioral block remains identical for every villager.
        social_chance = self.config.get("social_chance", 0.30)

        if best[0] == 1 and self.rng.random() >= social_chance:
            return None

        return best[3]

    def _start_social(self, partner_id: str, kind: str) -> None:
        """Prepare the shared social behavioral block for one partner."""

        self.social_partner = partner_id
        self.social_kind = kind
        self.social_stage = 0
        self.social_stage_tick = 0
        self.social_point_key = self._choose_point_target()
        self.social_sequence = self._build_social_sequence(kind)

    def _build_social_sequence(self, kind: str) -> tuple[tuple[str, int], ...]:
        """Create wave -> random emotes -> wave, always capped at 50 ticks."""

        config_key = (
            "visitor_middle_actions"
            if kind == "visitor"
            else "npc_middle_actions"
        )
        low, high = self.config.get(config_key, (2, 4))

        # With 4-tick bookend waves and at most 7 ticks per middle action,
        # six middle actions is the largest sequence that can fit in 50 ticks.
        max_middle = max(
            0,
            (self.MAX_SOCIAL_TICKS - 2 * self.SOCIAL_WAVE_TICKS)
            // self.SOCIAL_ACTION_TICKS[1],
        )
        high = min(high, max_middle)
        low = min(low, high)
        middle_count = self.rng.randint(low, high) if high > 0 else 0

        middle: list[tuple[str, int]] = []
        previous: str | None = None

        for _ in range(middle_count):
            choices = [
                emote
                for emote in self.SOCIAL_RANDOM_EMOTES
                if emote != previous
            ]
            emote = self.rng.choice(choices)
            duration = self.rng.randint(*self.SOCIAL_ACTION_TICKS)
            middle.append((emote, duration))
            previous = emote

        sequence = (
            (("wave", self.SOCIAL_WAVE_TICKS),)
            + tuple(middle)
            + (("wave", self.SOCIAL_WAVE_TICKS),)
        )

        # This should always hold by construction and documents the behavioral
        # block's interface/limit for future edits.
        assert sum(duration for _, duration in sequence) <= self.MAX_SOCIAL_TICKS
        return sequence

    def _block_socialize(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction | None:
        """Shared social block used by every role and both partner types."""

        if self.social_partner is None:
            return None

        partner = next(
            (p for p in self._nearby if p["id"] == self.social_partner),
            None,
        )
        if partner is None:
            self._finish_social(self._tick)
            return None

        if self.social_stage >= len(self.social_sequence):
            self._finish_social(self._tick)
            return None

        emote, duration = self.social_sequence[self.social_stage]

        if emote == "point" and self.social_point_key is not None:
            social_heading = geometry.heading_to(
                self._here,
                _cell_centre_from_key(self.social_point_key),
            )
        else:
            social_heading = geometry.heading_to(self._here, partner["position"])

        self.social_stage_tick += 1
        if self.social_stage_tick >= duration:
            self.social_stage += 1
            self.social_stage_tick = 0
            if self.social_stage >= len(self.social_sequence):
                self._finish_social(self._tick)

        return self._stand(social_heading, emote)

    def _choose_point_target(self) -> CellKey | None:
        # Pointing reinforces each role: the keeper indicates the pump, workers
        # indicate their work site, and sweepers indicate the next path stop.
        if self.role == self.ROLE_WELL_KEEPER and self.pump is not None:
            pump_key = _cell_key(self.pump["cell"])
            if pump_key is not None:
                return pump_key

        if self.role == self.ROLE_WORKER and self.work_target_key is not None:
            return self.work_target_key

        if self.role == self.ROLE_SWEEPER and self.sweep_route_keys:
            return self.sweep_route_keys[self.sweep_route_index]

        candidates: list[CellKey] = []

        if self.destination_key is not None:
            candidates.append(self.destination_key)

        if self.pump is not None:
            pump_key = _cell_key(self.pump["cell"])
            if pump_key is not None:
                candidates.append(pump_key)

        for bench in self.benches:
            bench_key = _cell_key(bench["cell"])
            if bench_key is not None:
                candidates.append(bench_key)

        if self._here_key is not None:
            component = self.component_of.get(self._here_key)
            outdoor = self.outdoor_by_component.get(component, []) if component is not None else []
            for _ in range(min(6, len(outdoor))):
                key = self.rng.choice(outdoor)
                if _manhattan_keys(self._here_key, key) >= 4:
                    candidates.append(key)
                    break

        return self.rng.choice(candidates) if candidates else None

    def _finish_social(self, tick: int) -> None:
        if self.social_partner is not None:
            if self.social_kind == "visitor":
                cooldown = self.rng.randint(220, 320)
            else:
                cooldown = self.rng.randint(170, 250)
            self.social_cooldowns[self.social_partner] = tick + cooldown

        # Role configuration controls how quickly each job becomes available
        # for another conversation after the shared social block finishes.
        low, high = self.config.get("social_cooldown", (60, 100))
        self.social_block_until = tick + self.rng.randint(low, high)
        self.social_partner = None
        self.social_kind = None
        self.social_point_key = None
        self.social_sequence = ()
        self.social_stage = 0
        self.social_stage_tick = 0

    def _passing_expression(self) -> str:
        """Occasionally wave while walking without interrupting the route."""

        if not self._seen:
            return "none"
        if self._tick - self.last_passing_wave_tick < 45:
            return "none"

        visitor_seen = any(people.is_visitor(person["id"]) for person in self._seen)
        wave_chance = self.config.get("passing_wave_chance", 0.25)
        if visitor_seen or self.rng.random() < wave_chance:
            self.last_passing_wave_tick = self._tick
            return "wave"
        return "none"

    # ------------------------------------------------------------------
    # Fast navigation
    # ------------------------------------------------------------------

    def _build_navigation_graph(self, observation: ThreeBranchesObservation) -> None:
        """Cache all static walkable cells and legal cardinal edges once."""

        cells_x = self.frame["cells_x"]
        cells_y = self.frame["cells_y"]

        for y in range(cells_y):
            for x in range(cells_x):
                key = (x, y)
                cell = {"x": x, "y": y}
                if not layout.walkable(observation, cell):
                    continue
                self.walkable_keys.add(key)
                self.ground_by_key[key] = layout.ground_at(observation, cell)

        # Evaluate can_step only once per undirected edge. The result is stored
        # in both directions because cardinal movement is static for this map.
        mutable_neighbors: dict[CellKey, list[CellKey]] = {
            key: [] for key in self.walkable_keys
        }

        for x, y in self.walkable_keys:
            current = {"x": x, "y": y}
            for neighbor_key in ((x + 1, y), (x, y + 1)):
                if neighbor_key not in self.walkable_keys:
                    continue
                neighbor = _cell_from_key(neighbor_key)
                if layout.can_step(observation, current, neighbor):
                    mutable_neighbors[(x, y)].append(neighbor_key)
                    mutable_neighbors[neighbor_key].append((x, y))

        self.nav_neighbors = {
            key: tuple(neighbors)
            for key, neighbors in mutable_neighbors.items()
        }

    def _build_components(self) -> None:
        """Label connected components and cache outdoor cells in each one."""

        component_id = 0

        for start in self.walkable_keys:
            if start in self.component_of:
                continue

            queue: deque[CellKey] = deque([start])
            self.component_of[start] = component_id
            outdoor: list[CellKey] = []

            while queue:
                current = queue.popleft()
                if self.ground_by_key.get(current) != "interior":
                    outdoor.append(current)

                for neighbor in self.nav_neighbors.get(current, ()):
                    if neighbor in self.component_of:
                        continue
                    self.component_of[neighbor] = component_id
                    queue.append(neighbor)

            self.outdoor_by_component[component_id] = outdoor
            component_id += 1

    def _block_travel_to(self, goal: CellKey | None) -> ThreeBranchesAction | None:
        if goal is None or self._here_key is None:
            return None
        if self._here_key == goal:
            return None

        if self.destination_key != goal:
            self._set_destination_key(goal)

        # Discard route cells already crossed.
        while self.route and self.route[0] == self._here_key:
            self.route.popleft()

        # Compute ONE route only when needed. BFS uses cached tuple neighbors and
        # therefore performs no layout.walkable/can_step calls.
        if not self.route:
            path = self._bfs_cached(self._here_key, goal)
            if not path:
                return None
            self.route = deque(path)

        next_key = self.route[0]
        if next_key not in self.nav_neighbors.get(self._here_key, ()):
            # We drifted to a different cell; cheaply invalidate and retry next
            # tick rather than repeatedly running expensive geometry checks.
            self.route.clear()
            return None

        return self._walk(
            geometry.heading_to(self._here, _cell_centre_from_key(next_key)),
            1.0,
            self._passing_expression(),
        )

    def _bfs_cached(self, start: CellKey, goal: CellKey) -> list[CellKey]:
        """BFS over the cached static graph; called only on destination changes."""

        if start == goal:
            return []
        if start not in self.nav_neighbors or goal not in self.nav_neighbors:
            return []
        if self.component_of.get(start) != self.component_of.get(goal):
            return []

        queue: deque[CellKey] = deque([start])
        came_from: dict[CellKey, CellKey | None] = {start: None}

        found = False
        while queue and not found:
            current = queue.popleft()
            for neighbor in self.nav_neighbors[current]:
                if neighbor in came_from:
                    continue
                came_from[neighbor] = current
                if neighbor == goal:
                    found = True
                    break
                queue.append(neighbor)

        if goal not in came_from:
            return []

        reversed_path: list[CellKey] = []
        cursor = goal
        while cursor != start:
            reversed_path.append(cursor)
            parent = came_from[cursor]
            if parent is None:
                return []
            cursor = parent

        reversed_path.reverse()
        return reversed_path

    def _choose_approach_key(
        self,
        approaches: tuple[CellKey, ...],
        exclude: CellKey | None = None,
    ) -> CellKey | None:
        """Choose a reachable static prop approach without running BFS."""

        if self._here_key is None or not approaches:
            return None

        component = self.component_of.get(self._here_key)
        candidates = [
            key
            for key in approaches
            if key != exclude and self.component_of.get(key) == component
        ]
        if not candidates:
            return None
        if self._here_key in candidates:
            return self._here_key

        candidates.sort(key=lambda key: _manhattan_keys(self._here_key, key))

        # Spread villagers across the nearest approach cells when possible.
        shortlist = candidates[: min(3, len(candidates))]
        return shortlist[self.player_slot % len(shortlist)]

    def _recover_from_stuck(self) -> ThreeBranchesAction | None:
        if self._here_key is None:
            self.stuck_ticks = 0
            return None

        neighbors = list(self.nav_neighbors.get(self._here_key, ()))
        if not neighbors:
            self.stuck_ticks = 0
            self.route.clear()
            return self._stand(self._heading, "shrug")

        self.rng.shuffle(neighbors)

        if self.destination_key is not None:
            neighbors.sort(key=lambda key: _manhattan_keys(key, self.destination_key))
            pool = neighbors[: min(2, len(neighbors))]
            target = self.rng.choice(pool)
        else:
            target = self.rng.choice(neighbors)

        # Keep the long-term destination, but force a fresh route after this
        # sidestep. No BFS is needed for the sidestep itself.
        self.route.clear()
        self.stuck_ticks = 0
        return self._walk(
            geometry.heading_to(self._here, _cell_centre_from_key(target)),
            0.65,
            "shrug",
        )

    # ------------------------------------------------------------------
    # Role assignment and role setup
    # ------------------------------------------------------------------

    @classmethod
    def _role_for(cls, player_id: str) -> str:
        """Map stable player IDs to visible village jobs."""

        index = cls._player_number(player_id)
        if index == 1:
            return cls.ROLE_WELL_KEEPER
        if 2 <= index <= 5:
            return cls.ROLE_WORKER
        if 6 <= index <= 8:
            return cls.ROLE_SWEEPER
        return cls.ROLE_WANDERER

    def _initialize_role_behavior(self) -> None:
        """Choose role-specific static targets without adding per-tick cost."""

        if self.role == self.ROLE_WELL_KEEPER:
            self.keeper_anchor_key, self.keeper_patrol_keys = self._build_keeper_patrol()
        elif self.role == self.ROLE_WORKER:
            self.work_target_key = self._choose_worker_spot()
        elif self.role == self.ROLE_SWEEPER:
            self.sweep_route_keys = self._build_sweeper_route()
            if self.sweep_route_keys:
                self.sweep_route_index = self.player_slot % len(self.sweep_route_keys)

    def _home_component(self) -> int | None:
        if self.home_door_key is not None:
            component = self.component_of.get(self.home_door_key)
            if component is not None:
                return component

        # Fallback for unusual doorway geometry: use any component containing
        # an exterior cell. This keeps the agent functional on variant maps.
        for component, cells in self.outdoor_by_component.items():
            if cells:
                return component
        return None

    def _build_keeper_patrol(self) -> tuple[CellKey | None, list[CellKey]]:
        """Return a pump-side anchor and a few nearby cells for short trips."""

        component = self._home_component()
        if component is None:
            return None, []

        approaches = [
            key
            for key in self.pump_approaches
            if self.component_of.get(key) == component
        ]
        if not approaches:
            return None, []

        if self.home_door_key is not None:
            approaches.sort(key=lambda key: _manhattan_keys(self.home_door_key, key))
        anchor = approaches[0]

        outdoor = self.outdoor_by_component.get(component, [])
        nearby = [
            key
            for key in outdoor
            if _manhattan_keys(anchor, key) <= 3
        ]
        if anchor not in nearby:
            nearby.append(anchor)

        # Limit the local set to keep decisions cheap and the job visually
        # centered on the well/pump.
        nearby.sort(key=lambda key: (_manhattan_keys(anchor, key), key))
        return anchor, nearby[:12]

    def _choose_worker_spot(self) -> CellKey | None:
        """Choose a stable outdoor work site well away from the worker's home."""

        component = self._home_component()
        if component is None:
            return None

        pool = list(self.outdoor_by_component.get(component, []))
        if not pool:
            return None

        origin = self.home_door_key
        if origin is None:
            return self.rng.choice(pool)

        candidates = [
            key
            for key in pool
            if _manhattan_keys(origin, key) >= 6
        ]
        if not candidates:
            candidates = [key for key in pool if key != origin]
        if not candidates:
            return None

        # Prefer farther sites, then use player ID to spread workers across the
        # shortlist so players 2-5 do not all commute to one square.
        candidates.sort(
            key=lambda key: (_manhattan_keys(origin, key), key[0], key[1]),
            reverse=True,
        )
        shortlist = candidates[: min(16, len(candidates))]
        return shortlist[(self.player_slot * 3) % len(shortlist)]

    def _build_sweeper_route(self) -> list[CellKey]:
        """Choose a small repeating patrol of well-separated outdoor cells."""

        component = self._home_component()
        if component is None:
            return []

        pool = list(self.outdoor_by_component.get(component, []))
        if not pool:
            return []

        # Shuffle with the villager's private deterministic RNG so each sweeper
        # gets a different-looking route while remaining repeatable.
        self.rng.shuffle(pool)
        selected: list[CellKey] = []
        origin = self.home_door_key

        for candidate in pool:
            if origin is not None and _manhattan_keys(origin, candidate) < 3:
                continue
            if selected and min(_manhattan_keys(candidate, key) for key in selected) < 4:
                continue
            selected.append(candidate)
            if len(selected) >= 5:
                break

        if len(selected) < 3:
            for candidate in pool:
                if candidate in selected:
                    continue
                selected.append(candidate)
                if len(selected) >= min(5, len(pool)):
                    break

        return selected

    # ------------------------------------------------------------------
    # Destination / prop selection
    # ------------------------------------------------------------------

    def _choose_roam_destination(self) -> CellKey | None:
        """Choose a destination without trial-running BFS many times."""

        if self._here_key is None:
            all_outdoor = [key for cells in self.outdoor_by_component.values() for key in cells]
            return self.rng.choice(all_outdoor) if all_outdoor else None

        component = self.component_of.get(self._here_key)
        if component is None:
            return None

        pool = self.outdoor_by_component.get(component, [])
        if not pool:
            return None

        # Sample a small fixed number of known-reachable cells and prefer one
        # that is several cells away. No pathfinding is performed here.
        best: CellKey | None = None
        best_score = -1
        tries = min(12, len(pool))

        for _ in range(tries):
            candidate = self.rng.choice(pool)
            if candidate == self._here_key or candidate == self.last_roam_key:
                continue

            distance = _manhattan_keys(self._here_key, candidate)
            score = distance
            if candidate not in self.visited_places:
                score += 4
            if distance < 4:
                score -= 8

            if score > best_score:
                best = candidate
                best_score = score

        if best is not None:
            return best

        # Cheap fallback when random sampling happened to select only poor cells.
        for candidate in pool:
            if candidate != self._here_key and candidate != self.last_roam_key:
                return candidate
        return None

    def _choose_bench_and_approach(self) -> tuple[dict | None, CellKey | None]:
        """Choose a bench using cached components/Manhattan distance only."""

        if self._here_key is None:
            return None, None

        component = self.component_of.get(self._here_key)
        choices: list[tuple[int, dict, CellKey]] = []

        for bench in self.benches:
            if bench["id"] in self.rest_failed_ids:
                continue

            approaches = [
                key
                for key in self.bench_approaches.get(bench["id"], ())
                if self.component_of.get(key) == component
            ]
            if not approaches:
                continue

            approaches.sort(key=lambda key: _manhattan_keys(self._here_key, key))
            nearest = approaches[0]
            distance = _manhattan_keys(self._here_key, nearest)
            choices.append((distance, bench, nearest))

        if not choices:
            return None, None

        choices.sort(key=lambda item: item[0])
        shortlist = choices[: min(3, len(choices))]
        _, bench, nearest = self.rng.choice(shortlist)

        # Spread agents among valid approach cells around the chosen bench.
        approach = self._choose_approach_key(self.bench_approaches.get(bench["id"], ()))
        return bench, approach if approach is not None else nearest

    def _prop_approach_keys(self, prop_cell: dict[str, int]) -> tuple[CellKey, ...]:
        prop_key = _cell_key(prop_cell)
        if prop_key is None:
            return ()

        x, y = prop_key
        candidates = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )
        return tuple(key for key in candidates if key in self.walkable_keys)

    def _find_home_interior_key(self, door_key: CellKey | None) -> CellKey | None:
        if door_key is None:
            return None

        if self.ground_by_key.get(door_key) == "interior":
            return door_key

        for key in self.nav_neighbors.get(door_key, ()):
            if self.ground_by_key.get(key) == "interior":
                return key
        return None

    # ------------------------------------------------------------------
    # Season 5 scripted-knowledge helpers
    # ------------------------------------------------------------------

    def _build_scripted_knowledge(self) -> dict[str, str]:
        """Return a private topic -> fact dictionary for this villager.

        The information is intentionally static for now. Later steps can add
        observed facts and remembered facts as separate sources rather than
        mixing them into this dictionary.
        """

        knowledge = dict(self.COMMON_SCRIPTED_KNOWLEDGE)
        knowledge.update(self.ROLE_SCRIPTED_KNOWLEDGE.get(self.role, {}))
        return knowledge

    def _scripted_fact(self, topic: str) -> str | None:
        """Look up one scripted fact without changing conversation state."""

        return self.scripted_knowledge.get(topic)

    # ------------------------------------------------------------------
    # Season 5 observed-knowledge helpers
    # ------------------------------------------------------------------

    def _update_observed_knowledge(self) -> None:
        """Refresh this villager's CURRENT visible-world snapshot.

        This deliberately reuses ``self._seen`` from the normal Season 3
        perception pass. It does not call ``people.seen`` again, so adding
        observed knowledge does not add another expensive perception query.
        Nothing in the behavior tree reads this dictionary yet; it is only
        preparation for later dialogue.
        """

        visible_people: list[str] = []
        visible_npcs: list[str] = []
        visible_visitors: list[str] = []
        visible_positions: dict[str, dict[str, float]] = {}
        visible_expressions: dict[str, str] = {}

        for person in self._seen:
            pid = person.get("id")
            if not isinstance(pid, str):
                continue

            visible_people.append(pid)

            if people.is_npc(pid):
                visible_npcs.append(pid)
            elif people.is_visitor(pid):
                visible_visitors.append(pid)

            position = person.get("position")
            if isinstance(position, dict) and "x" in position and "y" in position:
                visible_positions[pid] = {
                    "x": float(position["x"]),
                    "y": float(position["y"]),
                }

            expression = person.get("expression")
            if isinstance(expression, dict):
                expression_type = expression.get("type")
                if isinstance(expression_type, str) and expression_type != "none":
                    visible_expressions[pid] = expression_type

        self.observed_knowledge = {
            "tick": self._tick,
            "visible_people": tuple(visible_people),
            "visible_npcs": tuple(visible_npcs),
            "visible_visitors": tuple(visible_visitors),
            "visible_positions": visible_positions,
            "visible_expressions": visible_expressions,
        }

    def _observed_people(self) -> tuple[str, ...]:
        """Return the IDs this villager can currently see."""

        value = self.observed_knowledge.get("visible_people", ())
        return value if isinstance(value, tuple) else ()

    def _observed_npcs(self) -> tuple[str, ...]:
        """Return currently visible NPC IDs."""

        value = self.observed_knowledge.get("visible_npcs", ())
        return value if isinstance(value, tuple) else ()

    def _observed_visitors(self) -> tuple[str, ...]:
        """Return currently visible visitor IDs."""

        value = self.observed_knowledge.get("visible_visitors", ())
        return value if isinstance(value, tuple) else ()

    # ------------------------------------------------------------------
    # Season 5 simple-memory helpers
    # ------------------------------------------------------------------

    def _remember_current_observations(self) -> None:
        """Store selected facts from the current sight snapshot.

        This method deliberately reuses ``self._seen``. It never calls
        ``people.seen`` or ``people.nearby`` itself, so remembering does not add
        another perception query. The roster is small, so keeping one compact
        record per person has negligible memory cost.
        """

        people_memory = self.memory.get("people")
        if not isinstance(people_memory, dict):
            people_memory = {}
            self.memory["people"] = people_memory

        ever_seen_npcs = self.memory.get("ever_seen_npcs")
        if not isinstance(ever_seen_npcs, set):
            ever_seen_npcs = set()
            self.memory["ever_seen_npcs"] = ever_seen_npcs

        ever_seen_visitors = self.memory.get("ever_seen_visitors")
        if not isinstance(ever_seen_visitors, set):
            ever_seen_visitors = set()
            self.memory["ever_seen_visitors"] = ever_seen_visitors

        for person in self._seen:
            pid = person.get("id")
            if not isinstance(pid, str) or pid == self.player_id:
                continue

            if people.is_npc(pid):
                kind = "npc"
                ever_seen_npcs.add(pid)
            elif people.is_visitor(pid):
                kind = "visitor"
                ever_seen_visitors.add(pid)
            else:
                kind = "person"

            position = person.get("position")
            if isinstance(position, dict):
                last_position = {
                    "x": float(position.get("x", 0.0)),
                    "y": float(position.get("y", 0.0)),
                }
            else:
                last_position = None

            expression = person.get("expression", {})
            if isinstance(expression, dict):
                expression_type = expression.get("type", "none")
                if not isinstance(expression_type, str):
                    expression_type = "none"
            else:
                expression_type = "none"

            existing = people_memory.get(pid)
            if isinstance(existing, dict):
                first_seen_tick = existing.get("first_seen_tick", self._tick)
                times_seen = existing.get("times_seen", 0)
                if not isinstance(first_seen_tick, int):
                    first_seen_tick = self._tick
                if not isinstance(times_seen, int):
                    times_seen = 0
            else:
                first_seen_tick = self._tick
                times_seen = 0

            people_memory[pid] = {
                "kind": kind,
                "first_seen_tick": first_seen_tick,
                "last_seen_tick": self._tick,
                "last_position": last_position,
                "last_expression": expression_type,
                "times_seen": times_seen + 1,
            }

    def _remembered_person(self, person_id: str) -> dict[str, object] | None:
        """Return this villager's remembered record for one person."""

        people_memory = self.memory.get("people", {})
        if not isinstance(people_memory, dict):
            return None

        record = people_memory.get(person_id)
        return record if isinstance(record, dict) else None

    def _has_seen_person_before(self, person_id: str) -> bool:
        """Whether this villager has ever visually observed the person."""

        return self._remembered_person(person_id) is not None

    def _remembered_people(self) -> tuple[str, ...]:
        """Return IDs this villager remembers seeing at least once."""

        people_memory = self.memory.get("people", {})
        if not isinstance(people_memory, dict):
            return ()
        return tuple(sorted(pid for pid in people_memory if isinstance(pid, str)))

    def _last_seen_tick(self, person_id: str) -> int | None:
        """Return the most recent remembered sighting tick for one person."""

        record = self._remembered_person(person_id)
        if record is None:
            return None
        value = record.get("last_seen_tick")
        return value if isinstance(value, int) else None

    # ------------------------------------------------------------------
    # Season 5 scripted visitor dialogue
    # ------------------------------------------------------------------

    def _begin_scripted_conversation(self, sender: str, tick: int | None) -> str:
        """Start a numbered visitor conversation without changing game behavior."""

        self.in_conversation = True
        self.conversation_partner = sender
        self.conversation_topic = "main"
        self.conversation_started_tick = tick
        return self.MAIN_DIALOGUE_MENU

    def _end_scripted_conversation(self) -> str:
        """Close the text conversation and reset only dialogue memory."""

        self.in_conversation = False
        self.conversation_partner = None
        self.conversation_topic = "main"
        self.conversation_started_tick = None
        return "Goodbye. I should get back to my day."

    def _main_dialogue_reply(self, choice: str) -> str:
        """Handle one numbered choice from the main menu."""

        if choice == "1":
            self.conversation_topic = "work"
            role_fact = self._scripted_fact("role") or "I am one of the village residents."
            work_fact = self._scripted_fact("work") or "I have work to do around the village."
            return f"{role_fact} {work_fact} {self.WORK_DIALOGUE_MENU}"

        if choice == "2":
            self.conversation_topic = "day"
            village_fact = self._scripted_fact("village") or "Everyone has their own routine here."
            return f"{village_fact} {self.DAY_DIALOGUE_MENU}"

        if choice == "3":
            role_fact = self._scripted_fact("role") or "I am one of the village residents."
            return f"{role_fact} {self.MAIN_DIALOGUE_MENU}"

        if choice == "9":
            return self.MAIN_DIALOGUE_MENU

        return (
            "For now, please choose one of the numbered options. "
            + self.MAIN_DIALOGUE_MENU
        )

    def _work_dialogue_reply(self, choice: str) -> str:
        """Handle follow-up questions while the current topic is work."""

        if choice == "1":
            fact = self._scripted_fact("workplace") or "My work takes me around the village."
            return f"{fact} {self.WORK_DIALOGUE_MENU}"

        if choice == "2":
            fact = self._scripted_fact("routine") or "I follow my usual work routine."
            return f"{fact} {self.WORK_DIALOGUE_MENU}"

        if choice == "9":
            self.conversation_topic = "main"
            return self.MAIN_DIALOGUE_MENU

        return (
            "That choice is not in the work menu. "
            + self.WORK_DIALOGUE_MENU
        )

    def _day_dialogue_reply(self, choice: str) -> str:
        """Handle follow-up questions while the current topic is the village day."""

        if choice == "1":
            fact = self._scripted_fact("village") or "People here follow different routines."
            return f"{fact} {self.DAY_DIALOGUE_MENU}"

        if choice == "2":
            fact = self._scripted_fact("home") or "Residents return home later in the day."
            return f"{fact} {self.DAY_DIALOGUE_MENU}"

        if choice == "9":
            self.conversation_topic = "main"
            return self.MAIN_DIALOGUE_MENU

        return (
            "That choice is not in the day menu. "
            + self.DAY_DIALOGUE_MENU
        )

    def _detect_intention(self, visitor_text: str) -> str | None:
        """Detect a visitor request for an ACTION.

        Questions such as "Where do you work?" remain information requests and
        are handled by ``_keyword_command``. Only imperative/request language
        that asks the villager to physically do something becomes an intention.
        """

        text = visitor_text.strip().lower()
        if not text:
            return None

        show_work_phrases = (
            "show me where you work",
            "show me your workplace",
            "show me your work",
            "take me where you work",
            "take me to your workplace",
            "take me to your work",
            "lead me where you work",
            "lead me to your workplace",
            "lead me to your work",
            "bring me to your workplace",
            "bring me where you work",
            "can you show me where you work",
            "could you show me where you work",
            "will you show me where you work",
            "can you take me to your workplace",
            "could you take me to your workplace",
            "can you lead me to your workplace",
        )

        if any(phrase in text for phrase in show_work_phrases):
            return "show_work"

        return None

    def _store_intention(
        self,
        intention_type: str,
        sender: str,
        message_tick: int | None,
        visitor_text: str,
    ) -> None:
        """Store the visitor's latest action request without executing it."""

        self.pending_intention = {
            "type": intention_type,
            "requested_by": sender,
            "requested_tick": message_tick,
            "original_text": visitor_text,
            "status": "detected",
        }
        self.last_intention_type = intention_type

    def _can_accept_intention(self, intention_type: str) -> tuple[bool, str]:
        """Decide whether the villager can reasonably accept an action request.

        This decision uses only facts already present in the working agent:
        current FSM state, role, and whether the role has a real work target.
        It does NOT start movement.
        """

        if intention_type != "show_work":
            return False, "I do not know how to carry out that request."

        # Strategic/rest states outrank optional visitor requests.
        if self.state == self.RETURN_HOME:
            return False, "I am already heading home, so I cannot show you now."

        if self.state in {self.NAP_GO_HOME, self.NAP_SLEEP}:
            return False, "I am taking my rest, so I cannot show you now."

        if self.role == self.ROLE_WELL_KEEPER:
            if self.pump is None or not self.pump_approaches:
                return False, "I cannot reach the pump right now."
            return True, "I can show you the village pump where I work."

        if self.role == self.ROLE_WORKER:
            if self.work_target_key is None:
                return False, "I do not have a usable work location right now."
            return True, "I can show you my work area."

        if self.role == self.ROLE_SWEEPER:
            if not self.sweep_route_keys:
                return False, "I do not have a usable sweeping route right now."
            return True, "I can show you one of the paths where I work."

        # Wanderers deliberately have no fixed workplace in this design.
        if self.role == self.ROLE_WANDERER:
            return (
                False,
                "I roam around the village rather than working at one fixed place.",
            )

        return False, "I do not have a fixed workplace to show you."

    def _decide_pending_intention(self) -> tuple[str, str]:
        """Mark the current pending intention accepted or refused.

        Returns ``(status, reason)``. The status is conversational state only;
        no movement or behavior is triggered in this version.
        """

        if self.pending_intention is None:
            return "none", "There is no request to decide."

        intention_type = self.pending_intention.get("type")
        if not isinstance(intention_type, str):
            self.pending_intention["status"] = "refused"
            self.pending_intention["decision_reason"] = "Invalid request."
            return "refused", "I could not understand that request."

        accepted, reason = self._can_accept_intention(intention_type)
        status = "accepted" if accepted else "refused"

        self.pending_intention["status"] = status
        self.pending_intention["decision_reason"] = reason
        self.pending_intention["decided_tick"] = self._tick

        return status, reason

    def _intention_reply(self, intention_type: str) -> str:
        """Return the accept/refuse reply for the stored visitor intention."""

        status, reason = self._decide_pending_intention()

        if intention_type == "show_work":
            if status == "accepted":
                return (
                    "Yes. I understand that you want me to show you where I work. "
                    f"{reason} "
                    "I can help with that."
                )

            return (
                "I understand that you want me to show you where I work, "
                f"but I have to refuse for now. {reason}"
            )

        if status == "accepted":
            return f"I can help with that. {reason}"

        return f"I cannot do that right now. {reason}"

    def _show_work_target(self) -> CellKey | None:
        """Return this role's real workplace target for ``show_work``."""

        if self.role == self.ROLE_WELL_KEEPER:
            if self.keeper_anchor_key is not None:
                return self.keeper_anchor_key
            return self._choose_approach_key(self.pump_approaches)

        if self.role == self.ROLE_WORKER:
            return self.work_target_key

        if self.role == self.ROLE_SWEEPER and self.sweep_route_keys:
            index = self.sweep_route_index % len(self.sweep_route_keys)
            return self.sweep_route_keys[index]

        return None

    def _clear_intention_execution(self) -> None:
        """Clear temporary request movement while keeping request history."""

        self.intention_target_key = None
        self.intention_started_tick = None
        self.intention_attempt_ticks = 0
        self._clear_navigation(keep_destination=False)

    def _cancel_active_intention(self, reason: str) -> None:
        """Cancel an accepted/active request when a strategic state takes over."""

        if self.pending_intention is None:
            return

        status = self.pending_intention.get("status")
        if status not in {"accepted", "active"}:
            return

        self.pending_intention["status"] = "cancelled"
        self.pending_intention["completion_reason"] = reason
        self.pending_intention["completed_tick"] = self._tick
        self._clear_intention_execution()

    def _begin_pending_intention(self) -> bool:
        """Turn an accepted request into a temporary workplace navigation goal."""

        if self.pending_intention is None:
            return False
        if self.pending_intention.get("status") != "accepted":
            return False
        if self.pending_intention.get("type") != "show_work":
            self.pending_intention["status"] = "failed"
            self.pending_intention["completion_reason"] = "Unsupported intention."
            self.pending_intention["completed_tick"] = self._tick
            return False

        target = self._show_work_target()
        if target is None:
            self.pending_intention["status"] = "failed"
            self.pending_intention["completion_reason"] = "No workplace target is available."
            self.pending_intention["completed_tick"] = self._tick
            return False

        # The request temporarily interrupts lower-priority side activities only.
        if self.social_partner is not None:
            self._finish_social(self._tick)
        if self.random_prop is not None:
            self._cancel_random_prop_activity()

        self._clear_navigation(keep_destination=False)
        self.intention_target_key = target
        self.intention_started_tick = self._tick
        self.intention_attempt_ticks = 0

        self.pending_intention["status"] = "active"
        self.pending_intention["started_tick"] = self._tick
        self.pending_intention["target_key"] = target
        self._set_destination_key(target)
        return True

    def _complete_pending_intention(self) -> None:
        """Mark the request complete and release the temporary navigation goal."""

        if self.pending_intention is not None:
            self.pending_intention["status"] = "completed"
            self.pending_intention["completion_reason"] = "Arrived at workplace."
            self.pending_intention["completed_tick"] = self._tick

        self._clear_intention_execution()

    def _fail_pending_intention(self, reason: str) -> None:
        """Fail cleanly instead of blocking the villager's normal routine."""

        if self.pending_intention is not None:
            self.pending_intention["status"] = "failed"
            self.pending_intention["completion_reason"] = reason
            self.pending_intention["completed_tick"] = self._tick

        self._clear_intention_execution()

    def _act_pending_intention(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction | None:
        """Execute an accepted ``show_work`` request using existing navigation.

        This is a temporary behavior-tree interrupt. It does not replace the
        villager's FSM state or role-specific phase. When the workplace is
        reached, the temporary target is cleared and normal behavior resumes.
        """

        if self.pending_intention is None:
            return None

        status = self.pending_intention.get("status")
        if status not in {"accepted", "active"}:
            return None

        if self.state == self.RETURN_HOME:
            self._cancel_active_intention("Returning home has higher priority.")
            return None

        if self.state in {self.NAP_GO_HOME, self.NAP_SLEEP}:
            self._cancel_active_intention("Scheduled rest has higher priority.")
            return None

        if status == "accepted":
            if not self._begin_pending_intention():
                return None

        target = self.intention_target_key
        if target is None:
            self._fail_pending_intention("The workplace target became unavailable.")
            return None

        if self._here_key is None:
            self.intention_attempt_ticks += 1
            if self.intention_attempt_ticks > self.MAX_INTENTION_TRAVEL_TICKS:
                self._fail_pending_intention("Could not locate the villager on the map.")
                return self._stand(self._heading, "shrug")
            return self._stand(self._heading, "none")

        if self._here_key == target:
            self._complete_pending_intention()
            return self._stand(self._heading, "point")

        self.intention_attempt_ticks += 1
        if self.intention_attempt_ticks > self.MAX_INTENTION_TRAVEL_TICKS:
            self._fail_pending_intention("The workplace could not be reached in time.")
            return self._stand(self._heading, "shrug")

        if self.stuck_ticks >= 5:
            recovery = self._recover_from_stuck()
            if recovery is not None:
                return recovery

        move = self._block_travel_to(target)
        if move is not None:
            return move

        # Allow a transient route invalidation to rebuild on the next tick.
        return self._stand(self._heading, "none")

    def _keyword_command(self, visitor_text: str) -> str | None:
        """Map simple natural-language questions to scripted dialogue commands.

        This is intentionally rule-based rather than an LLM. It keeps Season 5
        dialogue predictable and uses the same topics as the numbered menus.
        More specific phrases are checked before broad words such as "work".
        """

        text = visitor_text.strip().lower()
        if not text:
            return None

        # Conversation control.
        if any(phrase in text for phrase in (
            "goodbye",
            "bye",
            "end conversation",
            "stop talking",
            "leave conversation",
        )):
            return "end"

        if any(phrase in text for phrase in (
            "main menu",
            "go back",
            "back to menu",
            "start over",
        )):
            return "main"

        # Identity.
        if any(phrase in text for phrase in (
            "who are you",
            "what is your role",
            "what's your role",
            "what are you",
        )):
            return "identity"

        # Work questions: specific questions come before the broad work topic.
        if any(phrase in text for phrase in (
            "where do you work",
            "where is your work",
            "where's your work",
            "your workplace",
            "workplace",
        )):
            return "workplace"

        if any(phrase in text for phrase in (
            "what is your routine",
            "what's your routine",
            "your routine",
            "what do you do each day",
            "what do you do every day",
            "how do you spend your day",
        )):
            return "routine"

        if any(phrase in text for phrase in (
            "what work do you do",
            "what is your work",
            "what's your work",
            "your job",
            "what is your job",
            "what's your job",
        )):
            return "work"

        # Village/day questions.
        if any(phrase in text for phrase in (
            "tell me about the village",
            "about the village",
            "village life",
            "what is the village like",
            "what's the village like",
        )):
            return "village"

        if any(phrase in text for phrase in (
            "when do you go home",
            "do you go home",
            "going home",
            "return home",
            "your home",
        )):
            return "home"

        if any(phrase in text for phrase in (
            "tell me about your day",
            "about your day",
            "your day",
            "daily life",
        )):
            return "day"

        # Broad single-word fallbacks come last to avoid stealing more specific
        # questions such as "Where do you work?"
        words = set(text.replace("?", " ").replace("!", " ").replace(",", " ").split())
        if "work" in words or "job" in words:
            return "work"
        if "village" in words:
            return "village"
        if "home" in words:
            return "home"

        return None

    def _reply_to_command(self, command: str) -> str:
        """Answer one semantic command using the same scripted knowledge/menu FSM."""

        if command == "end":
            return self._end_scripted_conversation()

        if command == "main":
            self.conversation_topic = "main"
            return self.MAIN_DIALOGUE_MENU

        if command == "identity":
            self.conversation_topic = "main"
            role_fact = self._scripted_fact("role") or "I am one of the village residents."
            return f"{role_fact} {self.MAIN_DIALOGUE_MENU}"

        if command == "work":
            self.conversation_topic = "work"
            role_fact = self._scripted_fact("role") or "I am one of the village residents."
            work_fact = self._scripted_fact("work") or "I have work to do around the village."
            return f"{role_fact} {work_fact} {self.WORK_DIALOGUE_MENU}"

        if command == "workplace":
            self.conversation_topic = "work"
            fact = self._scripted_fact("workplace") or "My work takes me around the village."
            return f"{fact} {self.WORK_DIALOGUE_MENU}"

        if command == "routine":
            self.conversation_topic = "work"
            fact = self._scripted_fact("routine") or "I follow my usual work routine."
            return f"{fact} {self.WORK_DIALOGUE_MENU}"

        if command == "day":
            self.conversation_topic = "day"
            village_fact = self._scripted_fact("village") or "Everyone has their own routine here."
            return f"{village_fact} {self.DAY_DIALOGUE_MENU}"

        if command == "village":
            self.conversation_topic = "day"
            fact = self._scripted_fact("village") or "People here follow different routines."
            return f"{fact} {self.DAY_DIALOGUE_MENU}"

        if command == "home":
            self.conversation_topic = "day"
            fact = self._scripted_fact("home") or "Residents return home later in the day."
            return f"{fact} {self.DAY_DIALOGUE_MENU}"

        return self.MAIN_DIALOGUE_MENU

    def chat(self, inbox):
        """Reply to visitor messages using numbered choices or keyword matching.

        The course chat interface supplies ordinary message dictionaries. At
        this stage only ``player_0`` (the visitor) is handled. Natural-language
        questions are mapped onto the SAME scripted topics used by the numbered
        menus, so there is still only one conversation state machine.

        A ``show_work`` request is detected, stored, then accepted/refused from
        the villager's actual role/state. Accepted requests are executed by
        act() as a temporary navigation interrupt using the existing travel
        system; the villager's durable role/FSM state is preserved.
        """

        replies = []

        for message in inbox:
            if not isinstance(message, dict):
                continue

            sender = message.get("from")
            if sender != self.VISITOR_ID:
                continue

            raw_text = message.get("text", "")
            visitor_text = str(raw_text).strip()
            choice = visitor_text.lower()
            self.last_visitor_message = visitor_text

            raw_tick = message.get("tick")
            message_tick = raw_tick if isinstance(raw_tick, int) else None

            # First distinguish action requests from information questions.
            # Example:
            #   "Where do you work?"          -> keyword information request
            #   "Show me where you work."     -> intention: show_work
            detected_intention = self._detect_intention(visitor_text)

            # Ordinary natural-language questions still use the working keyword
            # system from v6.
            keyword_command = self._keyword_command(visitor_text)

            if not self.in_conversation or self.conversation_partner != sender:
                self._begin_scripted_conversation(sender, message_tick)

                if choice == "0":
                    reply_text = self._end_scripted_conversation()
                elif detected_intention is not None:
                    self._store_intention(
                        detected_intention,
                        sender,
                        message_tick,
                        visitor_text,
                    )
                    reply_text = self._intention_reply(detected_intention)
                elif choice in {"1", "2", "3", "9"}:
                    reply_text = self._main_dialogue_reply(choice)
                elif keyword_command is not None:
                    reply_text = self._reply_to_command(keyword_command)
                else:
                    reply_text = (
                        "Hello. I can answer a few questions about village life. "
                        + self.MAIN_DIALOGUE_MENU
                    )

                replies.append({"to": sender, "text": reply_text})
                continue

            if choice == "0":
                replies.append({"to": sender, "text": self._end_scripted_conversation()})
                continue

            # Action requests outrank information-keyword matching. Detect,
            # store, and accept/refuse here; act() performs accepted movement.
            if detected_intention is not None:
                self._store_intention(
                    detected_intention,
                    sender,
                    message_tick,
                    visitor_text,
                )
                replies.append(
                    {"to": sender, "text": self._intention_reply(detected_intention)}
                )
                continue

            # Natural language can jump directly to any known scripted topic,
            # regardless of the current numbered submenu.
            if keyword_command is not None:
                replies.append({"to": sender, "text": self._reply_to_command(keyword_command)})
                continue

            # Preserve the working numbered-menu behavior.
            if self.conversation_topic == "work":
                reply_text = self._work_dialogue_reply(choice)
            elif self.conversation_topic == "day":
                reply_text = self._day_dialogue_reply(choice)
            else:
                self.conversation_topic = "main"
                reply_text = self._main_dialogue_reply(choice)

            replies.append({"to": sender, "text": reply_text})

        return replies

    # ------------------------------------------------------------------
    # Small state/action helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: str, observation: ThreeBranchesObservation) -> None:
        if self.state == state:
            return

        self.state = state
        self.state_started_tick = day.tick(observation)
        self._clear_navigation(keep_destination=False)

        if state == self.VISIT_PUMP:
            self.pump_target_key = None
            self.pump_wait_ticks = 0
        elif state == self.REST:
            self.rest_bench = None
            self.rest_target_key = None
            self.rest_ticks = 0
            self.rest_attempt_ticks = 0
            self.rest_failed_ids.clear()
        elif state != self.REST:
            self.rest_attempt_ticks = 0

    def _set_destination_key(self, key: CellKey) -> None:
        if self.destination_key != key:
            self.destination_key = key
            self.route.clear()

    def _clear_navigation(self, keep_destination: bool = False) -> None:
        if not keep_destination:
            self.destination_key = None
        self.route.clear()
        self.expected_to_move = False

    def _walk(self, heading: float, speed: float, expression: str) -> ThreeBranchesAction:
        self.expected_to_move = speed > 0.05
        return action.walk(heading, speed, expression)

    def _stand(self, heading: float, expression: str) -> ThreeBranchesAction:
        self.expected_to_move = False
        return action.stand(heading, expression)

    @staticmethod
    def _player_number(player_id: str) -> int:
        try:
            return int(player_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    # Numbered choices, keyword matching, intention decisions, and temporary
    # ``show_work`` navigation are active. More intention types can be added later.
