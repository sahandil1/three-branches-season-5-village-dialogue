"""Validated generation tuning for the seeded village.

Every number the generator draws with lives in ``generation.json``. Code and tests read values from
here instead of restating them, so a tuning pass is a data edit. The document is validated when the
generation package imports, and the cross-group checks at the end of ``load`` are the arithmetic that
keeps the guarantees reachable: three mouths fit the south edge, and the widest trunk enters inside
its band. They bound what tuning allows, and each stage still checks the village it actually drew.

A group lands when its stage lands, so every number shipped here has a consumer.

A ``budget`` is a candidate budget: how many placements a mandatory stage may draw and test before
it gives up, discards the layout, and draws again. ``step_budget`` is not one of those; it counts a
walker's steps. Lantern and pine placement carries no budget at all, because an invalid spot there
is skipped rather than redrawn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from math import ceil
from typing import Any

from ..catalog import PROP_BY_TOKEN
from ..rules import FRAME
from ..validation import mapping, nonnegative_number, positive_int, positive_number


class Retry(Exception):
    """A mandatory placement ran out of candidates, so the whole layout is drawn again."""


def _int_range(value: Any, name: str) -> tuple[int, int]:
    """Accept inclusive whole-number bounds the generator draws between."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element array")
    low, high = (positive_int(item, name) for item in value)
    if low > high:
        raise ValueError(f"{name} must run from low to high")
    return low, high


def _number_range(value: Any, name: str) -> tuple[float, float]:
    """Accept inclusive bounds the generator draws a per-seed weight between."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element array")
    low, high = (positive_number(item, name) for item in value)
    if low > high:
        raise ValueError(f"{name} must run from low to high")
    return low, high


def _percent(value: Any, name: str) -> float:
    """Accept a frame share, written as a percentage for readable tuning."""
    number = nonnegative_number(value, name)
    if number > 100.0:
        raise ValueError(f"{name} must fall between zero and 100")
    return number


def _scaled_cells(percent: float, name: str, axis: int) -> int:
    """Resolve one configured frame percentage to the nearest whole cell."""
    cells = round(percent * axis / 100.0)
    if cells < 1:
        raise ValueError(f"{name} must resolve to at least one cell")
    return cells


def _percent_range(value: Any, name: str, axis: int) -> tuple[int, int]:
    """Resolve inclusive percentage bounds to the whole cells consumers use."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element array")
    low_percent, high_percent = (_percent(item, name) for item in value)
    if low_percent > high_percent:
        raise ValueError(f"{name} must run from low to high")
    low, high = (_scaled_cells(percent, name, axis) for percent in (low_percent, high_percent))
    return low, high


def _travel_budget(value: Any, name: str) -> int:
    """Resolve a per-frame-cell travel allowance so a walker crosses the whole map."""
    return ceil(positive_number(value, name) * max(FRAME.cells_x, FRAME.cells_y))


def _fraction(value: Any, name: str) -> float:
    """Accept a share of the unit range, which is the scale the noise fields are normalised to."""
    number = nonnegative_number(value, name)
    if number > 1.0:
        raise ValueError(f"{name} must fall between zero and one")
    return number


@dataclass(frozen=True, slots=True)
class Octave:
    """One layer of a noise field. Layering a few builds detail on top of broad shape."""

    # Cells between lattice nodes, resolved from frame-relative JSON tuning at load time. Wide
    # spacing gives broad country, narrow spacing gives texture.
    spacing: int
    # How much this layer counts against the others before the field is normalised.
    amplitude: float


@dataclass(frozen=True, slots=True)
class Fields:
    """The two noise fields every later stage reads: how high the land is and how wet it is."""

    # Shapes the land the water runs down.
    elevation_octaves: tuple[Octave, ...]
    # Shapes where reeds and fields grow, and which ground the road will prefer.
    moisture_octaves: tuple[Octave, ...]
    # How much elevation is lifted toward the north, which is what sends the water south. Zero
    # leaves the noise alone and one flattens it into a pure slope.
    south_bias: float


@dataclass(frozen=True, slots=True)
class Walker:
    """How a carving brush steers.

    Every step the walker blends the weights below into one heading, moves `step` along it, and
    paints. The three drawn ranges are sampled once per course, so two rivers on one map still
    behave differently from each other.
    """

    # Cells the brush travels each step. Shorter steps trace a smoother line and cost more time.
    step: float
    # Most steps one course may take, resolved from a per-frame-cell allowance at load time. A walk
    # that never reaches its stop line abandons the layout.
    step_budget: int
    # Recent steps of its own trail a course ignores when testing contact. Its trail is directly
    # behind it, so without this a course would block on the ground it just covered.
    self_ignore: int
    # When a step is blocked, how many turns to try and how far each one turns, alternating sides.
    reroute_attempts: int
    reroute_degrees: float
    # Weight on the heading it already had. High momentum makes a long, lazy curve.
    momentum: tuple[float, float]
    # Weight on the downhill slope of the elevation field.
    downhill: tuple[float, float]
    # Weight on the straight line to wherever the course is headed. This is what makes it arrive.
    pull: tuple[float, float]
    # Strength of the random nudge each step, which keeps the line from looking drawn.
    wobble: float
    # Strength of the turn away from the east and west frame edges.
    edge_push: float
    # Strength of the turn away from water another course already carved.
    separation: float
    # How far past the blocking clearance a course senses another. Sensing further than it blocks
    # is what lets it turn away in time instead of stopping dead against what it found.
    look_ahead: float
    # Strength of the sideways swing across the line to the target, and the cells travelled per
    # full swing. This is what bends a course rather than letting it run at its target.
    meander: float
    meander_wavelength: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Water:
    """The trunk, the fork, and the three channels that give the village its name."""

    # Frame-relative JSON bounds resolved to cells at load time. Columns the trunk may enter the
    # north edge in. The run it paints stays inside the band, not
    # just the point it starts from.
    entry_band: tuple[int, int]
    # Rows below the north edge where the trunk ends and the three channels take over. A deep fork
    # leaves the channels a long way to run; a shallow one keeps the braid near the top of the map.
    fork_band: tuple[int, int]
    # Cells across the trunk and across each channel. Both must be odd, because that is what a
    # brush centred on a cell can carve exactly.
    trunk_width: tuple[int, int]
    channel_width: tuple[int, int]
    # Least distance allowed between neighbouring mouth centres on the south edge.
    mouth_separation: int
    # Extra separation demanded of the drawn targets on top of that, so a course can wander on its
    # way down and still land far enough from its neighbour.
    mouth_slack: int
    # Rows at a frame edge over which a course runs straight. That is what makes the run it paints
    # where it meets the edge come out at exactly the width it was carved with.
    edge_straight: int
    # Columns of land kept clear along the east and west frame edges, so no course hugs the side.
    edge_margin: int
    # Radius of the shared pool where the trunk ends.
    fork_radius: float
    # Opening steps of each channel that count as shared water too. The pool alone is not enough
    # room for three channels to part, so their first reaches join the area where they may meet.
    fork_steps: int
    # How far the outer channels are turned away from their target as they leave the fork. This is
    # what fans them apart fast enough to stand clear of each other below the shared area.
    fan_degrees: float
    # Cells of land kept between any two courses outside that shared area.
    clearance: float
    # Draws allowed to find three mouth targets that keep their separation.
    mouth_budget: int
    walker: Walker


@dataclass(frozen=True, slots=True)
class Grounds:
    """Reed flats and terraced fields, painted over the open ground the water left."""

    # Cells from water within which a bank cell may turn to reeds, and the moisture it needs.
    reed_distance: int
    reed_moisture: float
    # Rows above the south edge where a bank cell turns to reeds whatever its moisture, so every
    # channel mouth collects them.
    mouth_reed_depth: int
    # A field wants land that is low, dry, and flat, so it takes elevation and moisture at or below
    # these and a local slope no steeper than this.
    field_elevation: float
    field_moisture: float
    field_slope: float
    # Majority passes over the land classes. Per-cell thresholds always speckle, and this clears it.
    smoothing_passes: int


@dataclass(frozen=True, slots=True)
class RoadWalker:
    """How the road steers. It carries a brush the way ``Walker`` describes, with its own terms.

    The road climbs toward drier ground rather than running downhill, and it is fenced by its band
    and by the water instead of by the frame edges.
    """

    step: float
    # Resolved from a per-frame-cell allowance at load time.
    step_budget: int
    self_ignore: int
    reroute_attempts: int
    reroute_degrees: float
    momentum: tuple[float, float]
    # Weight on the climb toward drier ground. The road reads moisture and not elevation, because
    # the southward elevation bias runs one way across the band and would pin the road to its edge.
    dry: tuple[float, float]
    pull: tuple[float, float]
    wobble: float
    # Strength of the turn back toward the middle of the road band.
    band_push: float
    # Strength of the turn away from water the road is not crossing.
    water_push: float
    look_ahead: float
    meander: float
    meander_wavelength: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Roadway:
    """The raised road, its bridges, and the band it winds inside."""

    # Cells across the road. Odd, for the same reason a water course is.
    width: int
    # Frame-relative JSON bounds resolved to cells at load time. They are a hard wall: the road is
    # turned back before it reaches one.
    band: tuple[int, int]
    # Cells of dry bank a bridge lands on at each end.
    apron: int
    # Longest straight cut a bridge may take across a channel, aprons included.
    crossing_run: int
    # Cells of land the road keeps between itself and water it is not crossing.
    water_clearance: float
    # Rows a district anchor's road target may be drawn away from the anchor, so the road passes
    # beside it differently each seed, and how close counts as passing it.
    anchor_swing: int
    anchor_reach: float
    # Cells at each frame edge over which the road runs straight, so its ends come out square.
    edge_straight: int
    walker: RoadWalker


@dataclass(frozen=True, slots=True)
class PathWalker:
    """How a footpath is walked, once the search has decided what it joins.

    The pull toward where the walk is going counts as one, so every weight here is read against it.
    The two drawn ranges are sampled once per path, so no two paths bend alike.
    """

    # Cells the walk travels each step. Shorter steps trace a smoother line and cost more time.
    step: float
    # Most steps one leg may take, resolved from a per-frame-cell allowance at load time. A leg
    # that runs out of them is painted along the searched route
    # instead, which is what keeps a doorway the search could reach joined either way.
    step_budget: int
    # When a step is blocked, how many turns to try and how far each one turns, alternating sides.
    reroute_attempts: int
    reroute_degrees: float
    # Weight on the heading it already had. High momentum makes a long, lazy curve.
    momentum: tuple[float, float]
    # Strength of the random nudge each step, which keeps the line from looking drawn.
    wobble: float
    # Strength of the sideways swing across the line to the target, and the cells travelled per full
    # swing. This is what keeps a path from running at its door in a line.
    meander: float
    meander_wavelength: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Path:
    """Footpaths: what the search decides one joins, and how the walk wears it."""

    width: int
    # How much cheaper an existing path cell is to reuse, which is what makes spurs share a route.
    merge_discount: float
    # Longest straight run a footpath may take across a channel.
    crossing_run: int
    # What one cell of that crossing costs the search, keeping a bridge a last resort.
    crossing_cost: float
    walker: PathWalker


@dataclass(frozen=True, slots=True)
class Spawn:
    """Where the visitor opens the day."""

    # Cells east of the west frame edge, along the road's straight entry run.
    edge_inset: int
    # Radius around the spawn that no footprint may enter.
    clearance: float


@dataclass(frozen=True, slots=True)
class Network:
    road: Roadway
    path: Path
    spawn: Spawn


@dataclass(frozen=True, slots=True)
class Scores:
    """What a home is looking for. Higher weight means the term counts for more."""

    bank: float
    flat: float
    dry: float
    apart: float


@dataclass(frozen=True, slots=True)
class Sites:
    """District anchors and the building sites placed against them."""

    # Clear cells kept around every site, keeping its routes and dressing out of it.
    margin: int
    # Cells of open ground a home may leave between its wall and its garden plot, drawn per home.
    garden_gap: int
    # Cells past a home wall's ends its garden may extend to, drawn per home.
    garden_slide: int
    # How far the well plaza reaches around its centre.
    plaza_radius: float
    # Frame-relative JSON distance resolved to cells at load time. How far from what a search is
    # aimed at it may look: the plaza from the fork, and the inn and
    # the shed from their anchors. A home is not aimed at anything, so it looks everywhere it may.
    reach: int
    # Candidates drawn for one anchor or one building. Running out discards the layout and draws
    # it again.
    budget: int
    scores: Scores


@dataclass(frozen=True, slots=True)
class Spot:
    """A lone prop placed beside something already committed: the board, the bell, the pump."""

    budget: int


@dataclass(frozen=True, slots=True)
class Stall:
    """Market stalls, scattered along both sides of the road through the market."""

    count: int
    # Cells of road between stall stations, and the road either side of the market they spread over.
    spacing: int
    span: int
    budget: int


@dataclass(frozen=True, slots=True)
class Bench:
    """Benches, split between the three places people wait."""

    plaza: int
    market: int
    inn: int
    budget: int


@dataclass(frozen=True, slots=True)
class Shrine:
    """Roadside shrines, which stand where the road turns hardest."""

    count: int
    # Cells of road kept between two shrines, and the road either side of a turn its sharpness is
    # measured over. A wide window reads the shape of a bend rather than one step's wobble.
    separation: int
    window: int
    budget: int


@dataclass(frozen=True, slots=True)
class Lantern:
    """Lantern posts along the road, closer together where the market is. Optional: a blocked
    station is skipped rather than redrawn."""

    spacing: int
    market_spacing: int


@dataclass(frozen=True, slots=True)
class Crate:
    """Market crates, littered loosely around the market. Optional: a blocked spot is skipped."""

    # Crates tried per village, and the cells either side of the market anchor they scatter over.
    count: int
    span: int


@dataclass(frozen=True, slots=True)
class Pine:
    """Red pines, placed last. Optional, like lanterns."""

    # Cells of road between pine stations, and open cells tried away from the road.
    spacing: int
    scatter: int
    # Cells kept between two pines, so a stand reads as trees rather than a hedge.
    gap: int
    # How often a pine brings neighbours, and how many it may bring.
    companion_chance: float
    companions: int
    # Inclusive collision factor each planted pine draws. The factor scales the solid circle's
    # radius, and the renderer layers a per-type visual scale on top, so collision and look are
    # separate knobs. It must not fall below 1.0: a smaller value would let a tree's own solid
    # shrink inside its reserved cell.
    size: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Accessories:
    """Everything placed once the ground, the buildings, and the routes are committed."""

    # Cells between the road edge and anything standing beside it.
    setback: int
    stall: Stall
    board: Spot
    bench: Bench
    shrine: Shrine
    lantern: Lantern
    bell: Spot
    pump: Spot
    crate: Crate
    pine: Pine


@dataclass(frozen=True, slots=True)
class Generation:
    fields: Fields
    water: Water
    grounds: Grounds
    network: Network
    sites: Sites
    accessories: Accessories
    # Whole layouts one seed may draw before generation gives up and raises. A draw is discarded
    # when a mandatory stage runs out of room, and the next one continues on the same stream.
    redraw_cap: int


def _octaves(value: Any, name: str) -> tuple[Octave, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return tuple(
        Octave(
            _scaled_cells(
                _percent(entry["spacing_percent"], f"{name}.spacing_percent"),
                f"{name}.spacing_percent",
                max(FRAME.cells_x, FRAME.cells_y),
            ),
            positive_number(entry["amplitude"], f"{name}.amplitude"),
        )
        for entry in (mapping(item, name, {"spacing_percent", "amplitude"}) for item in value)
    )


def _walker(value: Any) -> Walker:
    data = mapping(
        value,
        "water.walker",
        set(Walker.__dataclass_fields__) - {"step_budget", "meander_wavelength"}
        | {"step_budget_per_frame_cell", "meander_wavelength_percent"},
    )
    return Walker(
        positive_number(data["step"], "water.walker.step"),
        _travel_budget(data["step_budget_per_frame_cell"], "water.walker.step_budget_per_frame_cell"),
        positive_int(data["self_ignore"], "water.walker.self_ignore"),
        positive_int(data["reroute_attempts"], "water.walker.reroute_attempts"),
        positive_number(data["reroute_degrees"], "water.walker.reroute_degrees"),
        _number_range(data["momentum"], "water.walker.momentum"),
        _number_range(data["downhill"], "water.walker.downhill"),
        _number_range(data["pull"], "water.walker.pull"),
        nonnegative_number(data["wobble"], "water.walker.wobble"),
        nonnegative_number(data["edge_push"], "water.walker.edge_push"),
        nonnegative_number(data["separation"], "water.walker.separation"),
        positive_number(data["look_ahead"], "water.walker.look_ahead"),
        nonnegative_number(data["meander"], "water.walker.meander"),
        _percent_range(
            data["meander_wavelength_percent"],
            "water.walker.meander_wavelength_percent",
            max(FRAME.cells_x, FRAME.cells_y),
        ),
    )


def _road_walker(value: Any) -> RoadWalker:
    data = mapping(
        value,
        "network.road.walker",
        set(RoadWalker.__dataclass_fields__) - {"step_budget", "meander_wavelength"}
        | {"step_budget_per_frame_cell", "meander_wavelength_percent"},
    )
    name = "network.road.walker"
    return RoadWalker(
        positive_number(data["step"], f"{name}.step"),
        _travel_budget(data["step_budget_per_frame_cell"], f"{name}.step_budget_per_frame_cell"),
        positive_int(data["self_ignore"], f"{name}.self_ignore"),
        positive_int(data["reroute_attempts"], f"{name}.reroute_attempts"),
        positive_number(data["reroute_degrees"], f"{name}.reroute_degrees"),
        _number_range(data["momentum"], f"{name}.momentum"),
        _number_range(data["dry"], f"{name}.dry"),
        _number_range(data["pull"], f"{name}.pull"),
        nonnegative_number(data["wobble"], f"{name}.wobble"),
        nonnegative_number(data["band_push"], f"{name}.band_push"),
        nonnegative_number(data["water_push"], f"{name}.water_push"),
        positive_number(data["look_ahead"], f"{name}.look_ahead"),
        nonnegative_number(data["meander"], f"{name}.meander"),
        _percent_range(
            data["meander_wavelength_percent"],
            f"{name}.meander_wavelength_percent",
            max(FRAME.cells_x, FRAME.cells_y),
        ),
    )


def _path_walker(value: Any) -> PathWalker:
    data = mapping(
        value,
        "network.path.walker",
        set(PathWalker.__dataclass_fields__) - {"step_budget", "meander_wavelength"}
        | {"step_budget_per_frame_cell", "meander_wavelength_percent"},
    )
    name = "network.path.walker"
    return PathWalker(
        positive_number(data["step"], f"{name}.step"),
        _travel_budget(data["step_budget_per_frame_cell"], f"{name}.step_budget_per_frame_cell"),
        positive_int(data["reroute_attempts"], f"{name}.reroute_attempts"),
        positive_number(data["reroute_degrees"], f"{name}.reroute_degrees"),
        _number_range(data["momentum"], f"{name}.momentum"),
        nonnegative_number(data["wobble"], f"{name}.wobble"),
        nonnegative_number(data["meander"], f"{name}.meander"),
        _percent_range(
            data["meander_wavelength_percent"],
            f"{name}.meander_wavelength_percent",
            max(FRAME.cells_x, FRAME.cells_y),
        ),
    )


def _network(value: Any) -> Network:
    data = mapping(value, "generation.network", set(Network.__dataclass_fields__))
    road_data = mapping(
        data["road"],
        "network.road",
        set(Roadway.__dataclass_fields__) - {"band", "anchor_swing", "anchor_reach", "edge_straight"}
        | {"band_percent", "anchor_swing_percent", "anchor_reach_percent", "edge_straight_percent"},
    )
    path_data = mapping(data["path"], "network.path", set(Path.__dataclass_fields__))
    spawn_data = mapping(data["spawn"], "network.spawn", set(Spawn.__dataclass_fields__))
    return Network(
        Roadway(
            positive_int(road_data["width"], "network.road.width"),
            _percent_range(road_data["band_percent"], "network.road.band_percent", FRAME.cells_y),
            positive_int(road_data["apron"], "network.road.apron"),
            positive_int(road_data["crossing_run"], "network.road.crossing_run"),
            positive_number(road_data["water_clearance"], "network.road.water_clearance"),
            _scaled_cells(
                _percent(road_data["anchor_swing_percent"], "network.road.anchor_swing_percent"),
                "network.road.anchor_swing_percent",
                FRAME.cells_y,
            ),
            _scaled_cells(
                _percent(road_data["anchor_reach_percent"], "network.road.anchor_reach_percent"),
                "network.road.anchor_reach_percent",
                FRAME.cells_x,
            ),
            _scaled_cells(
                _percent(road_data["edge_straight_percent"], "network.road.edge_straight_percent"),
                "network.road.edge_straight_percent",
                FRAME.cells_y,
            ),
            _road_walker(road_data["walker"]),
        ),
        Path(
            positive_int(path_data["width"], "network.path.width"),
            _fraction(path_data["merge_discount"], "network.path.merge_discount"),
            positive_int(path_data["crossing_run"], "network.path.crossing_run"),
            positive_number(path_data["crossing_cost"], "network.path.crossing_cost"),
            _path_walker(path_data["walker"]),
        ),
        Spawn(
            positive_int(spawn_data["edge_inset"], "network.spawn.edge_inset"),
            positive_number(spawn_data["clearance"], "network.spawn.clearance"),
        ),
    )


def _sites(value: Any) -> Sites:
    data = mapping(value, "generation.sites", set(Sites.__dataclass_fields__) - {"reach"} | {"reach_percent"})
    scores = mapping(data["scores"], "sites.scores", set(Scores.__dataclass_fields__))
    return Sites(
        positive_int(data["margin"], "sites.margin"),
        positive_int(data["garden_gap"], "sites.garden_gap"),
        positive_int(data["garden_slide"], "sites.garden_slide"),
        positive_number(data["plaza_radius"], "sites.plaza_radius"),
        _scaled_cells(
            _percent(data["reach_percent"], "sites.reach_percent"),
            "sites.reach_percent",
            max(FRAME.cells_x, FRAME.cells_y),
        ),
        positive_int(data["budget"], "sites.budget"),
        Scores(
            nonnegative_number(scores["bank"], "sites.scores.bank"),
            nonnegative_number(scores["flat"], "sites.scores.flat"),
            nonnegative_number(scores["dry"], "sites.scores.dry"),
            nonnegative_number(scores["apart"], "sites.scores.apart"),
        ),
    )


def _spot(value: Any, name: str) -> Spot:
    return Spot(positive_int(mapping(value, name, {"budget"})["budget"], f"{name}.budget"))


def _accessories(value: Any) -> Accessories:
    data = mapping(value, "generation.accessories", set(Accessories.__dataclass_fields__))
    stall = mapping(data["stall"], "accessories.stall", set(Stall.__dataclass_fields__))
    bench = mapping(data["bench"], "accessories.bench", set(Bench.__dataclass_fields__))
    shrine = mapping(data["shrine"], "accessories.shrine", set(Shrine.__dataclass_fields__))
    lantern = mapping(data["lantern"], "accessories.lantern", set(Lantern.__dataclass_fields__))
    crate = mapping(data["crate"], "accessories.crate", set(Crate.__dataclass_fields__))
    pine = mapping(data["pine"], "accessories.pine", set(Pine.__dataclass_fields__))
    return Accessories(
        positive_int(data["setback"], "accessories.setback"),
        Stall(
            positive_int(stall["count"], "accessories.stall.count"),
            positive_int(stall["spacing"], "accessories.stall.spacing"),
            positive_int(stall["span"], "accessories.stall.span"),
            positive_int(stall["budget"], "accessories.stall.budget"),
        ),
        _spot(data["board"], "accessories.board"),
        Bench(
            positive_int(bench["plaza"], "accessories.bench.plaza"),
            positive_int(bench["market"], "accessories.bench.market"),
            positive_int(bench["inn"], "accessories.bench.inn"),
            positive_int(bench["budget"], "accessories.bench.budget"),
        ),
        Shrine(
            positive_int(shrine["count"], "accessories.shrine.count"),
            positive_int(shrine["separation"], "accessories.shrine.separation"),
            positive_int(shrine["window"], "accessories.shrine.window"),
            positive_int(shrine["budget"], "accessories.shrine.budget"),
        ),
        Lantern(
            positive_int(lantern["spacing"], "accessories.lantern.spacing"),
            positive_int(lantern["market_spacing"], "accessories.lantern.market_spacing"),
        ),
        _spot(data["bell"], "accessories.bell"),
        _spot(data["pump"], "accessories.pump"),
        Crate(
            positive_int(crate["count"], "accessories.crate.count"),
            positive_int(crate["span"], "accessories.crate.span"),
        ),
        Pine(
            positive_int(pine["spacing"], "accessories.pine.spacing"),
            positive_int(pine["scatter"], "accessories.pine.scatter"),
            positive_int(pine["gap"], "accessories.pine.gap"),
            _fraction(pine["companion_chance"], "accessories.pine.companion_chance"),
            positive_int(pine["companions"], "accessories.pine.companions"),
            _number_range(pine["size"], "accessories.pine.size"),
        ),
    )


def load(data: Any) -> Generation:
    """Validate a decoded generation document without accepting unrecognised data."""
    root = mapping(
        data,
        "generation",
        {"fields", "water", "grounds", "network", "sites", "accessories", "redraw"},
    )
    fields_data = mapping(root["fields"], "generation.fields", set(Fields.__dataclass_fields__))
    field_tuning = Fields(
        _octaves(fields_data["elevation_octaves"], "fields.elevation_octaves"),
        _octaves(fields_data["moisture_octaves"], "fields.moisture_octaves"),
        _fraction(fields_data["south_bias"], "fields.south_bias"),
    )
    water_data = mapping(
        root["water"],
        "generation.water",
        set(Water.__dataclass_fields__)
        - {
            "entry_band",
            "fork_band",
            "mouth_separation",
            "mouth_slack",
            "edge_margin",
            "fork_radius",
            "fork_steps",
            "edge_straight",
        }
        | {
            "entry_band_percent",
            "fork_band_percent",
            "mouth_separation_percent",
            "mouth_slack_percent",
            "edge_margin_percent",
            "fork_radius_percent",
            "fork_steps_percent",
            "edge_straight_percent",
        },
    )
    water_tuning = Water(
        _percent_range(water_data["entry_band_percent"], "water.entry_band_percent", FRAME.cells_x),
        _percent_range(water_data["fork_band_percent"], "water.fork_band_percent", FRAME.cells_y),
        _int_range(water_data["trunk_width"], "water.trunk_width"),
        _int_range(water_data["channel_width"], "water.channel_width"),
        _scaled_cells(
            _percent(water_data["mouth_separation_percent"], "water.mouth_separation_percent"),
            "water.mouth_separation_percent",
            FRAME.cells_x,
        ),
        _scaled_cells(
            _percent(water_data["mouth_slack_percent"], "water.mouth_slack_percent"),
            "water.mouth_slack_percent",
            FRAME.cells_x,
        ),
        _scaled_cells(
            _percent(water_data["edge_straight_percent"], "water.edge_straight_percent"),
            "water.edge_straight_percent",
            FRAME.cells_y,
        ),
        _scaled_cells(
            _percent(water_data["edge_margin_percent"], "water.edge_margin_percent"),
            "water.edge_margin_percent",
            FRAME.cells_x,
        ),
        _scaled_cells(
            _percent(water_data["fork_radius_percent"], "water.fork_radius_percent"),
            "water.fork_radius_percent",
            max(FRAME.cells_x, FRAME.cells_y),
        ),
        _scaled_cells(
            _percent(water_data["fork_steps_percent"], "water.fork_steps_percent"),
            "water.fork_steps_percent",
            FRAME.cells_y,
        ),
        positive_number(water_data["fan_degrees"], "water.fan_degrees"),
        positive_number(water_data["clearance"], "water.clearance"),
        positive_int(water_data["mouth_budget"], "water.mouth_budget"),
        _walker(water_data["walker"]),
    )
    grounds_keys = set(Grounds.__dataclass_fields__) - {"mouth_reed_depth"} | {"mouth_reed_depth_percent"}
    grounds_data = mapping(root["grounds"], "generation.grounds", grounds_keys)
    grounds_tuning = Grounds(
        positive_int(grounds_data["reed_distance"], "grounds.reed_distance"),
        _fraction(grounds_data["reed_moisture"], "grounds.reed_moisture"),
        _scaled_cells(
            _percent(grounds_data["mouth_reed_depth_percent"], "grounds.mouth_reed_depth_percent"),
            "grounds.mouth_reed_depth_percent",
            FRAME.cells_y,
        ),
        _fraction(grounds_data["field_elevation"], "grounds.field_elevation"),
        _fraction(grounds_data["field_moisture"], "grounds.field_moisture"),
        positive_number(grounds_data["field_slope"], "grounds.field_slope"),
        positive_int(grounds_data["smoothing_passes"], "grounds.smoothing_passes"),
    )
    redraw_data = mapping(root["redraw"], "generation.redraw", {"cap"})
    tuning = Generation(
        field_tuning,
        water_tuning,
        grounds_tuning,
        _network(root["network"]),
        _sites(root["sites"]),
        _accessories(root["accessories"]),
        positive_int(redraw_data["cap"], "redraw.cap"),
    )
    _check_frame_arithmetic(tuning)
    return tuning


def _check_frame_arithmetic(tuning: Generation) -> None:
    """Reject tuning the shipped frame cannot satisfy, before any village is drawn."""
    water = tuning.water
    for name, bounds in (
        ("water.trunk_width", water.trunk_width),
        ("water.channel_width", water.channel_width),
    ):
        if not any(width % 2 for width in range(bounds[0], bounds[1] + 1)):
            raise ValueError(f"{name} must contain an odd width, which is what a cell-centred brush carves")
    if water.fork_band[1] >= FRAME.cells_y or water.entry_band[1] >= FRAME.cells_x:
        raise ValueError("water.fork_band and water.entry_band must fall inside the frame")
    trunk_reach = ceil(water.trunk_width[1] / 2)
    channel_reach = ceil(water.channel_width[1] / 2)
    mouth_span = FRAME.cells_x - 1 - 2 * (water.edge_margin + channel_reach)
    if 2 * (water.mouth_separation + water.mouth_slack) >= mouth_span:
        raise ValueError("water.mouth_separation and water.mouth_slack leave no room for three mouths")
    if water.entry_band[1] - water.entry_band[0] <= water.trunk_width[1]:
        raise ValueError("water.entry_band must be wider than the widest trunk it has to contain")
    if water.entry_band[0] - trunk_reach < water.edge_margin:
        raise ValueError("water.entry_band must keep the widest trunk inside water.edge_margin")
    if water.entry_band[1] + trunk_reach > FRAME.cells_x - 1 - water.edge_margin:
        raise ValueError("water.entry_band must keep the widest trunk inside water.edge_margin")
    road = tuning.network.road
    if road.width % 2 == 0:
        raise ValueError("network.road.width must be odd, which is what a cell-centred brush paints")
    if road.band[0] < 1 or road.band[1] >= FRAME.cells_y - 1:
        raise ValueError("network.road.band must fall inside the frame")
    if road.band[1] - road.band[0] + 1 < road.width + 4:
        raise ValueError("network.road.band must be deep enough for the road to wind inside it")
    if road.band[1] >= FRAME.cells_y - 1 - water.fork_band[1]:
        raise ValueError("network.road.band must stay south of the deepest fork it may meet")
    if road.crossing_run < water.channel_width[1] + 2 * road.apron:
        raise ValueError("network.road.crossing_run must span the widest channel and both aprons")
    path = tuning.network.path
    if path.crossing_run < water.channel_width[1]:
        raise ValueError("network.path.crossing_run must span the widest channel")
    if tuning.network.spawn.edge_inset > road.edge_straight:
        raise ValueError("network.spawn.edge_inset must land on the road's straight entry run")
    if tuning.sites.margin < PROP_BY_TOKEN["plot"].height:
        raise ValueError("sites.margin must keep at least a plot's depth of clearance around a site")
    stall = tuning.accessories.stall
    if 2 * (stall.span // stall.spacing) + 1 < stall.count:
        raise ValueError("accessories.stall.span and spacing leave too few stations for the stalls")


GENERATION = load(
    json.loads(resources.files(__package__).joinpath("generation.json").read_text(encoding="utf-8"))
)
