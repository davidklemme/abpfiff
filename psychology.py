"""
Psychological engine - Phase 1 (MVP).

Implements the foundational slice from
docs/architecture/psychological-engine.md section 7.1:
  - composure (already an existing Player attribute)
  - confidence (a per-player state, -1 to 1)
  - basic pressure (spatial + tactical)
  - simple outcome -> confidence feedback

Phases 2-4 (full cognitive traits, instinct bank, memory/trauma, mentorship)
are out of scope here - see the design doc for the full model.
"""
from dataclasses import dataclass
from typing import Optional

from models import Player, Team, MatchEvent, MatchState


@dataclass
class PressureContext:
    """psychological-engine.md section 2.2. All fields are 0-1."""
    spatial: float = 0.0
    temporal: float = 0.0
    tactical: float = 0.0
    psychological: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.spatial * 0.35 +
            self.temporal * 0.30 +
            self.tactical * 0.20 +
            self.psychological * 0.15
        )


# Confidence deltas per observable outcome (psychological-engine.md section 5.3),
# trimmed to the event types this engine can actually detect in Phase 1.
CONFIDENCE_DELTAS = {
    "goal": 0.30,
    "goal_conceded": -0.15,
    "shot_on_target": 0.05,
    "miss": -0.08,
    "miss_clear_chance": -0.25,
    "save_against": -0.05,
    "save_made": 0.10,
    "pass": 0.02,
    "miscontrol": -0.08,
    "interception_won": 0.05,
    "interception_conceded": -0.05,
    "turnover": -0.10,
    "tackle_won": 0.10,
    "tackled": -0.10,
    "foul_committed": -0.03,
    "foul_won": 0.03,
    "yellow_card": -0.10,
    "red_card": -0.30,
}

# Clear-chance range: shots closer than this to goal count as "should score".
CLEAR_CHANCE_DISTANCE = 15.0

CONFIDENCE_DECAY_PER_MINUTE = 0.015


def effective_composure(player: Player) -> float:
    """Appendix D: composure boosted by confidence, dulled by mental fatigue.

    Physical `fatigue` (0-100) stands in for mental_fatigue since Phase 1
    doesn't track cognitive fatigue separately.
    """
    base = player.composure / 100.0
    mental_fatigue = min(1.0, player.fatigue / 100.0)
    value = base + player.confidence * 0.3 - mental_fatigue * 0.2
    return max(0.05, min(1.0, value))


def system1_weight(pressure_total: float, player: Player) -> float:
    """Appendix D system blend formula: how much instinct dominates (0-1)."""
    comp = effective_composure(player)
    if comp <= 0:
        return 1.0
    return max(0.0, min(1.0, pressure_total / comp))


def calculate_pressure(player: Player, state: MatchState, opponents: Team) -> PressureContext:
    """Basic Phase 1 pressure: spatial (nearest opponent) + tactical (scoreline/time)."""
    nearest_opp = min(
        (opp.position.distance_to(player.position) for opp in opponents.players),
        default=50.0
    )
    spatial = max(0.0, min(1.0, 1.0 - nearest_opp / 12.0))

    is_home = player in state.home_team.players
    own_score = state.home_score if is_home else state.away_score
    opp_score = state.away_score if is_home else state.home_score
    time_factor = min(1.0, state.minute / 90.0)

    # Continuous formula (no branching): a losing_margin of 0 collapses to the
    # baseline "time pressure only" case; being behind scales it up further.
    losing_margin = max(0.0, opp_score - own_score)
    tactical = min(1.0, time_factor * 0.15 + losing_margin * (0.3 + time_factor * 0.25))

    # Temporal pressure is a flat placeholder for Phase 1 - refining "time to
    # decide" requires action-level timing that doesn't exist yet.
    return PressureContext(spatial=spatial, temporal=0.25, tactical=tactical, psychological=0.0)


def _adjust(player: Optional[Player], key: str) -> None:
    if player is None:
        return
    delta = CONFIDENCE_DELTAS.get(key, 0.0)
    if delta:
        player.confidence = max(-1.0, min(1.0, player.confidence + delta))


def _conceding_goalkeeper(state: MatchState, scorer: Optional[Player]) -> Optional[Player]:
    if scorer is None:
        return None
    scorer_is_home = scorer in state.home_team.players
    conceding_team = state.away_team if scorer_is_home else state.home_team
    return conceding_team.goalkeeper


def _feedback_goal(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.player, "goal")
    _adjust(_conceding_goalkeeper(state, event.player), "goal_conceded")


def _feedback_shot(event: MatchEvent, state: MatchState) -> None:
    if event.success:
        _adjust(event.player, "shot_on_target")


def _feedback_miss(event: MatchEvent, state: MatchState) -> None:
    # Measure "clear chance" against the goal the shooter was attacking
    if event.player is not None and event.player in state.away_team.players:
        goal = state.away_team.attacking_goal
    else:
        goal = state.home_team.attacking_goal
    dist = event.position.distance_to(goal) if event.position else 30
    key = "miss_clear_chance" if dist < CLEAR_CHANCE_DISTANCE else "miss"
    _adjust(event.player, key)


def _feedback_save(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.target_player, "save_against")  # the shooter
    _adjust(event.player, "save_made")  # the goalkeeper


def _feedback_pass_received(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.target_player, "pass")  # target_player is the original passer


def _feedback_miscontrol(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.player, "miscontrol")


def _feedback_interception(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.player, "interception_won")
    _adjust(event.target_player, "interception_conceded")


def _feedback_turnover(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.player, "turnover")


def _feedback_tackle(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.player, "tackle_won")
    _adjust(event.target_player, "tackled")


def _feedback_foul(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.player, "foul_committed")   # the fouler
    _adjust(event.target_player, "foul_won")  # the fouled player


def _feedback_card(event: MatchEvent, state: MatchState) -> None:
    _adjust(event.player, event.event_type)  # "yellow_card" / "red_card"


# Dispatch table: event_type -> handler. Replaces an if/elif chain so adding
# a new observable outcome is a one-line addition, not a new branch.
FEEDBACK_HANDLERS = {
    "goal": _feedback_goal,
    "shot": _feedback_shot,
    "miss": _feedback_miss,
    "save": _feedback_save,
    "pass_received": _feedback_pass_received,
    "miscontrol": _feedback_miscontrol,
    "interception": _feedback_interception,
    "turnover": _feedback_turnover,
    "tackle": _feedback_tackle,
    "foul": _feedback_foul,
    "yellow_card": _feedback_card,
    "red_card": _feedback_card,
}


def process_feedback(event: MatchEvent, state: MatchState) -> None:
    """Translate a MatchEvent into confidence deltas (section 5.3/5.4, simplified)."""
    handler = FEEDBACK_HANDLERS.get(event.event_type)
    if handler:
        handler(event, state)


def decay_confidence(player: Player, minutes_elapsed: float) -> None:
    """Confidence drifts back toward neutral without reinforcing events."""
    if player.confidence == 0:
        return
    decay = CONFIDENCE_DECAY_PER_MINUTE * minutes_elapsed
    if player.confidence > 0:
        player.confidence = max(0.0, player.confidence - decay)
    else:
        player.confidence = min(0.0, player.confidence + decay)


def decay_all(state: MatchState, minutes_elapsed: float) -> None:
    for player in state.home_team.players + state.away_team.players:
        decay_confidence(player, minutes_elapsed)
