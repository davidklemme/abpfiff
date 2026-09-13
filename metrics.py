"""
Match statistics collection for validation and tuning.

MatchMetrics is a passive observer: hook `on_event` into
MatchEngine.on_event and `on_tick` into MatchEngine.on_tick, run the
match, then read the per-team counters. It never mutates match state, so
it can be attached to any simulation without changing its outcome.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

from models import MatchEvent, MatchState, Team


@dataclass
class TeamCounters:
    """Raw per-team counters for one match."""
    goals: int = 0
    shots_on_target: int = 0
    shots_off_target: int = 0
    passes_launched: int = 0      # pass left the boot toward a teammate
    passes_completed: int = 0     # teammate controlled it
    passes_misplaced: int = 0     # never reached a teammate (turnover)
    possession_ticks: int = 0
    throw_ins: int = 0
    corners: int = 0
    goal_kicks: int = 0
    tackles_won: int = 0
    interceptions: int = 0

    @property
    def shots(self) -> int:
        return self.shots_on_target + self.shots_off_target

    @property
    def pass_attempts(self) -> int:
        return self.passes_launched + self.passes_misplaced

    @property
    def pass_completion(self) -> Optional[float]:
        attempts = self.pass_attempts
        if attempts == 0:
            return None
        return self.passes_completed / attempts


class MatchMetrics:
    """Collects one match's statistics from engine events and ticks."""

    def __init__(self, home: Team, away: Team):
        self.home_team = home
        self.away_team = away
        self.home = TeamCounters()
        self.away = TeamCounters()
        self.ticks = 0

    def _counters_for(self, player) -> Optional[TeamCounters]:
        if player is None:
            return None
        if player in self.home_team.players:
            return self.home
        if player in self.away_team.players:
            return self.away
        return None

    # -- observers ----------------------------------------------------------

    def on_event(self, event: MatchEvent) -> None:
        counters = self._counters_for(event.player)
        if counters is None:
            return

        etype = event.event_type
        if etype == "goal":
            counters.goals += 1
        elif etype == "shot":
            counters.shots_on_target += 1
        elif etype == "miss":
            counters.shots_off_target += 1
        elif etype == "pass":
            counters.passes_launched += 1
        elif etype == "pass_received":
            # player = receiver; the pass belongs to the same team
            counters.passes_completed += 1
        elif etype == "turnover" and "misplaced_pass" in event.description:
            counters.passes_misplaced += 1
        elif etype == "throw_in":
            counters.throw_ins += 1
        elif etype == "corner":
            counters.corners += 1
        elif etype == "goal_kick":
            counters.goal_kicks += 1
        elif etype == "tackle":
            counters.tackles_won += 1
        elif etype == "interception":
            counters.interceptions += 1

    def on_tick(self, state: MatchState) -> None:
        self.ticks += 1
        ball = state.ball
        controller = ball.holder or (ball.passer if ball.is_in_flight() else None)
        counters = self._counters_for(controller)
        if counters is not None:
            counters.possession_ticks += 1

    # -- derived ------------------------------------------------------------

    def possession_share(self) -> Optional[float]:
        """Home team's share of controlled ticks (0-1)."""
        controlled = self.home.possession_ticks + self.away.possession_ticks
        if controlled == 0:
            return None
        return self.home.possession_ticks / controlled

    def summary(self) -> Dict[str, float]:
        """Flat dict of the match's headline numbers."""
        return {
            "home_goals": self.home.goals,
            "away_goals": self.away.goals,
            "home_shots": self.home.shots,
            "away_shots": self.away.shots,
            "home_pass_attempts": self.home.pass_attempts,
            "away_pass_attempts": self.away.pass_attempts,
            "home_pass_completion": self.home.pass_completion or 0.0,
            "away_pass_completion": self.away.pass_completion or 0.0,
            "possession_home_share": self.possession_share() or 0.5,
            "throw_ins": self.home.throw_ins + self.away.throw_ins,
            "corners": self.home.corners + self.away.corners,
            "goal_kicks": self.home.goal_kicks + self.away.goal_kicks,
        }
