"""
Component interfaces for the match engine.

MatchEngine is an orchestrator composed of small, independently testable
units. Each unit is typed as a Protocol so alternative implementations
(simplified models for tests, experimental models for tuning) can be
injected without touching the engine loop.
"""
from typing import List, Optional, Protocol

from models import MatchEvent, MatchState, Player, Position, Team


class MovementModel(Protocol):
    """Moves every player one tick according to tactics and roles."""

    def update_positions(self, state: MatchState) -> None:
        ...


class ActionResolver(Protocol):
    """Resolves what happens with the ball this tick (at most one event)."""

    def resolve(self, state: MatchState) -> Optional[MatchEvent]:
        ...


class RestartPolicy(Protocol):
    """Puts the ball back into play: kickoffs and out-of-play restarts."""

    def kickoff(self, state: MatchState, home_kicks: bool) -> None:
        ...

    def resolve_out_of_bounds(self, state: MatchState, raw_pos: Position,
                              last_toucher: Player) -> MatchEvent:
        ...

    def goal_kick(self, state: MatchState, defending_team: Team,
                  description: str = "") -> MatchEvent:
        ...


class ConditioningModel(Protocol):
    """Per-tick physical/momentum bookkeeping (fatigue, momentum, ...)."""

    def update(self, state: MatchState, events: List[MatchEvent]) -> None:
        ...
