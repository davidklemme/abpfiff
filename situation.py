"""
Situation embedding: a compact, continuous description of the moment a
player must decide in (psychological-engine.md section 3.5 / appendix B,
Phase 2 slice).

Every dimension is 0-1 and derived purely from signals the engine already
computes - pressure, frame-aware field position, match clock/score,
opponent density, passing support - so the embedding is a pure function
of match state and can be tested in isolation. (The doc's
body_orientation dimension is deferred: the engine doesn't model facing.)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from models import MatchState, Player, Team
import psychology


@dataclass(frozen=True)
class SituationEmbedding:
    pressure: float          # 0-1: total pressure on the decision maker
    progression: float       # 0-1: how advanced the ball is (team frame)
    time_criticality: float  # 0-1: clock plus scoreline urgency
    spatial_density: float   # 0-1: opponents crowding the ball
    support: float           # 0-1: passing options available

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.pressure, self.progression, self.time_criticality,
                self.spatial_density, self.support)


def situation_for(holder: Player, attacking_team: Team, defending_team: Team,
                  state: MatchState, lanes: Optional[List] = None) -> SituationEmbedding:
    """Build the embedding for the current ball holder."""
    pressure = psychology.calculate_pressure(holder, state, defending_team).total

    progression = attacking_team.frame_y(holder.position.y) / 100.0

    is_home = attacking_team is state.home_team
    own = state.home_score if is_home else state.away_score
    opp = state.away_score if is_home else state.home_score
    clock = min(1.0, state.minute / 90.0)
    losing_boost = 0.25 if opp > own else 0.0
    time_criticality = min(1.0, clock * 0.75 + losing_boost)

    nearby = sum(1 for p in defending_team.players
                 if p.position.distance_to(holder.position) < 15)
    spatial_density = min(1.0, nearby / 5.0)

    if lanes is None:
        support = 0.5
    else:
        count_part = min(1.0, len(lanes) / 6.0)
        best_quality = max((q for _, q in lanes), default=0.0)
        support = min(1.0, 0.6 * count_part + 0.4 * min(1.0, best_quality * 2.0))

    return SituationEmbedding(
        pressure=max(0.0, min(1.0, pressure)),
        progression=max(0.0, min(1.0, progression)),
        time_criticality=time_criticality,
        spatial_density=spatial_density,
        support=support,
    )


def similarity(a: SituationEmbedding, b: SituationEmbedding) -> float:
    """0-1 similarity between two situations (1 = identical).

    Mean absolute difference over the dimensions, inverted - cheap,
    monotonic, and adequate for prototype matching in Phase 2."""
    ta, tb = a.as_tuple(), b.as_tuple()
    distance = sum(abs(x - y) for x, y in zip(ta, tb)) / len(ta)
    return 1.0 - distance
