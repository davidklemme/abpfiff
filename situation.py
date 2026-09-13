"""
Situation embedding: a compact, continuous description of the moment a
player must decide in (psychological-engine.md section 3.5 / appendix B,
Phase 2 slice).

Every dimension is 0-1 and derived purely from signals the engine already
computes - pressure, frame-aware field position, match clock/score,
opponent density, passing support, width, occasion load - so the
embedding is a pure function of match state and can be tested in
isolation. (The engine now derives facing from motion -
engine._update_velocities - so the doc's body_orientation dimension is
unlocked; the fixed-scale similarity kernel below makes adding it cheap,
no threshold retune needed.)
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
    width: float = 0.5       # 0-1: how wide the holder is (0=center, 1=touchline)
    # 0-1: the occasion's mental load AS EXPERIENCED (environment through
    # the player's sensitivity). A dimension, not a side-channel: memories
    # carry the load they were formed under, so recognition is
    # state-dependent - the veteran's big-night anchors sit at high-load
    # coordinates and match the next big night by similarity alone, while
    # a quiet league game leaves them dormant (trauma resurfacing works
    # the same way, doc section 3.7).
    load: float = 0.0

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.pressure, self.progression, self.time_criticality,
                self.spatial_density, self.support, self.width, self.load)


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
        width=min(1.0, abs(holder.position.x - 50.0) / 50.0),
        load=state.environment.psychological_load(holder.sensitivity),
    )


# Distance budget the similarity kernel measures against. FIXED, not the
# dimension count: dividing by len(dims) would compress every existing
# contrast each time the embedding grows (agreement on a new axis is not
# evidence of sameness - it just dilutes the axes that DO differ).
# Against a fixed scale, a new dimension costs similarity only where two
# situations actually differ on it, so prototype discrimination and the
# thresholds built on it (instincts.MEMORY_MERGE_SIMILARITY, familiarity
# levels) stay stable as dimensions are added.
SIMILARITY_SCALE = 6.0


def similarity(a: SituationEmbedding, b: SituationEmbedding) -> float:
    """0-1 similarity between two situations (1 = identical).

    Total absolute difference over the dimensions against a fixed
    distance budget, inverted and floored at 0 - cheap, monotonic, and
    adequate for prototype matching in Phase 2."""
    ta, tb = a.as_tuple(), b.as_tuple()
    distance = sum(abs(x - y) for x, y in zip(ta, tb)) / SIMILARITY_SCALE
    return max(0.0, 1.0 - distance)
