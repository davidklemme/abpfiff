"""
Perception: what the player on the ball actually sees.

Players are not omniscient. Inside their focus they perceive true
positions; outside it, experience fills the picture in with the
formation prior - where teammates and opponents SHOULD be. The perceived
world is a different input vector to the same decision pipeline
(embedding -> utilities -> instincts); no decision logic branches on it.

Beliefs can be wrong. A pass is aimed at the PERCEIVED position of the
receiver, and the ball physically flies there (models.Ball lead pass +
no-teleport arrival). When reality goes against the grain - the winger
who checked inside instead of holding width, the unseen presser sitting
in a lane that looked open - the ball runs loose or is intercepted, and
the learning layer records the lesson. Perception errors are
consequential through the existing machinery, with no new failure code.

Certainty per observed player:
  - falls off with distance from the holder,
  - the vision attribute widens the focus,
  - pressure narrows it (tunnel vision, via the System 1 weight - the
    same pressure-vs-composure blend that governs decisions),
  - players BEHIND the holder (frame-aware) are seen less.

Perceived position = certainty * true + (1 - certainty) * expected,
where expected is the formation prior (base position). A player near
their expected spot is 'seen' correctly even at low certainty; only the
unexpected can be misjudged - which is the point.
"""
from dataclasses import dataclass
from typing import List, Protocol

from models import MatchState, Player, Position, Team
import psychology


# Focus geometry (pitch units)
BASE_FOCUS_RANGE = 30.0     # what anyone takes in at a glance
VISION_FOCUS_RANGE = 40.0   # additional range scaled by the vision attribute
TUNNEL_VISION = 0.45        # how much full instinct-mode pressure narrows focus
BEHIND_CERTAINTY = 0.6      # players behind the holder are glimpsed, not seen
MIN_CERTAINTY = 0.05

# Environmental visibility (models.Environment.visibility): poor light
# shrinks everyone's focus, sensitive players' more (their attention is
# already taxed by the conditions).
VISIBILITY_IMPAIRMENT = 0.6

# Facing: derived from smoothed motion (velocity), not an attribute. A
# player moving fast enough has an orientation; the directional factor is
# CONTINUOUS in the angle (full sight ahead of motion, a glimpse behind).
# Stationary players fall back to the attack-direction proxy.
MIN_FACING_SPEED = 0.5


@dataclass
class Percept:
    """One observed (or assumed) player: belief, not necessarily truth."""
    player: Player
    position: Position
    certainty: float  # 0-1: how much of this is seen vs. filled in

    def effective_attribute(self, attr: str) -> float:
        return self.player.effective_attribute(attr)


@dataclass
class PerceivedWorld:
    """The holder's picture of the pitch at decision time."""
    teammates: List[Percept]
    opponents: List[Percept]


class PerceptionModel(Protocol):
    def perceive(self, holder: Player, attacking_team: Team,
                 defending_team: Team, state: MatchState) -> PerceivedWorld:
        ...


class OmniscientPerception:
    """The null model: perfect information (pre-perception behavior).
    Useful as a baseline and for tests that need ground truth."""

    def perceive(self, holder: Player, attacking_team: Team,
                 defending_team: Team, state: MatchState) -> PerceivedWorld:
        return PerceivedWorld(
            teammates=[Percept(p, p.position, 1.0)
                       for p in attacking_team.players if p is not holder],
            opponents=[Percept(p, p.position, 1.0)
                       for p in defending_team.players],
        )


class FocalPerception:
    """Limited focus filled in by the formation prior."""

    def perceive(self, holder: Player, attacking_team: Team,
                 defending_team: Team, state: MatchState) -> PerceivedWorld:
        focus_range = self._focus_range(holder, defending_team, state)
        f = attacking_team.frame_y
        holder_frame_y = f(holder.position.y)

        def percept_for(observed: Player) -> Percept:
            certainty = self._certainty(holder, observed, focus_range,
                                        f, holder_frame_y)
            perceived = Position(
                certainty * observed.position.x + (1 - certainty) * observed.base_position.x,
                certainty * observed.position.y + (1 - certainty) * observed.base_position.y,
            )
            return Percept(observed, perceived, certainty)

        return PerceivedWorld(
            teammates=[percept_for(p)
                       for p in attacking_team.players if p is not holder],
            opponents=[percept_for(p) for p in defending_team.players],
        )

    def _focus_range(self, holder: Player, defending_team: Team,
                     state: MatchState) -> float:
        """Vision widens the focus; pressure narrows it (tunnel vision);
        poor visibility shrinks it - more for sensitive players."""
        vision = holder.effective_attribute('vision') / 100.0
        base = BASE_FOCUS_RANGE + VISION_FOCUS_RANGE * vision

        # Tunnel vision: the pressure total already carries the
        # environment's mental load through the player's sensitivity
        pressure = psychology.calculate_pressure(holder, state, defending_team)
        sys1 = psychology.system1_weight(pressure.total, holder)
        base *= 1.0 - TUNNEL_VISION * sys1

        # Literal visibility (floodlights, fog): a perceptual channel,
        # distinct from psychological load
        murk = 1.0 - state.environment.visibility
        susceptibility = 0.4 + 0.6 * (holder.sensitivity / 100.0)
        return base * (1.0 - VISIBILITY_IMPAIRMENT * murk * susceptibility)

    def _certainty(self, holder: Player, observed: Player, focus_range: float,
                   frame_y, holder_frame_y: float) -> float:
        distance = holder.position.distance_to(observed.position)
        certainty = max(0.0, 1.0 - distance / max(1.0, focus_range))
        certainty *= self._direction_factor(holder, observed, frame_y,
                                            holder_frame_y)
        return max(MIN_CERTAINTY, min(1.0, certainty))

    def _direction_factor(self, holder: Player, observed: Player,
                          frame_y, holder_frame_y: float) -> float:
        """Continuous directional sight from the holder's facing (their
        smoothed motion): full certainty toward where they're heading,
        a glimpse behind. Stationary holders have no motion-derived
        facing and fall back to the attack-direction proxy."""
        vx, vy = holder.velocity_x, holder.velocity_y
        speed = (vx * vx + vy * vy) ** 0.5

        if speed >= MIN_FACING_SPEED:
            dx = observed.position.x - holder.position.x
            dy = observed.position.y - holder.position.y
            distance = max(1e-6, (dx * dx + dy * dy) ** 0.5)
            alignment = (vx * dx + vy * dy) / (speed * distance)  # cos angle
            # -0.5 (behind) -> glimpse, +0.5 (ahead) -> full sight
            t = max(0.0, min(1.0, alignment + 0.5))
            return BEHIND_CERTAINTY + (1.0 - BEHIND_CERTAINTY) * t

        # Attack-direction proxy for a player standing still
        if frame_y(observed.position.y) < holder_frame_y - 2.0:
            return BEHIND_CERTAINTY
        return 1.0
