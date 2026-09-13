"""
Pass play: target selection, lead passes, flight interception, arrival.

Extracted from the action resolver so pass mechanics are one testable
unit. All outcome probabilities flow through the execution-quality factor
stack (execution.py); "forward" is always measured in the attacking
team's frame.
"""
import random
from typing import Callable, Optional

from models import BallState, MatchEvent, MatchState, Player, Position, Team
from spatial import SpaceControl
from restarts import SimpleRestartPolicy, is_out_of_bounds
from execution import execution_quality
import psychology


# Pass-scoring multipliers keyed by under_pressure, used instead of inline
# if/else branches in resolve_pass.
LATERAL_PRESSURE_MULTIPLIER = {True: 1.0, False: 0.6}
BACKWARD_PRESSURE_MULTIPLIER = {True: 0.8, False: 0.2}

# How close a receiver must be to an arriving ball to take it under
# control - possession never teleports.
CONTROL_RADIUS = 6.0

# A defender intercepting a lofted ball near their own goal often can't
# control it - they clear it, sometimes behind for a corner or over the
# touchline for a throw-in.
CLEARANCE_ZONE_FRAME_Y = 22.0     # own-goal frame y below which headers are clearances
CLEARANCE_CHANCE = 0.5            # chance the interception is a clearance
CLEARANCE_OUT_CHANCE = 0.45       # cleared ball goes out of play
CLEARANCE_BEHIND_CHANCE = 0.4     # ...of which: behind for a corner (else throw-in)


def _real_player(target) -> Player:
    """A lane target may be a real Player or a perception.Percept
    (a belief about one); the ball's actual receiver is always real."""
    return getattr(target, "player", target)


def box_targets(holder: Player, attacking_team: Team) -> list:
    """Teammates positioned around the box (frame-aware) a cross could
    find - generous edges, since the delivery leads them further in."""
    f = attacking_team.frame_y
    return [p for p in attacking_team.players
            if p is not holder and p.role != "gk"
            and f(p.position.y) > 66 and 22 < p.position.x < 78]


class PassResolver:
    """Resolves everything between 'pass chosen' and 'ball controlled'."""

    def __init__(self, space_control: SpaceControl,
                 restart_policy: SimpleRestartPolicy,
                 rng: random.Random, randomness: float,
                 publish: Callable[[MatchEvent, MatchState], None]):
        self.space_control = space_control
        self.restarts = restart_policy
        self.rng = rng
        self.randomness = randomness
        self.publish = publish

    # -- launching -----------------------------------------------------------

    def lead_position(self, passer: Player, receiver: Player,
                      attacking_team: Team) -> Position:
        """Aim point for a pass: ahead of the receiver, into space they can
        actually reach during the ball's flight (frame-aware, so 'ahead'
        means toward the goal their team attacks)."""
        f = attacking_team.frame_y
        distance = passer.position.distance_to(receiver.position)
        flight_speed = 12 if distance > 25 else 18  # mirrors Ball flight speeds
        flight_ticks = max(1, int(distance / flight_speed))

        # Lead only as far as the receiver can run while the ball travels
        receiver_speed = (receiver.effective_attribute('pace') / 100) * 2.5
        lead = min(8.0, flight_ticks * receiver_speed * 0.8)

        lead_y_frame = min(98.0, f(receiver.position.y) + lead)
        return Position(receiver.position.x, f(lead_y_frame)).clamp()

    def resolve_pass(self, state: MatchState, passer: Player,
                     attacking_team: Team, defending_team: Team,
                     forward_bias: float = 0.0,
                     turnover: Optional[Callable] = None,
                     lanes: Optional[list] = None) -> Optional[MatchEvent]:
        """Resolve a pass attempt.

        `forward_bias` carries the decision layer's intent into target
        selection: positive (pass_forward) upweights progressive options,
        negative (pass_safe) upweights the safe ball. `turnover` resolves
        a misplaced pass (owned by the coordinating resolver).

        `lanes` are the (target, quality) options the DECISION saw - built
        on the holder's perceived world, so targets may be Percepts whose
        positions are beliefs. The pass is aimed at the believed position;
        if the belief was wrong, the ball physically goes to the wrong
        place. Without `lanes`, ground truth is used (omniscient)."""
        if lanes is None:
            lanes = self.space_control.find_passing_lanes(
                passer, attacking_team.players, defending_team.players
            )

        if not lanes:
            # No options, hold ball
            return None

        # Choose target based on game context
        directness = attacking_team.tactics.directness if attacking_team.tactics else 50

        # Check defensive pressure on passer
        pressure = sum(1 for d in defending_team.players
                      if d.position.distance_to(passer.position) < 10)
        under_pressure = pressure >= 2

        # Psychological pressure: how much this passer is leaning on instinct
        # (safe, comfort actions) vs. calm analysis (ambitious, forward play)
        psych_pressure = psychology.calculate_pressure(passer, state, defending_team)
        sys1 = psychology.system1_weight(psych_pressure.total, passer)

        # Score each passing option, with "forward" measured in the
        # attacking team's frame (toward the goal they are shooting at)
        f = attacking_team.frame_y
        scored_lanes = []
        for target, quality in lanes:
            score = quality

            # Calculate y-progress (positive = forward, negative = backward)
            y_diff = f(target.position.y) - f(passer.position.y)

            if y_diff > 10:
                # Strong forward pass - big bonus, but it takes composure and
                # confidence to actually play it under pressure
                score *= 1.5 + (directness / 100)
                score *= max(0.4, 1.0 + passer.confidence * 0.2 - sys1 * 0.3)
                score *= max(0.2, 1.0 + forward_bias * 0.6)
            elif y_diff > 0:
                # Slight forward - small bonus
                score *= 1.2
                score *= max(0.2, 1.0 + forward_bias * 0.3)
            elif y_diff > -10:
                # Lateral pass - slight penalty unless under pressure
                score *= LATERAL_PRESSURE_MULTIPLIER[under_pressure]
                score *= 1.0 + sys1 * 0.2
                score *= max(0.2, 1.0 - forward_bias * 0.3)
            else:
                # Backward pass - heavy penalty unless under pressure
                score *= BACKWARD_PRESSURE_MULTIPLIER[under_pressure]
                # Rattled, instinct-dominant players overvalue the safe ball
                score *= 1.0 + sys1 * 0.4
                score *= max(0.2, 1.0 - forward_bias * 0.5)

            # Avoid passing back to last passer (anti ping-pong)
            if state.ball.passer is _real_player(target):
                score *= 0.3

            # Bonus for passes into attacking third
            if f(target.position.y) > 70:
                score *= 1.3

            scored_lanes.append((target, max(0.05, score)))

        target = self._pick_weighted(scored_lanes)
        if target is None:
            return None

        # Pass success from the full factor stack (skill, fatigue, pressure,
        # confidence, team momentum) plus the geometry of the chosen lane.
        # Interceptions and first-touch checks price further risk downstream.
        exec_q = execution_quality(passer, 'passing', state, defending_team,
                                   momentum=attacking_team.momentum)
        lane_quality = next((q for t, q in lanes if t is target), 0.5)
        success_prob = 0.62 + exec_q * 0.40 + lane_quality * 0.08

        # Add randomness
        success_prob += (self.rng.random() - 0.5) * self.randomness

        if self.rng.random() < success_prob:
            # Aim ahead of the receiver's BELIEVED position: the ball flies
            # where the passer thinks their teammate will be. The real
            # receiver runs to meet it; if the belief was badly wrong, the
            # ball runs loose where it lands.
            lead = self.lead_position(passer, target, attacking_team)
            receiver = _real_player(target)
            distance = passer.position.distance_to(lead)
            is_lofted = distance > 25  # Long passes are lofted
            state.ball.start_pass(passer, receiver, is_lofted, lead_position=lead)
            return MatchEvent(
                minute=state.minute,
                event_type="pass",
                player=passer,
                target_player=receiver,
                position=passer.position,
                success=True,
                description=f"{passer.name} passes to {receiver.name}"
            )

        # Misplaced - the coordinating resolver turns it into a turnover
        if turnover is not None:
            return turnover(state, passer, "misplaced_pass")
        return None

    def _pick_weighted(self, scored_lanes) -> Optional[Player]:
        """Weighted random choice of pass target."""
        total = sum(q for _, q in scored_lanes)
        if total == 0:
            return None
        r = self.rng.random() * total
        cumulative = 0
        for target, quality in scored_lanes:
            cumulative += quality
            if r <= cumulative:
                return target
        return scored_lanes[0][0]

    def resolve_cross(self, state: MatchState, crosser: Player,
                      attacking_team: Team,
                      defending_team: Team) -> Optional[MatchEvent]:
        """Whip a lofted ball toward the best-placed teammate in the box."""
        targets = box_targets(crosser, attacking_team)
        if not targets:
            return None  # nobody to aim for; resolver falls back to a pass

        # Best target: the one with the most room around them
        def room(player):
            return min((d.position.distance_to(player.position)
                        for d in defending_team.players), default=50.0)

        target = max(targets, key=room)
        lead = self.lead_position(crosser, target, attacking_team)
        state.ball.start_pass(crosser, target, is_lofted=True, lead_position=lead)
        return MatchEvent(
            minute=state.minute,
            event_type="cross",
            player=crosser,
            target_player=target,
            position=crosser.position,
            success=True,
            description=f"{crosser.name} crosses toward {target.name}"
        )

    # -- in flight ------------------------------------------------------------

    def check_interception(self, state: MatchState) -> Optional[MatchEvent]:
        """Check if any defender intercepts a pass in flight"""
        ball = state.ball

        if ball.state == BallState.SHOT:
            return None  # Shots handled separately

        # Determine which team is defending
        if ball.passer and ball.passer in state.home_team.players:
            defending_team = state.away_team
        else:
            defending_team = state.home_team

        # Check each defender near the ball's path
        for defender in defending_team.players:
            dist_to_ball = defender.position.distance_to(ball.position)

            if dist_to_ball < 8:  # Within interception range
                # Interception from the defender's factor stack (positioning
                # skill, fatigue, pressure, confidence) plus pace to close.
                # NOTE: this check re-rolls every tick of flight for every
                # defender in range, so the per-roll chance must stay small.
                attacking_team = (state.home_team
                                  if defending_team is state.away_team
                                  else state.away_team)
                exec_q = execution_quality(defender, 'positioning', state,
                                           attacking_team,
                                           momentum=defending_team.momentum)
                base_chance = 0.01 + exec_q * 0.05

                # Ground passes easier to intercept than air
                if ball.state == BallState.GROUND_PASS:
                    base_chance += 0.015

                # Fast players close gap better
                pace = defender.effective_attribute('pace')
                base_chance += (pace - 50) / 400

                if self.rng.random() < base_chance:
                    return self._resolve_interception(state, defender,
                                                      defending_team)

        return None

    def _resolve_interception(self, state: MatchState, defender: Player,
                              defending_team: Team) -> MatchEvent:
        """The defender got to the ball first: controlled interception, or
        - for lofted balls near their own goal - a desperate clearance
        that can go anywhere, including out of play."""
        ball = state.ball
        interception = MatchEvent(
            minute=state.minute,
            event_type="interception",
            player=defender,
            target_player=ball.passer,
            position=ball.position,
            success=True,
            description=f"{defender.name} intercepts the pass"
        )

        near_own_goal = defending_team.frame_y(defender.position.y) < CLEARANCE_ZONE_FRAME_Y
        is_lofted = ball.state == BallState.AIR_PASS
        if is_lofted and near_own_goal and self.rng.random() < CLEARANCE_CHANCE:
            self.publish(interception, state)
            return self.clear_danger(state, defender, defending_team)

        ball.give_to(defender)
        state.switch_possession()
        return interception

    def clear_danger(self, state: MatchState, defender: Player,
                     defending_team: Team) -> MatchEvent:
        """Head/hack the ball away under pressure: upfield, over the
        touchline, or behind the goal for a corner. `defending_team` is
        the clearing player's own team (frame of reference)."""
        clearance = MatchEvent(
            minute=state.minute,
            event_type="clearance",
            player=defender,
            position=defender.position,
            success=True,
            description=f"{defender.name} clears the danger"
        )

        if self.rng.random() < CLEARANCE_OUT_CHANCE:
            self.publish(clearance, state)
            if self.rng.random() < CLEARANCE_BEHIND_CHANCE:
                # Behind their own goal line: corner
                raw = Position(10.0 if self.rng.random() < 0.5 else 90.0,
                               defending_team.frame_y(-1.0))
            else:
                # Over the nearest touchline: throw-in
                raw = Position(-1.0 if defender.position.x < 50 else 101.0,
                               defender.position.y)
            return self.restarts.resolve_out_of_bounds(state, raw, defender)

        # Hoofed upfield: lands loose in their own attacking half
        f = defending_team.frame_y
        target = Position(
            max(5.0, min(95.0, defender.position.x + self.rng.uniform(-20, 20))),
            f(min(95.0, f(defender.position.y) + self.rng.uniform(25, 45)))
        )
        state.ball.launch_clear(defender, target)
        return clearance

    # -- arrival ---------------------------------------------------------------

    def resolve_pass_arrival(self, state: MatchState) -> Optional[MatchEvent]:
        """Handle pass arriving at target"""
        ball = state.ball
        target = ball.target_player
        passer = ball.passer

        if target is None:
            ball.make_loose()
            return None

        # The receiver must actually be there: a lead pass they never
        # reached runs loose where it landed instead of teleporting to them
        if target.position.distance_to(ball.position) > CONTROL_RADIUS:
            ball.make_loose()
            return None

        # First touch from the receiver's full factor stack: a tired,
        # rattled or pressured receiver miscontrols far more often
        if target in state.home_team.players:
            own_team, opp_team = state.home_team, state.away_team
        else:
            own_team, opp_team = state.away_team, state.home_team
        exec_q = execution_quality(target, 'first_touch', state, opp_team,
                                   momentum=own_team.momentum)
        control_chance = 0.78 + exec_q * 0.22

        # Air balls harder to control
        if ball.state == BallState.AIR_PASS:
            control_chance -= 0.08

        if self.rng.random() < control_chance:
            ball.give_to(target)
            return MatchEvent(
                minute=state.minute,
                event_type="pass_received",
                player=target,
                target_player=passer,
                position=target.position,
                success=True,
                description=f"{target.name} receives from {passer.name if passer else 'unknown'}"
            )

        # Miscontrol - ball squirts loose nearby, possibly out of play
        raw = Position(
            target.position.x + self.rng.uniform(-5, 5),
            target.position.y + self.rng.uniform(-5, 5)
        )
        miscontrol = MatchEvent(
            minute=state.minute,
            event_type="miscontrol",
            player=target,
            position=target.position,
            success=False,
            description=f"{target.name} miscontrols the ball"
        )
        if is_out_of_bounds(raw):
            # Report the miscontrol, then restart play
            self.publish(miscontrol, state)
            return self.restarts.resolve_out_of_bounds(state, raw, target)

        ball.make_loose()
        ball.position = raw.clamp()
        return miscontrol
