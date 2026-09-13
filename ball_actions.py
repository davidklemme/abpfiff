"""
Ball action resolution: what the player in possession does, and how
passes, shots, dribbles, tackles and loose balls resolve.

Out-of-play situations (miscontrols over a line, deflected tackles,
parried shots, wide misses) are delegated to a RestartPolicy so the ball
always comes back via a rule-shaped restart (throw-in / goal kick /
corner) instead of teleporting.
"""
import random
from typing import Callable, Optional

from models import (
    Ball, BallState, MatchEvent, MatchState, Player, Position, Team
)
from spatial import SpaceControl
from restarts import SimpleRestartPolicy, is_out_of_bounds
from execution import execution_quality, contest
import psychology


# Pass-scoring multipliers keyed by under_pressure, used instead of inline
# if/else branches in _resolve_pass.
LATERAL_PRESSURE_MULTIPLIER = {True: 1.0, False: 0.6}
BACKWARD_PRESSURE_MULTIPLIER = {True: 0.8, False: 0.2}

# Chance a won tackle deflects the ball over the nearest touchline
TACKLE_DEFLECTION_CHANCE = 0.3
# Chance a save is parried behind for a corner instead of held
SAVE_PARRY_CHANCE = 0.3

# How close a player must be to a ball to take it under control. Possession
# never teleports: an arriving pass the receiver didn't reach runs loose,
# and a loose ball must be run down before it can be claimed.
CONTROL_RADIUS = 6.0
LOOSE_BALL_CLAIM_RADIUS = 4.0


class DefaultActionResolver:
    """Resolves the ball each tick; returns at most one primary MatchEvent.

    Secondary events that precede a restart (e.g. the miscontrol that put
    the ball out of play) are reported through the `publish` callback so the
    engine can emit them and feed psychology before the restart event.
    """

    def __init__(self, space_control: Optional[SpaceControl] = None,
                 restart_policy: Optional[SimpleRestartPolicy] = None,
                 rng: Optional[random.Random] = None,
                 randomness: float = 0.3,
                 publish: Optional[Callable[[MatchEvent, MatchState], None]] = None):
        self.space_control = space_control or SpaceControl(resolution=10)
        self.restarts = restart_policy or SimpleRestartPolicy(rng=rng)
        self.rng = rng or random.Random()
        self.randomness = randomness
        self.publish = publish or (lambda event, state: None)

    # -- main entry ---------------------------------------------------------

    def resolve(self, state: MatchState) -> Optional[MatchEvent]:
        """Resolve what happens with the ball"""
        ball = state.ball

        # Handle ball in flight (passes, shots traveling)
        if ball.is_in_flight():
            return self._resolve_ball_in_flight(state)

        holder = ball.holder

        # Handle loose ball
        if holder is None:
            return self._resolve_loose_ball(state)

        # IMPORTANT: Determine attacking/defending based on WHO HAS THE BALL
        holder_is_home = holder in state.home_team.players
        if holder_is_home:
            attacking_team = state.home_team
            defending_team = state.away_team
        else:
            attacking_team = state.away_team
            defending_team = state.home_team

        # Sync the home_attacking flag with reality
        if holder_is_home != state.home_attacking:
            state.home_attacking = holder_is_home

        # Ball position safety net
        ball.position = ball.position.clamp()

        # Determine action based on position and pressure
        pressure = self.space_control.pressing_effectiveness(
            defending_team, ball.position
        )
        num_pressers, pressure_value = pressure

        # Higher pressure = more likely to lose ball or rush action
        if pressure_value > 0.8 and self.rng.random() < 0.3:
            return self._resolve_turnover(state, holder, "pressed")

        # Decision: shoot, dribble, or pass (context-aware)
        action = self.decide_action(holder, attacking_team, defending_team, state)

        if action == "shoot":
            return self.resolve_shot(state, holder, attacking_team, defending_team)
        elif action == "dribble":
            return self._resolve_dribble(state, holder, attacking_team, defending_team)
        else:
            return self._resolve_pass(state, holder, attacking_team, defending_team)

    # -- decision -----------------------------------------------------------

    def decide_action(self, holder: Player, attacking_team: Team,
                      defending_team: Team, state: MatchState) -> str:
        """
        Decide whether to shoot, dribble, or pass based on context.
        Returns 'shoot', 'dribble', or 'pass'.
        """
        # Calculate factors
        in_shooting_range = self.in_shooting_range(holder.position, attacking_team)
        space_ahead = self._space_ahead(holder, attacking_team, defending_team)
        dribbling_skill = holder.effective_attribute('dribbling')
        passing_skill = holder.effective_attribute('passing')

        # Find nearest defender
        nearest_def_dist = min(
            (p.position.distance_to(holder.position) for p in defending_team.players),
            default=50
        )

        # Role-based tendencies
        dribble_roles = ['lw', 'rw', 'st', 'cf', 'am']  # These roles dribble more
        is_dribbler_role = holder.role in dribble_roles

        pressure = psychology.calculate_pressure(holder, state, defending_team)
        sys1 = psychology.system1_weight(pressure.total, holder)

        # SHOOTING
        if in_shooting_range:
            shoot_chance = 0.25
            # Better angle = more likely to shoot
            if 30 < holder.position.x < 70:
                shoot_chance += 0.15
            # High composure = clinical finisher
            if holder.effective_attribute('composure') > 70:
                shoot_chance += 0.1
            # Confidence pulls the trigger; panic under pressure holds it back
            shoot_chance += holder.confidence * 0.15 - sys1 * 0.1
            shoot_chance = max(0.05, min(0.9, shoot_chance))
            if self.rng.random() < shoot_chance:
                return "shoot"

        # DRIBBLING
        dribble_chance = 0.15  # Base chance

        # More space ahead = dribble more
        if space_ahead > 20:
            dribble_chance += 0.25
        elif space_ahead > 10:
            dribble_chance += 0.15

        # Good dribbler = dribble more
        if dribbling_skill > 75:
            dribble_chance += 0.2
        elif dribbling_skill > 60:
            dribble_chance += 0.1

        # Dribbling role = dribble more
        if is_dribbler_role:
            dribble_chance += 0.15

        # Nearest defender far away = dribble more
        if nearest_def_dist > 15:
            dribble_chance += 0.15

        # Poor passer = dribble more
        if passing_skill < 50:
            dribble_chance += 0.1

        # Confident players back themselves; instinct-dominant players under
        # pressure retreat to the simplest option instead of taking a risk
        dribble_chance += holder.confidence * 0.1 - sys1 * 0.1
        dribble_chance = max(0.02, dribble_chance)

        if self.rng.random() < dribble_chance:
            return "dribble"

        # Default to pass
        return "pass"

    def in_shooting_range(self, pos: Position, attacking_team: Team) -> bool:
        """Check if position is in shooting range of the attacked goal"""
        goal = attacking_team.attacking_goal
        distance = pos.distance_to(goal)
        return distance < 30 and attacking_team.frame_y(pos.y) > 70

    def _space_ahead(self, player: Player, attacking_team: Team,
                     defending_team: Team) -> float:
        """Calculate how much space a player has ahead of them"""
        # Check area in front of player (toward the goal they attack)
        f = attacking_team.frame_y
        check_y_frame = min(f(player.position.y) + 20, 100)
        check_pos = Position(player.position.x, f(check_y_frame))

        nearest_def = min(
            (p.position.distance_to(check_pos) for p in defending_team.players),
            default=50
        )
        return nearest_def

    # -- ball in flight -----------------------------------------------------

    def _resolve_ball_in_flight(self, state: MatchState) -> Optional[MatchEvent]:
        """Handle ball traveling through the air/ground"""
        ball = state.ball

        # Check for interceptions before ball moves
        interception_event = self._check_interception(state)
        if interception_event:
            return interception_event

        # Move ball
        arrived = ball.update_flight()

        if not arrived:
            # Ball still traveling - no event yet
            return None

        # Ball arrived at destination
        if ball.state == BallState.SHOT:
            return self._resolve_shot_arrival(state)
        else:
            return self._resolve_pass_arrival(state)

    def _check_interception(self, state: MatchState) -> Optional[MatchEvent]:
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
                    ball.give_to(defender)
                    state.switch_possession()

                    return MatchEvent(
                        minute=state.minute,
                        event_type="interception",
                        player=defender,
                        target_player=ball.passer,
                        position=ball.position,
                        success=True,
                        description=f"{defender.name} intercepts the pass"
                    )

        return None

    def _resolve_pass_arrival(self, state: MatchState) -> Optional[MatchEvent]:
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
        else:
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

    def _resolve_shot_arrival(self, state: MatchState) -> Optional[MatchEvent]:
        """Handle shot arriving at goal"""
        ball = state.ball
        shooter = ball.passer
        goal_pos = ball.target_position

        # Get goalkeeper
        goalkeeper = state.defending_team.goalkeeper

        if goalkeeper:
            # Goalkeeper save attempt: the keeper's own factor stack
            # (positioning skill, fatigue, pressure, confidence) vs. geometry
            gk_dist = goalkeeper.position.distance_to(goal_pos)
            gk_exec = execution_quality(goalkeeper, 'positioning', state,
                                        state.attacking_team,
                                        momentum=state.defending_team.momentum)
            gk_reactions = goalkeeper.effective_attribute('pace')  # Use pace for reactions

            # Save chance based on execution quality and distance
            save_chance = 0.25 + gk_exec * 0.45
            if gk_dist < 10:
                save_chance += 0.2
            save_chance += (gk_reactions / 100) * 0.1

            # Shot power affects save difficulty
            if shooter:
                shooting = shooter.effective_attribute('shooting')
                save_chance -= (shooting / 100) * 0.2

            save_chance = max(0.1, min(0.9, save_chance))

            if self.rng.random() < save_chance:
                save_event = MatchEvent(
                    minute=state.minute,
                    event_type="save",
                    player=goalkeeper,
                    target_player=shooter,
                    position=goal_pos,
                    success=True,
                    description=f"{goalkeeper.name} saves the shot!"
                )

                # Some saves are parried behind for a corner
                if self.rng.random() < SAVE_PARRY_CHANCE:
                    self.publish(save_event, state)
                    raw = Position(
                        10.0 if self.rng.random() < 0.5 else 90.0,
                        -1.0 if goal_pos.y == 0 else 101.0
                    )
                    return self.restarts.resolve_out_of_bounds(state, raw, goalkeeper)

                # Held: keeper keeps the ball
                ball.give_to(goalkeeper)
                state.switch_possession()
                return save_event

        # Goal scored!
        if state.home_attacking:
            state.home_score += 1
        else:
            state.away_score += 1

        # Reset for kickoff
        self.restarts.kickoff(state, home_kicks=not state.home_attacking)

        return MatchEvent(
            minute=state.minute,
            event_type="goal",
            player=shooter,
            position=goal_pos,
            success=True,
            description=f"GOAL! {shooter.name if shooter else 'Unknown'} scores!"
        )

    # -- loose ball ---------------------------------------------------------

    def _resolve_loose_ball(self, state: MatchState) -> Optional[MatchEvent]:
        """Resolve who gets a loose ball"""
        ball_pos = state.ball.position

        # Find nearest player from each team
        all_players = state.home_team.players + state.away_team.players
        nearest = min(all_players, key=lambda p: p.position.distance_to(ball_pos))

        # Nobody close enough yet: the ball stays loose and players run it
        # down through the movement model (loose-ball chase)
        if nearest.position.distance_to(ball_pos) > LOOSE_BALL_CLAIM_RADIUS:
            return None

        state.ball.give_to(nearest)

        # Update possession
        is_home_player = nearest in state.home_team.players
        if is_home_player != state.home_attacking:
            state.switch_possession()

        return MatchEvent(
            minute=state.minute,
            event_type="ball_won",
            player=nearest,
            position=ball_pos,
            success=True,
            description=f"{nearest.name} wins the ball"
        )

    # -- passing ------------------------------------------------------------

    def _resolve_pass(self, state: MatchState, passer: Player,
                      attacking_team: Team, defending_team: Team) -> Optional[MatchEvent]:
        """Resolve a pass attempt"""
        # Find passing options
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
            elif y_diff > 0:
                # Slight forward - small bonus
                score *= 1.2
            elif y_diff > -10:
                # Lateral pass - slight penalty unless under pressure
                score *= LATERAL_PRESSURE_MULTIPLIER[under_pressure]
                score *= 1.0 + sys1 * 0.2
            else:
                # Backward pass - heavy penalty unless under pressure
                score *= BACKWARD_PRESSURE_MULTIPLIER[under_pressure]
                # Rattled, instinct-dominant players overvalue the safe ball
                score *= 1.0 + sys1 * 0.4

            # Avoid passing back to last passer (anti ping-pong)
            if state.ball.passer == target:
                score *= 0.3

            # Bonus for passes into attacking third
            if f(target.position.y) > 70:
                score *= 1.3

            scored_lanes.append((target, max(0.05, score)))

        # Weighted random selection
        total = sum(q for _, q in scored_lanes)
        if total == 0:
            return None

        r = self.rng.random() * total
        cumulative = 0
        target = scored_lanes[0][0]

        for t, q in scored_lanes:
            cumulative += q
            if r <= cumulative:
                target = t
                break

        # Pass success from the full factor stack (skill, fatigue, pressure,
        # confidence, team momentum) plus the geometry of the chosen lane.
        # Interceptions and first-touch checks price further risk downstream.
        exec_q = execution_quality(passer, 'passing', state, defending_team,
                                   momentum=attacking_team.momentum)
        lane_quality = next((q for t, q in lanes if t == target), 0.5)
        success_prob = 0.62 + exec_q * 0.40 + lane_quality * 0.08

        # Add randomness
        success_prob += (self.rng.random() - 0.5) * self.randomness

        success = self.rng.random() < success_prob

        if success:
            # Start ball flight toward a lead position ahead of the receiver
            lead = self.lead_position(passer, target, attacking_team)
            distance = passer.position.distance_to(lead)
            is_lofted = distance > 25  # Long passes are lofted
            state.ball.start_pass(passer, target, is_lofted, lead_position=lead)
            return MatchEvent(
                minute=state.minute,
                event_type="pass",
                player=passer,
                target_player=target,
                position=passer.position,
                success=True,
                description=f"{passer.name} passes to {target.name}"
            )
        else:
            # Interception or misplaced
            return self._resolve_turnover(state, passer, "misplaced_pass")

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

    # -- dribbling / duels --------------------------------------------------

    def _resolve_dribble(self, state: MatchState, dribbler: Player,
                         attacking_team: Team,
                         defending_team: Team) -> Optional[MatchEvent]:
        """Resolve a dribble attempt"""
        # Find nearest defender
        defenders = [p for p in defending_team.players
                     if p.position.distance_to(dribbler.position) < 15]

        if not defenders:
            # Space to dribble
            self.move_dribbler(dribbler, state, attacking_team)
            return MatchEvent(
                minute=state.minute,
                event_type="dribble",
                player=dribbler,
                position=dribbler.position,
                success=True,
                description=f"{dribbler.name} carries the ball forward"
            )

        # Contested dribble: a duel of two full factor stacks - skill,
        # fatigue, pressure, confidence and momentum on BOTH sides
        nearest_def = min(defenders, key=lambda p: p.position.distance_to(dribbler.position))

        q_attacker = execution_quality(dribbler, 'dribbling', state,
                                       defending_team,
                                       momentum=attacking_team.momentum)
        q_defender = execution_quality(nearest_def, 'defending', state,
                                       attacking_team,
                                       momentum=defending_team.momentum)

        success_prob = contest(q_attacker, q_defender)
        success_prob += (self.rng.random() - 0.5) * self.randomness

        if self.rng.random() < success_prob:
            self.move_dribbler(dribbler, state, attacking_team)
            return MatchEvent(
                minute=state.minute,
                event_type="dribble",
                player=dribbler,
                position=dribbler.position,
                success=True,
                description=f"{dribbler.name} beats {nearest_def.name}"
            )
        else:
            return self._resolve_tackle(state, dribbler, nearest_def)

    def move_dribbler(self, dribbler: Player, state: MatchState,
                      attacking_team: Team):
        """Move dribbler forward toward the goal their team attacks"""
        target = attacking_team.attacking_goal
        new_pos = dribbler.position.move_towards(target, 2)
        dribbler.position = new_pos.clamp()
        state.ball.position = dribbler.position

    def _resolve_tackle(self, state: MatchState, attacker: Player,
                        defender: Player) -> MatchEvent:
        """Resolve a tackle attempt"""
        tackle_event = MatchEvent(
            minute=state.minute,
            event_type="tackle",
            player=defender,
            target_player=attacker,
            position=defender.position,
            success=True,
            description=f"{defender.name} tackles {attacker.name}"
        )

        # Some tackles deflect the ball over the nearest touchline:
        # throw-in for the attacker's team (defender touched it last)
        if self.rng.random() < TACKLE_DEFLECTION_CHANCE:
            self.publish(tackle_event, state)
            raw = Position(
                -1.0 if attacker.position.x < 50 else 101.0,
                attacker.position.y
            )
            return self.restarts.resolve_out_of_bounds(state, raw, defender)

        state.ball.give_to(defender)
        state.switch_possession()
        return tackle_event

    def _resolve_turnover(self, state: MatchState, loser: Player,
                          reason: str) -> MatchEvent:
        """Resolve a turnover"""
        # Find nearest opponent
        defending_team = state.defending_team
        nearest_opponent = min(
            defending_team.players,
            key=lambda p: p.position.distance_to(loser.position)
        )

        state.ball.give_to(nearest_opponent)
        state.switch_possession()

        return MatchEvent(
            minute=state.minute,
            event_type="turnover",
            player=loser,
            target_player=nearest_opponent,
            position=loser.position,
            success=False,
            description=f"{loser.name} loses the ball ({reason})"
        )

    # -- shooting -----------------------------------------------------------

    def resolve_shot(self, state: MatchState, shooter: Player,
                     attacking_team: Team, defending_team: Team) -> MatchEvent:
        """Start a shot attempt - ball will travel to goal"""
        goal = attacking_team.attacking_goal
        distance = shooter.position.distance_to(goal)

        # Shot quality from the shooter's full factor stack (shooting skill,
        # fatigue, pressure, confidence, momentum) shaped by the geometry
        exec_q = execution_quality(shooter, 'shooting', state, defending_team,
                                   momentum=attacking_team.momentum)
        composure = shooter.effective_attribute('composure')

        # Distance penalty
        distance_factor = max(0.2, 1 - (distance / 40))

        # Angle factor (shots from center are easier)
        angle_factor = 1 - abs(shooter.position.x - 50) / 100

        shot_quality = exec_q * distance_factor * angle_factor * 0.7
        shot_quality += (composure / 100) * 0.3
        shot_quality += (self.rng.random() - 0.5) * self.randomness

        # Check if shot is on target
        if self.rng.random() < shot_quality:
            # On target - start ball flying toward goal
            state.ball.start_shot(shooter, goal)
            return MatchEvent(
                minute=state.minute,
                event_type="shot",
                player=shooter,
                position=shooter.position,
                success=True,
                description=f"{shooter.name} shoots!"
            )
        else:
            # Missed - goes wide, restart with a goal kick for the defenders
            goal_kick_event = self.restarts.goal_kick(state, defending_team)
            self.publish(goal_kick_event, state)
            return MatchEvent(
                minute=state.minute,
                event_type="miss",
                player=shooter,
                position=shooter.position,
                success=False,
                description=f"{shooter.name}'s shot goes wide"
            )
