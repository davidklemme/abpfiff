"""
Match simulation engine.
Ties together spatial control, tactical principles, and event resolution.
"""
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import random
import math

from models import (
    Player, Team, Ball, Position, MatchState, MatchEvent, Phase, BallState
)
from spatial import SpaceControl, PassingGraph
from tactics import TacticalSetup, TacticalPrinciple, MovementInstruction
import psychology


# Pass-scoring multipliers keyed by under_pressure, used instead of inline
# if/else branches in _resolve_pass.
LATERAL_PRESSURE_MULTIPLIER = {True: 1.0, False: 0.6}
BACKWARD_PRESSURE_MULTIPLIER = {True: 0.8, False: 0.2}


@dataclass
class SimulationConfig:
    """Configuration for the simulation"""
    ticks_per_minute: int = 10  # Higher = more detailed simulation
    fatigue_rate: float = 0.1  # Fatigue gain per tick
    momentum_decay: float = 0.95  # How quickly momentum normalizes
    randomness: float = 0.3  # 0-1, how much randomness affects outcomes
    debug: bool = False


class MatchEngine:
    """
    The main match simulation engine.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.space_control = SpaceControl(resolution=10)
        self.event_handlers: List[Callable] = []

    def on_event(self, handler: Callable):
        """Register event handler"""
        self.event_handlers.append(handler)

    def _emit_event(self, event: MatchEvent):
        """Emit event to all handlers"""
        for handler in self.event_handlers:
            handler(event)

    def simulate_match(self, state: MatchState, minutes: int = 90) -> MatchState:
        """Simulate a full match or portion of it"""
        for minute in range(state.minute, state.minute + minutes):
            state.minute = minute
            self._simulate_minute(state)

            # Half time
            if minute == 45:
                self._half_time(state)

        return state

    def simulate_tick(self, state: MatchState) -> List[MatchEvent]:
        """Simulate a single tick of the match"""
        events = []

        # 0. Update tick counters
        state.tick()

        # 1. Apply tactical movements
        self._apply_tactical_movements(state)

        # 2. Update space control (implicit in spatial calculations)

        # 3. Resolve ball action
        action_event = self._resolve_ball_action(state)
        if action_event:
            events.append(action_event)
            self._emit_event(action_event)
            psychology.process_feedback(action_event, state)

        # 4. Update fatigue
        self._update_fatigue(state)

        # 5. Update momentum
        self._update_momentum(state, events)

        # 6. Confidence drifts back toward neutral over time
        psychology.decay_all(state, minutes_elapsed=1.0 / self.config.ticks_per_minute)

        return events

    def _simulate_minute(self, state: MatchState) -> List[MatchEvent]:
        """Simulate one minute of play"""
        all_events = []

        for _ in range(self.config.ticks_per_minute):
            events = self.simulate_tick(state)
            all_events.extend(events)

        return all_events

    def _apply_tactical_movements(self, state: MatchState):
        """Move players according to tactical principles"""
        # Home team
        self._apply_team_movements(
            state.home_team, state, state.home_attacking
        )
        # Away team (note: their y-axis is flipped conceptually)
        self._apply_team_movements(
            state.away_team, state, not state.home_attacking
        )

    def _apply_team_movements(self, team: Team, state: MatchState,
                              team_attacking: bool):
        """Apply tactical movements for one team"""
        tactics = team.tactics
        other_team = state.away_team if team == state.home_team else state.home_team

        for player in team.players:
            # Skip ball holder - they move via dribble
            if player.has_ball:
                continue

            # Get active principles for this player
            principles = []
            if tactics:
                principles = tactics.get_active_principles(player, state, team_attacking)

            if principles:
                # Apply highest priority principle
                principle = principles[0]
                self._apply_movement(player, principle.movement, state, team)
            else:
                # Smart default movement based on role and game state
                self._apply_default_movement(player, state, team, team_attacking, other_team)

    def _apply_movement(self, player: Player, movement: MovementInstruction,
                        state: MatchState, team: Team):
        """Apply a movement instruction to a player"""
        target_x = player.position.x
        target_y = player.position.y

        # Absolute target
        if movement.target_x is not None:
            target_x = movement.target_x
        if movement.target_y is not None:
            target_y = movement.target_y

        # Relative movement
        target_x += movement.relative_x
        target_y += movement.relative_y

        # Move toward ball
        if movement.towards_ball > 0:
            ball_pos = state.ball.position
            target_x += (ball_pos.x - player.position.x) * movement.towards_ball
            target_y += (ball_pos.y - player.position.y) * movement.towards_ball

        # Move toward open space
        if movement.towards_space > 0:
            open_spaces = self.space_control.find_open_spaces(
                team, state.away_team if state.home_attacking else state.home_team
            )
            if open_spaces:
                # Find nearest advantageous space
                best_space = min(
                    open_spaces,
                    key=lambda s: player.position.distance_to(s)
                )
                target_x += (best_space.x - player.position.x) * movement.towards_space * 0.3
                target_y += (best_space.y - player.position.y) * movement.towards_space * 0.3

        # Maintain shape (blend toward base position)
        if movement.maintain_shape > 0:
            target_x = target_x * (1 - movement.maintain_shape) + player.base_position.x * movement.maintain_shape
            target_y = target_y * (1 - movement.maintain_shape) + player.base_position.y * movement.maintain_shape

        # Calculate movement speed based on pace and fatigue
        max_speed = (player.effective_attribute('pace') / 100) * 3  # Max 3 units per tick
        target_pos = Position(target_x, target_y).clamp()

        # Move toward target
        new_pos = player.position.move_towards(target_pos, max_speed)
        player.position = new_pos.clamp()

    def _move_toward_base(self, player: Player):
        """Default movement: drift toward base position"""
        new_pos = player.position.move_towards(player.base_position, 0.5)
        player.position = new_pos.clamp()

    def _apply_default_movement(self, player: Player, state: MatchState,
                                 team: Team, team_attacking: bool, other_team: Team):
        """
        Smart default movement when no tactical principle applies.
        This is the heart of emergent tactical behavior.
        """
        ball_pos = state.ball.position
        ball_holder = state.ball.holder

        # Calculate target position based on game state
        target_x = player.position.x
        target_y = player.position.y

        if team_attacking:
            # ATTACKING: Support play, find space, make runs
            target_x, target_y = self._attacking_movement(
                player, ball_pos, ball_holder, team, other_team, state
            )
        else:
            # DEFENDING: Track runners, maintain shape, close space
            target_x, target_y = self._defending_movement(
                player, ball_pos, ball_holder, team, other_team, state
            )

        # Movement speed based on player attributes and urgency
        pace = player.effective_attribute('pace')
        workrate = player.effective_attribute('workrate')
        urgency = self._calculate_urgency(player, ball_pos, team_attacking)

        max_speed = (pace / 100) * 2.5 * (0.5 + workrate / 200) * urgency

        # Apply movement
        target_pos = Position(target_x, target_y).clamp()
        new_pos = player.position.move_towards(target_pos, max_speed)
        player.position = new_pos.clamp()

    def _attacking_movement(self, player: Player, ball_pos: Position,
                            ball_holder: Optional[Player], team: Team,
                            other_team: Team, state: MatchState) -> tuple:
        """Calculate attacking movement for player"""
        target_x = player.position.x
        target_y = player.position.y

        # Role-based attacking behavior
        role = player.role.lower()
        distance_to_ball = player.position.distance_to(ball_pos)

        if role in ['gk']:
            # GK stays back but supports buildup
            target_y = player.base_position.y + 5
            target_x = player.base_position.x

        elif role in ['cb', 'cb_l', 'cb_r']:
            # CBs provide safety, spread for buildup
            target_y = min(player.base_position.y + 15, ball_pos.y - 20)
            # Spread horizontally when team has ball
            spread = 10 if player.position.x < 50 else -10
            target_x = player.base_position.x + spread

        elif role in ['lb', 'rb', 'lwb', 'rwb']:
            # Fullbacks push forward and wide to stretch play
            target_y = min(ball_pos.y + 10, 85)
            target_x = 15 if 'l' in role else 85
            # If ball is on their side, get even higher
            if (player.position.x < 50) == (ball_pos.x < 50):
                target_y = min(target_y + 10, 90)

        elif role in ['dm', 'cdm']:
            # DM links defense and midfield
            target_y = max(ball_pos.y - 15, player.base_position.y)
            target_x = 50 + (ball_pos.x - 50) * 0.3

        elif role in ['cm', 'cm_l', 'cm_r']:
            # CMs support ball, find pockets of space
            target_y = ball_pos.y + random.uniform(-5, 10)
            # Move to half-spaces
            if player.position.x < 50:
                target_x = max(25, ball_pos.x - 15)
            else:
                target_x = min(75, ball_pos.x + 15)

        elif role in ['am', 'cam']:
            # AM finds space between lines
            target_y = ball_pos.y + random.uniform(5, 20)
            target_x = 50 + (ball_pos.x - 50) * 0.5

        elif role in ['lw', 'rw', 'lm', 'rm']:
            # Wingers: width and depth
            target_x = 10 if 'l' in role else 90
            target_y = max(ball_pos.y, 60)
            # If ball is on opposite side, come narrower for cutback
            if (player.position.x < 50) != (ball_pos.x < 50):
                target_x = 30 if 'l' in role else 70
                target_y = ball_pos.y + 15

        elif role in ['st', 'cf']:
            # Strikers: stretch defense, make runs
            target_y = min(ball_pos.y + 25, 95)
            # Drift across to find space
            target_x = 50 + random.uniform(-20, 20)
            # Stay onside (simplified)
            def_line = self._get_defensive_line(other_team)
            target_y = min(target_y, def_line + 5)

        # Add some unpredictability
        target_x += random.uniform(-3, 3)
        target_y += random.uniform(-2, 2)

        # When attacking, prioritize calculated position over base (more fluid).
        # Confidence drives how much a player commits to the advanced, ball-
        # seeking position vs. retreating toward their safe base position -
        # a rattled player hides, a confident one demands involvement.
        ball_seeking = max(0.5, min(0.95, 0.8 + player.confidence * 0.15))
        target_x = target_x * ball_seeking + player.base_position.x * (1 - ball_seeking)
        target_y = target_y * ball_seeking + player.base_position.y * (1 - ball_seeking)

        # Push all outfield players forward when team has ball
        if ball_holder and ball_holder in [p for p in team.players]:
            target_y = min(target_y + 5, 95)

        return target_x, target_y

    def _defending_movement(self, player: Player, ball_pos: Position,
                            ball_holder: Optional[Player], team: Team,
                            other_team: Team, state: MatchState) -> tuple:
        """Calculate defending movement for player"""
        target_x = player.position.x
        target_y = player.position.y

        role = player.role.lower()
        distance_to_ball = player.position.distance_to(ball_pos)

        if role in ['gk']:
            # GK adjusts position based on ball
            target_y = 5
            target_x = 50 + (ball_pos.x - 50) * 0.15

        elif role in ['cb', 'cb_l', 'cb_r']:
            # CBs: maintain line, cover central areas
            target_y = min(35, ball_pos.y - 10)
            # Shift toward ball side
            target_x = player.base_position.x + (ball_pos.x - 50) * 0.15

        elif role in ['lb', 'rb', 'lwb', 'rwb']:
            # Fullbacks: track wingers, stay compact
            target_y = min(40, ball_pos.y - 5)
            # Tuck in if ball is on far side
            if (player.position.x < 50) != (ball_pos.x < 50):
                target_x = 30 if 'l' in role else 70
            else:
                target_x = player.base_position.x

        elif role in ['dm', 'cdm']:
            # DM screens defense
            target_y = min(45, ball_pos.y - 5)
            target_x = 50 + (ball_pos.x - 50) * 0.4

        elif role in ['cm', 'cm_l', 'cm_r']:
            # CMs: press or cover
            if distance_to_ball < 25 and player.effective_attribute('workrate') > 60:
                # Press
                target_x = ball_pos.x
                target_y = ball_pos.y - 5
            else:
                # Cover passing lanes
                target_y = max(35, ball_pos.y - 15)
                target_x = player.base_position.x + (ball_pos.x - 50) * 0.3

        elif role in ['am', 'cam']:
            # AM drops into midfield when defending
            target_y = max(45, ball_pos.y - 10)
            target_x = 50 + (ball_pos.x - 50) * 0.3

        elif role in ['lw', 'rw', 'lm', 'rm']:
            # Wingers: track back or press
            if ball_pos.y > 60 and player.effective_attribute('workrate') > 55:
                # Press high
                target_y = ball_pos.y + 5
                target_x = ball_pos.x + (10 if 'l' in role else -10)
            else:
                # Track back
                target_y = max(40, ball_pos.y - 10)
                target_x = 25 if 'l' in role else 75

        elif role in ['st', 'cf']:
            # Strikers: light press or stay high for counter
            if ball_pos.y > 50:
                # Press from front
                target_y = ball_pos.y + 8
                target_x = ball_pos.x + random.uniform(-10, 10)
            else:
                # Stay high for counter-attack
                target_y = 65
                target_x = 50

        # Blend with base (more defensive = more base-weighted)
        target_x = target_x * 0.5 + player.base_position.x * 0.5
        target_y = target_y * 0.5 + player.base_position.y * 0.5

        return target_x, target_y

    def _get_defensive_line(self, team: Team) -> float:
        """Get the y-position of a team's defensive line"""
        defenders = [p for p in team.players if p.role in ['cb', 'cb_l', 'cb_r', 'lb', 'rb']]
        if defenders:
            return max(p.position.y for p in defenders)
        return 30

    def _calculate_urgency(self, player: Player, ball_pos: Position,
                           team_attacking: bool) -> float:
        """Calculate movement urgency (affects speed)"""
        distance = player.position.distance_to(ball_pos)

        if distance < 20:
            base = 1.2  # Close to ball = urgent
        elif distance < 40:
            base = 1.0
        else:
            base = 0.7  # Far from ball = less urgent

        # Confident players engage more urgently, rattled players drag their feet
        return max(0.3, base + player.confidence * 0.15)

    def _resolve_ball_action(self, state: MatchState) -> Optional[MatchEvent]:
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

        # Determine action based on position and pressure
        pressure = self.space_control.pressing_effectiveness(
            defending_team, ball.position
        )
        num_pressers, pressure_value = pressure

        # Higher pressure = more likely to lose ball or rush action
        if pressure_value > 0.8 and random.random() < 0.3:
            return self._resolve_turnover(state, holder, "pressed")

        # Decision: shoot, dribble, or pass (context-aware)
        action = self._decide_action(holder, attacking_team, defending_team, state)

        if action == "shoot":
            return self._resolve_shot(state, holder)
        elif action == "dribble":
            return self._resolve_dribble(state, holder, defending_team)
        else:
            return self._resolve_pass(state, holder, attacking_team, defending_team)

    def _decide_action(self, holder: Player, attacking_team: Team,
                       defending_team: Team, state: MatchState) -> str:
        """
        Decide whether to shoot, dribble, or pass based on context.
        Returns 'shoot', 'dribble', or 'pass'.
        """
        # Calculate factors
        in_shooting_range = self._in_shooting_range(holder.position)
        space_ahead = self._space_ahead(holder, defending_team)
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
            if random.random() < shoot_chance:
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

        if random.random() < dribble_chance:
            return "dribble"

        # Default to pass
        return "pass"

    def _space_ahead(self, player: Player, defending_team: Team) -> float:
        """Calculate how much space a player has ahead of them"""
        # Check area in front of player (toward opponent goal)
        check_y = min(player.position.y + 20, 100)
        check_pos = Position(player.position.x, check_y)

        nearest_def = min(
            (p.position.distance_to(check_pos) for p in defending_team.players),
            default=50
        )
        return nearest_def

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
                # Interception chance based on positioning and anticipation
                base_chance = 0.15

                # Ground passes easier to intercept than air
                if ball.state == BallState.GROUND_PASS:
                    base_chance += 0.1

                # Good positioning = better interception
                positioning = defender.effective_attribute('positioning')
                base_chance += (positioning - 50) / 200

                # Fast players close gap better
                pace = defender.effective_attribute('pace')
                base_chance += (pace - 50) / 300

                if random.random() < base_chance:
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

        # Check if target controlled the ball
        first_touch = target.effective_attribute('first_touch')
        control_chance = 0.7 + (first_touch / 100) * 0.25

        # Air balls harder to control
        if ball.state == BallState.AIR_PASS:
            control_chance -= 0.15

        if random.random() < control_chance:
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
            # Miscontrol - ball becomes loose nearby
            ball.make_loose()
            ball.position = Position(
                target.position.x + random.uniform(-5, 5),
                target.position.y + random.uniform(-5, 5)
            ).clamp()
            return MatchEvent(
                minute=state.minute,
                event_type="miscontrol",
                player=target,
                position=target.position,
                success=False,
                description=f"{target.name} miscontrols the ball"
            )

    def _resolve_shot_arrival(self, state: MatchState) -> Optional[MatchEvent]:
        """Handle shot arriving at goal"""
        ball = state.ball
        shooter = ball.passer
        goal_pos = ball.target_position

        # Get goalkeeper
        goalkeeper = state.defending_team.goalkeeper

        if goalkeeper:
            # Goalkeeper save attempt
            gk_dist = goalkeeper.position.distance_to(goal_pos)
            gk_positioning = goalkeeper.effective_attribute('positioning')
            gk_reactions = goalkeeper.effective_attribute('pace')  # Use pace for reactions

            # Save chance based on positioning and distance
            save_chance = 0.3 + (gk_positioning / 100) * 0.4
            if gk_dist < 10:
                save_chance += 0.2
            save_chance += (gk_reactions / 100) * 0.1

            # Shot power affects save difficulty
            if shooter:
                shooting = shooter.effective_attribute('shooting')
                save_chance -= (shooting / 100) * 0.2

            save_chance = max(0.1, min(0.9, save_chance))

            if random.random() < save_chance:
                # Saved!
                ball.give_to(goalkeeper)
                state.switch_possession()
                return MatchEvent(
                    minute=state.minute,
                    event_type="save",
                    player=goalkeeper,
                    target_player=shooter,
                    position=goal_pos,
                    success=True,
                    description=f"{goalkeeper.name} saves the shot!"
                )

        # Goal scored!
        if state.home_attacking:
            state.home_score += 1
        else:
            state.away_score += 1

        # Reset for kickoff
        self._kickoff(state, not state.home_attacking)

        return MatchEvent(
            minute=state.minute,
            event_type="goal",
            player=shooter,
            position=goal_pos,
            success=True,
            description=f"GOAL! {shooter.name if shooter else 'Unknown'} scores!"
        )

    def _resolve_loose_ball(self, state: MatchState) -> Optional[MatchEvent]:
        """Resolve who gets a loose ball"""
        ball_pos = state.ball.position

        # Find nearest player from each team
        all_players = state.home_team.players + state.away_team.players
        nearest = min(all_players, key=lambda p: p.position.distance_to(ball_pos))

        # Give ball to nearest player (with some randomness based on reactions)
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

        # Score each passing option
        scored_lanes = []
        for target, quality in lanes:
            score = quality

            # Calculate y-progress (positive = forward, negative = backward)
            y_diff = target.position.y - passer.position.y

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
            if target.position.y > 70:
                score *= 1.3

            scored_lanes.append((target, max(0.05, score)))

        # Weighted random selection
        total = sum(q for _, q in scored_lanes)
        if total == 0:
            return None

        r = random.random() * total
        cumulative = 0
        target = scored_lanes[0][0]

        for t, q in scored_lanes:
            cumulative += q
            if r <= cumulative:
                target = t
                break

        # Calculate pass success
        base_success = (passer.effective_attribute('passing') / 100) * 0.7
        lane_quality = next((q for t, q in lanes if t == target), 0.5)
        success_prob = base_success + lane_quality * 0.3

        # Add randomness
        success_prob += (random.random() - 0.5) * self.config.randomness

        success = random.random() < success_prob

        if success:
            # Start ball flight - don't give instantly
            distance = passer.position.distance_to(target.position)
            is_lofted = distance > 25  # Long passes are lofted
            state.ball.start_pass(passer, target, is_lofted)
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

    def _resolve_dribble(self, state: MatchState, dribbler: Player,
                         defending_team: Team) -> Optional[MatchEvent]:
        """Resolve a dribble attempt"""
        # Find nearest defender
        defenders = [p for p in defending_team.players
                     if p.position.distance_to(dribbler.position) < 15]

        if not defenders:
            # Space to dribble
            self._move_dribbler(dribbler, state)
            return MatchEvent(
                minute=state.minute,
                event_type="dribble",
                player=dribbler,
                position=dribbler.position,
                success=True,
                description=f"{dribbler.name} carries the ball forward"
            )

        # Contested dribble
        nearest_def = min(defenders, key=lambda p: p.position.distance_to(dribbler.position))

        dribble_skill = dribbler.effective_attribute('dribbling')
        defend_skill = nearest_def.effective_attribute('defending')

        success_prob = (dribble_skill / (dribble_skill + defend_skill))
        success_prob += (random.random() - 0.5) * self.config.randomness

        if random.random() < success_prob:
            self._move_dribbler(dribbler, state)
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

    def _move_dribbler(self, dribbler: Player, state: MatchState):
        """Move dribbler forward"""
        # Move toward opponent goal
        target = Position(50, 100)  # Opponent goal
        new_pos = dribbler.position.move_towards(target, 2)
        dribbler.position = new_pos.clamp()
        state.ball.position = dribbler.position

    def _resolve_tackle(self, state: MatchState, attacker: Player,
                        defender: Player) -> MatchEvent:
        """Resolve a tackle attempt"""
        state.ball.give_to(defender)
        state.switch_possession()

        return MatchEvent(
            minute=state.minute,
            event_type="tackle",
            player=defender,
            target_player=attacker,
            position=defender.position,
            success=True,
            description=f"{defender.name} tackles {attacker.name}"
        )

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

    def _in_shooting_range(self, pos: Position) -> bool:
        """Check if position is in shooting range"""
        goal = Position(50, 100)
        distance = pos.distance_to(goal)
        return distance < 30 and pos.y > 70

    def _resolve_shot(self, state: MatchState, shooter: Player) -> MatchEvent:
        """Start a shot attempt - ball will travel to goal"""
        goal = Position(50, 100)
        distance = shooter.position.distance_to(goal)

        # Base shot quality (determines if on target)
        shooting = shooter.effective_attribute('shooting')
        composure = shooter.effective_attribute('composure')

        # Distance penalty
        distance_factor = max(0.2, 1 - (distance / 40))

        # Angle factor (shots from center are easier)
        angle_factor = 1 - abs(shooter.position.x - 50) / 100

        shot_quality = (shooting / 100) * distance_factor * angle_factor * 0.7
        shot_quality += (composure / 100) * 0.3
        shot_quality += (random.random() - 0.5) * self.config.randomness

        # Check if shot is on target
        if random.random() < shot_quality:
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
            # Missed - goes wide
            state.ball.make_loose()
            state.ball.position = Position(50, 5)  # Goal kick area
            state.switch_possession()
            return MatchEvent(
                minute=state.minute,
                event_type="miss",
                player=shooter,
                position=shooter.position,
                success=False,
                description=f"{shooter.name}'s shot goes wide"
            )

    def _kickoff(self, state: MatchState, home_kicks: bool):
        """Reset for kickoff"""
        state.home_attacking = home_kicks

        # Reset positions
        for player in state.home_team.players:
            player.position = Position(player.base_position.x, player.base_position.y)
        for player in state.away_team.players:
            player.position = Position(player.base_position.x, player.base_position.y)

        # Give ball to kicking team's striker
        kicking_team = state.home_team if home_kicks else state.away_team
        striker = next((p for p in kicking_team.players if p.role in ['st', 'cf']), kicking_team.players[0])
        state.ball.give_to(striker)
        state.ball.position = Position(50, 50)

    def _half_time(self, state: MatchState):
        """Handle half time"""
        # Swap sides (flip y positions)
        for player in state.home_team.players + state.away_team.players:
            player.position.y = 100 - player.position.y
            player.base_position.y = 100 - player.base_position.y
            player.fatigue *= 0.5  # Some recovery

    def _update_fatigue(self, state: MatchState):
        """Update player fatigue"""
        for player in state.home_team.players + state.away_team.players:
            # Base fatigue
            base_rate = self.config.fatigue_rate

            # High workrate = more fatigue
            workrate_mod = player.workrate / 100

            # Stamina reduces fatigue rate
            stamina_mod = 1 - (player.stamina / 200)

            fatigue_gain = base_rate * workrate_mod * stamina_mod
            player.fatigue = min(100, player.fatigue + fatigue_gain)

    def _update_momentum(self, state: MatchState, events: List[MatchEvent]):
        """Update team momentum based on events"""
        for event in events:
            if event.event_type == "goal":
                if event.player in state.home_team.players:
                    state.home_team.momentum = min(100, state.home_team.momentum + 20)
                    state.away_team.momentum = max(0, state.away_team.momentum - 10)
                else:
                    state.away_team.momentum = min(100, state.away_team.momentum + 20)
                    state.home_team.momentum = max(0, state.home_team.momentum - 10)

        # Decay toward 50
        state.home_team.momentum = 50 + (state.home_team.momentum - 50) * self.config.momentum_decay
        state.away_team.momentum = 50 + (state.away_team.momentum - 50) * self.config.momentum_decay
