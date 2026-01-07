"""
Match simulation engine.
Ties together spatial control, tactical principles, and event resolution.
"""
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import random
import math

from models import (
    Player, Team, Ball, Position, MatchState, MatchEvent, Phase
)
from spatial import SpaceControl, PassingGraph
from tactics import TacticalSetup, TacticalPrinciple, MovementInstruction


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

        # 1. Apply tactical movements
        self._apply_tactical_movements(state)

        # 2. Update space control (implicit in spatial calculations)

        # 3. Resolve ball action
        action_event = self._resolve_ball_action(state)
        if action_event:
            events.append(action_event)
            self._emit_event(action_event)

        # 4. Update fatigue
        self._update_fatigue(state)

        # 5. Update momentum
        self._update_momentum(state, events)

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
        if not tactics:
            return

        for player in team.players:
            # Get active principles for this player
            principles = tactics.get_active_principles(player, state, team_attacking)

            if not principles:
                # Default: drift toward base position
                self._move_toward_base(player)
                continue

            # Apply highest priority principle
            principle = principles[0]
            self._apply_movement(player, principle.movement, state, team)

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

    def _resolve_ball_action(self, state: MatchState) -> Optional[MatchEvent]:
        """Resolve what happens with the ball"""
        ball = state.ball
        holder = ball.holder

        if holder is None:
            return self._resolve_loose_ball(state)

        attacking_team = state.attacking_team
        defending_team = state.defending_team

        # Determine action based on position and pressure
        pressure = self.space_control.pressing_effectiveness(
            defending_team, ball.position
        )
        num_pressers, pressure_value = pressure

        # Higher pressure = more likely to lose ball or rush action
        if pressure_value > 0.8 and random.random() < 0.3:
            # Forced error under pressure
            return self._resolve_turnover(state, holder, "pressed")

        # Decision: pass, dribble, or shoot
        if self._in_shooting_range(holder.position) and random.random() < 0.2:
            return self._resolve_shot(state, holder)

        if random.random() < 0.7:
            return self._resolve_pass(state, holder, attacking_team, defending_team)
        else:
            return self._resolve_dribble(state, holder, defending_team)

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

        # Choose target (weighted by lane quality and tactical directness)
        directness = attacking_team.tactics.directness if attacking_team.tactics else 50

        # Prefer progressive passes if high directness
        scored_lanes = []
        for target, quality in lanes:
            progress_bonus = 0
            if target.position.y > passer.position.y:
                progress_bonus = (directness / 100) * 0.3

            scored_lanes.append((target, quality + progress_bonus))

        # Weighted random selection
        total = sum(q for _, q in scored_lanes)
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
            state.ball.give_to(target)
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
        """Resolve a shot attempt"""
        goal = Position(50, 100)
        distance = shooter.position.distance_to(goal)

        # Base shot quality
        shooting = shooter.effective_attribute('shooting')
        composure = shooter.effective_attribute('composure')

        # Distance penalty
        distance_factor = max(0.2, 1 - (distance / 40))

        # Angle factor (shots from center are easier)
        angle_factor = 1 - abs(shooter.position.x - 50) / 100

        shot_quality = (shooting / 100) * distance_factor * angle_factor * 0.7
        shot_quality += (composure / 100) * 0.3

        # Goalkeeper save chance
        goalkeeper = state.defending_team.goalkeeper
        if goalkeeper:
            save_ability = goalkeeper.effective_attribute('positioning')
            save_chance = (save_ability / 100) * 0.6
        else:
            save_chance = 0.3

        # Resolve
        roll = random.random()
        shot_quality += (random.random() - 0.5) * self.config.randomness

        if roll < shot_quality * (1 - save_chance):
            # GOAL!
            if state.home_attacking:
                state.home_score += 1
            else:
                state.away_score += 1

            # Reset
            self._kickoff(state, not state.home_attacking)

            return MatchEvent(
                minute=state.minute,
                event_type="goal",
                player=shooter,
                position=shooter.position,
                success=True,
                description=f"GOAL! {shooter.name} scores!"
            )
        elif roll < shot_quality:
            # Saved
            if goalkeeper:
                state.ball.give_to(goalkeeper)
                state.switch_possession()
            return MatchEvent(
                minute=state.minute,
                event_type="save",
                player=shooter,
                target_player=goalkeeper,
                position=shooter.position,
                success=False,
                description=f"{shooter.name}'s shot saved by {goalkeeper.name if goalkeeper else 'keeper'}"
            )
        else:
            # Missed
            state.ball.holder = None
            state.ball.position = Position(50, 5)  # Goal kick
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
