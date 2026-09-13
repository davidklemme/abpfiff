"""
Shot play: taking shots and resolving them at the goalmouth.

Extracted from the action resolver so shooting mechanics are one testable
unit. Shot and save quality flow through the execution-quality factor
stack (execution.py); the target goal is always the attacking team's.
"""
import random
from typing import Callable, Optional

from models import MatchEvent, MatchState, Player, Position, Team
from restarts import SimpleRestartPolicy
from execution import execution_quality

# Chance a save is parried behind for a corner instead of held
SAVE_PARRY_CHANCE = 0.3


class ShotResolver:
    """Resolves everything between 'shoot chosen' and goal/save/miss."""

    def __init__(self, restart_policy: SimpleRestartPolicy,
                 rng: random.Random, randomness: float,
                 publish: Callable[[MatchEvent, MatchState], None]):
        self.restarts = restart_policy
        self.rng = rng
        self.randomness = randomness
        self.publish = publish

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

    def resolve_shot_arrival(self, state: MatchState) -> Optional[MatchEvent]:
        """Handle shot arriving at goal"""
        ball = state.ball
        shooter = ball.passer
        goal_pos = ball.target_position

        # Get goalkeeper
        goalkeeper = state.defending_team.goalkeeper

        if goalkeeper:
            save_event = self._attempt_save(state, goalkeeper, shooter, goal_pos)
            if save_event is not None:
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

    def _attempt_save(self, state: MatchState, goalkeeper: Player,
                      shooter: Optional[Player],
                      goal_pos: Position) -> Optional[MatchEvent]:
        """The keeper's save attempt; None means the shot beat them."""
        # The keeper's own factor stack (positioning skill, fatigue,
        # pressure, confidence) vs. geometry
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

        if self.rng.random() >= save_chance:
            return None  # Beaten

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
        state.ball.give_to(goalkeeper)
        state.switch_possession()
        return save_event
