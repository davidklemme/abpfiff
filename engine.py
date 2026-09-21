"""
Match simulation engine.

MatchEngine is a thin orchestrator: each concern lives in its own
independently testable component (see interfaces.py for the Protocols):

  - movement.RoleMovementModel      where players without the ball move
  - ball_actions.DefaultActionResolver  what happens with the ball
  - restarts.SimpleRestartPolicy    kickoffs and out-of-play restarts
  - conditioning.FatigueModel / MomentumModel  physical bookkeeping
  - psychology                      pressure, confidence and feedback

Components are injected through the constructor and share one
random.Random instance, so a seeded config reproduces a match exactly.
"""
from typing import Callable, List, Optional
from dataclasses import dataclass
import random

from models import MatchState, MatchEvent, Position, Team, Player
from interfaces import (
    ActionResolver, ConditioningModel, MovementModel, RestartPolicy
)
from spatial import SpaceControl
from movement import RoleMovementModel
from ball_actions import DefaultActionResolver
from restarts import SimpleRestartPolicy
from conditioning import FatigueModel, MomentumModel
from decisions import DualProcessDecisionModel
from minds import MindRegistry
from learning import ExperienceLearning
from tactics import MovementInstruction
import psychology


@dataclass
class SimulationConfig:
    """Configuration for the simulation"""
    ticks_per_minute: int = 10  # Higher = more detailed simulation
    fatigue_rate: float = 0.1  # Fatigue gain per tick
    momentum_decay: float = 0.95  # How quickly momentum normalizes
    randomness: float = 0.3  # 0-1, how much randomness affects outcomes
    seed: Optional[int] = None  # Seed the RNG for reproducible matches
    debug: bool = False


class MatchEngine:
    """
    The main match simulation engine: orchestrates the components and
    owns the tick/minute/match loops plus event distribution.
    """

    def __init__(self, config: SimulationConfig = None,
                 movement: Optional[MovementModel] = None,
                 actions: Optional[ActionResolver] = None,
                 restart_policy: Optional[RestartPolicy] = None,
                 conditioning: Optional[List[ConditioningModel]] = None,
                 minds: Optional[MindRegistry] = None):
        self.config = config or SimulationConfig()
        self.rng = random.Random(self.config.seed)

        self.space_control = SpaceControl(resolution=10)
        self.restarts = restart_policy or SimpleRestartPolicy(rng=self.rng)
        self.movement = movement or RoleMovementModel(
            space_control=self.space_control, rng=self.rng)
        # Decisions read the same minds that outcomes write (learning).
        # Injectable so a series runner can carry one registry across
        # matches and engines (docs/specs/temporal-continuity.md).
        self.minds = minds if minds is not None else MindRegistry()
        self.learning = ExperienceLearning(self.minds)
        self.actions = actions or DefaultActionResolver(
            space_control=self.space_control,
            restart_policy=self.restarts,
            rng=self.rng,
            randomness=self.config.randomness,
            publish=self._publish_event,
            decision_model=DualProcessDecisionModel(rng=self.rng, minds=self.minds),
        )
        self.conditioning: List[ConditioningModel] = conditioning if conditioning is not None else [
            FatigueModel(self.config.fatigue_rate),
            MomentumModel(self.config.momentum_decay),
        ]

        self.event_handlers: List[Callable] = []
        self.tick_handlers: List[Callable] = []
        # Per-player last positions, for deriving motion (facing) each tick
        self._last_positions: dict = {}

    # -- events -------------------------------------------------------------

    def on_event(self, handler: Callable):
        """Register event handler"""
        self.event_handlers.append(handler)

    def on_tick(self, handler: Callable):
        """Register a per-tick observer, called with the MatchState after
        every simulated tick (used by metrics/validation harnesses)."""
        self.tick_handlers.append(handler)

    def _emit_event(self, event: MatchEvent):
        """Emit event to all handlers"""
        for handler in self.event_handlers:
            handler(event)

    def _publish_event(self, event: MatchEvent, state: MatchState):
        """Emit a secondary event (e.g. the miscontrol before a restart)
        and feed it to psychology and learning - used by components
        mid-resolution."""
        self._emit_event(event)
        psychology.process_feedback(event, state)
        self.learning.on_event(event, state)

    # -- simulation loops ---------------------------------------------------

    def simulate_match(self, state: MatchState, minutes: int = 90) -> MatchState:
        """Simulate a full match or portion of it"""
        # Opening kickoff if the caller didn't set one up
        if (state.minute == 0 and state.ball.holder is None
                and not state.ball.is_in_flight()):
            self.restarts.kickoff(state, home_kicks=True)

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
        self.movement.update_positions(state)

        # 2. Resolve ball action
        action_event = self.actions.resolve(state)
        if action_event:
            events.append(action_event)
            self._emit_event(action_event)
            psychology.process_feedback(action_event, state)
            self.learning.on_event(action_event, state)

        # 3. Physical bookkeeping (fatigue, momentum)
        for model in self.conditioning:
            model.update(state, events)

        # 4. Psychology fades. Memory retention advances only at the explicit
        # match boundary, so a match cannot accidentally apply two clocks.
        minutes_elapsed = 1.0 / self.config.ticks_per_minute
        psychology.decay_all(state, minutes_elapsed=minutes_elapsed)

        # 5. Update motion state (smoothed velocity -> facing for perception)
        self._update_velocities(state)

        # 6. Notify per-tick observers
        for handler in self.tick_handlers:
            handler(state)

        return events

    # A player covers at most ~3 units/tick under their own power; any
    # larger step is a restart teleport (kickoff reset, throw-in spot,
    # cross-match reuse of the engine) and must not become "motion".
    MAX_PHYSICAL_STEP = 6.0

    def _update_velocities(self, state: MatchState):
        """Derive each player's smoothed velocity from their movement this
        tick. Orientation is state, not an attribute: perception reads the
        facing straight from where the player is actually heading.

        Teleports (restarts placing players, goal kickoffs resetting all
        22, a reused engine starting a fresh match) are detected by step
        size and reset the motion state instead of polluting the facing."""
        for player in state.home_team.players + state.away_team.players:
            last = self._last_positions.get(player.player_id)
            if last is not None:
                dx = player.position.x - last[0]
                dy = player.position.y - last[1]
                if (dx * dx + dy * dy) ** 0.5 > self.MAX_PHYSICAL_STEP:
                    player.velocity_x = player.velocity_y = 0.0
                else:
                    player.velocity_x = 0.6 * player.velocity_x + 0.4 * dx
                    player.velocity_y = 0.6 * player.velocity_y + 0.4 * dy
            self._last_positions[player.player_id] = (player.position.x,
                                                      player.position.y)

    def _simulate_minute(self, state: MatchState) -> List[MatchEvent]:
        """Simulate one minute of play"""
        all_events = []

        for _ in range(self.config.ticks_per_minute):
            events = self.simulate_tick(state)
            all_events.extend(events)

        return all_events

    def _half_time(self, state: MatchState):
        """Handle half time"""
        # Swap sides (flip y positions AND attack directions)
        for player in state.home_team.players + state.away_team.players:
            player.position.y = 100 - player.position.y
            player.base_position.y = 100 - player.base_position.y
            player.fatigue *= 0.5  # Some recovery
        state.home_team.attacks_up = not state.home_team.attacks_up
        state.away_team.attacks_up = not state.away_team.attacks_up

        # Side swap invalidates motion history (positions jumped)
        self._last_positions.clear()
        for player in state.home_team.players + state.away_team.players:
            player.velocity_x = player.velocity_y = 0.0

        # Second half: the away team kicks off (home kicked off the first)
        self.restarts.kickoff(state, home_kicks=False)

    # -- compatibility delegators -------------------------------------------
    # Thin forwards to the components, kept so existing scripts and tests
    # can address the engine directly. New code should use the components.

    def _kickoff(self, state: MatchState, home_kicks: bool):
        self.restarts.kickoff(state, home_kicks)

    def _apply_movement(self, player: Player, movement_instruction: MovementInstruction,
                        state: MatchState, team: Team):
        self.movement.apply_movement(player, movement_instruction, state, team)

    def _attacking_movement(self, player: Player, ball_pos: Position,
                            ball_holder: Optional[Player], team: Team,
                            other_team: Team, state: MatchState) -> tuple:
        return self.movement.attacking_movement(
            player, ball_pos, ball_holder, team, other_team, state)

    def _defending_movement(self, player: Player, ball_pos: Position,
                            ball_holder: Optional[Player], team: Team,
                            other_team: Team, state: MatchState) -> tuple:
        return self.movement.defending_movement(
            player, ball_pos, ball_holder, team, other_team, state)

    def _decide_action(self, holder: Player, attacking_team: Team,
                       defending_team: Team, state: MatchState) -> str:
        return self.actions.decide_action(holder, attacking_team, defending_team, state)

    def _in_shooting_range(self, pos: Position, attacking_team: Team) -> bool:
        return self.actions.in_shooting_range(pos, attacking_team)

    def _resolve_shot(self, state: MatchState, shooter: Player,
                      attacking_team: Team, defending_team: Team) -> MatchEvent:
        return self.actions.resolve_shot(state, shooter, attacking_team, defending_team)

    def _move_dribbler(self, dribbler: Player, state: MatchState,
                       attacking_team: Team):
        self.actions.move_dribbler(dribbler, state, attacking_team)
