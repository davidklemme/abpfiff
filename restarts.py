"""
Restart policy: how the ball comes back into play.

Phase 0.5 of the roadmap (docs/reviews/2026-09-10-critical-review.md):
throw-ins, goal kicks and corners are modeled as simple possession
restarts - the right team gets the ball at the right spot - without any
set-piece choreography (that stays on the TODO list).
"""
import random
from typing import Optional

from models import MatchEvent, MatchState, Player, Position, Team


def is_out_of_bounds(pos: Position) -> bool:
    """True if a raw (unclamped) position lies outside the pitch."""
    return pos.x < 0 or pos.x > 100 or pos.y < 0 or pos.y > 100


class SimpleRestartPolicy:
    """Minimal, rule-shaped restarts: correct team + correct spot only."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    # -- kickoffs -----------------------------------------------------------

    def kickoff(self, state: MatchState, home_kicks: bool) -> None:
        """Reset positions and give the ball to the kicking team's striker."""
        state.home_attacking = home_kicks

        for player in state.home_team.players + state.away_team.players:
            player.position = Position(player.base_position.x, player.base_position.y)

        kicking_team = state.home_team if home_kicks else state.away_team
        striker = next((p for p in kicking_team.players if p.role in ['st', 'cf']),
                       kicking_team.players[0])
        striker.position = Position(50, 50)
        state.ball.give_to(striker)

    # -- out of play --------------------------------------------------------

    def resolve_out_of_bounds(self, state: MatchState, raw_pos: Position,
                              last_toucher: Player) -> MatchEvent:
        """The ball crossed a line at `raw_pos`, last touched by
        `last_toucher`. Decide throw-in / goal kick / corner and restart."""
        if last_toucher in state.home_team.players:
            last_team, other_team = state.home_team, state.away_team
        else:
            last_team, other_team = state.away_team, state.home_team

        # Over a touchline: throw-in for the other team
        if raw_pos.x < 0 or raw_pos.x > 100:
            spot = Position(0.0 if raw_pos.x < 0 else 100.0,
                            max(2.0, min(98.0, raw_pos.y)))
            return self._restart_possession(
                state, other_team, spot, "throw_in",
                f"Throw-in for {other_team.name}")

        # Over a goal line: corner if the defending-end team touched it last,
        # goal kick if the attacking team overhit it
        line_y = 0.0 if raw_pos.y < 0 else 100.0
        if state.home_team.own_goal.y == line_y:
            end_defenders = state.home_team
        else:
            end_defenders = state.away_team
        end_attackers = state.away_team if end_defenders is state.home_team else state.home_team

        if last_team is end_defenders:
            corner_spot = Position(0.0 if raw_pos.x < 50 else 100.0, line_y)
            return self._restart_possession(
                state, end_attackers, corner_spot, "corner",
                f"Corner for {end_attackers.name}")

        return self.goal_kick(state, end_defenders,
                              f"Goal kick for {end_defenders.name}")

    def free_kick(self, state: MatchState, fouled: Player,
                  team: Team) -> MatchEvent:
        """Restart after a foul: the fouled player's team takes the free
        kick from where it happened (no direct free kicks or penalties
        yet - a simple possession restart)."""
        spot = Position(fouled.position.x, fouled.position.y)
        return self._restart_possession(
            state, team, spot, "free_kick",
            f"Free kick for {team.name}")

    def goal_kick(self, state: MatchState, defending_team: Team,
                  description: str = "") -> MatchEvent:
        """Restart with the defending team's goalkeeper."""
        keeper = defending_team.goalkeeper
        if keeper is None:
            own = defending_team.own_goal
            spot = Position(own.x, defending_team.frame_y(5.0))
            return self._restart_possession(
                state, defending_team, spot, "goal_kick",
                description or f"Goal kick for {defending_team.name}")

        state.ball.give_to(keeper)
        self._sync_possession(state, defending_team)
        return MatchEvent(
            minute=state.minute,
            event_type="goal_kick",
            player=keeper,
            position=Position(keeper.position.x, keeper.position.y),
            success=True,
            description=description or f"Goal kick for {defending_team.name}"
        )

    # -- helpers ------------------------------------------------------------

    def _restart_possession(self, state: MatchState, team: Team, spot: Position,
                            event_type: str, description: str) -> MatchEvent:
        """Give the ball to `team` at `spot` (nearest player takes it)."""
        candidates = team.outfield_players or team.players
        taker = min(candidates, key=lambda p: p.position.distance_to(spot))
        taker.position = Position(spot.x, spot.y)
        state.ball.give_to(taker)
        self._sync_possession(state, team)
        return MatchEvent(
            minute=state.minute,
            event_type=event_type,
            player=taker,
            position=spot,
            success=True,
            description=description
        )

    def _sync_possession(self, state: MatchState, team_in_possession: Team) -> None:
        is_home = team_in_possession is state.home_team
        if is_home != state.home_attacking:
            state.switch_possession()
