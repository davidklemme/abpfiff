"""
Physical conditioning and momentum bookkeeping - the per-tick updates
that are not about the ball: fatigue accumulation and team momentum.
"""
from typing import List

from models import MatchEvent, MatchState


class FatigueModel:
    """Per-tick fatigue accumulation based on workrate and stamina."""

    def __init__(self, fatigue_rate: float = 0.1):
        self.fatigue_rate = fatigue_rate

    def update(self, state: MatchState, events: List[MatchEvent]) -> None:
        for player in state.home_team.players + state.away_team.players:
            # High workrate = more fatigue
            workrate_mod = player.workrate / 100

            # Stamina reduces fatigue rate
            stamina_mod = 1 - (player.stamina / 200)

            fatigue_gain = self.fatigue_rate * workrate_mod * stamina_mod
            player.fatigue = min(100, player.fatigue + fatigue_gain)


class MomentumModel:
    """Event-driven team momentum with decay toward neutral."""

    def __init__(self, momentum_decay: float = 0.95):
        self.momentum_decay = momentum_decay

    def update(self, state: MatchState, events: List[MatchEvent]) -> None:
        for event in events:
            if event.event_type == "goal":
                if event.player in state.home_team.players:
                    state.home_team.momentum = min(100, state.home_team.momentum + 20)
                    state.away_team.momentum = max(0, state.away_team.momentum - 10)
                else:
                    state.away_team.momentum = min(100, state.away_team.momentum + 20)
                    state.home_team.momentum = max(0, state.home_team.momentum - 10)

        # Decay toward 50
        state.home_team.momentum = 50 + (state.home_team.momentum - 50) * self.momentum_decay
        state.away_team.momentum = 50 + (state.away_team.momentum - 50) * self.momentum_decay
