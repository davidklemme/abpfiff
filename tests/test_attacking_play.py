#!/usr/bin/env python3
"""
Tests for attack construction: crossing (situation-vector driven, no
boolean gates) and clearances, plus their frame-awareness for both teams.

Run directly: python3 tests/test_attacking_play.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, make_match, run_tests
from models import Ball, BallState, MatchState, Position, Team
from engine import MatchEngine, SimulationConfig
from passing import box_targets
from situation import SituationEmbedding
from decisions import DualProcessDecisionModel, DecisionContext
from teams import create_tactical_matchup


def make_context(holder, width, progression, box_count):
    home = Team(name="H", players=[holder])
    away = Team(name="A", players=[make_player("D")])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    teammate = make_player("T")
    return DecisionContext(
        holder=holder, attacking_team=home, defending_team=away, state=state,
        situation=SituationEmbedding(0.4, progression, 0.5, 0.4, 0.5, width),
        in_shooting_range=False, space_ahead=15.0, nearest_defender_dist=10.0,
        lanes=[(teammate, 0.3)], best_forward_lane=0.3, best_safe_lane=0.3,
        box_targets=box_count,
    )


# ---------------------------------------------------------------------------
# Crossing is continuous in the situation vector
# ---------------------------------------------------------------------------

def test_cross_utility_grows_with_width_and_progression():
    model = DualProcessDecisionModel(rng=random.Random(1))
    winger = make_player("W", role="rw")

    wide_high = model.utilities(make_context(winger, width=0.9, progression=0.8, box_count=2))
    central_deep = model.utilities(make_context(winger, width=0.1, progression=0.3, box_count=2))

    assert wide_high["cross"] > central_deep["cross"] * 3, (
        wide_high["cross"], central_deep["cross"])


def test_cross_needs_someone_to_aim_at():
    model = DualProcessDecisionModel(rng=random.Random(2))
    winger = make_player("W", role="rw")
    for _ in range(100):
        action = model.decide(make_context(winger, width=0.95, progression=0.9,
                                           box_count=0))
        assert action != "cross"


def test_wide_advanced_wingers_actually_cross():
    model = DualProcessDecisionModel(rng=random.Random(3))
    winger = make_player("W", role="rw", x=88, y=75)
    crosses = sum(1 for _ in range(300)
                  if model.decide(make_context(winger, width=0.8,
                                               progression=0.75,
                                               box_count=2)) == "cross")
    assert crosses > 30, f"only {crosses}/300 crosses"


def test_box_targets_are_frame_aware():
    """For the away team, 'in the box' means LOW absolute y."""
    holder = make_player("AwayWinger", role="rw", x=85, y=25)
    striker_in_box = make_player("AwaySt", role="st", x=50, y=15)
    striker_deep = make_player("AwayCm", role="cm", x=50, y=70)
    away = Team(name="A", players=[holder, striker_in_box, striker_deep],
                attacks_up=False)

    targets = box_targets(holder, away)
    assert striker_in_box in targets
    assert striker_deep not in targets


def test_resolve_cross_delivers_lofted_ball_toward_box_target():
    engine = MatchEngine(SimulationConfig(seed=4))
    winger = make_player("W", role="rw", x=85, y=78)
    striker = make_player("St", role="st", x=50, y=80)
    state = make_match([winger, striker], [make_player("CB", x=45, y=80)])
    state.ball.give_to(winger)

    event = engine.actions.passes.resolve_cross(
        state, winger, state.home_team, state.away_team)

    assert event is not None and event.event_type == "cross"
    assert state.ball.state == BallState.AIR_PASS
    assert state.ball.target_player is striker


# ---------------------------------------------------------------------------
# Clearances
# ---------------------------------------------------------------------------

def test_clear_danger_sends_ball_away_from_own_goal():
    engine = MatchEngine(SimulationConfig(seed=5))
    cb = make_player("AwayCB", role="cb", x=50, y=85)  # away defends y=100
    state = make_match([make_player("HomeSt", x=50, y=80)], [cb])
    state.ball.give_to(cb)

    away = state.away_team
    event = engine.actions.passes.clear_danger(state, cb, away)

    assert event.event_type in ("clearance", "corner", "throw_in")
    if event.event_type == "clearance" and state.ball.is_in_flight():
        # Hoofed upfield: toward LOWER absolute y for the away team
        assert state.ball.target_position.y < cb.position.y


def test_matches_produce_crosses_and_clearances():
    counts = {"cross": 0, "clearance": 0}
    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=6))
    engine.on_event(lambda ev: counts.__setitem__(
        ev.event_type, counts[ev.event_type] + 1)
        if ev.event_type in counts else None)

    for _ in range(4):
        home, away = create_tactical_matchup("balanced", "balanced")
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        engine.simulate_match(state, minutes=90)

    assert counts["cross"] > 10, counts
    assert counts["clearance"] > 0, counts


if __name__ == "__main__":
    run_tests(globals())
