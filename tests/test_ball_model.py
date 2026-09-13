#!/usr/bin/env python3
"""
Tests for the Phase 2a ball model: lead passes into space, receivers
moving to meet the arriving ball, and proximity-gated possession
(the ball never teleports to a player who isn't there).

No test framework dependency - plain asserts, run directly:
    python3 tests/test_ball_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, BallState, MatchState, Player, Position, Team
from engine import MatchEngine, SimulationConfig
from ball_actions import CONTROL_RADIUS, LOOSE_BALL_CLAIM_RADIUS
from teams import create_tactical_matchup


def make_player(name="P", role="cm", x=50.0, y=50.0, **attrs) -> Player:
    p = Player(name=name, number=1, role=role, **attrs)
    p.position = Position(x, y)
    p.base_position = Position(x, y)
    return p


def make_state(home_players, away_players, away_flipped=True):
    home = Team(name="H", players=home_players)
    away = Team(name="A", players=away_players, attacks_up=not away_flipped)
    return MatchState(home_team=home, away_team=away, ball=Ball())


# ---------------------------------------------------------------------------
# Lead passes
# ---------------------------------------------------------------------------

def test_lead_position_is_ahead_of_receiver_in_team_frame():
    engine = MatchEngine(SimulationConfig(seed=1))
    passer_h = make_player("HP", x=50, y=40)
    receiver_h = make_player("HR", x=60, y=55, pace=80)
    home = Team(name="H", players=[passer_h, receiver_h])

    lead_h = engine.actions.lead_position(passer_h, receiver_h, home)
    assert lead_h.y > receiver_h.position.y  # home attacks up

    passer_a = make_player("AP", x=50, y=60)
    receiver_a = make_player("AR", x=60, y=45, pace=80)
    away = Team(name="A", players=[passer_a, receiver_a], attacks_up=False)

    lead_a = engine.actions.lead_position(passer_a, receiver_a, away)
    assert lead_a.y < receiver_a.position.y  # away attacks down


def test_lead_is_reachable_during_flight():
    """The lead must not outrun the receiver: bounded by pace x flight time."""
    engine = MatchEngine(SimulationConfig(seed=1))
    passer = make_player("P", x=50, y=40)
    slow = make_player("Slow", x=50, y=50, pace=30)
    fast = make_player("Fast", x=60, y=50, pace=95)
    home = Team(name="H", players=[passer, slow, fast])

    lead_slow = engine.actions.lead_position(passer, slow, home)
    lead_fast = engine.actions.lead_position(passer, fast, home)

    assert lead_fast.y - fast.position.y > lead_slow.y - slow.position.y
    assert lead_slow.y - slow.position.y <= 8.0 + 1e-9


def test_start_pass_targets_lead_position():
    passer = make_player("P", x=50, y=40)
    receiver = make_player("R", x=50, y=55)
    ball = Ball()
    ball.give_to(passer)
    lead = Position(50, 60)

    ball.start_pass(passer, receiver, lead_position=lead)

    assert ball.target_position.x == 50 and ball.target_position.y == 60
    assert ball.target_player is receiver
    assert ball.is_in_flight()


# ---------------------------------------------------------------------------
# Receivers meet the ball; loose balls are chased
# ---------------------------------------------------------------------------

def test_receiver_runs_to_meet_the_arriving_ball():
    engine = MatchEngine(SimulationConfig(seed=1))
    passer = make_player("P", x=50, y=30)
    receiver = make_player("R", x=60, y=50, pace=80)
    state = make_state([passer, receiver], [make_player("Opp", x=20, y=80)])

    state.ball.give_to(passer)
    state.ball.start_pass(passer, receiver, lead_position=Position(60, 58))
    dist_before = receiver.position.distance_to(Position(60, 58))

    engine.movement.update_positions(state)

    dist_after = receiver.position.distance_to(Position(60, 58))
    assert dist_after < dist_before


def test_nearest_player_chases_a_loose_ball():
    engine = MatchEngine(SimulationConfig(seed=1))
    near = make_player("Near", x=40, y=50)
    far = make_player("Far", x=10, y=10)
    state = make_state([near, far], [make_player("Opp", x=90, y=90)])
    state.ball.make_loose()
    state.ball.position = Position(50, 50)

    dist_before = near.position.distance_to(state.ball.position)
    engine.movement.update_positions(state)
    dist_after = near.position.distance_to(state.ball.position)

    assert dist_after < dist_before


# ---------------------------------------------------------------------------
# No teleported possession
# ---------------------------------------------------------------------------

def test_pass_the_receiver_never_reached_runs_loose():
    engine = MatchEngine(SimulationConfig(seed=1))
    passer = make_player("P", x=50, y=30)
    receiver = make_player("R", x=50, y=40)
    state = make_state([passer, receiver], [make_player("Opp", x=10, y=90)])

    state.ball.give_to(passer)
    state.ball.start_pass(passer, receiver, lead_position=Position(50, 44))
    # Receiver got dragged far from the arrival point
    receiver.position = Position(50 + CONTROL_RADIUS + 10, 44)
    # Force arrival
    state.ball.position = Position(50, 44)
    state.ball.flight_ticks_remaining = 0

    event = engine.actions._resolve_pass_arrival(state)

    assert event is None
    assert state.ball.holder is None
    assert state.ball.state == BallState.LOOSE
    assert state.ball.position.y == 44  # loose where it landed


def test_loose_ball_is_not_claimed_from_distance():
    engine = MatchEngine(SimulationConfig(seed=1))
    player = make_player("P", x=50, y=50 + LOOSE_BALL_CLAIM_RADIUS + 10)
    state = make_state([player], [make_player("Opp", x=10, y=10)])
    state.ball.make_loose()
    state.ball.position = Position(50, 50)

    event = engine.actions._resolve_loose_ball(state)

    assert event is None
    assert state.ball.holder is None


def test_loose_ball_is_claimed_when_close():
    engine = MatchEngine(SimulationConfig(seed=1))
    player = make_player("P", x=50, y=51)
    state = make_state([player], [make_player("Opp", x=10, y=10)])
    state.ball.make_loose()
    state.ball.position = Position(50, 50)

    event = engine.actions._resolve_loose_ball(state)

    assert event is not None and event.event_type == "ball_won"
    assert state.ball.holder is player


# ---------------------------------------------------------------------------
# Integration: matches still complete and improve
# ---------------------------------------------------------------------------

def test_full_match_runs_and_ball_is_always_accounted_for():
    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=9))
    home, away = create_tactical_matchup("balanced", "balanced")
    state = MatchState(home_team=home, away_team=away, ball=Ball())

    def check(s):
        holders = [p for p in s.home_team.players + s.away_team.players if p.has_ball]
        assert len(holders) <= 1, "two players hold the ball"
        if s.ball.holder is not None:
            assert s.ball.holder.has_ball

    engine.on_tick(check)
    engine.simulate_match(state, minutes=90)
    assert state.home_score >= 0 and state.away_score >= 0


TESTS = [obj for name, obj in list(globals().items())
         if name.startswith("test_") and callable(obj)]


def main():
    failures = []
    for test in TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            failures.append(test.__name__)
            print(f"  FAIL  {test.__name__}: {e}")
        except Exception as e:
            failures.append(test.__name__)
            print(f"  ERROR {test.__name__}: {e!r}")

    print()
    print(f"{len(TESTS) - len(failures)}/{len(TESTS)} passed")
    if failures:
        print("Failed:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
