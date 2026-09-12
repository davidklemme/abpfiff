#!/usr/bin/env python3
"""
Tests for Phase 0.5: out-of-play restarts (throw-in / goal kick / corner)
and kickoffs, plus the engine's component decomposition.

The restart policy is a standalone unit (restarts.SimpleRestartPolicy),
so most tests exercise it directly without a full engine.

No test framework dependency - plain asserts, run directly:
    python3 tests/test_restarts.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchState, Player, Position, Team
from engine import MatchEngine, SimulationConfig
from restarts import SimpleRestartPolicy, is_out_of_bounds
from teams import create_tactical_matchup


def make_player(name="P", role="cm", x=50.0, y=50.0) -> Player:
    p = Player(name=name, number=1, role=role)
    p.position = Position(x, y)
    p.base_position = Position(x, y)
    return p


def make_fixture():
    """Home attacks toward y=100, away (flipped) toward y=0."""
    home = Team(name="Home", players=[
        make_player("HomeGK", "gk", 50, 5),
        make_player("HomeCB", "cb", 50, 20),
        make_player("HomeST", "st", 50, 75),
    ])
    away = Team(name="Away", players=[
        make_player("AwayGK", "gk", 50, 95),
        make_player("AwayCB", "cb", 50, 80),
        make_player("AwayST", "st", 50, 25),
    ], attacks_up=False)
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    return home, away, state


# ---------------------------------------------------------------------------
# Unit tests: SimpleRestartPolicy
# ---------------------------------------------------------------------------

def test_is_out_of_bounds():
    assert is_out_of_bounds(Position(-1, 50))
    assert is_out_of_bounds(Position(101, 50))
    assert is_out_of_bounds(Position(50, -0.5))
    assert is_out_of_bounds(Position(50, 100.5))
    assert not is_out_of_bounds(Position(0, 0))
    assert not is_out_of_bounds(Position(100, 100))


def test_throw_in_goes_to_the_other_team():
    home, away, state = make_fixture()
    policy = SimpleRestartPolicy(rng=random.Random(1))
    last_toucher = home.players[1]  # home CB touched it last

    event = policy.resolve_out_of_bounds(state, Position(-3, 40), last_toucher)

    assert event.event_type == "throw_in"
    assert state.ball.holder in away.players
    assert state.ball.position.x == 0.0
    assert not state.home_attacking  # possession synced to away


def test_corner_when_defenders_put_ball_behind_their_own_goal():
    home, away, state = make_fixture()
    policy = SimpleRestartPolicy(rng=random.Random(1))
    # Home defends the y=0 end; home CB deflects it over that line
    event = policy.resolve_out_of_bounds(state, Position(30, -2), home.players[1])

    assert event.event_type == "corner"
    assert state.ball.holder in away.players
    assert state.ball.position.y == 0.0
    assert state.ball.position.x == 0.0  # near-side corner


def test_goal_kick_when_attackers_overhit_behind_the_goal():
    home, away, state = make_fixture()
    policy = SimpleRestartPolicy(rng=random.Random(1))
    # Away attacks the y=0 end; away striker overhits behind it
    event = policy.resolve_out_of_bounds(state, Position(60, -2), away.players[2])

    assert event.event_type == "goal_kick"
    assert state.ball.holder is home.goalkeeper
    assert state.home_attacking  # possession synced to home


def test_corner_is_direction_aware_at_the_high_end():
    home, away, state = make_fixture()
    policy = SimpleRestartPolicy(rng=random.Random(1))
    # Away defends the y=100 end; away CB deflects it over that line
    event = policy.resolve_out_of_bounds(state, Position(80, 103), away.players[1])

    assert event.event_type == "corner"
    assert state.ball.holder in home.players
    assert state.ball.position.y == 100.0
    assert state.ball.position.x == 100.0


def test_kickoff_places_striker_at_center_with_ball():
    home, away, state = make_fixture()
    policy = SimpleRestartPolicy(rng=random.Random(1))
    policy.kickoff(state, home_kicks=False)

    holder = state.ball.holder
    assert holder in away.players and holder.role == "st"
    assert holder.position.x == 50 and holder.position.y == 50
    assert not state.home_attacking


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_opening_kickoff_is_automatic():
    home, away = create_tactical_matchup("balanced", "balanced")
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    engine = MatchEngine(SimulationConfig(seed=1))

    engine.simulate_match(state, minutes=0)  # kickoff only, no play

    assert state.ball.holder in home.players


def test_second_half_kickoff_goes_to_away_team():
    home, away = create_tactical_matchup("balanced", "balanced")
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    engine = MatchEngine(SimulationConfig(seed=1))

    engine._half_time(state)

    assert state.ball.holder in away.players
    assert not home.attacks_up  # sides swapped


def test_seeded_engines_reproduce_the_same_match():
    def run(seed):
        home, away = create_tactical_matchup("balanced", "balanced")
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=seed))
        engine.simulate_match(state, minutes=90)
        return (state.home_score, state.away_score)

    assert run(1234) == run(1234)


def test_matches_produce_all_restart_types():
    """Throw-ins (deflected tackles, miscontrols out), corners (parried
    saves, defensive miscontrols) and goal kicks (wide shots) must all
    actually occur in play."""
    counts = {"throw_in": 0, "corner": 0, "goal_kick": 0}

    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=77))
    engine.on_event(lambda ev: counts.__setitem__(
        ev.event_type, counts[ev.event_type] + 1)
        if ev.event_type in counts else None)

    for _ in range(4):
        home, away = create_tactical_matchup("balanced", "balanced")
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        engine.simulate_match(state, minutes=90)

    assert counts["throw_in"] > 0, counts
    assert counts["corner"] > 0, counts
    assert counts["goal_kick"] > 0, counts


def test_restart_events_do_not_move_confidence():
    """Restarts are neutral possession events in Phase 1 psychology."""
    home, away, state = make_fixture()
    policy = SimpleRestartPolicy(rng=random.Random(1))
    import psychology
    event = policy.resolve_out_of_bounds(state, Position(-3, 40), home.players[1])
    psychology.process_feedback(event, state)
    assert all(p.confidence == 0.0 for p in home.players + away.players)


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
