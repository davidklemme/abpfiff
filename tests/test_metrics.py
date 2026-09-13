#!/usr/bin/env python3
"""
Tests for the match metrics collector (metrics.py) and the validation
harness (validate.py).

No test framework dependency - plain asserts, run directly:
    python3 tests/test_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchEvent, MatchState, Player, Position, Team
from metrics import MatchMetrics
from validate import run_validation, evaluate, BANDS


def make_player(name="P", role="cm") -> Player:
    return Player(name=name, number=1, role=role)


def make_fixture():
    home = Team(name="H", players=[make_player("H1"), make_player("HGK", "gk")])
    away = Team(name="A", players=[make_player("A1"), make_player("AGK", "gk")],
                attacks_up=False)
    return home, away, MatchMetrics(home, away)


def ev(event_type, player, description="", target=None):
    return MatchEvent(minute=1, event_type=event_type, player=player,
                      target_player=target, position=Position(50, 50),
                      description=description)


# ---------------------------------------------------------------------------
# MatchMetrics unit tests
# ---------------------------------------------------------------------------

def test_events_are_attributed_to_the_right_team():
    home, away, m = make_fixture()
    m.on_event(ev("goal", home.players[0]))
    m.on_event(ev("shot", home.players[0]))
    m.on_event(ev("miss", away.players[0]))
    m.on_event(ev("corner", away.players[0]))

    assert m.home.goals == 1 and m.away.goals == 0
    assert m.home.shots == 1 and m.away.shots == 1  # miss counts as a shot
    assert m.away.corners == 1 and m.home.corners == 0


def test_pass_completion_accounts_for_misplaced_passes():
    home, away, m = make_fixture()
    passer, receiver = home.players[0], home.players[1]
    m.on_event(ev("pass", passer))
    m.on_event(ev("pass_received", receiver, target=passer))
    m.on_event(ev("pass", passer))  # launched but never received (intercepted)
    m.on_event(ev("turnover", passer, description="loses the ball (misplaced_pass)"))

    # 3 attempts (2 launched + 1 misplaced), 1 completed
    assert m.home.pass_attempts == 3
    assert abs(m.home.pass_completion - 1 / 3) < 1e-9


def test_possession_counted_from_ball_control():
    home, away, m = make_fixture()
    state = MatchState(home_team=home, away_team=away, ball=Ball())

    state.ball.give_to(home.players[0])
    m.on_tick(state)
    m.on_tick(state)
    state.ball.give_to(away.players[0])
    m.on_tick(state)

    assert m.home.possession_ticks == 2
    assert m.away.possession_ticks == 1
    assert abs(m.possession_share() - 2 / 3) < 1e-9


def test_unattributed_events_are_ignored():
    home, away, m = make_fixture()
    stranger = make_player("Stranger")
    m.on_event(ev("goal", stranger))
    assert m.home.goals == 0 and m.away.goals == 0


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

def test_validation_run_produces_populated_aggregate():
    aggregate = run_validation(matches=2, seed=5, ticks_per_minute=4)

    assert aggregate.matches == 2
    assert aggregate.pass_attempts > 50  # matches actually happened
    assert aggregate.possession_home_ticks + aggregate.possession_away_ticks > 0
    assert aggregate.home_shots + aggregate.away_shots > 0


def test_evaluate_flags_gate_and_target_violations():
    """Evaluation machinery: gate breaches fail, target misses only warn.
    (The real reference-sample gate runs in CI: validate.py --gate with 20
    matches - small samples are too noisy for the ratio bands.)"""
    from validate import Aggregate, Band

    agg = Aggregate(matches=10, home_goals=10, away_goals=10,
                    home_shots=100, away_shots=100,
                    pass_attempts=1000, passes_completed=800,
                    possession_home_ticks=500, possession_away_ticks=500)
    bands = [
        Band("in target", Aggregate.goals_per_match, 1.0, 4.0, 1.5, 3.0),
        Band("off target only", Aggregate.pass_completion, 0.5, 0.98, 0.9, 0.95),
        Band("gate breach", Aggregate.shots_per_team_per_match, 20.0, 40.0, 25.0, 30.0),
    ]
    gate_failures, target_warnings, rows = evaluate(agg, bands)

    assert gate_failures == ["gate breach"]
    assert target_warnings == ["off target only"]
    assert [status for _, _, _, status in rows] == ["ok", "off target", "GATE FAIL"]


def test_bands_are_well_formed():
    for band in BANDS:
        assert band.gate_lo <= band.target_lo <= band.target_hi <= band.gate_hi, \
            f"{band.name}: target band must sit inside gate band"


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
