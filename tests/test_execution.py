#!/usr/bin/env python3
"""
Tests for the execution-quality model (execution.py): every contested
action's outcome probability must be driven by the full factor stack -
skill, physical fatigue, psychological pressure, confidence, and team
momentum - each moving the outcome in the right direction.

No test framework dependency - plain asserts, run directly:
    python3 tests/test_execution.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchState, Player, Position, Team
from execution import execution_quality, contest, QUALITY_FLOOR, QUALITY_CEIL


def make_player(name="P", role="cm", x=50.0, y=50.0, **attrs) -> Player:
    p = Player(name=name, number=1, role=role, **attrs)
    p.position = Position(x, y)
    p.base_position = Position(x, y)
    return p


def make_state(player, opponent_distance=40.0):
    """A minimal state with one opponent at a controllable distance."""
    opponent = make_player("Opp", x=50.0, y=50.0 + opponent_distance)
    home = Team(name="H", players=[player])
    away = Team(name="A", players=[opponent])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    return state, away


def quality(player, opponent_distance=40.0, momentum=None):
    state, opponents = make_state(player, opponent_distance)
    return execution_quality(player, "passing", state, opponents, momentum=momentum)


# ---------------------------------------------------------------------------
# Each factor moves execution the right way
# ---------------------------------------------------------------------------

def test_skill_raises_quality():
    weak = make_player(passing=40)
    strong = make_player(passing=85)
    assert quality(strong) > quality(weak)


def test_physical_fatigue_lowers_quality():
    fresh = make_player(passing=70, fatigue=0.0)
    tired = make_player(passing=70, fatigue=90.0)
    assert quality(tired) < quality(fresh)


def test_confidence_raises_quality():
    rattled = make_player(passing=70)
    rattled.confidence = -0.8
    confident = make_player(passing=70)
    confident.confidence = 0.8
    assert quality(confident) > quality(rattled)


def test_opponent_pressure_lowers_quality():
    player_open = make_player(passing=70)
    player_pressed = make_player(passing=70)
    assert quality(player_pressed, opponent_distance=2.0) < \
           quality(player_open, opponent_distance=40.0)


def test_composure_resists_pressure():
    """Under identical pressure, the composed player executes better."""
    nervy = make_player(passing=70, composure=25)
    composed = make_player(passing=70, composure=90)
    assert quality(composed, opponent_distance=2.0) > \
           quality(nervy, opponent_distance=2.0)


def test_team_momentum_raises_quality():
    player = make_player(passing=70)
    low = quality(make_player(passing=70), momentum=20.0)
    high = quality(player, momentum=80.0)
    assert high > low


def test_quality_is_clamped():
    superman = make_player(passing=100, composure=100)
    superman.confidence = 1.0
    wreck = make_player(passing=1, composure=1, fatigue=100.0)
    wreck.confidence = -1.0
    assert quality(superman, momentum=100.0) <= QUALITY_CEIL
    assert quality(wreck, opponent_distance=1.0, momentum=0.0) >= QUALITY_FLOOR


# ---------------------------------------------------------------------------
# Duels combine both sides' stacks
# ---------------------------------------------------------------------------

def test_contest_favors_higher_quality():
    assert contest(0.8, 0.4) > 0.5
    assert contest(0.4, 0.8) < 0.5
    assert contest(0.5, 0.5) == 0.5


def test_factor_stack_spreads_outcomes():
    """The point of the model: same attribute, different physical/mental
    state, meaningfully different execution."""
    same_skill_best = make_player(passing=70, composure=85, fatigue=0.0)
    same_skill_best.confidence = 0.7
    same_skill_worst = make_player(passing=70, composure=30, fatigue=85.0)
    same_skill_worst.confidence = -0.7

    q_best = quality(same_skill_best, opponent_distance=3.0, momentum=75.0)
    q_worst = quality(same_skill_worst, opponent_distance=3.0, momentum=25.0)

    # State alone should swing execution by a large factor
    assert q_best > q_worst * 1.5, (q_best, q_worst)


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
