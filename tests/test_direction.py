#!/usr/bin/env python3
"""
Tests for per-team attack direction (the Phase 0 correctness sprint).

Before this fix, every spatial rule in the engine was hardcoded to the home
team's frame (attack toward y=100): the away team shot at its own end,
tactical zones evaluated against the wrong half, and the half-time flip
inverted which team was broken. These tests pin the corrected behavior.

No test framework dependency - plain asserts, run directly:
    python3 tests/test_direction.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchState, Player, Position, Team
from engine import MatchEngine, SimulationConfig
from tactics import MovementInstruction, TacticalPrinciple, TriggerType
from teams import create_tactical_matchup, create_433_team, flip_team_positions


def make_player(name="P", role="cm", x=50.0, y=50.0, **kwargs) -> Player:
    p = Player(name=name, number=1, role=role, **kwargs)
    p.position = Position(x, y)
    p.base_position = Position(x, y)
    return p


# ---------------------------------------------------------------------------
# Unit tests: frame mapping
# ---------------------------------------------------------------------------

def test_frame_y_is_identity_for_home_and_mirror_for_away():
    home = Team(name="H", players=[])
    away = Team(name="A", players=[], attacks_up=False)
    assert home.frame_y(30) == 30
    assert away.frame_y(30) == 70
    # Involution: converting twice returns the original value
    for y in (0, 12.5, 50, 88, 100):
        assert home.frame_y(home.frame_y(y)) == y
        assert away.frame_y(away.frame_y(y)) == y


def test_goal_positions_follow_attack_direction():
    home = Team(name="H", players=[])
    away = Team(name="A", players=[], attacks_up=False)
    assert home.attacking_goal.y == 100 and home.own_goal.y == 0
    assert away.attacking_goal.y == 0 and away.own_goal.y == 100


def test_flip_team_positions_flips_attack_direction():
    team = create_433_team("Test")
    assert team.attacks_up
    gk_y_before = team.goalkeeper.position.y
    flip_team_positions(team)
    assert not team.attacks_up
    assert team.goalkeeper.position.y == 100 - gk_y_before


def test_half_time_flips_directions_and_positions():
    home, away = create_tactical_matchup("balanced", "balanced")
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    engine = MatchEngine(SimulationConfig())
    engine._half_time(state)
    assert not home.attacks_up
    assert away.attacks_up
    # Keepers swapped ends
    assert home.goalkeeper.position.y > 50
    assert away.goalkeeper.position.y < 50


# ---------------------------------------------------------------------------
# Unit tests: engine spatial logic respects direction
# ---------------------------------------------------------------------------

def test_shooting_range_is_direction_aware():
    engine = MatchEngine(SimulationConfig())
    home = Team(name="H", players=[])
    away = Team(name="A", players=[], attacks_up=False)
    near_top = Position(50, 90)
    near_bottom = Position(50, 10)

    assert engine._in_shooting_range(near_top, home)
    assert not engine._in_shooting_range(near_bottom, home)
    assert engine._in_shooting_range(near_bottom, away)
    assert not engine._in_shooting_range(near_top, away)


def test_away_shot_targets_the_low_goal():
    random.seed(11)
    engine = MatchEngine(SimulationConfig(randomness=0.0))
    for _ in range(50):
        shooter = make_player("AwaySt", role="st", x=50, y=8,
                              shooting=95, composure=95)
        keeper = make_player("HomeGK", role="gk", x=50, y=5)
        away = Team(name="A", players=[shooter], attacks_up=False)
        home = Team(name="H", players=[keeper])
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        state.ball.give_to(shooter)
        state.home_attacking = False

        event = engine._resolve_shot(state, shooter, away, home)
        if event.event_type == "shot":  # on target
            assert state.ball.target_position.y == 0.0
            return
    assert False, "no on-target shot in 50 attempts"


def test_dribbler_moves_toward_attacked_goal():
    engine = MatchEngine(SimulationConfig())
    dribbler = make_player("AwayWinger", role="lw", x=50, y=50)
    away = Team(name="A", players=[dribbler], attacks_up=False)
    home = Team(name="H", players=[])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    state.ball.give_to(dribbler)

    engine._move_dribbler(dribbler, state, away)
    assert dribbler.position.y < 50  # toward y=0


def test_attacking_movement_pushes_away_striker_toward_low_y():
    random.seed(5)
    engine = MatchEngine(SimulationConfig())
    striker = make_player("AwaySt", role="st", x=50, y=50)
    defender = make_player("HomeCB", role="cb", x=50, y=20)
    away = Team(name="A", players=[striker], attacks_up=False)
    home = Team(name="H", players=[defender])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    state.ball.position = Position(50, 50)

    _, ty = engine._attacking_movement(striker, state.ball.position, None,
                                       away, home, state)
    assert ty < 50, f"away striker should attack toward y=0, got target y={ty}"


def test_defending_movement_keeps_away_cb_near_high_goal():
    random.seed(5)
    engine = MatchEngine(SimulationConfig())
    cb = make_player("AwayCB", role="cb", x=50, y=80)  # flipped base
    away = Team(name="A", players=[cb], attacks_up=False)
    home = Team(name="H", players=[])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    state.ball.position = Position(50, 50)

    _, ty = engine._defending_movement(cb, state.ball.position, None,
                                       away, home, state)
    assert ty > 60, f"away CB should defend near y=100, got target y={ty}"


def test_tactical_ball_zone_evaluated_in_team_frame():
    """'Only in the opponent half' (ball_zone_y_min=50) must mean the
    opponent's half for BOTH teams."""
    principle = TacticalPrinciple(
        name="press_opponent_half",
        trigger=TriggerType.ALWAYS,
        applies_to_roles=[],
        movement=MovementInstruction(towards_ball=0.5),
        ball_zone_y_min=50,
    )
    player = make_player("AwayCM", role="cm")
    away = Team(name="A", players=[player], attacks_up=False)
    home = Team(name="H", players=[])
    state = MatchState(home_team=home, away_team=away, ball=Ball())

    state.ball.position = Position(50, 30)  # opponent half for the away team
    assert principle.check_conditions(player, state, True, away)

    state.ball.position = Position(50, 70)  # away team's own half
    assert not principle.check_conditions(player, state, True, away)


def test_towards_center_x_is_side_aware():
    """A 'push to the touchline' instruction must move the left winger left
    and the right winger right - previously both moved the same way."""
    engine = MatchEngine(SimulationConfig())
    lw = make_player("LW", role="lw", x=20, y=60, pace=100)
    rw = make_player("RW", role="rw", x=80, y=60, pace=100)
    team = Team(name="H", players=[lw, rw])
    state = MatchState(home_team=team,
                       away_team=Team(name="A", players=[]),
                       ball=Ball())
    wide = MovementInstruction(towards_center_x=-20)

    engine._apply_movement(lw, wide, state, team)
    engine._apply_movement(rw, wide, state, team)
    assert lw.position.x < 20
    assert rw.position.x > 80

    tuck = MovementInstruction(towards_center_x=20)
    engine._apply_movement(lw, tuck, state, team)
    engine._apply_movement(rw, tuck, state, team)
    assert lw.position.x > 20 - 3  # moved back toward center
    assert rw.position.x < 80 + 3


# ---------------------------------------------------------------------------
# Integration: full matches produce direction-consistent, two-sided football
# ---------------------------------------------------------------------------

def _run_matches(n, seed, on_event=None, minutes=90):
    random.seed(seed)
    config = SimulationConfig(ticks_per_minute=6, randomness=0.3)
    engine = MatchEngine(config)
    totals = {"home_goals": 0, "away_goals": 0}
    for _ in range(n):
        home, away = create_tactical_matchup("balanced", "balanced")
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        striker = next((p for p in home.players if p.role == "st"), home.players[-1])
        state.ball.give_to(striker)
        if on_event:
            engine.event_handlers = []
            engine.on_event(lambda ev, h=home, a=away: on_event(ev, h, a))
        state = engine.simulate_match(state, minutes=minutes)
        totals["home_goals"] += state.home_score
        totals["away_goals"] += state.away_score
    return totals


def test_all_shots_taken_in_shooters_attacking_third():
    violations = []

    def on_event(ev, home, away):
        if ev.event_type in ("shot", "miss") and ev.position is not None:
            team = home if ev.player in home.players else away
            frame_y = team.frame_y(ev.position.y)
            if frame_y < 65:
                violations.append((ev.player.name, round(frame_y, 1)))

    _run_matches(4, seed=101, on_event=on_event)
    assert not violations, f"shots from outside the attacking third: {violations[:5]}"


def test_both_teams_score_and_outcomes_are_roughly_symmetric():
    totals = _run_matches(16, seed=202)
    hg, ag = totals["home_goals"], totals["away_goals"]
    assert hg > 0, "home team never scored"
    assert ag > 0, "away team never scored"
    ratio = hg / ag
    assert 0.4 <= ratio <= 2.5, (
        f"identical teams should be roughly symmetric, got home {hg} vs away {ag}"
    )


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
