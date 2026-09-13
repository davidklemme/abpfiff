#!/usr/bin/env python3
"""
Tests for the Phase 1 psychological engine (psychology.py) and its
integration into engine.py.

No test framework dependency (the project has none) - plain asserts,
run directly: python3 tests/test_psychology.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchEvent, MatchState, Player, Position, Team
import psychology as psych
from engine import MatchEngine, SimulationConfig
from teams import create_tactical_matchup


def make_player(name="P", composure=50, confidence=0.0, fatigue=0.0, **kwargs) -> Player:
    return Player(name=name, number=1, composure=composure, confidence=confidence,
                  fatigue=fatigue, **kwargs)


def make_team(name: str, players, role_default="cm") -> Team:
    for p in players:
        if not p.role or p.role == "default":
            p.role = role_default
    return Team(name=name, players=players)


# ---------------------------------------------------------------------------
# Unit tests: formulas
# ---------------------------------------------------------------------------

def test_effective_composure_boosted_by_confidence():
    calm = make_player(composure=50, confidence=0.0)
    confident = make_player(composure=50, confidence=0.8)
    assert psych.effective_composure(confident) > psych.effective_composure(calm)


def test_effective_composure_reduced_by_fatigue():
    fresh = make_player(composure=50, confidence=0.0, fatigue=0.0)
    tired = make_player(composure=50, confidence=0.0, fatigue=100.0)
    assert psych.effective_composure(tired) < psych.effective_composure(fresh)


def test_effective_composure_clamped_0_to_1():
    maxed = make_player(composure=100, confidence=1.0, fatigue=0.0)
    minned = make_player(composure=0, confidence=-1.0, fatigue=100.0)
    assert 0.0 < psych.effective_composure(maxed) <= 1.0
    assert 0.0 <= psych.effective_composure(minned) < 1.0


def test_pressure_triggers_system1():
    """Mirrors psychological-engine.md E.1: low pressure -> analysis, high -> instinct."""
    player = make_player(composure=50, confidence=0.0)
    low_weight = psych.system1_weight(0.15, player)
    high_weight = psych.system1_weight(0.95, player)
    assert low_weight < 0.5
    assert high_weight > 0.8


def test_confidence_buffs_system2_access():
    """Higher confidence -> more system2 (analysis) weight under the same pressure."""
    pressure_total = 0.5
    low_conf = make_player(composure=50, confidence=-0.5)
    high_conf = make_player(composure=50, confidence=0.8)
    assert psych.system1_weight(pressure_total, high_conf) < psych.system1_weight(pressure_total, low_conf)


def test_calculate_pressure_spatial_rises_with_proximity():
    defender = make_player(name="Defender")
    attacker = make_player(name="Attacker")
    opponents = make_team("Opp", [defender])
    state = MatchState(home_team=make_team("Home", [attacker]), away_team=opponents, ball=Ball())

    defender.position = Position(50, 50)

    attacker.position = Position(50, 51)  # very close
    close_pressure = psych.calculate_pressure(attacker, state, opponents)

    attacker.position = Position(90, 90)  # far away
    far_pressure = psych.calculate_pressure(attacker, state, opponents)

    assert close_pressure.spatial > far_pressure.spatial


def test_calculate_pressure_tactical_rises_when_losing_late():
    attacker = make_player(name="Attacker")
    opponents = make_team("Opp", [make_player(name="D")])
    home = make_team("Home", [attacker])
    state = MatchState(home_team=home, away_team=opponents, ball=Ball(),
                        minute=10, home_score=0, away_score=0)
    early_drawing = psych.calculate_pressure(attacker, state, opponents)

    state.minute = 85
    state.home_score = 0
    state.away_score = 2
    late_losing = psych.calculate_pressure(attacker, state, opponents)

    assert late_losing.tactical > early_drawing.tactical


# ---------------------------------------------------------------------------
# Unit tests: feedback -> confidence
# ---------------------------------------------------------------------------

def _state_with(scorer_home=True):
    home_gk = make_player(name="HomeGK", role="gk")
    away_gk = make_player(name="AwayGK", role="gk")
    home = make_team("Home", [make_player(name="HomeScorer"), home_gk])
    away = make_team("Away", [make_player(name="AwayScorer"), away_gk])
    return MatchState(home_team=home, away_team=away, ball=Ball())


def test_goal_increases_scorer_confidence_and_hurts_keeper():
    state = _state_with()
    scorer = state.home_team.players[0]
    keeper = state.away_team.goalkeeper
    event = MatchEvent(minute=10, event_type="goal", player=scorer, success=True)

    psych.process_feedback(event, state)

    assert scorer.confidence > 0
    assert keeper.confidence < 0


def test_missed_clear_chance_hurts_more_than_a_speculative_miss():
    state = _state_with()
    close_shooter = state.home_team.players[0]
    far_shooter = state.away_team.players[0]

    psych.process_feedback(MatchEvent(minute=1, event_type="miss", player=close_shooter,
                                       position=Position(50, 95), success=False), state)
    psych.process_feedback(MatchEvent(minute=1, event_type="miss", player=far_shooter,
                                       position=Position(50, 72), success=False), state)

    assert close_shooter.confidence < far_shooter.confidence < 0


def test_pass_received_credits_the_original_passer():
    state = _state_with()
    passer = state.home_team.players[0]
    receiver = state.away_team.players[0]
    event = MatchEvent(minute=5, event_type="pass_received", player=receiver,
                        target_player=passer, success=True)

    psych.process_feedback(event, state)

    assert passer.confidence > 0
    assert receiver.confidence == 0  # only the passer is credited in Phase 1


def test_tackle_rewards_defender_and_punishes_attacker():
    state = _state_with()
    defender = state.away_team.players[0]
    attacker = state.home_team.players[0]
    event = MatchEvent(minute=20, event_type="tackle", player=defender,
                        target_player=attacker, success=True)

    psych.process_feedback(event, state)

    assert defender.confidence > 0
    assert attacker.confidence < 0


def test_confidence_deltas_clamp_to_valid_range():
    state = _state_with()
    scorer = state.home_team.players[0]
    scorer.confidence = 0.95
    event = MatchEvent(minute=10, event_type="goal", player=scorer, success=True)

    psych.process_feedback(event, state)

    assert scorer.confidence <= 1.0


def test_unhandled_event_type_leaves_confidence_unchanged():
    state = _state_with()
    player = state.home_team.players[0]
    event = MatchEvent(minute=1, event_type="ball_won", player=player, success=True)

    psych.process_feedback(event, state)

    assert player.confidence == 0.0


# ---------------------------------------------------------------------------
# Unit tests: decay
# ---------------------------------------------------------------------------

def test_decay_moves_confidence_toward_zero_from_both_sides():
    happy = make_player(confidence=0.5)
    sad = make_player(confidence=-0.5)

    psych.decay_confidence(happy, minutes_elapsed=5)
    psych.decay_confidence(sad, minutes_elapsed=5)

    assert 0.0 < happy.confidence < 0.5
    assert -0.5 < sad.confidence < 0.0


def test_decay_does_not_overshoot_past_zero():
    player = make_player(confidence=0.01)
    psych.decay_confidence(player, minutes_elapsed=100)
    assert player.confidence == 0.0


# ---------------------------------------------------------------------------
# Integration tests: psychology visibly changes engine behavior
# ---------------------------------------------------------------------------

def test_rattled_player_positions_closer_to_base_than_confident_player():
    """psychological-engine.md E.2: a rattled player's ball_seeking should be lower."""
    engine = MatchEngine(SimulationConfig())

    def build(confidence):
        player = make_player(name="Winger", role="lw", confidence=confidence)
        player.position = Position(50, 50)
        player.base_position = Position(10, 60)
        team = make_team("Home", [player])
        opponents = make_team("Away", [make_player(name="Marker")])
        state = MatchState(home_team=team, away_team=opponents, ball=Ball())
        state.ball.position = Position(55, 55)
        return player, opponents, state

    # attacking movement adds random jitter; seed the engine's own RNG
    # identically so confidence is the only thing that differs.
    rattled, opp1, state1 = build(confidence=-0.8)
    engine.rng.seed(7)
    tx_r, ty_r = engine._attacking_movement(rattled, state1.ball.position, None,
                                             state1.home_team, opp1, state1)

    confident, opp2, state2 = build(confidence=0.8)
    engine.rng.seed(7)
    tx_c, ty_c = engine._attacking_movement(confident, state2.ball.position, None,
                                             state2.home_team, opp2, state2)

    dist_rattled_to_base = Position(tx_r, ty_r).distance_to(rattled.base_position)
    dist_confident_to_base = Position(tx_c, ty_c).distance_to(confident.base_position)

    assert dist_rattled_to_base < dist_confident_to_base


def test_confident_player_shoots_more_often_than_rattled_player():
    engine = MatchEngine(SimulationConfig(randomness=0.0, seed=42))

    def attempt_counts(confidence, trials=300):
        shooter = make_player(name="Striker", role="st", confidence=confidence, composure=50)
        shooter.position = Position(50, 90)  # in shooting range, central
        defender = make_player(name="CB")
        defender.position = Position(50, 20)  # far away, low pressure
        attacking = make_team("Home", [shooter])
        defending = make_team("Away", [defender])
        state = MatchState(home_team=attacking, away_team=defending, ball=Ball(), minute=45)

        shots = sum(1 for _ in range(trials)
                    if engine._decide_action(shooter, attacking, defending, state) == "shoot")
        return shots

    rattled_shots = attempt_counts(confidence=-0.8)
    confident_shots = attempt_counts(confidence=0.8)

    assert confident_shots > rattled_shots


def test_full_match_produces_varied_confidence_and_stays_in_range():
    """End-to-end smoke test: a full simulated match updates confidence and
    never lets it escape [-1, 1]."""
    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, randomness=0.3))
    home, away = create_tactical_matchup("gegenpressing", "low_block_counter")

    ball = Ball()
    state = MatchState(home_team=home, away_team=away, ball=ball)
    striker = next((p for p in home.players if p.role == "st"), home.players[-1])
    ball.give_to(striker)

    state = engine.simulate_match(state, minutes=90)

    all_players = home.players + away.players
    for p in all_players:
        assert -1.0 <= p.confidence <= 1.0

    assert any(p.confidence != 0.0 for p in all_players)


from support import run_tests

if __name__ == "__main__":
    run_tests(globals())
