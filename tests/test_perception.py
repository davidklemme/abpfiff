#!/usr/bin/env python3
"""
Tests for perception: limited visual/spatial focus with experience
fill-in. A player sees what's in focus; the rest is the formation prior -
and when reality goes against the grain, the ball goes where the belief
was, not where the teammate is.

Run directly: python3 tests/test_perception.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, make_match, run_tests
from models import Ball, MatchState, Position, Team
from engine import MatchEngine, SimulationConfig
from perception import (
    FocalPerception, OmniscientPerception, BEHIND_CERTAINTY
)
from metrics import MatchMetrics
from teams import create_tactical_matchup


def perceive_one(holder, observed, opponents=None, holder_team_extra=None):
    """Perceive a single teammate; returns their Percept."""
    teammates = [holder, observed] + (holder_team_extra or [])
    state = make_match(teammates, opponents or [make_player("Opp", x=10, y=10)])
    world = FocalPerception().perceive(holder, state.home_team,
                                       state.away_team, state)
    return next(p for p in world.teammates if p.player is observed)


# ---------------------------------------------------------------------------
# Certainty: distance, vision, pressure, direction
# ---------------------------------------------------------------------------

def test_certainty_falls_with_distance():
    holder = make_player("H", x=50, y=50)
    near = perceive_one(holder, make_player("Near", x=55, y=55))
    far = perceive_one(holder, make_player("Far", x=90, y=90))
    assert near.certainty > far.certainty


def test_vision_widens_the_focus():
    observed_pos = dict(x=80, y=75)
    hawk = make_player("Hawk", x=50, y=50, vision=95)
    mole = make_player("Mole", x=50, y=50, vision=25)
    seen_by_hawk = perceive_one(hawk, make_player("T", **observed_pos))
    seen_by_mole = perceive_one(mole, make_player("T", **observed_pos))
    assert seen_by_hawk.certainty > seen_by_mole.certainty


def test_pressure_narrows_the_focus():
    """Tunnel vision: a challenged holder sees less of the far pitch."""
    holder_open = make_player("Open", x=50, y=50, composure=50)
    holder_pressed = make_player("Pressed", x=50, y=50, composure=50)
    teammate = dict(x=80, y=70)

    open_view = perceive_one(holder_open, make_player("T", **teammate),
                             opponents=[make_player("Opp", x=10, y=10)])
    pressed_view = perceive_one(holder_pressed, make_player("T", **teammate),
                                opponents=[make_player("Opp", x=51, y=51)])
    assert pressed_view.certainty < open_view.certainty


def test_players_behind_are_seen_less():
    """Frame-aware: 'behind' the away holder is toward HIGHER absolute y."""
    holder = make_player("AwayCm", x=50, y=50)
    ahead = make_player("Ahead", x=50, y=35)    # toward y=0: forward for away
    behind = make_player("Behind", x=50, y=65)  # toward own goal

    away = Team(name="A", players=[holder, ahead, behind], attacks_up=False)
    home = Team(name="H", players=[make_player("Opp", x=10, y=10)])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    world = FocalPerception().perceive(holder, away, home, state)

    seen = {p.player.name: p.certainty for p in world.teammates}
    assert seen["Behind"] < seen["Ahead"]
    assert abs(seen["Behind"] - seen["Ahead"] * BEHIND_CERTAINTY) < 1e-6


# ---------------------------------------------------------------------------
# Belief vs. truth: experience fills in, and can be wrong
# ---------------------------------------------------------------------------

def test_low_certainty_blends_toward_the_formation_prior():
    holder = make_player("H", x=50, y=20, vision=30)
    roamer = make_player("Roamer", x=85, y=85)   # actually way out right
    roamer.base_position = Position(15, 85)      # expected way out left

    percept = perceive_one(holder, roamer)

    assert percept.certainty < 0.4
    # Belief sits far closer to where they SHOULD be than where they are
    assert abs(percept.position.x - 15) < abs(percept.position.x - 85)


def test_player_near_their_prior_is_perceived_correctly_anyway():
    """Only the unexpected can be misjudged: low certainty about a player
    standing where expected still yields an accurate belief."""
    holder = make_player("H", x=50, y=10, vision=30)
    predictable = make_player("Predictable", x=80, y=80)
    predictable.base_position = Position(80, 80)

    percept = perceive_one(holder, predictable)

    assert percept.certainty < 0.5
    assert percept.position.distance_to(predictable.position) < 1e-6


def test_omniscient_model_returns_ground_truth():
    holder = make_player("H", x=50, y=50)
    roamer = make_player("Roamer", x=85, y=85)
    roamer.base_position = Position(15, 85)
    state = make_match([holder, roamer], [make_player("Opp")])

    world = OmniscientPerception().perceive(holder, state.home_team,
                                            state.away_team, state)
    percept = next(p for p in world.teammates if p.player is roamer)
    assert percept.certainty == 1.0
    assert percept.position.x == 85


# ---------------------------------------------------------------------------
# Consequences: the ball goes where the belief was
# ---------------------------------------------------------------------------

def test_pass_is_aimed_at_the_believed_position():
    """The winger checked inside, but the deep passer believes they held
    width - the ball flies toward the belief, not the player."""
    engine = MatchEngine(SimulationConfig(randomness=0.0, seed=8))
    passer = make_player("Passer", x=50, y=15, vision=30, passing=90)
    winger = make_player("Winger", x=55, y=60)      # actually came inside
    winger.base_position = Position(10, 60)         # expected: hugging the line
    state = make_match([passer, winger], [make_player("Opp", x=90, y=10)])
    state.ball.give_to(passer)

    context = engine.actions.build_decision_context(
        passer, state.home_team, state.away_team, state)
    percept = context.lanes[0][0]
    assert percept.position.x < 30  # belief: out wide

    for _ in range(30):
        event = engine.actions.passes.resolve_pass(
            state, passer, state.home_team, state.away_team,
            lanes=context.lanes, turnover=lambda *a: None)
        if event is not None and event.event_type == "pass":
            break
        state.ball.give_to(passer)
    else:
        raise AssertionError("no pass launched in 30 attempts")

    assert event.target_player is winger      # the ball's receiver is real
    assert state.ball.target_position.x < 35  # ...but it flies to the belief


def test_perception_errors_cost_completion():
    """Focal perception must not outperform omniscience (determinism-
    checked over a seeded sample)."""
    def completion(perception_model, seed):
        engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=seed))
        engine.actions.perception = perception_model
        completed = attempts = 0
        for _ in range(4):
            home, away = create_tactical_matchup("balanced", "balanced")
            state = MatchState(home_team=home, away_team=away, ball=Ball())
            m = MatchMetrics(home, away)
            engine.event_handlers = [m.on_event]
            engine.simulate_match(state, minutes=90)
            completed += m.home.passes_completed + m.away.passes_completed
            attempts += m.home.pass_attempts + m.away.pass_attempts
        return completed / attempts

    focal = completion(FocalPerception(), seed=31)
    omniscient = completion(OmniscientPerception(), seed=31)
    assert focal <= omniscient + 0.02, (focal, omniscient)


def test_vision_now_has_perceptual_value():
    """A high-vision squad completes more passes than a low-vision one -
    the attribute governs what players actually see. Aggregated over
    several seeds: the effect is real but a single 4-match sample is
    noisy (adversarial review: seed 17 alone inverts)."""
    def completion(vision, seed):
        engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=seed))
        completed = attempts = 0
        for _ in range(4):
            home, away = create_tactical_matchup("balanced", "balanced")
            for p in home.players + away.players:
                p.vision = vision
            state = MatchState(home_team=home, away_team=away, ball=Ball())
            m = MatchMetrics(home, away)
            engine.event_handlers = [m.on_event]
            engine.simulate_match(state, minutes=90)
            completed += m.home.passes_completed + m.away.passes_completed
            attempts += m.home.pass_attempts + m.away.pass_attempts
        return completed / attempts

    seeds = (17, 23, 31)
    sharp = sum(completion(vision=90, seed=s) for s in seeds)
    blind = sum(completion(vision=20, seed=s) for s in seeds)
    assert sharp > blind, (sharp, blind)


if __name__ == "__main__":
    run_tests(globals())
