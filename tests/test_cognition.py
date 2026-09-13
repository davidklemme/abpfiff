#!/usr/bin/env python3
"""
Tests for cognitive load as a core mechanism:

  load = match pressure + environment x sensitivity + novelty
  familiarity (pre-exposure via the instinct bank) buys load relief and
  gives System 1 something real to offer; overload - load beyond what
  composure absorbs - degrades System 1 itself. Facing is derived state
  (motion), and literal visibility is a separate perceptual channel.

Run directly: python3 tests/test_cognition.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, make_match, run_tests
from models import Ball, Environment, MatchState, Position, Team
from engine import MatchEngine, SimulationConfig
from situation import SituationEmbedding
from instincts import InstinctBank, Instinct, BOLD_ACTIONS
from decisions import DualProcessDecisionModel, DecisionContext
from perception import FocalPerception
import psychology

BIG_NIGHT = Environment(stakes=0.9, crowd_intensity=0.9)
S = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5)


def make_context(holder, state, pressure=None):
    """Synthetic context; unless overridden, the situation's pressure is
    computed through psychology.calculate_pressure so the environment
    channel flows exactly as in production (situation_for does the same)."""
    if pressure is None:
        pressure = min(1.0, 0.35 + psychology.calculate_pressure(
            holder, state, state.away_team).total)
    teammate = make_player("T")
    return DecisionContext(
        holder=holder, attacking_team=state.home_team,
        defending_team=state.away_team, state=state,
        situation=SituationEmbedding(pressure, 0.6, 0.5, 0.5, 0.5),
        in_shooting_range=True, space_ahead=15.0, nearest_defender_dist=10.0,
        lanes=[(teammate, 0.35)], best_forward_lane=0.35, best_safe_lane=0.35,
    )


def action_shares(model, context_fn, trials=400):
    counts = {}
    for _ in range(trials):
        action = model.decide(context_fn())
        counts[action] = counts.get(action, 0) + 1
    return {a: c / trials for a, c in counts.items()}


# ---------------------------------------------------------------------------
# Environment -> mental load through sensitivity
# ---------------------------------------------------------------------------

def test_neutral_environment_imposes_no_load():
    assert Environment().psychological_load(sensitivity=90) == 0.0


def test_sensitivity_converts_environment_into_load():
    thin_skinned = BIG_NIGHT.psychological_load(sensitivity=90)
    thick_skinned = BIG_NIGHT.psychological_load(sensitivity=10)
    assert thin_skinned > thick_skinned > 0.0


def test_environmental_load_flows_into_pressure():
    player = make_player("P", sensitivity=85)
    state = make_match([player], [make_player("Opp", x=10, y=10)])

    calm = psychology.calculate_pressure(player, state, state.away_team)
    state.environment = BIG_NIGHT
    big = psychology.calculate_pressure(player, state, state.away_team)

    assert calm.psychological == 0.0
    assert big.psychological > 0.4
    assert big.total > calm.total


def test_sensitive_players_play_safer_on_big_nights():
    """The limelight consumes capacity: under identical situations the
    sensitive player retreats to the safe ball more when the stage grows.
    (Composure high enough that load shifts the blend without crossing
    into overload - past absorption, even the safe habit scrambles, see
    test_overload_blurs_even_trained_instinct.)"""
    nervous = make_player("Nervous", role="cm", sensitivity=95, composure=75)
    model = DualProcessDecisionModel(rng=random.Random(5))

    def shares(env):
        state = make_match([nervous], [make_player("Opp", x=10, y=10)])
        state.environment = env
        return action_shares(model, lambda: make_context(nervous, state))

    quiet = shares(Environment())
    big = shares(BIG_NIGHT)
    assert big.get("pass_safe", 0) > quiet.get("pass_safe", 0)


# ---------------------------------------------------------------------------
# Familiarity: pre-exposure is load relief and System 1 content
# ---------------------------------------------------------------------------

def test_familiarity_reflects_pre_exposure():
    empty = InstinctBank([])
    assert empty.familiarity(S) == 0.0

    exposed = InstinctBank([])
    exposed.learn(S, "shoot", valence=1.0, significance=0.9)
    assert exposed.familiarity(S) > 0.2
    far = SituationEmbedding(1.0, 0.05, 1.0, 0.05, 1.0, 0.05)
    assert exposed.familiarity(far) < exposed.familiarity(S)


def test_veterans_keep_their_game_under_the_lights():
    """Same attributes, same big night: the player whose bank KNOWS this
    situation (pre-exposure) acts on it; the newcomer's flat instinct
    retreats to the fallback."""
    def bold_share_for(pre_exposed, seed):
        player = make_player("P", role="st", sensitivity=80, composure=50)
        model = DualProcessDecisionModel(rng=random.Random(seed))
        mind = model.minds.mind_for(player)
        mind.bank = InstinctBank([])
        if pre_exposed:
            for _ in range(6):  # six career nights like this one
                mind.bank.learn(SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5),
                                "shoot", valence=1.0, significance=0.8)
        state = make_match([player], [make_player("Opp", x=10, y=10)])
        state.environment = BIG_NIGHT
        shares = action_shares(model, lambda: make_context(player, state))
        return sum(shares.get(a, 0) for a in BOLD_ACTIONS)

    assert bold_share_for(True, seed=9) > bold_share_for(False, seed=9)


# ---------------------------------------------------------------------------
# Overload degrades System 1 itself; composure is the absorption
# ---------------------------------------------------------------------------

def test_no_overload_while_composure_absorbs_the_load():
    iceman = make_player("Iceman", composure=90)
    assert psychology.overload(0.5, iceman) == 0.0
    assert psychology.system1_integrity(0.5, iceman) == 1.0


def test_overload_grows_past_absorption_and_composure_gates_it():
    nervy = make_player("Nervy", composure=25)
    iceman = make_player("Iceman", composure=90)
    heavy_load = 0.9

    assert psychology.overload(heavy_load, nervy) > psychology.overload(heavy_load, iceman)
    assert psychology.system1_integrity(heavy_load, nervy) < \
           psychology.system1_integrity(heavy_load, iceman)


def test_overload_blurs_even_trained_instinct():
    """A strong shoot instinct decides less under crushing load: the
    action distribution of the overloaded player is measurably flatter."""
    def shoot_share(composure, seed):
        player = make_player("P", role="st", sensitivity=95, composure=composure)
        model = DualProcessDecisionModel(rng=random.Random(seed))
        mind = model.minds.mind_for(player)
        mind.bank = InstinctBank([Instinct(
            "drilled", SituationEmbedding(0.9, 0.6, 0.5, 0.5, 0.5),
            {"shoot": 0.9, "dribble": 0.03, "pass_forward": 0.03,
             "cross": 0.01, "pass_safe": 0.03})])
        state = make_match([player], [make_player("Opp", x=10, y=10)])
        state.environment = BIG_NIGHT
        shares = action_shares(model,
                               lambda: make_context(player, state, pressure=0.9))
        return shares.get("shoot", 0)

    assert shoot_share(composure=90, seed=13) > shoot_share(composure=15, seed=13)


# ---------------------------------------------------------------------------
# Perceptual channel: visibility; orientation: derived facing
# ---------------------------------------------------------------------------

def test_poor_visibility_narrows_focus_more_for_the_sensitive():
    def certainty(sensitivity, visibility):
        holder = make_player("H", x=50, y=50, sensitivity=sensitivity)
        far_mate = make_player("T", x=70, y=65)  # in range of a clear day
        state = make_match([holder, far_mate], [make_player("Opp", x=10, y=10)])
        state.environment = Environment(visibility=visibility)
        world = FocalPerception().perceive(holder, state.home_team,
                                           state.away_team, state)
        return world.teammates[0].certainty

    assert certainty(50, visibility=0.3) < certainty(50, visibility=1.0)
    clear_gap = certainty(90, 1.0) - certainty(90, 0.3)
    calm_gap = certainty(10, 1.0) - certainty(10, 0.3)
    assert clear_gap > calm_gap  # murk costs the sensitive more


def test_facing_is_derived_from_motion():
    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=3))
    from teams import create_tactical_matchup
    home, away = create_tactical_matchup("balanced", "balanced")
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    engine.simulate_match(state, minutes=5)

    moving = [p for p in home.players + away.players
              if abs(p.velocity_x) + abs(p.velocity_y) > 0.1]
    assert moving, "five minutes of play should leave players in motion"


def test_moving_player_sees_ahead_of_motion_best():
    holder = make_player("H", x=50, y=50)
    holder.velocity_x, holder.velocity_y = 0.0, 2.0  # heading up-pitch
    ahead = make_player("Ahead", x=50, y=70)
    behind = make_player("Behind", x=50, y=30)
    state = make_match([holder, ahead, behind], [make_player("Opp", x=10, y=10)])

    world = FocalPerception().perceive(holder, state.home_team,
                                       state.away_team, state)
    seen = {p.player.name: p.certainty for p in world.teammates}
    assert seen["Ahead"] > seen["Behind"]


def test_facing_beats_the_frame_proxy_when_moving_backwards():
    """A player retreating toward their own goal sees the 'backward'
    teammate they are running toward better than the frame proxy would."""
    retreating = make_player("H", x=50, y=50)
    retreating.velocity_x, retreating.velocity_y = 0.0, -2.0  # toward own goal
    deep_mate = make_player("Deep", x=50, y=30)
    state = make_match([retreating, deep_mate], [make_player("Opp", x=10, y=10)])

    world = FocalPerception().perceive(retreating, state.home_team,
                                       state.away_team, state)
    moving_view = world.teammates[0].certainty

    retreating.velocity_x = retreating.velocity_y = 0.0  # frame fallback
    world = FocalPerception().perceive(retreating, state.home_team,
                                       state.away_team, state)
    stationary_view = world.teammates[0].certainty

    assert moving_view > stationary_view


if __name__ == "__main__":
    run_tests(globals())
