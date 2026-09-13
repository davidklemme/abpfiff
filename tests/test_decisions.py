#!/usr/bin/env python3
"""
Tests for Phase 2b decision depth: situation embeddings, seeded instinct
banks, and the dual-process (System 1/2) decision model.

The behavioral-separation tests are the acceptance criteria of this
phase: players with different personalities must make measurably
different decisions in identical situations.

No test framework dependency - plain asserts, run directly:
    python3 tests/test_decisions.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchState, Player, Position, Team
from engine import MatchEngine, SimulationConfig
from situation import SituationEmbedding, situation_for, similarity
from instincts import default_bank_for, BOLD_ACTIONS
from decisions import DecisionContext, DualProcessDecisionModel


def make_player(name="P", role="cm", x=50.0, y=50.0, **attrs) -> Player:
    p = Player(name=name, number=1, role=role, **attrs)
    p.position = Position(x, y)
    p.base_position = Position(x, y)
    return p


def make_context(holder, pressure=0.4, progression=0.6, density=0.4,
                 support=0.5, in_range=False, space_ahead=15.0,
                 nearest_def=10.0, best_forward=0.3, best_safe=0.3):
    """A synthetic decision context - lanes list is non-empty so passing
    stays possible."""
    home = Team(name="H", players=[holder])
    away = Team(name="A", players=[make_player("D")])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    situation = SituationEmbedding(pressure, progression, 0.5, density, support)
    teammate = make_player("T")
    return DecisionContext(
        holder=holder, attacking_team=home, defending_team=away, state=state,
        situation=situation, in_shooting_range=in_range,
        space_ahead=space_ahead, nearest_defender_dist=nearest_def,
        lanes=[(teammate, best_safe)], best_forward_lane=best_forward,
        best_safe_lane=best_safe,
    )


def action_shares(model, context_fn, trials=500):
    counts = {}
    for _ in range(trials):
        action = model.decide(context_fn())
        counts[action] = counts.get(action, 0) + 1
    return {a: c / trials for a, c in counts.items()}


def bold_share(shares):
    return sum(shares.get(a, 0.0) for a in BOLD_ACTIONS)


# ---------------------------------------------------------------------------
# Situation embedding
# ---------------------------------------------------------------------------

def test_situation_fields_are_bounded():
    holder = make_player("H", role="st", x=50, y=80)
    home = Team(name="H", players=[holder])
    away = Team(name="A", players=[make_player("D", x=52, y=81)])
    state = MatchState(home_team=home, away_team=away, ball=Ball(), minute=85)

    sit = situation_for(holder, home, away, state, lanes=[])
    for value in sit.as_tuple():
        assert 0.0 <= value <= 1.0, sit


def test_progression_is_frame_aware():
    """An away player at absolute y=20 is deep in the opponent half."""
    holder = make_player("A", role="st", x=50, y=20)
    away = Team(name="A", players=[holder], attacks_up=False)
    home = Team(name="H", players=[make_player("D", x=10, y=90)])
    state = MatchState(home_team=home, away_team=away, ball=Ball())

    sit = situation_for(holder, away, home, state)
    assert sit.progression == 0.8


def test_similarity_identity_and_ordering():
    a = SituationEmbedding(0.5, 0.5, 0.5, 0.5, 0.5)
    near = SituationEmbedding(0.55, 0.5, 0.5, 0.5, 0.5)
    far = SituationEmbedding(1.0, 0.0, 1.0, 0.0, 1.0)
    assert similarity(a, a) == 1.0
    assert similarity(a, near) > similarity(a, far)


# ---------------------------------------------------------------------------
# Instinct banks
# ---------------------------------------------------------------------------

def test_role_seeds_differ_between_striker_and_defender():
    box = SituationEmbedding(0.5, 0.85, 0.5, 0.6, 0.5)
    striker_bank = default_bank_for(make_player("St", role="st"))
    defender_bank = default_bank_for(make_player("Cb", role="cb"))

    striker_instinct = striker_bank.query(box)
    defender_instinct = defender_bank.query(box)

    assert striker_instinct["shoot"] > defender_instinct["shoot"]
    assert defender_instinct["pass_safe"] > striker_instinct["pass_safe"]


def test_confidence_tilts_instincts_toward_bold_or_safe():
    situation = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5)
    bank = default_bank_for(make_player("Cm", role="cm"))

    confident = bank.query(situation, confidence=0.8)
    rattled = bank.query(situation, confidence=-0.8)

    assert bold_share(confident) > bold_share(rattled)
    assert rattled["pass_safe"] > confident["pass_safe"]


def test_aggression_shapes_seeded_bank():
    situation = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5)
    aggressive = default_bank_for(make_player("Agg", role="cm", aggression=90))
    timid = default_bank_for(make_player("Tim", role="cm", aggression=20))

    assert bold_share(aggressive.query(situation)) > bold_share(timid.query(situation))


# ---------------------------------------------------------------------------
# Dual-process decision model: behavioral separation
# ---------------------------------------------------------------------------

def test_personality_creates_decision_variance():
    """psychological-engine.md E.2: identical situation, different
    personalities, measurably different action distributions."""
    chaos = make_player("Chaos", role="cm", composure=35, aggression=90)
    metronome = make_player("Metronome", role="cm", composure=90, aggression=25)

    model = DualProcessDecisionModel(rng=random.Random(42))
    ctx = lambda p: make_context(p, pressure=0.55, in_range=True,
                                 best_forward=0.35, best_safe=0.35)

    chaos_shares = action_shares(model, lambda: ctx(chaos))
    metronome_shares = action_shares(model, lambda: ctx(metronome))

    assert bold_share(chaos_shares) > bold_share(metronome_shares) * 1.3, (
        chaos_shares, metronome_shares)


def test_pressure_pushes_low_composure_players_to_the_safe_ball():
    """Firm pressure - below total overload - drives the nervy player to
    the safe ball. (At crushing load, overload scrambles even the safe
    habit: psychology.system1_integrity; see tests/test_cognition.py.)"""
    nervy = make_player("Nervy", role="cm", composure=35, aggression=50)

    # Paired comparison: each condition gets an identically-seeded model,
    # so the only difference between the samples is the pressure
    calm_shares = action_shares(
        DualProcessDecisionModel(rng=random.Random(7)),
        lambda: make_context(nervy, pressure=0.1, density=0.2))
    pressed_shares = action_shares(
        DualProcessDecisionModel(rng=random.Random(7)),
        lambda: make_context(nervy, pressure=0.75, density=0.7))

    assert pressed_shares.get("pass_safe", 0) > calm_shares.get("pass_safe", 0)


def test_composure_preserves_analysis_under_pressure():
    """Under identical high pressure, the composed player stays closer to
    the analytical (forward-looking) game than the nervy one."""
    nervy = make_player("Nervy", role="cm", composure=30, aggression=50)
    iceman = make_player("Iceman", role="cm", composure=95, aggression=50)
    model = DualProcessDecisionModel(rng=random.Random(11))

    pressed = lambda p: make_context(p, pressure=0.8, density=0.8,
                                     best_forward=0.5, best_safe=0.2)
    nervy_shares = action_shares(model, lambda: pressed(nervy))
    iceman_shares = action_shares(model, lambda: pressed(iceman))

    assert iceman_shares.get("pass_forward", 0) > nervy_shares.get("pass_forward", 0)


def test_shoot_never_chosen_out_of_range():
    shooter = make_player("St", role="st", aggression=95)
    model = DualProcessDecisionModel(rng=random.Random(3))
    for _ in range(200):
        action = model.decide(make_context(shooter, in_range=False))
        assert action != "shoot"


# ---------------------------------------------------------------------------
# The DecisionModel seam
# ---------------------------------------------------------------------------

class AlwaysDribble:
    def decide(self, context) -> str:
        return "dribble"


def test_decision_model_is_injectable():
    from ball_actions import DefaultActionResolver
    resolver = DefaultActionResolver(rng=random.Random(1),
                                     decision_model=AlwaysDribble())
    holder = make_player("H", role="cm", x=50, y=50)
    home = Team(name="H", players=[holder])
    away = Team(name="A", players=[make_player("D", x=60, y=60)])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    state.ball.give_to(holder)

    assert resolver.decide_action(holder, home, away, state) == "dribble"


def test_resolver_builds_frame_aware_context():
    engine = MatchEngine(SimulationConfig(seed=5))
    holder = make_player("A", role="cm", x=50, y=60)
    mate_forward = make_player("Mate", role="st", x=50, y=40)  # forward for away
    away = Team(name="A", players=[holder, mate_forward], attacks_up=False)
    home = Team(name="H", players=[make_player("D", x=20, y=20)])
    state = MatchState(home_team=home, away_team=away, ball=Ball())
    state.ball.give_to(holder)

    ctx = engine.actions.build_decision_context(holder, away, home, state)

    # The teammate at lower absolute y is a FORWARD option for the away team
    assert ctx.best_forward_lane > 0
    assert ctx.situation.progression == away.frame_y(60) / 100


from support import run_tests

if __name__ == "__main__":
    run_tests(globals())
