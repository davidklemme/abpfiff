#!/usr/bin/env python3
"""
Tests for Phase 3 learning & memory: outcomes write success anchors and
traumas into instinct banks, memories merge/decay/prune, confidence
selects between memory classes, and the full decision->outcome->memory
loop runs inside real matches.

Run directly: python3 tests/test_learning.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, make_match, run_tests
from models import Ball, MatchEvent, MatchState, Position
from engine import MatchEngine, SimulationConfig
from situation import SituationEmbedding
from instincts import InstinctBank
from minds import MindRegistry
from learning import ExperienceLearning
from teams import create_tactical_matchup


BOX = SituationEmbedding(0.5, 0.85, 0.5, 0.6, 0.5)
SIMILAR_BOX = SituationEmbedding(0.55, 0.8, 0.5, 0.65, 0.5)
DEEP = SituationEmbedding(0.4, 0.15, 0.3, 0.3, 0.5)


def empty_bank() -> InstinctBank:
    return InstinctBank([])


# ---------------------------------------------------------------------------
# Business logic: memory formation in the bank
# ---------------------------------------------------------------------------

def test_success_forms_an_anchor_that_reinforces_the_action():
    bank = empty_bank()
    before = bank.query(BOX)["shoot"]
    bank.learn(BOX, "shoot", valence=1.0, significance=0.9)
    after = bank.query(BOX)["shoot"]
    assert after > before
    assert len(bank.learned_memories()) == 1


def test_trauma_suppresses_the_action_in_similar_situations():
    bank = empty_bank()
    # A neutral anchor for another action so query has positive mass
    bank.learn(BOX, "pass_safe", valence=0.5, significance=0.5)
    baseline = bank.query(SIMILAR_BOX)["shoot"]
    bank.learn(BOX, "shoot", valence=-1.0, significance=0.9)
    suppressed = bank.query(SIMILAR_BOX)["shoot"]
    assert suppressed <= baseline


def test_similar_memories_merge_instead_of_duplicating():
    bank = empty_bank()
    bank.learn(BOX, "shoot", valence=1.0, significance=0.6)
    mass_before = bank.learned_memories()[0].mass
    bank.learn(SIMILAR_BOX, "shoot", valence=1.0, significance=0.6)

    memories = bank.learned_memories()
    assert len(memories) == 1  # merged, not appended
    assert memories[0].mass > mass_before


def test_dissimilar_memories_stay_separate():
    bank = empty_bank()
    bank.learn(BOX, "shoot", valence=1.0, significance=0.6)
    bank.learn(DEEP, "shoot", valence=1.0, significance=0.6)
    assert len(bank.learned_memories()) == 2


def test_mass_is_logarithmic_and_retention_is_power_law():
    bank = empty_bank()
    bank.learn(BOX, "shoot", valence=1.0, significance=0.5)
    trace = bank.learned_memories()[0]
    first_mass = trace.mass
    first_retained = trace.retained_mass()
    bank.learn(BOX, "shoot", valence=1.0, significance=0.5)

    assert first_mass < trace.mass < first_mass * 2
    bank.decay(1.0)
    after_one = trace.retained_mass()
    bank.decay(9.0)
    after_ten = trace.retained_mass()
    assert trace.mass > first_mass  # decay never destroys stored evidence
    assert after_ten < after_one < trace.mass
    assert after_ten > 0.0


def test_reinforcement_adapts_each_axis_independently():
    bank = empty_bank()
    bank.learn(BOX, "shoot", valence=1.0, significance=0.6)
    trace = bank.learned_memories()[0]
    before = trace.widths
    varied = SituationEmbedding(0.5, 0.50, 0.5, 0.6, 0.5)
    bank.learn(varied, "shoot", valence=1.0, significance=0.6)
    assert trace.widths[0] < before[0]       # pressure agreed
    assert trace.widths[1] > trace.widths[0]  # progression varied


def test_spaced_reinforcement_flattens_retention():
    massed, spaced = empty_bank(), empty_bank()
    for bank in (massed, spaced):
        bank.learn(BOX, "shoot", valence=1.0, significance=0.5)
    massed.learn(BOX, "shoot", valence=1.0, significance=0.5)
    spaced.decay(4.0)
    spaced.learn(BOX, "shoot", valence=1.0, significance=0.5)
    massed.decay(10.0)
    spaced.decay(10.0)
    assert spaced.learned_memories()[0].retained_mass() > \
           massed.learned_memories()[0].retained_mass()


def test_confidence_selects_between_anchor_and_trauma():
    """Doc section 5.3: confident players sample success anchors, rattled
    players feel their traumas."""
    bank = empty_bank()
    bank.learn(BOX, "shoot", valence=1.0, significance=0.8)      # anchor
    bank.learn(BOX, "pass_safe", valence=0.6, significance=0.5)  # comparison mass

    confident = bank.query(BOX, confidence=0.8)["shoot"]
    rattled = bank.query(BOX, confidence=-0.8)["shoot"]
    assert confident > rattled

    trauma_bank = empty_bank()
    trauma_bank.learn(BOX, "pass_safe", valence=0.6, significance=0.5)
    trauma_bank.learn(BOX, "shoot", valence=-1.0, significance=0.8)  # trauma
    confident_t = trauma_bank.query(BOX, confidence=0.8)["shoot"]
    rattled_t = trauma_bank.query(BOX, confidence=-0.8)["shoot"]
    assert rattled_t <= confident_t


# ---------------------------------------------------------------------------
# Business logic: pairing decisions with outcomes
# ---------------------------------------------------------------------------

def _learning_fixture():
    scorer = make_player("Scorer", role="st")
    keeper = make_player("GK", role="gk")
    state = make_match([scorer], [keeper])
    minds = MindRegistry()
    learning = ExperienceLearning(minds)
    return scorer, keeper, state, minds, learning


def test_goal_outcome_reinforces_the_pending_decision():
    scorer, _, state, minds, learning = _learning_fixture()
    mind = minds.mind_for(scorer)
    role_mass = sum(trace.mass for trace in mind.bank.instincts)
    mind.remember_decision(BOX, "shoot")

    goal = MatchEvent(minute=10, event_type="goal", player=scorer,
                      position=Position(50, 100))
    learning.on_event(goal, state)

    # A compatible success can reinforce schooling directly; it need not
    # manufacture a parallel "experience" class.
    assert sum(trace.mass for trace in mind.bank.instincts) > role_mass
    assert mind.pending_decision is None  # consumed


def test_interception_traumatizes_the_passer_not_the_interceptor():
    passer = make_player("Passer", role="cm")
    interceptor = make_player("Interceptor", role="cm")
    state = make_match([passer], [interceptor])
    minds = MindRegistry()
    learning = ExperienceLearning(minds)
    minds.mind_for(passer).remember_decision(BOX, "pass_forward")

    event = MatchEvent(minute=5, event_type="interception", player=interceptor,
                       target_player=passer, position=Position(50, 50))
    learning.on_event(event, state)

    assert minds.mind_for(passer).bank.learned_memories()[0].source == "trauma"
    assert minds.mind_for(interceptor).bank.learned_memories() == []


def test_outcome_without_pending_decision_is_ignored():
    scorer, _, state, minds, learning = _learning_fixture()
    goal = MatchEvent(minute=10, event_type="goal", player=scorer,
                      position=Position(50, 100))
    learning.on_event(goal, state)
    assert minds.mind_for(scorer).bank.learned_memories() == []


def test_unmapped_event_types_do_not_learn_or_consume():
    scorer, _, state, minds, learning = _learning_fixture()
    mind = minds.mind_for(scorer)
    mind.remember_decision(BOX, "shoot")

    launch = MatchEvent(minute=3, event_type="pass", player=scorer)
    learning.on_event(launch, state)

    assert mind.pending_decision is not None  # still awaiting a real outcome
    assert mind.bank.learned_memories() == []


# ---------------------------------------------------------------------------
# Integration: the loop runs inside real matches
# ---------------------------------------------------------------------------

def test_players_accumulate_memories_over_a_match():
    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=21))
    home, away = create_tactical_matchup("balanced", "balanced")
    state = MatchState(home_team=home, away_team=away, ball=Ball())

    engine.simulate_match(state, minutes=90)

    learned = sum(len(mind.bank.learned_memories())
                  for mind in engine.minds._minds.values())
    assert learned > 0, "a full match should form some memories"


def test_learning_persists_across_matches_with_the_same_players():
    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=22))
    home, away = create_tactical_matchup("balanced", "balanced")

    for _ in range(2):  # same Player objects, two matches
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        engine.simulate_match(state, minutes=45)

    experienced = [mind for mind in engine.minds._minds.values()
                   if mind.bank.learned_memories()]
    assert experienced, "memories should survive into the next match"


if __name__ == "__main__":
    run_tests(globals())
