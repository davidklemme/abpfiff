#!/usr/bin/env python3
"""
Tests for temporal continuity, first slice
(docs/specs/temporal-continuity.md): the match boundary protocol,
mind serialization, and the series runner.

Run directly: python3 tests/test_continuity.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, run_tests
from minds import (MindRegistry, CONFIDENCE_RETENTION_PER_REST_DAY,
                   FATIGUE_REMAINING_PER_REST_DAY)
from situation import SituationEmbedding
from series import SeriesRunner, occasion_schedule

S = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.0)
BIG = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.9)


def registry_with_memories():
    """A registry with one experienced player: an anchor and a trauma."""
    player = make_player("Vet", role="cm")
    registry = MindRegistry()
    mind = registry.mind_for(player)
    mind.bank.learn(S, "pass_forward", valence=1.0, significance=0.8)
    mind.bank.learn(BIG, "shoot", valence=-1.0, significance=0.9)
    return registry, player, mind


# ---------------------------------------------------------------------------
# Match boundary protocol
# ---------------------------------------------------------------------------

def test_close_match_flushes_pending_decisions():
    registry, player, mind = registry_with_memories()
    mind.remember_decision(S, "dribble")

    registry.close_match([player])

    assert mind.pending_decision is None
    # The flushed decision was never learned from
    assert all("dribble" not in i.action_weights
               for i in mind.bank.learned_memories())


def test_close_match_advances_one_power_law_clock_for_every_trace():
    registry, player, mind = registry_with_memories()
    before = [(i.age, i.retained_mass()) for i in mind.bank.instincts]

    registry.close_match([player], rest_days=7.0)

    after = [(i.age, i.retained_mass()) for i in mind.bank.instincts]
    assert all(new_age == old_age + 1.0
               for (old_age, _), (new_age, _) in zip(before, after))
    assert all(new_mass < old_mass
               for (_, old_mass), (_, new_mass) in zip(before, after))


def test_close_match_reverts_state_but_not_traumas():
    registry, player, mind = registry_with_memories()
    player.confidence = 0.8
    player.fatigue = 60.0
    bystander = make_player("Bench", role="cb")  # never decided anything
    bystander.confidence = -0.6
    bystander.fatigue = 40.0

    registry.close_match([player, bystander], rest_days=2.0)

    conf = CONFIDENCE_RETENTION_PER_REST_DAY ** 2.0
    fat = FATIGUE_REMAINING_PER_REST_DAY ** 2.0
    assert abs(player.confidence - 0.8 * conf) < 1e-12
    assert abs(bystander.confidence - (-0.6) * conf) < 1e-12
    assert abs(player.fatigue - 60.0 * fat) < 1e-12
    assert abs(bystander.fatigue - 40.0 * fat) < 1e-12
    # The trauma is still there, still suppressing shooting on big nights
    traumas = [i for i in mind.bank.learned_memories() if i.source == "trauma"]
    assert traumas and traumas[0].action_weights.get("shoot", 0) < 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_serialization_round_trips_byte_stable():
    registry, player, _ = registry_with_memories()
    player.confidence = 0.31

    first = json.dumps(registry.to_dict(), sort_keys=True)
    fresh_player = make_player("Vet", role="cm")  # same identity, new object
    restored = MindRegistry.from_dict(json.loads(first), [fresh_player])
    second = json.dumps(restored.to_dict(), sort_keys=True)

    assert first == second
    assert fresh_player.confidence == 0.31


def test_restored_minds_behave_like_the_originals():
    registry, player, mind = registry_with_memories()
    blob = registry.to_dict()

    fresh_player = make_player("Vet", role="cm")
    restored_mind = MindRegistry.from_dict(blob, [fresh_player]).mind_for(fresh_player)

    for probe in (S, BIG):
        assert restored_mind.bank.query(probe) == mind.bank.query(probe)
        assert restored_mind.bank.familiarity(probe) == mind.bank.familiarity(probe)


def test_serialization_is_dimension_forward():
    """A mind stored before the embedding grew restores with neutral
    defaults for the missing trailing dimensions."""
    old_record = {"schema": 1, "minds": {"Vet#1": {
        "confidence": 0.0,
        "instincts": [{"name": "experience_shoot",
                       "prototype": [0.5, 0.8, 0.5, 0.5, 0.5],  # 5 dims
                       "weights": {"shoot": 1.0},
                       "strength": 0.4, "source": "experience"}],
    }}}
    player = make_player("Vet", role="st")
    mind = MindRegistry.from_dict(old_record, [player]).mind_for(player)

    memory = mind.bank.learned_memories()[0]
    assert memory.prototype.width == 0.5   # dataclass neutral default
    assert memory.prototype.load == 0.0
    assert len(memory.prototype.as_tuple()) == len(S.as_tuple())


def test_unknown_schema_is_rejected():
    try:
        MindRegistry.from_dict({"schema": 999, "minds": {}})
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unknown schema silently accepted")


def test_unknown_identities_stay_dormant_and_survive_reserialization():
    registry, player, _ = registry_with_memories()
    blob = registry.to_dict()

    stranger = make_player("Other", role="st")
    restored = MindRegistry.from_dict(blob, [stranger])

    assert restored.get(player.player_id) is None      # not revived
    assert player.player_id in restored.to_dict()["minds"]  # not lost

    # First sighting revives the record
    same_identity = make_player("Vet", role="cm")
    mind = restored.mind_for(same_identity)
    assert mind.bank.learned_memories()


# ---------------------------------------------------------------------------
# Series runner
# ---------------------------------------------------------------------------

def test_occasion_schedule_is_seeded_and_mixed():
    a = occasion_schedule(40, random.Random(3))
    b = occasion_schedule(40, random.Random(3))
    assert [(e.stakes, e.crowd_intensity) for e in a] == \
           [(e.stakes, e.crowd_intensity) for e in b]
    stakes = [e.stakes for e in a]
    assert any(s >= 0.6 for s in stakes)   # some big nights
    assert any(s < 0.3 for s in stakes)    # mostly routine


def test_series_is_deterministic_under_a_seed():
    first = SeriesRunner(matches=2, seed=11, minutes=30).play()
    second = SeriesRunner(matches=2, seed=11, minutes=30).play()
    assert first.snapshots == second.snapshots


def test_experience_accumulates_across_the_series():
    runner = SeriesRunner(matches=3, seed=5, minutes=30).play()
    assert runner.snapshots[-1].learned_mean > runner.snapshots[0].learned_mean
    assert runner.snapshots[-1].learned_mean > 0


def test_serialize_restore_mid_series_changes_nothing():
    """Acceptance gate: serialize -> restore between matches leaves the
    rest of the series identical (round-trip determinism)."""
    straight = SeriesRunner(matches=3, seed=13, minutes=30).play()

    staged = SeriesRunner(matches=3, seed=13, minutes=30)
    staged.play(upto=1)
    blob = json.dumps(staged.minds.to_dict(), sort_keys=True)
    staged.minds = MindRegistry.from_dict(
        json.loads(blob), staged.rosters[0] + staged.rosters[1])
    staged.play()

    assert staged.snapshots == straight.snapshots


def test_squad_reset_restores_match_scoped_state():
    runner = SeriesRunner(matches=1, seed=17, minutes=30)
    victim = runner.rosters[0][3]
    victim.yellow_cards = 2
    victim.sent_off = True
    runner.home.players = [p for p in runner.home.players if p is not victim]
    runner.home.attacks_up = False   # as a finished odd-half match leaves it
    runner.home.momentum = 80.0
    victim.position.x = 1.0

    runner._reset_squads()

    assert victim in runner.home.players
    assert victim.yellow_cards == 0 and not victim.sent_off
    assert runner.home.attacks_up is True
    assert runner.home.momentum == 50.0
    assert victim.position.x == victim.base_position.x


if __name__ == "__main__":
    run_tests(globals())
