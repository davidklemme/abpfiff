#!/usr/bin/env python3
"""
Tests for the parameter sweep harness and the knobs it sweeps.

The load-bearing claim is that every knob DEFAULTS to the behavior the
engine shipped with - otherwise `baseline` in a sweep report is not the
engine anyone has validated, and every comparison drawn against it is
against a phantom. Most of this file pins that down.

Run directly: python3 tests/test_sweeps.py
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, run_tests
import instincts
import learning
from instincts import InstinctBank, default_bank_for
from situation import SituationEmbedding
from sweeps import (
    KNOBS, MEASUREMENTS, BASELINE, Variant, overrides, run_variants,
    approach_variants, side_effect_variants,
)

QUIET = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.0)
BIG = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.9)


def trained_bank():
    """A bank carrying one quiet-night and one big-night memory."""
    bank = default_bank_for(make_player("Vet", role="cm"))
    bank.learn(QUIET, "pass_forward", valence=1.0, significance=0.8)
    bank.learn(BIG, "shoot", valence=1.0, significance=0.9)
    return bank


# ---------------------------------------------------------------------------
# Defaults are the shipped behavior
# ---------------------------------------------------------------------------

def test_knob_defaults_preserve_shipped_behavior():
    """Each new knob's default is the constant the engine shipped with:
    no-op top-k fold, no decay shelter, the original 0.6 ceilings."""
    assert instincts.FAMILIARITY_TOP_K == 1
    assert instincts.FAMILIARITY_BLEND_DECAY == 0.0
    assert instincts.HIGH_LOAD_DECAY_SHELTER == 0.0
    assert instincts.MAX_GAIN_PER_OUTCOME == 0.6
    assert instincts.MAX_NEW_MEMORY_STRENGTH == 0.6


def test_top_k_with_zero_blend_decay_is_exactly_the_max():
    """Folding over more memories with zero blend weight must reproduce
    the bare max, or the K=1 fast path and the K>1 path disagree about
    what the same configuration means."""
    bank = trained_bank()
    baseline = bank.familiarity(BIG)
    with overrides({"top_k": 8, "blend_decay": 0.0}):
        assert abs(bank.familiarity(BIG) - baseline) < 1e-12


def test_top_k_blend_adds_supporting_memories():
    bank = trained_bank()
    baseline = bank.familiarity(BIG)
    with overrides({"top_k": 8, "blend_decay": 0.6}):
        assert bank.familiarity(BIG) > baseline


def test_zero_shelter_decays_every_memory_alike():
    bank = trained_bank()
    before = [i.strength for i in bank.learned_memories()]
    bank.decay(0.5)
    after = [i.strength for i in bank.learned_memories()]
    assert all(abs(a - b * 0.5) < 1e-12 for a, b in zip(after, before))


def test_shelter_protects_high_load_memories_only():
    """The mechanism's whole point: a big night should outlast a quiet
    Tuesday, not everything decay together."""
    bank = trained_bank()
    quiet = next(i for i in bank.learned_memories() if i.prototype.load == 0.0)
    big = next(i for i in bank.learned_memories() if i.prototype.load == 0.9)
    quiet_before, big_before = quiet.strength, big.strength

    with overrides({"load_shelter": 1.0}):
        bank.decay(0.5)

    assert abs(quiet.strength - quiet_before * 0.5) < 1e-12  # unsheltered
    assert big.strength > big_before * 0.5                   # sheltered
    assert big.strength <= big_before


def test_strength_ceiling_knob_raises_new_memory_strength():
    bank = InstinctBank([])
    bank.learn(BIG, "shoot", valence=1.0, significance=1.0)
    assert bank.learned_memories()[0].strength == 0.6  # shipped ceiling

    stronger = InstinctBank([])
    with overrides({"gain_cap": 1.0, "new_strength_cap": 1.0}):
        stronger.learn(BIG, "shoot", valence=1.0, significance=1.0)
    assert stronger.learned_memories()[0].strength > 0.6


# ---------------------------------------------------------------------------
# Override mechanics
# ---------------------------------------------------------------------------

def test_overrides_restore_previous_values():
    before = instincts.ROLE_SCHOOLING_FAMILIARITY
    with overrides({"role_schooling": 0.1}):
        assert instincts.ROLE_SCHOOLING_FAMILIARITY == 0.1
    assert instincts.ROLE_SCHOOLING_FAMILIARITY == before


def test_overrides_restore_even_when_the_body_raises():
    """A leaked knob would silently contaminate every later variant."""
    before = learning.DECAY_PER_MINUTE
    try:
        with overrides({"in_match_decay": 0.5}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert learning.DECAY_PER_MINUTE == before


def test_unknown_knob_is_rejected():
    try:
        with overrides({"no_such_knob": 1.0}):
            pass
    except KeyError as error:
        assert "no_such_knob" in str(error)
    else:
        raise AssertionError("unknown knob silently accepted")


def test_every_declared_knob_exists_on_its_module():
    """Guards against a constant being renamed out from under a knob."""
    for knob, (module, attr) in KNOBS.items():
        assert hasattr(module, attr), f"{knob} -> {module.__name__}.{attr}"


def test_no_knob_is_captured_as_a_default_argument():
    """A constant used as a default argument value binds at IMPORT, so
    overriding the module attribute afterwards changes nothing and the
    knob silently sweeps nothing at all.

    This is not hypothetical: `_proto(load=SCHOOLING_LOAD)` made the
    schooling-load knob inert, and a whole sweep reported "this
    parameter has no effect" when what it had measured was the override
    failing to bind."""
    offenders = []
    for knob, (module, attr) in KNOBS.items():
        tree = ast.parse(open(module.__file__).read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None]
            for default in defaults:
                if isinstance(default, ast.Name) and default.id == attr:
                    offenders.append(
                        f"{knob}: {module.__name__}.py:{node.lineno} "
                        f"{node.name}(... ={attr})")
    assert not offenders, "knob(s) frozen at import: " + "; ".join(offenders)


def test_schooling_load_knob_reaches_role_seeds():
    """ROLE_SEEDS is built at import, so the seeds must be stamped with
    the current schooling load when a bank is built, not when the module
    loads."""
    player = make_player("Rookie", role="cm")
    assert default_bank_for(player).instincts[0].prototype.load == \
        instincts.SCHOOLING_LOAD
    with overrides({"schooling_load": 0.4}):
        seeded = default_bank_for(player)
        assert all(i.prototype.load == 0.4 for i in seeded.instincts)


def test_every_variant_only_uses_declared_knobs():
    for variant in approach_variants() + side_effect_variants():
        unknown = set(variant.params) - set(KNOBS)
        assert not unknown, f"{variant.name}: {unknown}"


# ---------------------------------------------------------------------------
# Harness plumbing
# ---------------------------------------------------------------------------

def test_run_variants_measures_every_metric():
    """One tiny series end to end: the harness runs, measures, and the
    baseline variant leaves the engine's constants where it found them."""
    before = instincts.ROLE_SCHOOLING_FAMILIARITY
    variants = [BASELINE, Variant("probe", "approach", {"role_schooling": 0.2})]
    results = run_variants(variants, matches=1, seeds=[3], minutes=10,
                           workers=1)

    assert len(results) == 2
    for result in results:
        assert set(result.per_seed[3]) == set(MEASUREMENTS)
        assert result.per_seed[3]["mem/plyr"] is not None
    assert instincts.ROLE_SCHOOLING_FAMILIARITY == before


def test_results_aggregate_across_seeds():
    results = run_variants([BASELINE], matches=1, seeds=[3, 4], minutes=10,
                           workers=1)
    result = results[0]
    low, high = result.spread("mem/plyr")
    assert low <= result.mean("mem/plyr") <= high


if __name__ == "__main__":
    run_tests(globals())
