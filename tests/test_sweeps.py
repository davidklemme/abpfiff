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
# Geometry knobs are live and meaningful
# ---------------------------------------------------------------------------

def test_width_knob_changes_new_trace_geometry():
    narrow = InstinctBank([])
    with overrides({"plastic_width": 0.5}):
        narrow.learn(BIG, "shoot", valence=1.0, significance=1.0)
    wide = InstinctBank([])
    with overrides({"plastic_width": 1.1}):
        wide.learn(BIG, "shoot", valence=1.0, significance=1.0)
    assert narrow.learned_memories()[0].widths[0] < wide.learned_memories()[0].widths[0]


def test_recognition_scale_changes_familiarity_not_trace_content():
    bank = trained_bank()
    traces = list(bank.instincts)
    with overrides({"recognition_scale": 0.05}):
        low = bank.familiarity(BIG)
    with overrides({"recognition_scale": 0.2}):
        high = bank.familiarity(BIG)
    assert high > low
    assert bank.instincts == traces


def test_retention_knob_changes_aged_evidence():
    bank = InstinctBank([])
    bank.learn(BIG, "shoot", valence=1.0, significance=1.0)
    trace = bank.learned_memories()[0]
    bank.decay(10)
    with overrides({"retention_exponent": 0.2}):
        slow = trace.retained_mass()
    with overrides({"retention_exponent": 0.6}):
        fast = trace.retained_mass()
    assert slow > fast


# ---------------------------------------------------------------------------
# Override mechanics
# ---------------------------------------------------------------------------

def test_overrides_restore_previous_values():
    before = instincts.RECOGNITION_SCALE
    with overrides({"recognition_scale": 0.1}):
        assert instincts.RECOGNITION_SCALE == 0.1
    assert instincts.RECOGNITION_SCALE == before


def test_overrides_restore_even_when_the_body_raises():
    """A leaked knob would silently contaminate every later variant."""
    before = instincts.RETENTION_EXPONENT
    try:
        with overrides({"retention_exponent": 0.5}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert instincts.RETENTION_EXPONENT == before


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
    before = instincts.RECOGNITION_SCALE
    variants = [BASELINE, Variant("probe", "approach", {"recognition_scale": 0.2})]
    results = run_variants(variants, matches=1, seeds=[3], minutes=10,
                           workers=1)

    assert len(results) == 2
    for result in results:
        assert set(result.per_seed[3]) == set(MEASUREMENTS)
        assert result.per_seed[3]["mem/plyr"] is not None
    assert instincts.RECOGNITION_SCALE == before


def test_results_aggregate_across_seeds():
    results = run_variants([BASELINE], matches=1, seeds=[3, 4], minutes=10,
                           workers=1)
    result = results[0]
    low, high = result.spread("mem/plyr")
    assert low <= result.mean("mem/plyr") <= high


if __name__ == "__main__":
    run_tests(globals())
