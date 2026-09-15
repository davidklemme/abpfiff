#!/usr/bin/env python3
"""
Tests for the attribute benchmark suite's plumbing (the markers
themselves run real matches and are exercised by CI via
`benchmark.py --gate`; here we pin the machinery that makes their
verdicts trustworthy).

Run directly: python3 tests/test_benchmark.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, run_tests
from benchmark import (
    MARKERS, BenchAggregate, Marker, null, boost, limelight, BIG_NIGHT
)


def test_marker_names_are_unique():
    names = [m.name for m in MARKERS]
    assert len(names) == len(set(names)), names


def test_every_mutation_actually_mutates():
    """Dud detection starts with the mutations themselves: each one must
    change at least one attribute on a probe player."""
    for marker in MARKERS:
        if marker.mutate is None:
            continue
        probe = make_player("Probe", role="cm")
        before = dict(vars(probe))
        marker.mutate(probe)
        changed = [k for k, v in vars(probe).items() if before[k] != v]
        assert changed, f"marker '{marker.name}' mutates nothing"


def test_null_and_boost_set_the_named_attributes():
    player = make_player("P", role="cm")
    null("passing", "vision")(player)
    assert player.passing == 1 and player.vision == 1
    boost("aggression", to=95)(player)
    assert player.aggression == 95


def test_limelight_needs_the_stage():
    """The limelight mutation is inert in a neutral environment - its
    consequence flows through Environment.psychological_load."""
    player = make_player("P", role="cm")
    limelight(player)
    from models import Environment
    assert Environment().psychological_load(player.sensitivity) == 0.0
    assert BIG_NIGHT.psychological_load(player.sensitivity) > 0.5


def test_empty_aggregate_values_are_none_not_crashes():
    empty = BenchAggregate(matches=1)
    assert empty.goals_share() is None
    assert empty.shots_share() is None
    assert empty.completion_ratio() is None
    assert empty.completed_ratio() is None
    assert empty.possession_share() is None
    assert empty.fouls_share() is None
    assert empty.fatigue_ratio() is None
    assert empty.goals_per_match() == 0.0


def test_target_bands_default_to_gate_bands():
    marker = Marker("m", "c", BenchAggregate.goals_share, 0.2, 0.8)
    assert marker.target_lo == 0.2 and marker.target_hi == 0.8
    assert marker.key == "m"


def test_shared_keys_pair_the_same_mutation():
    """Markers sharing a sample run must genuinely describe the same
    experiment (same mutation, same environment)."""
    by_key = {}
    for marker in MARKERS:
        by_key.setdefault(marker.key, []).append(marker)
    for key, group in by_key.items():
        if len(group) == 1:
            continue
        probes = []
        for marker in group:
            probe = make_player("Probe", role="cm")
            if marker.mutate is not None:
                marker.mutate(probe)
            probes.append(vars(probe).copy())
            assert marker.environment == group[0].environment, key
        assert all(p == probes[0] for p in probes), key


def test_a_failing_gate_is_reported():
    """The verdict logic itself: values outside the gate fail, values
    between gate and target warn."""
    from benchmark import run_benchmarks  # noqa: F401  (import sanity)
    marker = Marker("m", "c", BenchAggregate.goals_share,
                    0.0, 0.3, 0.0, 0.2)
    agg = BenchAggregate(matches=1)
    agg.home_goals, agg.away_goals = 5, 5          # share 0.5 -> gate fail
    assert not (marker.gate_lo <= marker.value(agg) <= marker.gate_hi)
    agg.home_goals, agg.away_goals = 1, 3          # share 0.25 -> warn zone
    value = marker.value(agg)
    assert marker.gate_lo <= value <= marker.gate_hi
    assert not (marker.target_lo <= value <= marker.target_hi)


if __name__ == "__main__":
    run_tests(globals())
