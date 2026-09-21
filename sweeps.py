#!/usr/bin/env python3
"""
Parameter sweep harness: does a candidate change produce the effect it
promises, and what does it cost everywhere else?

`validate.py` asks "is the engine still sane" and `series.py` asks "does
a trajectory stay healthy". Both answer for ONE configuration. This asks
a different question: run the same seeded series under many parameter
settings and put the results side by side, so a mechanism claim can be
checked against a control instead of against intuition.

It exists because of a specific failure. The load dimension promises a
veteran effect - squads that have lived big nights should recognize the
next one better than fresh squads - and `series.py` measures it as
"Big-night familiarity gain (trained - fresh)". That number sits at
~0.001-0.012 against a >=0.01 target while every gate passes, because
the gate band (-0.02..1) cannot fail. A single number that small has
many possible causes, so this harness measures the CANDIDATE FIXES
against a baseline, and measures what each one does to match quality at
the same time: a fix that manufactures recognition by wrecking the
football is not a fix.

Three things make the comparison trustworthy:

  - Every knob defaults to current behavior, so `baseline` here is the
    shipped engine, not an approximation of it.
  - Every variant runs the same seeds over the same occasion schedule,
    so differences are the parameter, not the draw.
  - Seed spread is reported next to the mean. The metric under
    investigation swings 12x across seeds at 10 matches; a single-seed
    improvement means nothing.

Usage:
    python3 sweeps.py                        # approaches + side effects
    python3 sweeps.py --only approach        # just the candidate fixes
    python3 sweeps.py --matches 20 --seeds 42,7,99,3
    python3 sweeps.py --full                 # every metric column
"""
import argparse
import multiprocessing
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import instincts
import minds
import series
from series import SeriesRunner, SERIES_BANDS
from validate import evaluate

# ---------------------------------------------------------------------------
# Sweepable surface
# ---------------------------------------------------------------------------
# Every knob is a module-level constant read at call time, so overriding
# the attribute is enough - no plumbing through constructors. Listing
# them explicitly (rather than accepting "module.ATTR" strings) means a
# typo in a variant fails at startup instead of silently sweeping
# nothing, and documents in one place what this harness can actually
# move.
KNOBS: Dict[str, Tuple[object, str]] = {
    # -- trace geometry and retention ---------------------------------------
    "schooling_width": (instincts, "SCHOOLING_WIDTH"),
    "rigid_width": (instincts, "LIVED_WIDTH_RIGID"),
    "plastic_width": (instincts, "LIVED_WIDTH_PLASTIC"),
    "retention_exponent": (instincts, "RETENTION_EXPONENT"),
    "spacing_gain": (instincts, "SPACING_GAIN"),
    "merge_kernel": (instincts, "MERGE_KERNEL_THRESHOLD"),
    "recognition_scale": (instincts, "RECOGNITION_SCALE"),
    "max_traces": (instincts, "MAX_TRACES"),
    "schooling_load": (instincts, "SCHOOLING_LOAD"),
    "rest_days": (series, "DEFAULT_REST_DAYS"),
    # -- retrieval and psychology -------------------------------------------
    "confidence_tilt": (instincts, "CONFIDENCE_TILT"),
    "anchor_gain": (instincts, "ANCHOR_CONFIDENCE_GAIN"),
    "trauma_gain": (instincts, "TRAUMA_RATTLED_GAIN"),
    "confidence_retention": (minds, "CONFIDENCE_RETENTION_PER_REST_DAY"),
}


@contextmanager
def overrides(params: Dict[str, float]):
    """Apply knob values for the duration of the block, then restore.

    Restoring matters even though most runs are one-shot subprocesses:
    the harness runs variants in-process when parallelism is off, and a
    leaked knob would silently contaminate every later variant."""
    unknown = set(params) - set(KNOBS)
    if unknown:
        raise KeyError(f"unknown knob(s): {', '.join(sorted(unknown))}")

    previous = {}
    for name, value in params.items():
        module, attr = KNOBS[name]
        previous[name] = getattr(module, attr)
        setattr(module, attr, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            module, attr = KNOBS[name]
            setattr(module, attr, value)


# ---------------------------------------------------------------------------
# What gets measured
# ---------------------------------------------------------------------------
# Table-driven so a new question is one row. The first four are the
# mechanism under investigation; the rest are the side-effect watch -
# any of them moving is the cost of the fix.
MEASUREMENTS: Dict[str, Callable[[SeriesRunner], Optional[float]]] = {
    "fam gain": SeriesRunner.big_night_familiarity_gain,
    "lrn wins": SeriesRunner.learned_wins_share,
    "role rec": SeriesRunner.role_recognition,
    "lrn rec": SeriesRunner.learned_recognition,
    "mem/plyr": SeriesRunner.final_learned_mean,
    "anchor%": SeriesRunner.anchor_share,
    "surv": SeriesRunner.play_survival,
    "pinned": SeriesRunner.max_pinned_share,
    "goals": SeriesRunner.goals_per_match,
    "shots": SeriesRunner.shots_per_match,
    "pass%": SeriesRunner.pass_completion,
    "|conf|": SeriesRunner.mean_abs_confidence,
}

# Shown by default; the rest need --full. These are the mechanism plus
# the match-quality metrics a broken fix would show up in first.
PRIMARY = ("fam gain", "lrn wins", "role rec", "lrn rec",
           "mem/plyr", "goals", "shots", "pass%", "surv")

# The target the whole exercise is aimed at (series.SERIES_BANDS).
TARGET_GAIN = 0.01


@dataclass
class Variant:
    """One parameter configuration to run."""
    name: str
    category: str              # "baseline" | "approach" | "side-effect"
    params: Dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass
class Result:
    """One variant's outcome, aggregated over seeds."""
    variant: Variant
    per_seed: Dict[int, Dict[str, Optional[float]]]
    gate_failures: Dict[int, List[str]]

    def mean(self, metric: str) -> Optional[float]:
        values = [m[metric] for m in self.per_seed.values()
                  if m.get(metric) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def spread(self, metric: str) -> Tuple[Optional[float], Optional[float]]:
        values = [m[metric] for m in self.per_seed.values()
                  if m.get(metric) is not None]
        if not values:
            return None, None
        return min(values), max(values)

    @property
    def broken_gates(self) -> List[str]:
        """Gate bands this variant violated on any seed (deduplicated)."""
        seen: List[str] = []
        for failures in self.gate_failures.values():
            for failure in failures:
                if failure not in seen:
                    seen.append(failure)
        return seen


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def measure(runner: SeriesRunner) -> Dict[str, Optional[float]]:
    return {name: fn(runner) for name, fn in MEASUREMENTS.items()}


def _run_job(job: Tuple[str, Dict[str, float], int, int, int]):
    """One (variant, seed) series. Module level so it survives pickling
    into a worker process."""
    name, params, matches, seed, minutes = job
    with overrides(params):
        runner = SeriesRunner(matches=matches, seed=seed,
                              minutes=minutes).play()
        gate_failures, _, _ = evaluate(runner, SERIES_BANDS)
        return name, seed, measure(runner), gate_failures


def run_variants(variants: Sequence[Variant], matches: int,
                 seeds: Sequence[int], minutes: int = 90,
                 workers: Optional[int] = None) -> List[Result]:
    """Run every (variant, seed) pair, in parallel where possible.

    Uses spawn rather than fork: the workers re-import the engine and
    apply their own knobs, so there is no shared mutated state to
    inherit, and fork-after-import is not safe on macOS."""
    jobs = [(v.name, v.params, matches, seed, minutes)
            for v in variants for seed in seeds]
    workers = workers if workers is not None else min(len(jobs),
                                                      multiprocessing.cpu_count())

    collected: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {
        v.name: {} for v in variants}
    gates: Dict[str, Dict[int, List[str]]] = {v.name: {} for v in variants}

    if workers <= 1:
        finished = (_run_job(job) for job in jobs)
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=workers)
        finished = pool.imap_unordered(_run_job, jobs)

    done = 0
    results_iter = finished
    for name, seed, metrics, gate_failures in results_iter:
        collected[name][seed] = metrics
        gates[name][seed] = gate_failures
        done += 1
        print(f"\r  {done}/{len(jobs)} runs", end="", file=sys.stderr,
              flush=True)
    print("", file=sys.stderr)

    if workers > 1:
        pool.close()
        pool.join()

    return [Result(variant=v, per_seed=collected[v.name],
                   gate_failures=gates[v.name]) for v in variants]


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

BASELINE = Variant("baseline", "baseline", {}, "shipped engine")


def approach_variants() -> List[Variant]:
    """Sensitivity analysis for the shipped trace-geometry mechanism."""
    return [
        Variant("narrow load fields", "approach", {"plastic_width": 0.65}),
        Variant("wide load fields", "approach", {"plastic_width": 1.10}),
        Variant("faster forgetting", "approach", {"retention_exponent": 0.50}),
        Variant("slower forgetting", "approach", {"retention_exponent": 0.22}),
        Variant("weak spacing", "approach", {"spacing_gain": 0.25}),
        Variant("strong spacing", "approach", {"spacing_gain": 0.85}),
    ]


def side_effect_variants() -> List[Variant]:
    """Knobs NOT aimed at the familiarity problem, swept to see what they
    move.

    Two purposes. First, a control: if half the engine's constants
    happen to move the familiarity gain as much as the targeted fixes
    do, then the gain is just noise and no fix is credible. Second,
    mapping - these are the parameters a future tuning round will reach
    for, and nobody has yet measured what they cost."""
    sweeps: List[Tuple[str, str, List[float]]] = [
        ("merge_kernel", "merge kernel", [0.2, 0.5]),
        ("recognition_scale", "recognition", [0.08, 0.18]),
        ("max_traces", "max traces", [80, 240]),
        ("schooling_load", "schooling load", [0.0, 0.3]),
        ("rest_days", "rest days", [2.0, 14.0]),
        ("confidence_tilt", "conf tilt", [0.1, 0.6]),
        ("anchor_gain", "anchor gain", [0.0, 1.0]),
        ("trauma_gain", "trauma gain", [0.0, 1.0]),
        ("confidence_retention", "conf retention", [0.5, 0.99]),
    ]
    variants = []
    for knob, label, values in sweeps:
        for value in values:
            variants.append(Variant(f"{label} {value:g}", "side-effect",
                                    {knob: value}))
    return variants


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _cell(value: Optional[float], width: int = 8) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.3f}"


def _delta_cell(value: Optional[float], base: Optional[float],
                width: int = 8) -> str:
    if value is None or base is None:
        return f"{'-':>{width}}"
    delta = value - base
    if abs(delta) < 5e-4:
        return f"{'.':>{width}}"
    return f"{delta:>+{width}.3f}"


def report(results: List[Result], baseline: Result, columns: Sequence[str],
           title: str, as_delta: bool) -> None:
    name_width = max(len(r.variant.name) for r in results) + 2
    header = f"{'variant':<{name_width}}" + "".join(
        f"{c:>9}" for c in columns) + "   verdict"
    print(f"\n{title}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for result in results:
        cells = ""
        for column in columns:
            value = result.mean(column)
            if as_delta and result is not baseline:
                cells += _delta_cell(value, baseline.mean(column), 9)
            else:
                cells += _cell(value, 9)
        print(f"{result.variant.name:<{name_width}}{cells}   "
              f"{verdict(result)}")
    print("-" * len(header))


def verdict(result: Result) -> str:
    """Did it hit the target, and did anything break on the way?"""
    gain = result.mean("fam gain")
    low, _ = result.spread("fam gain")
    broken = result.broken_gates

    if gain is None:
        return "no data"
    if broken:
        return f"GATE BREAK: {broken[0]}"
    if gain >= TARGET_GAIN and low is not None and low >= TARGET_GAIN:
        return "on target, every seed"
    if gain >= TARGET_GAIN:
        return "on target on average only"
    return ""


def print_spread(results: List[Result], metric: str = "fam gain") -> None:
    """Per-seed values for the headline metric.

    Printed in full because the mean hides the thing that made this
    investigation necessary: at 10 matches the shipped engine scores
    0.012 on one seed and 0.001 on another, and a fix that only works on
    one seed is not a fix."""
    print(f"\nper-seed {metric} (the mean above hides this)")
    seeds = sorted({seed for r in results for seed in r.per_seed})
    name_width = max(len(r.variant.name) for r in results) + 2
    print("-" * (name_width + 9 * len(seeds)))
    print(f"{'variant':<{name_width}}" + "".join(f"{s:>9}" for s in seeds))
    print("-" * (name_width + 9 * len(seeds)))
    for result in results:
        cells = "".join(_cell(result.per_seed.get(seed, {}).get(metric), 9)
                        for seed in seeds)
        print(f"{result.variant.name:<{name_width}}{cells}")
    print("-" * (name_width + 9 * len(seeds)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parameter sweeps for the familiarity mechanism")
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="42,7,99",
                        help="comma-separated series seeds")
    parser.add_argument("--minutes", type=int, default=90)
    parser.add_argument("--only", choices=["approach", "side-effect"],
                        help="run only one family of variants")
    parser.add_argument("--full", action="store_true",
                        help="every metric column, not just the primary ones")
    parser.add_argument("--filter", type=str, default=None,
                        help="run only variants whose name contains this "
                             "(the baseline is always kept, as the control)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [BASELINE]
    if args.only != "side-effect":
        variants += approach_variants()
    if args.only != "approach":
        variants += side_effect_variants()

    if args.filter:
        variants = [v for v in variants
                    if v is BASELINE or args.filter.lower() in v.name.lower()]
        if len(variants) == 1:
            print(f"no variant matches {args.filter!r}")
            return 1

    columns = list(MEASUREMENTS) if args.full else list(PRIMARY)

    print(f"ANSTOSS ENGINE PARAMETER SWEEPS  ({len(variants)} variants x "
          f"{len(seeds)} seeds x {args.matches} matches)")
    print(f"seeds: {seeds}   target: big-night familiarity gain >= "
          f"{TARGET_GAIN}")

    results = run_variants(variants, args.matches, seeds, args.minutes,
                           args.workers)
    baseline = results[0]

    approaches = [r for r in results if r.variant.category != "side-effect"]
    side_effects = [r for r in results if r.variant.category == "side-effect"]

    if len(approaches) > 1:
        report(approaches, baseline, columns,
               "CANDIDATE FIXES (absolute values; baseline first)", False)
        print_spread(approaches)
    if side_effects:
        report([baseline] + side_effects, baseline, columns,
               "SIDE-EFFECT SWEEPS (delta vs baseline; '.' = unmoved)", True)

    notes = [r.variant for r in results if r.variant.note]
    if notes:
        print("\nnotes")
        for variant in notes:
            print(f"  {variant.name}: {variant.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
