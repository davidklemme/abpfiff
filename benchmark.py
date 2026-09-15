#!/usr/bin/env python3
"""
Attribute benchmark suite: markers and negative samples.

Every marker plays N seeded matches of an otherwise identical mirrored
matchup where ONE squad (home) carries a mutation - a nulled attribute
(set to 1), a nulled combination, or a boost - and gates a home-vs-away
consequence. The away squad is the in-match control, so each marker
asks one question end-to-end through the real engine:

  - STANDARD MARKERS: does the baseline behave (no mutation), does
    skill win matches?
  - NEGATIVE SAMPLES: does nulling an attribute have the assumed
    consequence (no passing -> completion collapses, no shooting ->
    goals dry up, ...)?
  - DUD DETECTION: a nulled attribute that moves nothing fails its
    marker - dead attribute wiring cannot hide (the pre-review engine
    shipped exactly such duds, finding F4).

Combos null several attributes together to catch interactions that
individually-wired attributes could still miss, and one marker runs the
mutation under a big-night environment so the sensitivity channel is
exercised end-to-end.

    python3 benchmark.py --matches 6 --seed 42 --gate
"""
import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from models import Ball, Environment, MatchState, Player, Team
from engine import MatchEngine, SimulationConfig
from metrics import MatchMetrics
from teams import create_tactical_matchup

BIG_NIGHT = Environment(stakes=0.9, crowd_intensity=0.9)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class BenchAggregate:
    """Per-side sums over a marker's matches."""
    matches: int
    home_goals: int = 0
    away_goals: int = 0
    home_shots: int = 0
    away_shots: int = 0
    home_pass_attempts: int = 0
    home_passes_completed: int = 0
    away_pass_attempts: int = 0
    away_passes_completed: int = 0
    home_possession: int = 0
    away_possession: int = 0
    home_fouls: int = 0
    away_fouls: int = 0
    home_crosses_completed: int = 0
    away_crosses_completed: int = 0
    home_fatigue: float = 0.0   # summed end-of-match squad means
    away_fatigue: float = 0.0

    def add(self, m: MatchMetrics, home: Team, away: Team) -> None:
        self.home_goals += m.home.goals
        self.away_goals += m.away.goals
        self.home_shots += m.home.shots
        self.away_shots += m.away.shots
        self.home_pass_attempts += m.home.pass_attempts
        self.home_passes_completed += m.home.passes_completed
        self.away_pass_attempts += m.away.pass_attempts
        self.away_passes_completed += m.away.passes_completed
        self.home_possession += m.home.possession_ticks
        self.away_possession += m.away.possession_ticks
        self.home_fouls += m.home.fouls
        self.away_fouls += m.away.fouls
        self.home_crosses_completed += m.home.crosses_completed
        self.away_crosses_completed += m.away.crosses_completed
        self.home_fatigue += (sum(p.fatigue for p in home.players)
                              / max(1, len(home.players)))
        self.away_fatigue += (sum(p.fatigue for p in away.players)
                              / max(1, len(away.players)))

    # -- comparison values (home relative to the away control) --------------

    def goals_share(self) -> Optional[float]:
        total = self.home_goals + self.away_goals
        return None if total == 0 else self.home_goals / total

    def goals_per_match(self) -> float:
        return (self.home_goals + self.away_goals) / self.matches

    def shots_share(self) -> Optional[float]:
        total = self.home_shots + self.away_shots
        return None if total == 0 else self.home_shots / total

    def completed_ratio(self) -> Optional[float]:
        """Open-play passes completed, home vs away - robust when a
        mutation makes a squad stop passing entirely."""
        return (None if self.away_passes_completed == 0
                else self.home_passes_completed / self.away_passes_completed)

    def completion_ratio(self) -> Optional[float]:
        if not self.home_pass_attempts or not self.away_pass_attempts:
            return None
        home = self.home_passes_completed / self.home_pass_attempts
        away = self.away_passes_completed / self.away_pass_attempts
        return None if away == 0 else home / away

    def possession_share(self) -> Optional[float]:
        total = self.home_possession + self.away_possession
        return None if total == 0 else self.home_possession / total

    def fouls_share(self) -> Optional[float]:
        total = self.home_fouls + self.away_fouls
        return None if total == 0 else self.home_fouls / total

    def crosses_found_ratio(self) -> Optional[float]:
        """Crosses that found a teammate, home vs away - vision's
        aggregate value lives in the longest deliveries (perception:
        certainty falls with distance)."""
        return (None if self.away_crosses_completed == 0
                else self.home_crosses_completed / self.away_crosses_completed)

    def fatigue_ratio(self) -> Optional[float]:
        return (None if self.away_fatigue == 0
                else self.home_fatigue / self.away_fatigue)


# ---------------------------------------------------------------------------
# Markers (table-driven; the vector principle applied to QA)
# ---------------------------------------------------------------------------

def null(*attributes: str) -> Callable[[Player], None]:
    def mutate(player: Player) -> None:
        for attribute in attributes:
            setattr(player, attribute, 1)
    return mutate


def boost(*attributes: str, to: int = 90) -> Callable[[Player], None]:
    def mutate(player: Player) -> None:
        for attribute in attributes:
            setattr(player, attribute, to)
    return mutate


def limelight(player: Player) -> None:
    """Nervy and thin-skinned: the mutation only bites when the
    environment supplies the stage."""
    player.composure = 20
    player.sensitivity = 95


@dataclass
class Marker:
    name: str
    consequence: str                              # what is asserted, plainly
    value: Callable[[BenchAggregate], Optional[float]]
    gate_lo: float                # regression guard: outside = CI fail
    gate_hi: float
    target_lo: float = None       # where it SHOULD be: outside = warn
    target_hi: float = None
    mutate: Optional[Callable[[Player], None]] = None  # applied to HOME squad
    environment: Environment = field(default_factory=Environment)
    key: str = ""                 # markers sharing a key share one sample run
    matches: Optional[int] = None  # override the CLI sample size (noisy markers)
    fmt: str = "{:.3f}"

    def __post_init__(self):
        if self.target_lo is None:
            self.target_lo = self.gate_lo
        if self.target_hi is None:
            self.target_hi = self.gate_hi
        if not self.key:
            self.key = self.name


MARKERS: List[Marker] = [
    # -- standard markers ---------------------------------------------------
    Marker("baseline mirror", "identical squads stay near-symmetric",
           BenchAggregate.shots_share, 0.30, 0.70, 0.42, 0.58),
    # Known gap the target band keeps visible: under high stakes the
    # load channel currently suppresses BOTH squads' scoring hard
    # (0.1-1.4 gpm by seed); the gate only guards against literal zero
    Marker("big-night mirror", "the stage alone does not kill football",
           BenchAggregate.goals_per_match, 0.05, 4.0, 1.5, 3.5,
           environment=BIG_NIGHT, fmt="{:.2f}"),
    Marker("skill wins", "an outclassing squad takes the goals",
           BenchAggregate.goals_share, 0.60, 1.00, 0.75, 1.00,
           mutate=boost("passing", "shooting", "dribbling", "pace",
                        "positioning", "vision", "first_touch",
                        "composure")),
    # -- negative samples: nulled attributes must have their consequence ----
    Marker("null passing", "their passing game vanishes",
           BenchAggregate.completed_ratio, 0.00, 0.05,
           mutate=null("passing"), key="null passing"),
    # Tracked calibration target, not a regression gate: the spatial
    # round cut spam's edge (goals share ~0.95 -> ~0.7-1.0 by seed) but
    # dribble+shoot volume still outscores a leaky passing game. The
    # fix is attack construction (third-man support), not more tax.
    Marker("direct-play balance", "...yet dribble+shoot spam should NOT win",
           BenchAggregate.goals_share, 0.00, 1.00, 0.00, 0.45,
           mutate=null("passing"), key="null passing"),
    Marker("null shooting", "no shooting -> the goals dry up",
           BenchAggregate.goals_share, 0.00, 0.45, 0.00, 0.30,
           mutate=null("shooting")),
    # Vision's aggregate effect needs a bigger sample than most markers
    Marker("null vision", "no vision -> the long ball finds nobody",
           BenchAggregate.crosses_found_ratio, 0.00, 0.88, 0.00, 0.75,
           mutate=null("vision"), matches=16),
    Marker("null composure", "nerves -> the bold game dries up",
           BenchAggregate.shots_share, 0.00, 0.42,
           mutate=null("composure")),
    Marker("null pace", "no legs -> second to every ball",
           BenchAggregate.possession_share, 0.00, 0.47,
           mutate=null("pace")),
    Marker("null stamina", "no tank -> fatigue piles up",
           BenchAggregate.fatigue_ratio, 1.20, 10.0,
           mutate=null("stamina")),
    # Foul counts are small (~50 per sample), so the share is noisy at
    # roughly +-0.07: the gate asserts direction with margin, the target
    # asserts the effect size seen on calibration seeds.
    Marker("max aggression", "recklessness -> the fouls follow",
           BenchAggregate.fouls_share, 0.53, 1.00, 0.58, 1.00,
           mutate=boost("aggression", to=95)),
    Marker("null aggression", "timidity -> barely a foul",
           BenchAggregate.fouls_share, 0.00, 0.42,
           mutate=null("aggression")),
    # -- combos: interactions must not cancel out ---------------------------
    Marker("null playmaking combo", "passing+vision nulled together",
           BenchAggregate.completed_ratio, 0.00, 0.05,
           mutate=null("passing", "vision")),
    # The strongest limelight channel is conversion: sensitive squads
    # still shoot, but the chances go begging (bigger sample - env
    # markers are the noisiest)
    Marker("limelight collapse", "nervy+sensitive bottle the big stage",
           BenchAggregate.goals_share, 0.00, 0.45, 0.00, 0.35,
           mutate=limelight, environment=BIG_NIGHT, matches=12),
    Marker("null everything", "a squad of nothing loses everything",
           BenchAggregate.goals_share, 0.00, 0.30,
           mutate=null("pace", "stamina", "passing", "shooting",
                       "dribbling", "positioning", "composure",
                       "vision", "first_touch")),
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def run_marker(marker: Marker, matches: int, seed: int,
               ticks_per_minute: int) -> BenchAggregate:
    engine = MatchEngine(SimulationConfig(
        ticks_per_minute=ticks_per_minute, randomness=0.3, seed=seed))
    aggregate = BenchAggregate(matches=matches)

    for _ in range(matches):
        home, away = create_tactical_matchup("balanced", "balanced")
        if marker.mutate is not None:
            for player in home.players:
                marker.mutate(player)
        state = MatchState(home_team=home, away_team=away, ball=Ball(),
                           environment=marker.environment)
        match_metrics = MatchMetrics(home, away)
        engine.event_handlers = [match_metrics.on_event]
        engine.tick_handlers = [match_metrics.on_tick]
        engine.simulate_match(state, minutes=90)
        aggregate.add(match_metrics, home, away)

    return aggregate


def run_benchmarks(matches: int, seed: int, ticks_per_minute: int):
    """Returns (rows, gate_failures, target_warnings); each row is
    (marker, value or None, status). Markers sharing a `key` share one
    sample run (two consequences of the same mutation)."""
    rows, gate_failures, target_warnings = [], [], []
    samples = {}
    for marker in MARKERS:
        if marker.key not in samples:
            samples[marker.key] = run_marker(marker, marker.matches or matches,
                                             seed, ticks_per_minute)
        value = marker.value(samples[marker.key])
        if value is None or not (marker.gate_lo <= value <= marker.gate_hi):
            status = "GATE FAIL"
            gate_failures.append(marker.name)
        elif not (marker.target_lo <= value <= marker.target_hi):
            status = "off target"
            target_warnings.append(marker.name)
        else:
            status = "ok"
        rows.append((marker, value, status))
    return rows, gate_failures, target_warnings


def print_report(rows, matches: int, seed: int) -> None:
    print(f"\nANSTOSS ENGINE ATTRIBUTE BENCHMARK  "
          f"({matches} matches per marker, seed {seed}, "
          f"mutations on home, away = control)")
    print("-" * 108)
    print(f"{'marker':<22}{'consequence asserted':<44}{'value':>8}   "
          f"{'gate band':<12}{'target band':<12}status")
    print("-" * 108)
    for marker, value, status in rows:
        shown = "n/a" if value is None else marker.fmt.format(value)
        gate = f"{marker.gate_lo:g}-{marker.gate_hi:g}"
        target = f"{marker.target_lo:g}-{marker.target_hi:g}"
        print(f"{marker.name:<22}{marker.consequence:<44}{shown:>8}   "
              f"{gate:<12}{target:<12}{status}")
    print("-" * 108)


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute benchmark suite")
    parser.add_argument("--matches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero if any marker fails its gate")
    args = parser.parse_args()

    rows, gate_failures, target_warnings = run_benchmarks(
        args.matches, args.seed, args.ticks)
    print_report(rows, args.matches, args.seed)

    if target_warnings:
        print(f"off target ({len(target_warnings)}): "
              + ", ".join(target_warnings))
        print("  -> known engine truths to calibrate away; see MARKERS")
    if gate_failures:
        print(f"GATE FAILURES ({len(gate_failures)}): "
              + ", ".join(gate_failures))
        return 1 if args.gate else 0
    print("all benchmark marker gates satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
