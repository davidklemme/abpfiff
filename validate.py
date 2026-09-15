#!/usr/bin/env python3
"""
Statistical validation harness (Phase 1.5 of the roadmap).

Runs N seeded matches between identical teams and checks headline match
statistics against two kinds of bands:

  - GATE bands: regression guards the current engine must satisfy; CI runs
    `validate.py --gate` and fails the build if any is violated.
  - TARGET bands: real-football realism goals the engine is being tuned
    toward; deviations are reported as warnings, not failures.

Usage:
    python3 validate.py                     # report only
    python3 validate.py --gate              # exit 1 on gate violations
    python3 validate.py --matches 50 --seed 7 --home gegenpressing
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

from models import Ball, MatchState
from engine import MatchEngine, SimulationConfig
from metrics import MatchMetrics
from teams import create_tactical_matchup


@dataclass
class Band:
    """An acceptance band for one aggregated metric."""
    name: str
    value_fn: Callable[["Aggregate"], Optional[float]]
    gate_lo: float
    gate_hi: float
    target_lo: float
    target_hi: float
    fmt: str = "{:.2f}"


@dataclass
class Aggregate:
    """Aggregated results over all validation matches."""
    matches: int
    home_goals: int = 0
    away_goals: int = 0
    home_shots: int = 0
    away_shots: int = 0
    pass_attempts: int = 0
    passes_completed: int = 0
    possession_home_ticks: int = 0
    possession_away_ticks: int = 0
    throw_ins: int = 0
    corners: int = 0
    goal_kicks: int = 0
    crosses: int = 0
    fouls: int = 0
    yellow_cards: int = 0

    def add(self, m: MatchMetrics) -> None:
        self.home_goals += m.home.goals
        self.away_goals += m.away.goals
        self.home_shots += m.home.shots
        self.away_shots += m.away.shots
        self.pass_attempts += m.home.pass_attempts + m.away.pass_attempts
        self.passes_completed += m.home.passes_completed + m.away.passes_completed
        self.possession_home_ticks += m.home.possession_ticks
        self.possession_away_ticks += m.away.possession_ticks
        self.throw_ins += m.home.throw_ins + m.away.throw_ins
        self.corners += m.home.corners + m.away.corners
        self.goal_kicks += m.home.goal_kicks + m.away.goal_kicks
        self.crosses += m.home.crosses + m.away.crosses
        self.fouls += m.home.fouls + m.away.fouls
        self.yellow_cards += m.home.yellow_cards + m.away.yellow_cards

    # -- derived metrics ----------------------------------------------------

    def goals_per_match(self) -> float:
        return (self.home_goals + self.away_goals) / self.matches

    def shots_per_team_per_match(self) -> float:
        return (self.home_shots + self.away_shots) / (2 * self.matches)

    def pass_completion(self) -> Optional[float]:
        if self.pass_attempts == 0:
            return None
        return self.passes_completed / self.pass_attempts

    def possession_home_share(self) -> Optional[float]:
        total = self.possession_home_ticks + self.possession_away_ticks
        if total == 0:
            return None
        return self.possession_home_ticks / total

    def goal_ratio_home_away(self) -> Optional[float]:
        if self.away_goals == 0:
            return None
        return self.home_goals / self.away_goals

    def corners_per_match(self) -> float:
        return self.corners / self.matches

    def throw_ins_per_match(self) -> float:
        return self.throw_ins / self.matches

    def crosses_per_match(self) -> float:
        return self.crosses / self.matches

    def fouls_per_match(self) -> float:
        return self.fouls / self.matches

    def yellows_per_match(self) -> float:
        return self.yellow_cards / self.matches


# Bands for identical mirrored teams ("balanced" vs "balanced").
# GATE bounds hold for the current engine (regression guard); TARGET bounds
# are the real-football calibration goal. When tuning closes the gap,
# tighten the gate toward the target.
BANDS: List[Band] = [
    # Gate floors re-based 2026-09 when live spatial dynamics (closing
    # down, density-scaled dribbles, corridor pricing) reset the attack
    # economy; targets hold the realism line the engine still owes.
    Band("Total goals / match", Aggregate.goals_per_match,
         gate_lo=0.8, gate_hi=4.0, target_lo=2.0, target_hi=3.5),
    Band("Shots / team / match", Aggregate.shots_per_team_per_match,
         gate_lo=3.0, gate_hi=20.0, target_lo=10.0, target_hi=18.0),
    Band("Pass completion", Aggregate.pass_completion,
         gate_lo=0.50, gate_hi=0.98, target_lo=0.70, target_hi=0.90),
    Band("Possession, home share (identical teams)", Aggregate.possession_home_share,
         gate_lo=0.40, gate_hi=0.60, target_lo=0.45, target_hi=0.55),
    Band("Goal ratio home/away (identical teams)", Aggregate.goal_ratio_home_away,
         gate_lo=0.50, gate_hi=2.00, target_lo=0.80, target_hi=1.30),
    Band("Corners / match", Aggregate.corners_per_match,
         gate_lo=0.5, gate_hi=25.0, target_lo=6.0, target_hi=14.0),
    Band("Throw-ins / match", Aggregate.throw_ins_per_match,
         gate_lo=0.5, gate_hi=80.0, target_lo=25.0, target_hi=50.0),
    Band("Crosses / match", Aggregate.crosses_per_match,
         gate_lo=1.0, gate_hi=60.0, target_lo=15.0, target_hi=40.0),
    Band("Fouls / match", Aggregate.fouls_per_match,
         gate_lo=2.0, gate_hi=60.0, target_lo=18.0, target_hi=30.0),
    Band("Yellow cards / match", Aggregate.yellows_per_match,
         gate_lo=0.0, gate_hi=12.0, target_lo=2.0, target_hi=6.0),
]


def behavioral_separation(seed: int = 42, trials: int = 400) -> float:
    """Archetype divergence: how much bolder the low-composure/high-
    aggression 'chaos engine' plays than the composed metronome in an
    identical situation. Returns the bold-action share ratio (>1 means
    personalities measurably separate); deterministic for a given seed."""
    import random as random_module
    from models import Ball, Player, Position, Team
    from situation import SituationEmbedding
    from decisions import DecisionContext, DualProcessDecisionModel
    from instincts import BOLD_ACTIONS

    def make_player(name, composure, aggression):
        player = Player(name=name, number=1, role="cm",
                        composure=composure, aggression=aggression)
        player.position = Position(50, 70)
        player.base_position = Position(50, 70)
        return player

    def context_for(player):
        home = Team(name="H", players=[player])
        teammate = Player(name="T", number=2, role="cm")
        away = Team(name="A", players=[Player(name="D", number=3, role="cm")])
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        return DecisionContext(
            holder=player, attacking_team=home, defending_team=away,
            state=state,
            situation=SituationEmbedding(0.55, 0.6, 0.5, 0.4, 0.5),
            in_shooting_range=True, space_ahead=15.0,
            nearest_defender_dist=10.0, lanes=[(teammate, 0.35)],
            best_forward_lane=0.35, best_safe_lane=0.35,
        )

    model = DualProcessDecisionModel(rng=random_module.Random(seed))

    def bold_share(player):
        bold = sum(1 for _ in range(trials)
                   if model.decide(context_for(player)) in BOLD_ACTIONS)
        return bold / trials

    chaos = bold_share(make_player("Chaos", composure=35, aggression=90))
    metronome = bold_share(make_player("Metronome", composure=90, aggression=25))
    return chaos / metronome if metronome > 0 else float("inf")


# Archetypes must separate at least this much for the gate to pass
SEPARATION_GATE = 1.15
SEPARATION_TARGET = 1.30


def run_validation(matches: int = 20, seed: int = 42, ticks_per_minute: int = 6,
                   home_style: str = "balanced",
                   away_style: str = "balanced") -> Aggregate:
    """Run the validation sample and return aggregated metrics."""
    engine = MatchEngine(SimulationConfig(
        ticks_per_minute=ticks_per_minute, randomness=0.3, seed=seed))
    aggregate = Aggregate(matches=matches)

    for _ in range(matches):
        home, away = create_tactical_matchup(home_style, away_style)
        state = MatchState(home_team=home, away_team=away, ball=Ball())

        match_metrics = MatchMetrics(home, away)
        engine.event_handlers = [match_metrics.on_event]
        engine.tick_handlers = [match_metrics.on_tick]

        engine.simulate_match(state, minutes=90)
        aggregate.add(match_metrics)

    return aggregate


def evaluate(aggregate: Aggregate, bands: List[Band] = BANDS):
    """Evaluate bands; returns (gate_failures, target_warnings, rows)."""
    gate_failures, target_warnings, rows = [], [], []

    for band in bands:
        value = band.value_fn(aggregate)
        if value is None:
            status = "GATE FAIL (no data)"
            gate_failures.append(band.name)
            rows.append((band.name, "n/a", band, status))
            continue

        if not (band.gate_lo <= value <= band.gate_hi):
            status = "GATE FAIL"
            gate_failures.append(band.name)
        elif not (band.target_lo <= value <= band.target_hi):
            status = "off target"
            target_warnings.append(band.name)
        else:
            status = "ok"
        rows.append((band.name, band.fmt.format(value), band, status))

    return gate_failures, target_warnings, rows


def print_report(aggregate: Aggregate, rows) -> None:
    print(f"\nANSTOSS ENGINE VALIDATION  ({aggregate.matches} matches, "
          f"identical mirrored teams)")
    print("-" * 92)
    header = (f"{'metric':<44}{'value':>8}   {'gate band':<14}"
              f"{'target band':<14}status")
    print(header)
    print("-" * 92)
    for name, value, band, status in rows:
        gate = f"{band.gate_lo:g}-{band.gate_hi:g}"
        target = f"{band.target_lo:g}-{band.target_hi:g}"
        print(f"{name:<44}{value:>8}   {gate:<14}{target:<14}{status}")
    print("-" * 92)


def main() -> int:
    parser = argparse.ArgumentParser(description="Engine statistical validation")
    parser.add_argument("--matches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--home", default="balanced")
    parser.add_argument("--away", default="balanced")
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero if any gate band fails")
    args = parser.parse_args()

    aggregate = run_validation(args.matches, args.seed, args.ticks,
                               args.home, args.away)
    gate_failures, target_warnings, rows = evaluate(aggregate)
    print_report(aggregate, rows)

    # Behavioral separation: personalities must measurably diverge
    separation = behavioral_separation(seed=args.seed)
    sep_status = ("ok" if separation >= SEPARATION_TARGET
                  else "off target" if separation >= SEPARATION_GATE
                  else "GATE FAIL")
    print(f"{'Archetype bold-action ratio (chaos/metronome)':<44}"
          f"{separation:>8.2f}   {f'>={SEPARATION_GATE:g}':<14}"
          f"{f'>={SEPARATION_TARGET:g}':<14}{sep_status}")
    if sep_status == "GATE FAIL":
        gate_failures.append("Behavioral separation")
    elif sep_status == "off target":
        target_warnings.append("Behavioral separation")

    if target_warnings:
        print(f"off target ({len(target_warnings)}): "
              + ", ".join(target_warnings))
        print("  -> calibration goals, not failures; see BANDS in validate.py")
    if gate_failures:
        print(f"GATE FAILURES ({len(gate_failures)}): "
              + ", ".join(gate_failures))
        return 1 if args.gate else 0

    print("all gate bands satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
