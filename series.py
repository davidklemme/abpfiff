#!/usr/bin/env python3
"""
Series runner: temporal continuity, first slice
(docs/specs/temporal-continuity.md).

Persistent squads play N matches with one shared MindRegistry; between
matches the boundary protocol runs (minds.close_match) and physical/
match-scoped state resets. An OCCASION SCHEDULE - a seeded Environment
per match, mostly routine with some big nights - provides the league's
interface (stakes variance for the load dimension) without its
machinery.

Trajectory metrics watch the evolution for degenerate states, with the
same gate/target band mechanics as validate.py:

    python3 series.py --matches 10 --seed 42 --gate

Each match runs on a fresh engine with a seed derived from
(series seed, match index) so a series is reproducible match-by-match
and mind serialization can be verified not to change the future.
"""
import argparse
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from models import Ball, Environment, MatchState, Position
from engine import MatchEngine, SimulationConfig
from metrics import MatchMetrics
from minds import MindRegistry, DEFAULT_REST_DAYS
from instincts import MAX_LEARNED_MEMORIES, default_bank_for
from situation import SituationEmbedding
from teams import create_tactical_matchup
from validate import Band, evaluate

# Occasion schedule: share of matches that are big nights, and how the
# environment is drawn for each kind.
BIG_NIGHT_CHANCE = 0.25
BIG_NIGHT_STAKES = (0.6, 1.0)
BIG_NIGHT_CROWD = (0.6, 1.0)
ROUTINE_STAKES = (0.0, 0.3)
ROUTINE_CROWD = (0.1, 0.5)

# A big-night decision moment, for probing familiarity after a series
# (load ~0.9 is what a stakes/crowd ~0.9 environment produces for a
# mid-sensitivity player - models.Environment.psychological_load).
BIG_NIGHT_PROBE = SituationEmbedding(0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.9)

CONFIDENCE_PINNED = 0.95  # |confidence| beyond this counts as pinned


def occasion_schedule(matches: int, rng: random.Random) -> List[Environment]:
    """Seeded Environment per match: the stakes variance a league would
    generate, without the league."""
    schedule = []
    for _ in range(matches):
        if rng.random() < BIG_NIGHT_CHANCE:
            schedule.append(Environment(
                stakes=rng.uniform(*BIG_NIGHT_STAKES),
                crowd_intensity=rng.uniform(*BIG_NIGHT_CROWD)))
        else:
            schedule.append(Environment(
                stakes=rng.uniform(*ROUTINE_STAKES),
                crowd_intensity=rng.uniform(*ROUTINE_CROWD)))
    return schedule


@dataclass
class MatchSnapshot:
    """Trajectory sample, taken at each match's end BEFORE the boundary
    protocol reverts state (so pinning/composition are measured as the
    match left them)."""
    big_night: bool
    goals: int
    shots: int
    crosses: int
    learned_mean: float        # learned memories per player
    anchor_strength: float     # total strength of success anchors
    trauma_strength: float     # total strength of traumas
    pinned_share: float        # players with |confidence| > CONFIDENCE_PINNED
    mean_abs_confidence: float
    pass_attempts: int = 0     # match quality, for side-effect watching
    passes_completed: int = 0


class SeriesRunner:
    """N matches, persistent squads, one shared mind registry."""

    def __init__(self, matches: int = 10, seed: int = 42,
                 ticks_per_minute: int = 6, minutes: int = 90,
                 home_style: str = "balanced", away_style: str = "balanced"):
        self.matches = matches
        self.seed = seed
        self.ticks_per_minute = ticks_per_minute
        self.minutes = minutes
        self.minds = MindRegistry()
        self.home, self.away = create_tactical_matchup(home_style, away_style)
        self.rosters = (list(self.home.players), list(self.away.players))
        # Canonical formation frame: base positions and attack direction
        # as created, so resets don't depend on how many half-time flips
        # a match applied (sent-off players miss flips, e.g.).
        self._frame = {
            player.player_id: (player.base_position.x, player.base_position.y)
            for roster in self.rosters for player in roster
        }
        self._attacks_up = (self.home.attacks_up, self.away.attacks_up)
        self.schedule = occasion_schedule(matches, random.Random(seed))
        self.snapshots: List[MatchSnapshot] = []

    # -- running -------------------------------------------------------------

    def play(self, upto: Optional[int] = None) -> "SeriesRunner":
        """Play matches up to index `upto` (exclusive; default: all).
        Callable in stages - a partial series continues where it left
        off, which is how serialization mid-series is verified."""
        upto = self.matches if upto is None else min(upto, self.matches)
        while len(self.snapshots) < upto:
            self._play_one(len(self.snapshots))
        return self

    def _play_one(self, index: int) -> None:
        self._reset_squads()
        environment = self.schedule[index]
        engine = MatchEngine(
            SimulationConfig(ticks_per_minute=self.ticks_per_minute,
                             randomness=0.3, seed=self.seed * 1000 + index),
            minds=self.minds)
        state = MatchState(home_team=self.home, away_team=self.away,
                           ball=Ball(), environment=environment)
        match_metrics = MatchMetrics(self.home, self.away)
        engine.event_handlers = [match_metrics.on_event]
        engine.tick_handlers = [match_metrics.on_tick]
        engine.simulate_match(state, minutes=self.minutes)

        self.snapshots.append(self._snapshot(match_metrics, environment))
        everyone = self.rosters[0] + self.rosters[1]
        self.minds.close_match(everyone, rest_days=DEFAULT_REST_DAYS)

    def _reset_squads(self) -> None:
        """Match-scoped state resets; continuous state (confidence,
        memories, residual fatigue) is close_match's business."""
        for team, roster, attacks_up in zip(
                (self.home, self.away), self.rosters, self._attacks_up):
            team.players = list(roster)  # sent-off players return
            team.momentum = 50.0
            team.attacks_up = attacks_up
            for player in roster:
                base_x, base_y = self._frame[player.player_id]
                player.base_position = Position(base_x, base_y)
                player.position = Position(base_x, base_y)
                player.velocity_x = player.velocity_y = 0.0
                player.yellow_cards = 0
                player.sent_off = False

    # -- trajectory measurement ----------------------------------------------

    def _snapshot(self, m: MatchMetrics,
                  environment: Environment) -> MatchSnapshot:
        everyone = self.rosters[0] + self.rosters[1]
        learned_counts, anchor_strength, trauma_strength = [], 0.0, 0.0
        for player in everyone:
            mind = self.minds.get(player.player_id)
            if mind is None:
                learned_counts.append(0)
                continue
            learned = mind.bank.learned_memories()
            learned_counts.append(len(learned))
            for instinct in learned:
                if instinct.source == "experience":
                    anchor_strength += instinct.strength
                else:
                    trauma_strength += instinct.strength

        pinned = sum(1 for p in everyone
                     if abs(p.confidence) > CONFIDENCE_PINNED)
        return MatchSnapshot(
            big_night=environment.stakes >= BIG_NIGHT_STAKES[0],
            goals=m.home.goals + m.away.goals,
            shots=m.home.shots + m.away.shots,
            crosses=m.home.crosses + m.away.crosses,
            learned_mean=sum(learned_counts) / len(learned_counts),
            anchor_strength=anchor_strength,
            trauma_strength=trauma_strength,
            pinned_share=pinned / len(everyone),
            mean_abs_confidence=(sum(abs(p.confidence) for p in everyone)
                                 / len(everyone)),
            pass_attempts=m.home.pass_attempts + m.away.pass_attempts,
            passes_completed=m.home.passes_completed + m.away.passes_completed,
        )

    # -- series-level metrics (Band value functions) -------------------------

    @property
    def final(self) -> MatchSnapshot:
        return self.snapshots[-1]

    def final_learned_mean(self) -> float:
        return self.final.learned_mean

    def first_match_learned_mean(self) -> float:
        return self.snapshots[0].learned_mean

    def anchor_share(self) -> Optional[float]:
        """Share of learned-memory strength that is success anchors."""
        total = self.final.anchor_strength + self.final.trauma_strength
        if total == 0:
            return None
        return self.final.anchor_strength / total

    def max_pinned_share(self) -> float:
        return max(s.pinned_share for s in self.snapshots)

    def play_survival(self) -> Optional[float]:
        """Attacking output late in the series relative to early: the
        degenerate failure of learning is trauma pile-up driving
        everyone to the safe ball until shots vanish."""
        third = max(1, len(self.snapshots) // 3)
        early = sum(s.shots for s in self.snapshots[:third])
        late = sum(s.shots for s in self.snapshots[-third:])
        if early == 0:
            return None
        return late / early

    def big_night_familiarity_gain(self) -> float:
        """Veteran effect: how much better the series-trained squads
        recognize a big-night moment than fresh identical squads would
        (the load dimension's promise, spec section 4)."""
        everyone = self.rosters[0] + self.rosters[1]
        trained = fresh = 0.0
        for player in everyone:
            mind = self.minds.get(player.player_id)
            bank = mind.bank if mind is not None else default_bank_for(player)
            trained += bank.familiarity(BIG_NIGHT_PROBE)
            fresh += default_bank_for(player).familiarity(BIG_NIGHT_PROBE)
        return (trained - fresh) / len(everyone)

    # -- composition diagnostics ---------------------------------------------
    #
    # The gain above is a difference of two folded numbers, so a zero can
    # mean "nothing was learned" or "plenty was learned but schooling
    # still dominates the fold". These split it.

    def _recognition_split(self) -> List[Tuple[float, float]]:
        """Per player, (role, learned) recognition of the big-night probe."""
        everyone = self.rosters[0] + self.rosters[1]
        split = []
        for player in everyone:
            mind = self.minds.get(player.player_id)
            bank = mind.bank if mind is not None else default_bank_for(player)
            split.append(bank.recognition_split(BIG_NIGHT_PROBE))
        return split

    def role_recognition(self) -> float:
        split = self._recognition_split()
        return sum(role for role, _ in split) / len(split)

    def learned_recognition(self) -> float:
        split = self._recognition_split()
        return sum(learned for _, learned in split) / len(split)

    def learned_wins_share(self) -> float:
        """Share of players whose lived memories out-recognize their
        schooling at a big-night moment. This is the veteran effect
        stated as a count: at 0 no player recognizes a big night from
        having lived one."""
        split = self._recognition_split()
        return sum(1 for role, learned in split if learned > role) / len(split)

    # -- match quality (side-effect watching) --------------------------------

    def goals_per_match(self) -> float:
        return sum(s.goals for s in self.snapshots) / len(self.snapshots)

    def shots_per_match(self) -> float:
        return sum(s.shots for s in self.snapshots) / len(self.snapshots)

    def pass_completion(self) -> Optional[float]:
        attempts = sum(s.pass_attempts for s in self.snapshots)
        if attempts == 0:
            return None
        return sum(s.passes_completed for s in self.snapshots) / attempts

    def mean_abs_confidence(self) -> float:
        return (sum(s.mean_abs_confidence for s in self.snapshots)
                / len(self.snapshots))


# Degenerate-state gates (regression guards) and evolution targets.
# Reuses validate.Band/evaluate: value_fn is called with the runner.
SERIES_BANDS: List[Band] = [
    Band("Learned memories / player (series end)",
         SeriesRunner.final_learned_mean,
         gate_lo=0.5, gate_hi=float(MAX_LEARNED_MEMORIES),
         target_lo=1.5, target_hi=float(MAX_LEARNED_MEMORIES) * 0.9),
    Band("Learned memories / player (after match 1)",
         SeriesRunner.first_match_learned_mean,
         gate_lo=0.05, gate_hi=MAX_LEARNED_MEMORIES * 0.9,
         target_lo=0.2, target_hi=6.0),
    Band("Anchor share of learned strength",
         SeriesRunner.anchor_share,
         gate_lo=0.10, gate_hi=0.95, target_lo=0.30, target_hi=0.80),
    Band("Max pinned-confidence share",
         SeriesRunner.max_pinned_share,
         gate_lo=0.0, gate_hi=0.50, target_lo=0.0, target_hi=0.25),
    Band("Play survival (late shots / early shots)",
         SeriesRunner.play_survival,
         gate_lo=0.40, gate_hi=3.0, target_lo=0.70, target_hi=1.40),
    Band("Big-night familiarity gain (trained - fresh)",
         SeriesRunner.big_night_familiarity_gain,
         gate_lo=-0.02, gate_hi=1.0, target_lo=0.01, target_hi=1.0,
         fmt="{:.3f}"),
]


def run_series(matches: int = 10, seed: int = 42,
               ticks_per_minute: int = 6, minutes: int = 90) -> SeriesRunner:
    return SeriesRunner(matches=matches, seed=seed,
                        ticks_per_minute=ticks_per_minute,
                        minutes=minutes).play()


def print_series_report(runner: SeriesRunner, rows) -> None:
    print(f"\nANSTOSS ENGINE SERIES VALIDATION  ({runner.matches} matches, "
          f"persistent squads, seed {runner.seed})")
    big_nights = sum(1 for s in runner.snapshots if s.big_night)
    print(f"occasions: {big_nights} big nights, "
          f"{runner.matches - big_nights} routine")
    print("-" * 92)
    print(f"{'trajectory metric':<44}{'value':>8}   {'gate band':<14}"
          f"{'target band':<14}status")
    print("-" * 92)
    for name, value, band, status in rows:
        gate = f"{band.gate_lo:g}-{band.gate_hi:g}"
        target = f"{band.target_lo:g}-{band.target_hi:g}"
        print(f"{name:<44}{value:>8}   {gate:<14}{target:<14}{status}")
    print("-" * 92)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Temporal continuity series validation")
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--minutes", type=int, default=90)
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero if any gate band fails")
    args = parser.parse_args()

    runner = run_series(args.matches, args.seed, args.ticks, args.minutes)
    gate_failures, target_warnings, rows = evaluate(runner, SERIES_BANDS)
    print_series_report(runner, rows)

    if target_warnings:
        print(f"off target ({len(target_warnings)}): "
              + ", ".join(target_warnings))
    if gate_failures:
        print(f"GATE FAILURES ({len(gate_failures)}): "
              + ", ".join(gate_failures))
        return 1 if args.gate else 0

    print("all series gate bands satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
