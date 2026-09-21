"""Career trajectory simulator: 100 players, 15 seasons, two memory models.

Careers cannot be validated by reasoning and cannot be run through the real
engine at scale (15 seasons x 38 matches x 100 players is ~60k matches). So
this replays the EMPIRICAL learning stream captured from real matches
(stream.json: situations, actions, valences, significances the engine
actually produces) at career scale, under two interchangeable trace models:

  current  - scalar strength, merge/prune/decay exactly as shipped
  proposed - mass + per-dimension widths, log accumulation, power-law
             retention, elasticity-driven birth width

The question is not "which scores higher" but "which produces a plausible
DISTRIBUTION of careers": most players improving into their mid-to-late
twenties, a spread of peaks, specialists and generalists, and some who
never make it.
"""
import json
import math
import pathlib
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DIMS = 7
DIM_NAMES = ["pressure", "progression", "time_crit", "density",
             "support", "width", "load"]

MATCHES_PER_SEASON = 38
SEASONS = 15
EVENTS_PER_MATCH = 5.7          # measured: 1244 events / 22 players / 10 matches
START_AGE = 17

# -- current model ----------------------------------------------------------
CUR_MERGE_SIM = 0.8
CUR_MAX_MEMORIES = 12
CUR_MIN_STRENGTH = 0.05
CUR_DECAY_PER_MATCH = 0.999 ** 90 * 0.99 ** 7   # 0.852
CUR_SIM_SCALE = 6.0
CUR_ROLE_FAMILIARITY = 0.6

# -- proposed model ---------------------------------------------------------
BIRTH_WIDTH_PLASTIC = 0.90      # a wide-eyed teenager generalises broadly
BIRTH_WIDTH_RIGID = 0.30        # a set veteran learns only what literally happened
ELASTICITY_HALFLIFE_REPS = 900.0
NARROW_RATE = 0.85              # width multiplier on an agreeing axis
WIDEN_RATE = 1.12               # width multiplier on a disagreeing axis
WIDTH_FLOOR = 0.08
WIDTH_CEIL = 1.60
RETENTION_D0 = 0.35             # power-law exponent for a one-off trace
SPACING_GAIN = 0.55             # how much repetition flattens the curve
MATCH_KERNEL_THRESHOLD = 0.35   # kernel above which a trace counts as "the same"
MAX_TRACES = 160
SCHOOLING_REPS = 24.0           # drilled for years before the career starts
SCHOOLING_WIDTH = 1.30          # broad and shallow: covers everything loosely


def load_stream(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Current model (as shipped)
# ---------------------------------------------------------------------------

class CurrentBank:
    def __init__(self, rng, elasticity=50):
        # role seeds: 2 prototypes, strength 1.0, never decay, never reinforced
        self.seeds = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.05] for _ in range(2)]
        self.traces = []      # (prototype, strength)

    def _sim(self, a, b):
        return max(0.0, 1.0 - sum(abs(x - y) for x, y in zip(a, b)) / CUR_SIM_SCALE)

    def learn(self, situation, significance, valence, match_index):
        gained = min(0.6, abs(valence) * significance)
        for trace in self.traces:
            if self._sim(situation, trace[0]) >= CUR_MERGE_SIM:
                trace[1] = min(1.0, trace[1] + gained * 0.5)
                trace[0] = [0.8 * o + 0.2 * n for o, n in zip(trace[0], situation)]
                return
        self.traces.append([list(situation), min(0.6, 0.15 + gained)])
        if len(self.traces) > CUR_MAX_MEMORIES:
            self.traces.remove(min(self.traces, key=lambda t: t[1]))

    def end_match(self):
        for trace in self.traces:
            trace[1] *= CUR_DECAY_PER_MATCH
        self.traces = [t for t in self.traces if t[1] >= CUR_MIN_STRENGTH]

    def recognition(self, situation):
        best = 0.0
        for seed in self.seeds:
            s = self._sim(situation, seed)
            best = max(best, s * s * 1.0 * CUR_ROLE_FAMILIARITY)
        for proto, strength in self.traces:
            s = self._sim(situation, proto)
            best = max(best, s * s * strength)
        return min(1.0, best)

    @property
    def trace_count(self):
        return len(self.traces)


# ---------------------------------------------------------------------------
# Proposed model: mass + per-dimension receptive fields
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    prototype: List[float]
    width: List[float]
    reps: float                 # accumulated evidence count (significance-weighted)
    last_fired: int             # match index

    def kernel(self, situation) -> float:
        distance = sum(abs(a - b) / w
                       for a, b, w in zip(situation, self.prototype, self.width))
        return max(0.0, 1.0 - distance)

    def evidence(self, match_index: int) -> float:
        """Log-accumulated mass, faded by power-law retention whose exponent
        flattens with spaced repetition."""
        mass = math.log1p(self.reps)
        elapsed = max(0, match_index - self.last_fired)
        exponent = RETENTION_D0 / (1.0 + SPACING_GAIN * math.log1p(self.reps))
        return mass * (1.0 + elapsed) ** -exponent


class ProposedBank:
    def __init__(self, rng, elasticity=50):
        self.trait = elasticity / 100.0
        self.traces: List[Trace] = []
        self.total_reps = 0.0
        # schooling: high mass, wide widths - same physics, different shape
        for _ in range(2):
            self.traces.append(Trace(
                prototype=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.05],
                width=[SCHOOLING_WIDTH] * DIMS,
                reps=SCHOOLING_REPS, last_fired=0))

    def elasticity(self) -> float:
        """Trait, consumed by accumulated experience."""
        spent = math.exp(-self.total_reps / ELASTICITY_HALFLIFE_REPS)
        return self.trait * spent

    def birth_width(self) -> float:
        e = self.elasticity()
        return BIRTH_WIDTH_RIGID + (BIRTH_WIDTH_PLASTIC - BIRTH_WIDTH_RIGID) * e

    def learn(self, situation, significance, valence, match_index):
        self.total_reps += significance
        best, best_k = None, MATCH_KERNEL_THRESHOLD
        for trace in self.traces:
            k = trace.kernel(situation)
            if k > best_k:
                best, best_k = trace, k

        if best is None:
            w = self.birth_width()
            self.traces.append(Trace(prototype=list(situation),
                                     width=[w] * DIMS,
                                     reps=significance, last_fired=match_index))
            self._consolidate()
            return

        # reinforce: mass up, widths learn WHICH axes were diagnostic,
        # prototype moves less the more established the trace is
        best.reps += significance
        best.last_fired = match_index
        move = 1.0 / (1.0 + best.reps)
        for d in range(DIMS):
            delta = abs(situation[d] - best.prototype[d])
            if delta < best.width[d] * 0.5:      # this axis agreed
                best.width[d] = max(WIDTH_FLOOR, best.width[d] * NARROW_RATE)
            else:                                 # this axis varied
                best.width[d] = min(WIDTH_CEIL, best.width[d] * WIDEN_RATE)
            best.prototype[d] += move * (situation[d] - best.prototype[d])

    def _consolidate(self):
        """Redundancy-based, not count-based: merge the most overlapping pair.
        Under power-law retention 'drop the weakest' would always drop the
        newest trace, and the bank would never turn over."""
        if len(self.traces) <= MAX_TRACES:
            return
        # Sampled rather than exhaustive: the exhaustive O(n^2) scan runs on
        # EVERY insertion once the bank is saturated, which made a 15-season
        # population run intractable. Sampling finds a near-most-redundant
        # pair at a fraction of the cost, and consolidation is not a decision
        # that needs to be optimal.
        import random as _random
        candidates = _random.sample(range(len(self.traces)),
                                    min(24, len(self.traces)))
        best_pair, best_overlap = None, -1.0
        for a_idx in range(len(candidates)):
            for b_idx in range(a_idx + 1, len(candidates)):
                i, j = candidates[a_idx], candidates[b_idx]
                o = self.traces[i].kernel(self.traces[j].prototype)
                if o > best_overlap:
                    best_pair, best_overlap = (min(i, j), max(i, j)), o
        i, j = best_pair
        a, b = self.traces[i], self.traces[j]
        total = a.reps + b.reps
        a.prototype = [(a.reps * x + b.reps * y) / total
                       for x, y in zip(a.prototype, b.prototype)]
        a.width = [max(wa, wb) for wa, wb in zip(a.width, b.width)]
        a.reps = total
        a.last_fired = max(a.last_fired, b.last_fired)
        self.traces.pop(j)

    def end_match(self):
        pass   # retention is computed from last_fired, not applied per match

    def recognition(self, situation, match_index=0):
        return min(1.0, sum(t.kernel(situation) * t.evidence(match_index)
                            for t in self.traces))

    @property
    def trace_count(self):
        return len(self.traces)


# ---------------------------------------------------------------------------
# Population and careers
# ---------------------------------------------------------------------------

@dataclass
class Player:
    pid: int
    elasticity: int
    opportunity: float          # share of available matches actually played
    bank: object = None
    history: List[Tuple[int, float, float, int]] = field(default_factory=list)


def make_population(n, rng, model):
    players = []
    for pid in range(n):
        elasticity = max(5, min(95, int(rng.gauss(50, 18))))
        # opportunity is lognormal-ish: a few play everything, many play little
        opportunity = min(1.0, max(0.05, rng.betavariate(2.0, 2.2)))
        bank = model(rng, elasticity)
        players.append(Player(pid=pid, elasticity=elasticity,
                              opportunity=opportunity, bank=bank))
    return players


def capability(bank, probes, match_index):
    """How much of the football a player ACTUALLY meets do they recognise?
    Mean recognition over situations drawn from the empirical distribution -
    weighted by how often those situations occur, because recognising rare
    moments is worth less than recognising common ones."""
    if isinstance(bank, ProposedBank):
        return statistics.mean(bank.recognition(p, match_index) for p in probes)
    return statistics.mean(bank.recognition(p) for p in probes)


def run_population(stream, model, n_players=100, seed=1):
    rng = random.Random(seed)
    players = make_population(n_players, rng, model)
    situations = [r["situation"] for r in stream]
    probes = [situations[rng.randrange(len(situations))] for _ in range(120)]

    total_matches = SEASONS * MATCHES_PER_SEASON
    for match_index in range(total_matches):
        for player in players:
            if rng.random() > player.opportunity:
                continue                      # not selected this match
            events = int(EVENTS_PER_MATCH) + (1 if rng.random() < 0.7 else 0)
            for _ in range(events):
                record = stream[rng.randrange(len(stream))]
                player.bank.learn(record["situation"], record["significance"],
                                  record["valence"], match_index)
            player.bank.end_match()

        if (match_index + 1) % MATCHES_PER_SEASON == 0:
            season = (match_index + 1) // MATCHES_PER_SEASON
            for player in players:
                player.history.append((
                    season,
                    capability(player.bank, probes, match_index),
                    player.bank.elasticity() if isinstance(player.bank, ProposedBank) else 0.0,
                    player.bank.trace_count,
                ))
    return players


def describe(players, label):
    print(f"\n{'=' * 78}")
    print(f"{label}")
    print('=' * 78)
    print(f"{'season':>7}{'age':>5}{'capability':>24}{'traces':>9}{'elastic':>9}")
    print(f"{'':>12}{'p10':>7}{'median':>8}{'p90':>8}{'median':>9}{'median':>9}")
    for s in range(1, SEASONS + 1):
        caps = sorted(p.history[s - 1][1] for p in players)
        traces = statistics.median(p.history[s - 1][3] for p in players)
        elas = statistics.median(p.history[s - 1][2] for p in players)
        def pct(q): return caps[min(len(caps) - 1, int(q * len(caps)))]
        print(f"{s:>7}{START_AGE + s:>5}{pct(0.10):>7.3f}{pct(0.50):>8.3f}"
              f"{pct(0.90):>8.3f}{traces:>9.0f}{elas:>9.2f}")

    peaks = []
    for p in players:
        caps = [h[1] for h in p.history]
        peaks.append(START_AGE + 1 + caps.index(max(caps)))
    print()
    print(f"peak age: median {statistics.median(peaks):.0f}, "
          f"mean {statistics.mean(peaks):.1f}, "
          f"range {min(peaks)}-{max(peaks)}")
    dist = {}
    for pk in peaks:
        bucket = "<=21" if pk <= 21 else "22-25" if pk <= 25 else "26-29" if pk <= 29 else "30+"
        dist[bucket] = dist.get(bucket, 0) + 1
    print("peak-age distribution: " + "  ".join(
        f"{k}: {v}%" for k, v in sorted(dist.items())))

    finals = [p.history[-1][1] for p in players]
    firsts = [p.history[0][1] for p in players]
    improved = sum(1 for a, b in zip(firsts, finals) if b > a * 1.05)
    flat = sum(1 for a, b in zip(firsts, finals) if a * 0.95 <= b <= a * 1.05)
    print(f"career outcome: {improved}% improved, {flat}% flat, "
          f"{len(players)-improved-flat}% declined")
    print(f"spread at peak season 8: p90/p10 = "
          f"{sorted(p.history[7][1] for p in players)[89]:.3f} / "
          f"{sorted(p.history[7][1] for p in players)[9]:.3f}")


if __name__ == "__main__":
    stream = load_stream(pathlib.Path(__file__).parent / 'stream.json')
    print(f"replaying {len(stream)} empirical learning events, "
          f"{SEASONS} seasons x {MATCHES_PER_SEASON} matches, 100 players")

    describe(run_population(stream, CurrentBank), "CURRENT MODEL (as shipped)")
    describe(run_population(stream, ProposedBank), "PROPOSED MODEL (trace geometry)")
