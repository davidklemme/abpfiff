"""Receptive-field traces for fast, automatic action preferences (System 1).

Every trace has the same physics.  Role schooling starts broad and well-rehearsed;
lived anchors and traumas start with a width set by player elasticity.  Retrieval
is anisotropic, evidence accumulates logarithmically, and retention follows a
power law advanced only at match boundaries.
"""
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

from models import Player
from situation import SituationEmbedding

BOLD_ACTIONS = ("shoot", "dribble", "pass_forward", "cross")
SAFE_ACTIONS = ("pass_safe",)
ALL_ACTIONS = BOLD_ACTIONS + SAFE_ACTIONS

CONFIDENCE_TILT = 0.35
ANCHOR_CONFIDENCE_GAIN = 0.5
TRAUMA_RATTLED_GAIN = 0.5
SCHOOLING_LOAD = 0.05

DIMENSIONS = 7
SCHOOLING_WIDTH = 1.45
SCHOOLING_REPETITIONS = 24.0
LIVED_WIDTH_RIGID = 0.30
LIVED_WIDTH_PLASTIC = 0.90
ELASTICITY_EVIDENCE_HALFLIFE = 90.0
WIDTH_FLOOR = 0.08
WIDTH_CEILING = 1.60
WIDTH_EXPONENT = 1.0 / (DIMENSIONS + 4.0)
WIDTH_PRIOR_EVIDENCE = 3.0
RETENTION_EXPONENT = 0.35
SPACING_GAIN = 0.55
MERGE_KERNEL_THRESHOLD = 0.35
MAX_TRACES = 160
MIN_RETAINED_MASS = 0.01
RECOGNITION_SCALE = 0.12
LEARNED_SOURCES = ("experience", "trauma")


def _source_factor(source: str, confidence: float) -> float:
    if source == "experience":
        return 1.0 + ANCHOR_CONFIDENCE_GAIN * max(0.0, confidence)
    if source == "trauma":
        return 1.0 + TRAUMA_RATTLED_GAIN * max(0.0, -confidence)
    return 1.0


def aggression_tilted(weights: Dict[str, float], aggression: float) -> Dict[str, float]:
    """Tilt a role/fallback action map once, from the player's personality."""
    tilt = (aggression - 50) / 100.0
    return {
        action: max(0.01, weight * (1.0 + tilt * 0.6 if action in BOLD_ACTIONS
                                    else 1.0 - tilt * 0.6))
        for action, weight in weights.items()
    }


@dataclass
class Instinct:
    """A situation prototype with its own per-axis receptive field."""
    name: str
    prototype: SituationEmbedding
    action_weights: Dict[str, float]
    mass: float = field(default_factory=lambda: math.log1p(SCHOOLING_REPETITIONS))
    source: str = "role"
    widths: Tuple[float, ...] = field(default_factory=lambda: (SCHOOLING_WIDTH,) * DIMENSIONS)
    repetitions: float = SCHOOLING_REPETITIONS
    spacing: float = SCHOOLING_REPETITIONS
    age: float = 0.0
    birth_width: float = SCHOOLING_WIDTH

    def __post_init__(self) -> None:
        self.widths = tuple(float(value) for value in self.widths)
        if len(self.widths) != len(self.prototype.as_tuple()):
            raise ValueError("trace widths must match the situation dimensions")
        if any(value <= 0.0 for value in self.widths):
            raise ValueError("trace widths must be positive")
        if (self.mass < 0.0 or self.repetitions < 0.0
                or self.spacing < 0.0 or self.age < 0.0):
            raise ValueError("trace evidence and age cannot be negative")

    def kernel(self, situation: SituationEmbedding) -> float:
        distance = sum(abs(a - b) / width for a, b, width in
                       zip(situation.as_tuple(), self.prototype.as_tuple(), self.widths))
        return max(0.0, 1.0 - distance)

    def retained_mass(self) -> float:
        exponent = RETENTION_EXPONENT / (
            1.0 + SPACING_GAIN * math.log1p(self.spacing))
        return self.mass * (1.0 + self.age) ** -exponent


class InstinctBank:
    """A bounded, similarity-queried store of automatic responses."""

    def __init__(self, instincts: List[Instinct], elasticity: int = 50):
        self.instincts = instincts
        self.elasticity_trait = max(0.0, min(1.0, elasticity / 100.0))
        self.accumulated_evidence = sum(
            trace.repetitions for trace in instincts if isinstance(trace, Instinct)
            and trace.source in LEARNED_SOURCES)

    def _traces(self):
        # Some clients attach sentinels to banks; they are metadata, not traces.
        return (trace for trace in self.instincts if isinstance(trace, Instinct))

    def recall(self, situation: SituationEmbedding, confidence: float = 0.0):
        weights = {action: 0.0 for action in ALL_ACTIONS}
        recognition_evidence = 0.0
        for trace in self._traces():
            kernel = trace.kernel(situation)
            retained = trace.retained_mass()
            match = kernel * retained * _source_factor(trace.source, confidence)
            for action, weight in trace.action_weights.items():
                weights[action] = weights.get(action, 0.0) + match * weight
            recognition_evidence += kernel * retained

        for action in list(weights):
            factor = (1.0 + CONFIDENCE_TILT * confidence if action in BOLD_ACTIONS
                      else 1.0 - CONFIDENCE_TILT * confidence)
            weights[action] = max(0.0, weights[action] * factor)
        total = sum(weights.values())
        if total <= 0.0:
            weights = {action: 1.0 / len(ALL_ACTIONS) for action in ALL_ACTIONS}
        else:
            weights = {action: value / total for action, value in weights.items()}
        familiarity = 1.0 - math.exp(-RECOGNITION_SCALE * recognition_evidence)
        return weights, familiarity

    def query(self, situation: SituationEmbedding, confidence: float = 0.0) -> Dict[str, float]:
        return self.recall(situation, confidence)[0]

    def familiarity(self, situation: SituationEmbedding) -> float:
        return self.recall(situation)[1]

    def recognition_split(self, situation: SituationEmbedding) -> Tuple[float, float]:
        role = learned = 0.0
        for trace in self._traces():
            evidence = trace.kernel(situation) * trace.retained_mass()
            if trace.source == "role":
                role += evidence
            else:
                learned += evidence
        convert = lambda value: 1.0 - math.exp(-RECOGNITION_SCALE * value)
        return convert(role), convert(learned)

    def _birth_width(self) -> float:
        plasticity = self.elasticity_trait * math.exp(
            -self.accumulated_evidence / ELASTICITY_EVIDENCE_HALFLIFE)
        return LIVED_WIDTH_RIGID + (LIVED_WIDTH_PLASTIC - LIVED_WIDTH_RIGID) * plasticity

    def learn(self, situation: SituationEmbedding, action: str,
              valence: float, significance: float) -> None:
        if valence == 0.0 or action not in ALL_ACTIONS or significance <= 0.0:
            return
        source = "experience" if valence > 0.0 else "trauma"
        evidence = abs(valence) * significance
        self.accumulated_evidence += evidence
        if not self._merge_into_existing(situation, action, source, evidence):
            width = self._birth_width()
            self.instincts.append(Instinct(
                name=f"{source}_{action}", prototype=situation,
                action_weights={action: 1.0 if valence > 0.0 else -1.0},
                mass=math.log1p(evidence), source=source,
                widths=(width,) * DIMENSIONS, repetitions=evidence,
                spacing=evidence, birth_width=width))
        self._consolidate()

    @staticmethod
    def _compatible(trace: Instinct, action: str, source: str) -> bool:
        if action not in trace.action_weights:
            return False
        # Schooling is reinforceable by success. Opposite-signed outcomes remain
        # separate so consolidation cannot erase trauma/anchor semantics.
        return trace.source == source or (trace.source == "role" and source == "experience")

    def _merge_into_existing(self, situation: SituationEmbedding, action: str,
                             source: str, evidence: float) -> bool:
        candidates = [(trace.kernel(situation), trace) for trace in self._traces()
                      if self._compatible(trace, action, source)]
        if not candidates:
            return False
        kernel, trace = max(candidates, key=lambda item: item[0])
        if kernel < MERGE_KERNEL_THRESHOLD:
            return False
        old_repetitions = trace.repetitions
        # Reinforcement after an interval carries more retention information
        # than the same repetitions massed into one episode.
        trace.spacing += evidence * (1.0 + 0.25 * min(trace.age, 4.0))
        trace.repetitions += evidence
        trace.mass = math.log1p(trace.repetitions)
        trace.age = 0.0
        move = evidence / (1.0 + old_repetitions + evidence)
        old_values = trace.prototype.as_tuple()
        new_values = situation.as_tuple()
        trace.prototype = SituationEmbedding(*(
            old + move * (new - old) for old, new in zip(old_values, new_values)))
        shrink = ((old_repetitions + WIDTH_PRIOR_EVIDENCE) /
                  (trace.repetitions + WIDTH_PRIOR_EVIDENCE)) ** WIDTH_EXPONENT
        adapted = []
        for old, new, current in zip(old_values, new_values, trace.widths):
            delta = abs(new - old)
            target = max(trace.birth_width, 2.0 * delta)
            learned = current + move * (target - current)
            adapted.append(max(WIDTH_FLOOR, min(WIDTH_CEILING, learned * shrink)))
        trace.widths = tuple(adapted)
        return True

    def _consolidate(self) -> None:
        traces = list(self._traces())
        while len(traces) > MAX_TRACES:
            pairs = []
            for left_index, left in enumerate(traces):
                for right_index in range(left_index + 1, len(traces)):
                    right = traces[right_index]
                    if left.source != right.source or left.action_weights != right.action_weights:
                        continue
                    overlap = min(left.kernel(right.prototype), right.kernel(left.prototype))
                    pairs.append((overlap, left.name, right.name, left_index, right_index))
            if not pairs:
                # No semantically safe merge exists; retain the newest trace and
                # evict the stalest negligible one only as a hard safety valve.
                victim = min(traces[:-1], key=lambda trace: (trace.retained_mass(), trace.name))
                self.instincts.remove(victim)
            else:
                _, _, _, left_index, right_index = max(pairs)
                left, right = traces[left_index], traces[right_index]
                total = left.repetitions + right.repetitions
                if total <= 0.0:
                    total = 1.0
                left.prototype = SituationEmbedding(*(
                    (left.repetitions * a + right.repetitions * b) / total
                    for a, b in zip(left.prototype.as_tuple(), right.prototype.as_tuple())))
                left.widths = tuple(max(a, b) for a, b in zip(left.widths, right.widths))
                left.repetitions += right.repetitions
                left.spacing += right.spacing
                left.mass = math.log1p(left.repetitions)
                left.age = min(left.age, right.age)
                self.instincts.remove(right)
            traces = list(self._traces())

    def decay(self, matches: float = 1.0) -> None:
        """Advance the power-law retention clock by match boundaries."""
        for trace in self._traces():
            trace.age += max(0.0, matches)
        self.instincts = [trace for trace in self.instincts
                          if not isinstance(trace, Instinct)
                          or trace.source == "role"
                          or trace.retained_mass() >= MIN_RETAINED_MASS]

    def learned_memories(self) -> List[Instinct]:
        return [trace for trace in self._traces() if trace.source in LEARNED_SOURCES]

# ---------------------------------------------------------------------------
# Seeding: role comfort actions, shaped by attributes
# ---------------------------------------------------------------------------

def _proto(pressure=0.5, progression=0.5, time_criticality=0.5,
           spatial_density=0.5, support=0.5, width=0.5,
           load=None) -> SituationEmbedding:
    load = SCHOOLING_LOAD if load is None else load
    return SituationEmbedding(pressure, progression, time_criticality,
                              spatial_density, support, width, load)


def _at_schooling_load(prototype: SituationEmbedding) -> SituationEmbedding:
    """Stamp a role seed with the CURRENT schooling load.

    ROLE_SEEDS is built once at import, so the load its prototypes were
    written with freezes there. Stamping at bank construction instead
    keeps SCHOOLING_LOAD a live dial - it is documented as a tuning
    knob ('a flat familiarity tax on every ordinary-environment
    situation'), and an import-time constant that silently ignores every
    later change is not one. Returns the prototype unchanged when it
    already carries the current load, so this is free at the default."""
    if prototype.load == SCHOOLING_LOAD:
        return prototype
    return replace(prototype, load=SCHOOLING_LOAD)


# Per role-group: (situation prototype, action weights) comfort patterns.
ROLE_SEEDS = {
    "finisher": [  # st, cf
        Instinct("box_instinct", _proto(progression=0.85, spatial_density=0.6),
                 {"shoot": 0.55, "dribble": 0.15, "pass_forward": 0.15, "pass_safe": 0.15}),
        Instinct("link_play", _proto(progression=0.55, pressure=0.6),
                 {"shoot": 0.10, "dribble": 0.15, "pass_forward": 0.35, "pass_safe": 0.40}),
    ],
    "wide_creator": [  # lw, rw, lm, rm, am, cam
        Instinct("take_on", _proto(progression=0.7, spatial_density=0.4, width=0.7),
                 {"shoot": 0.12, "dribble": 0.38, "pass_forward": 0.20,
                  "cross": 0.18, "pass_safe": 0.12}),
        Instinct("whip_it_in", _proto(progression=0.85, spatial_density=0.5, width=0.9),
                 {"shoot": 0.08, "dribble": 0.12, "pass_forward": 0.12,
                  "cross": 0.55, "pass_safe": 0.13}),
        Instinct("recycle_under_pressure", _proto(pressure=0.8, spatial_density=0.7),
                 {"shoot": 0.05, "dribble": 0.18, "pass_forward": 0.20,
                  "cross": 0.12, "pass_safe": 0.45}),
    ],
    "midfield_organizer": [  # cm, dm, cdm variants
        Instinct("progress_in_space", _proto(pressure=0.25, support=0.7),
                 {"shoot": 0.05, "dribble": 0.15, "pass_forward": 0.55, "pass_safe": 0.25}),
        Instinct("keep_it_simple", _proto(pressure=0.8, spatial_density=0.7),
                 {"shoot": 0.02, "dribble": 0.08, "pass_forward": 0.20, "pass_safe": 0.70}),
    ],
    "defender": [  # cb, lb, rb, wb variants, gk
        Instinct("safety_first", _proto(pressure=0.6, progression=0.25),
                 {"shoot": 0.01, "dribble": 0.09, "pass_forward": 0.30, "pass_safe": 0.60}),
        Instinct("step_out_in_space", _proto(pressure=0.15, support=0.7),
                 {"shoot": 0.02, "dribble": 0.18, "pass_forward": 0.45, "pass_safe": 0.35}),
    ],
}

_ROLE_GROUP = {
    "st": "finisher", "cf": "finisher",
    "lw": "wide_creator", "rw": "wide_creator", "lm": "wide_creator",
    "rm": "wide_creator", "am": "wide_creator", "cam": "wide_creator",
    "cm": "midfield_organizer", "cm_l": "midfield_organizer",
    "cm_r": "midfield_organizer", "dm": "midfield_organizer",
    "cdm": "midfield_organizer",
}


def default_bank_for(player: Player) -> InstinctBank:
    """Seed a bank from the player's role, shaped by aggression:
    aggressive players' comfort actions skew bold, timid ones' skew safe."""
    group = _ROLE_GROUP.get(player.role.lower(), "defender")
    return InstinctBank([
        Instinct(seed.name, _at_schooling_load(seed.prototype),
                 aggression_tilted(seed.action_weights, player.aggression),
                 mass=seed.mass, source="role",
                 widths=(SCHOOLING_WIDTH,) * DIMENSIONS,
                 repetitions=seed.repetitions, spacing=seed.spacing, age=seed.age,
                 birth_width=SCHOOLING_WIDTH)
        for seed in ROLE_SEEDS[group]
    ], elasticity=player.elasticity)
