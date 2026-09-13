"""
Instinct bank: a player's fast, automatic action preferences (System 1),
queried by situation similarity, written by experience
(psychological-engine.md sections 3.5-3.7).

Banks start SEEDED from role and attributes ("comfort actions" - what
this kind of player reaches for without thinking). Outcomes then write
into them: successes form success anchors that reinforce an action in
similar situations, failures form trauma entries that suppress it.
Confidence decides which memory class dominates retrieval (section 5.3):
confident players sample their success anchors, rattled players feel
their traumas and retreat to the safe ball. The action vocabulary matches
the decision layer: shoot, dribble, pass_forward, pass_safe.
"""
from dataclasses import dataclass
from typing import Dict, List

from models import Player
from situation import SituationEmbedding, similarity

BOLD_ACTIONS = ("shoot", "dribble", "pass_forward", "cross")
SAFE_ACTIONS = ("pass_safe",)
ALL_ACTIONS = BOLD_ACTIONS + SAFE_ACTIONS

# How strongly confidence tilts instinct sampling between bold comfort
# actions and the safe default (design doc section 5.3).
CONFIDENCE_TILT = 0.35

# Source-level sampling: how strongly confidence amplifies success anchors
# and how strongly being rattled amplifies trauma (section 5.3).
ANCHOR_CONFIDENCE_GAIN = 0.5
TRAUMA_RATTLED_GAIN = 0.5

# Learning shape: new memories merge into a sufficiently similar existing
# memory of the same action and kind instead of piling up duplicates.
# NOTE: mean-absolute similarity compresses toward 1 as embedding
# dimensions grow - retune this when SituationEmbedding gains dimensions.
MEMORY_MERGE_SIMILARITY = 0.8
PROTOTYPE_BLEND = 0.2          # how far a merge moves the prototype
MAX_LEARNED_MEMORIES = 12      # per bank; weakest dropped beyond this
MIN_MEMORY_STRENGTH = 0.05     # decayed below this = forgotten

LEARNED_SOURCES = ("experience", "trauma")


@dataclass
class Instinct:
    """One situation prototype -> action preference mapping."""
    name: str
    prototype: SituationEmbedding
    action_weights: Dict[str, float]
    strength: float = 1.0  # How ingrained (0-1); seeds are fully ingrained
    source: str = "role"   # "role" | "experience" | "trauma"


def _source_factor(source: str, confidence: float) -> float:
    """Confidence-dependent sampling weight per memory class."""
    if source == "experience":  # success anchors: confident players lean in
        return 1.0 + ANCHOR_CONFIDENCE_GAIN * max(0.0, confidence)
    if source == "trauma":      # trauma: rattled players feel it more
        return 1.0 + TRAUMA_RATTLED_GAIN * max(0.0, -confidence)
    return 1.0


def _blend_prototypes(old: SituationEmbedding,
                      new: SituationEmbedding) -> SituationEmbedding:
    keep = 1.0 - PROTOTYPE_BLEND
    return SituationEmbedding(*(keep * a + PROTOTYPE_BLEND * b
                                for a, b in zip(old.as_tuple(), new.as_tuple())))


class InstinctBank:
    """Similarity-queried store of a player's automatic responses."""

    def __init__(self, instincts: List[Instinct]):
        self.instincts = instincts

    # -- retrieval (System 1) ------------------------------------------------

    def query(self, situation: SituationEmbedding,
              confidence: float = 0.0) -> Dict[str, float]:
        """Similarity-weighted action preferences for this situation.

        Confidence acts twice: it selects between memory classes (success
        anchors vs trauma) and tilts the final mix bold vs safe. Returns
        weights normalized to sum 1 (uniform if the bank is empty).
        """
        weights = {action: 0.0 for action in ALL_ACTIONS}

        for instinct in self.instincts:
            match = (similarity(situation, instinct.prototype)
                     * instinct.strength
                     * _source_factor(instinct.source, confidence))
            for action, weight in instinct.action_weights.items():
                weights[action] = weights.get(action, 0.0) + match * weight

        # Confidence tilt
        for action in list(weights):
            if action in BOLD_ACTIONS:
                weights[action] *= 1.0 + CONFIDENCE_TILT * confidence
            else:
                weights[action] *= 1.0 - CONFIDENCE_TILT * confidence
            weights[action] = max(0.0, weights[action])

        total = sum(weights.values())
        if total <= 0:
            return {action: 1.0 / len(ALL_ACTIONS) for action in ALL_ACTIONS}
        return {action: weight / total for action, weight in weights.items()}

    # -- learning (written by outcomes) ---------------------------------------

    def learn(self, situation: SituationEmbedding, action: str,
              valence: float, significance: float) -> None:
        """Record an outcome: positive valence reinforces `action` in
        similar situations (success anchor), negative suppresses it
        (trauma). Similar memories merge instead of duplicating."""
        if valence == 0 or action not in ALL_ACTIONS:
            return

        source = "experience" if valence > 0 else "trauma"
        gained = min(0.6, abs(valence) * significance)

        merged = self._merge_into_existing(situation, action, source, gained)
        if not merged:
            weight = 1.0 if valence > 0 else -1.0
            self.instincts.append(Instinct(
                name=f"{source}_{action}",
                prototype=situation,
                action_weights={action: weight},
                strength=min(0.6, 0.15 + gained),
                source=source,
            ))
        self._prune()

    def _merge_into_existing(self, situation: SituationEmbedding, action: str,
                             source: str, gained: float) -> bool:
        for instinct in self.instincts:
            if (instinct.source == source
                    and action in instinct.action_weights
                    and similarity(situation, instinct.prototype) >= MEMORY_MERGE_SIMILARITY):
                instinct.strength = min(1.0, instinct.strength + gained * 0.5)
                instinct.prototype = _blend_prototypes(instinct.prototype, situation)
                return True
        return False

    def _prune(self) -> None:
        learned = [i for i in self.instincts if i.source in LEARNED_SOURCES]
        if len(learned) <= MAX_LEARNED_MEMORIES:
            return
        weakest = min(learned, key=lambda i: i.strength)
        self.instincts.remove(weakest)

    def decay(self, factor: float) -> None:
        """Fade learned memories (seeds don't fade); forget the negligible."""
        for instinct in self.instincts:
            if instinct.source in LEARNED_SOURCES:
                instinct.strength *= factor
        self.instincts = [i for i in self.instincts
                          if i.source not in LEARNED_SOURCES
                          or i.strength >= MIN_MEMORY_STRENGTH]

    def learned_memories(self) -> List[Instinct]:
        return [i for i in self.instincts if i.source in LEARNED_SOURCES]


# ---------------------------------------------------------------------------
# Seeding: role comfort actions, shaped by attributes
# ---------------------------------------------------------------------------

def _proto(pressure=0.5, progression=0.5, time_criticality=0.5,
           spatial_density=0.5, support=0.5, width=0.5) -> SituationEmbedding:
    return SituationEmbedding(pressure, progression, time_criticality,
                              spatial_density, support, width)


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
    aggression_tilt = (player.aggression - 50) / 100.0  # -0.5 .. +0.5

    instincts = []
    for seed in ROLE_SEEDS[group]:
        weights = {}
        for action, weight in seed.action_weights.items():
            if action in BOLD_ACTIONS:
                weight *= 1.0 + aggression_tilt * 0.6
            else:
                weight *= 1.0 - aggression_tilt * 0.6
            weights[action] = max(0.01, weight)
        instincts.append(Instinct(seed.name, seed.prototype, weights,
                                  strength=seed.strength, source="role"))

    return InstinctBank(instincts)
