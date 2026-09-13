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
# 0.8 = a total distance budget of 1.2 across the embedding
# (situation.SIMILARITY_SCALE is fixed, so this threshold no longer
# needs retuning when SituationEmbedding gains dimensions).
MEMORY_MERGE_SIMILARITY = 0.8

# Role schooling is acquired on the training pitch - quieter than any
# real match - so its prototypes sit at near-zero occasion-load.
# Big-night situations are therefore DISSIMILAR to pure schooling in the
# load dimension: the debutant effect emerges from geometry, and only
# lived high-load memories close the gap. Keep this near zero: it is a
# flat familiarity tax on every ordinary-environment situation.
SCHOOLING_LOAD = 0.05

# Recognition ("I have LIVED this moment") is a sharper judgment than
# response retrieval: familiarity uses a squared similarity kernel (so
# the dimension-compressed mean-absolute metric regains discrimination)
# and discounts generic role schooling - a textbook covers everything
# loosely, lived memories cover their situations exactly.
ROLE_SCHOOLING_FAMILIARITY = 0.6
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


def aggression_tilted(weights: Dict[str, float],
                      aggression: float) -> Dict[str, float]:
    """The one place personality tilts an action-weight map bold/safe:
    used for seeding banks AND for the untrained-instinct fallback."""
    tilt = (aggression - 50) / 100.0  # -0.5 .. +0.5
    return {
        action: max(0.01, weight * (1.0 + tilt * 0.6 if action in BOLD_ACTIONS
                                    else 1.0 - tilt * 0.6))
        for action, weight in weights.items()
    }


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

    def recall(self, situation: SituationEmbedding,
               confidence: float = 0.0):
        """One pass over the bank: (action weights, familiarity).

        The weights are similarity-weighted action preferences (System 1
        content); familiarity is recognition - how well this player KNOWS
        the situation. Both derive from the same prototype comparisons,
        computed once (this runs on every on-ball decision).

        Confidence acts twice on the weights: it selects between memory
        classes (success anchors vs trauma) and tilts the mix bold/safe.
        Familiarity uses a squared kernel and discounts role schooling:
        pre-exposure - lived memories matching this moment - is what
        makes a veteran of fifty big nights recognize the fifty-first.
        """
        weights = {action: 0.0 for action in ALL_ACTIONS}
        familiarity = 0.0

        for instinct in self.instincts:
            sim = similarity(situation, instinct.prototype)

            match = sim * instinct.strength * _source_factor(instinct.source,
                                                             confidence)
            for action, weight in instinct.action_weights.items():
                weights[action] = weights.get(action, 0.0) + match * weight

            recognition = sim * sim * instinct.strength
            if instinct.source == "role":
                recognition *= ROLE_SCHOOLING_FAMILIARITY
            familiarity = max(familiarity, recognition)

        # Confidence tilt
        for action in list(weights):
            if action in BOLD_ACTIONS:
                weights[action] *= 1.0 + CONFIDENCE_TILT * confidence
            else:
                weights[action] *= 1.0 - CONFIDENCE_TILT * confidence
            weights[action] = max(0.0, weights[action])

        total = sum(weights.values())
        if total <= 0:
            weights = {action: 1.0 / len(ALL_ACTIONS) for action in ALL_ACTIONS}
        else:
            weights = {action: weight / total
                       for action, weight in weights.items()}
        return weights, min(1.0, familiarity)

    def query(self, situation: SituationEmbedding,
              confidence: float = 0.0) -> Dict[str, float]:
        """Action preferences only (see recall)."""
        weights, _ = self.recall(situation, confidence)
        return weights

    def familiarity(self, situation: SituationEmbedding) -> float:
        """Recognition only (see recall)."""
        _, familiarity = self.recall(situation)
        return familiarity

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
           spatial_density=0.5, support=0.5, width=0.5,
           load=SCHOOLING_LOAD) -> SituationEmbedding:
    return SituationEmbedding(pressure, progression, time_criticality,
                              spatial_density, support, width, load)


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
        Instinct(seed.name, seed.prototype,
                 aggression_tilted(seed.action_weights, player.aggression),
                 strength=seed.strength, source="role")
        for seed in ROLE_SEEDS[group]
    ])
