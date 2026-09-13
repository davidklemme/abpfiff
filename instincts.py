"""
Instinct bank: a player's fast, automatic action preferences (System 1),
queried by situation similarity (psychological-engine.md section 3.5,
Phase 2 slice).

Phase 2 banks are SEEDED from role and attributes ("comfort actions" -
what this kind of player reaches for without thinking); Phase 3 will make
them learned from experience (success anchors, trauma). The action
vocabulary matches the decision layer: shoot, dribble, pass_forward,
pass_safe.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from models import Player
from situation import SituationEmbedding, similarity

BOLD_ACTIONS = ("shoot", "dribble", "pass_forward")
SAFE_ACTIONS = ("pass_safe",)
ALL_ACTIONS = BOLD_ACTIONS + SAFE_ACTIONS

# How strongly confidence tilts instinct sampling between bold comfort
# actions and the safe default (section 5.3 of the design doc, simplified:
# no learned anchors yet, so confidence tilts the seeded weights).
CONFIDENCE_TILT = 0.35


@dataclass
class Instinct:
    """One situation prototype -> action preference mapping."""
    name: str
    prototype: SituationEmbedding
    action_weights: Dict[str, float]
    strength: float = 1.0  # How ingrained (0-1); seeds are fully ingrained
    source: str = "role"   # "role" now; "experience"/"mentor" in Phase 3


class InstinctBank:
    """Similarity-queried store of a player's automatic responses."""

    def __init__(self, instincts: List[Instinct]):
        self.instincts = instincts

    def query(self, situation: SituationEmbedding,
              confidence: float = 0.0) -> Dict[str, float]:
        """Similarity-weighted action preferences for this situation.

        Confidence tilts the result: a confident player's instinct reaches
        for bold comfort actions, a rattled player's for the safe ball.
        Returns weights normalized to sum 1 (uniform if the bank is empty).
        """
        weights = {action: 0.0 for action in ALL_ACTIONS}

        for instinct in self.instincts:
            match = similarity(situation, instinct.prototype) * instinct.strength
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


# ---------------------------------------------------------------------------
# Seeding: role comfort actions, shaped by attributes
# ---------------------------------------------------------------------------

def _proto(pressure=0.5, progression=0.5, time_criticality=0.5,
           spatial_density=0.5, support=0.5) -> SituationEmbedding:
    return SituationEmbedding(pressure, progression, time_criticality,
                              spatial_density, support)


# Per role-group: (situation prototype, action weights) comfort patterns.
ROLE_SEEDS = {
    "finisher": [  # st, cf
        Instinct("box_instinct", _proto(progression=0.85, spatial_density=0.6),
                 {"shoot": 0.55, "dribble": 0.15, "pass_forward": 0.15, "pass_safe": 0.15}),
        Instinct("link_play", _proto(progression=0.55, pressure=0.6),
                 {"shoot": 0.10, "dribble": 0.15, "pass_forward": 0.35, "pass_safe": 0.40}),
    ],
    "wide_creator": [  # lw, rw, lm, rm, am, cam
        Instinct("take_on", _proto(progression=0.7, spatial_density=0.4),
                 {"shoot": 0.15, "dribble": 0.45, "pass_forward": 0.25, "pass_safe": 0.15}),
        Instinct("recycle_under_pressure", _proto(pressure=0.8, spatial_density=0.7),
                 {"shoot": 0.05, "dribble": 0.20, "pass_forward": 0.25, "pass_safe": 0.50}),
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
