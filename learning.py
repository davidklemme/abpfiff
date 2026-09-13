"""
Experience learning: match outcomes write into players' instinct banks
(psychological-engine.md sections 3.6-3.7 and 5.4, Phase 3 slice).

The loop: the decision layer records each on-ball decision (situation +
chosen action) as the player's pending decision; when the outcome event
arrives, this module pairs them and writes a memory - a success anchor
that reinforces the action in similar situations, or a trauma that
suppresses it. System 1 then retrieves what was learned; confidence
selects between the memory classes at query time (instincts.py).

Outcome mapping is table-driven: one row per observable event type.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from models import MatchEvent, MatchState
from minds import MindRegistry

# One row per event type the engine can attribute to a prior decision:
#   who         - which event field names the player whose decision resolved
#   valence     - direction and magnitude of the lesson (-1..1)
#   significance- how strongly it imprints (0..1)
@dataclass(frozen=True)
class OutcomeRule:
    who: str          # "player" | "target_player"
    valence: float
    significance: float


OUTCOME_RULES: Dict[str, OutcomeRule] = {
    "goal":          OutcomeRule("player", +1.0, 0.9),
    "dribble":       OutcomeRule("player", +0.4, 0.25),
    "pass_received": OutcomeRule("target_player", +0.5, 0.3),   # the passer
    "miss":          OutcomeRule("player", -0.5, 0.5),
    "save":          OutcomeRule("target_player", -0.3, 0.4),   # the shooter
    "interception":  OutcomeRule("target_player", -0.7, 0.5),   # the passer
    "turnover":      OutcomeRule("player", -0.6, 0.4),
    "tackle":        OutcomeRule("target_player", -0.6, 0.4),   # the dribbler
}

# Learned memories fade slowly; applied once per simulated minute.
DECAY_PER_MINUTE = 0.999


class ExperienceLearning:
    """Pairs pending decisions with their outcomes and writes memories."""

    def __init__(self, minds: MindRegistry,
                 rules: Optional[Dict[str, OutcomeRule]] = None):
        self.minds = minds
        self.rules = rules if rules is not None else OUTCOME_RULES

    def on_event(self, event: MatchEvent, state: MatchState) -> None:
        """Learn from an outcome event, if it resolves a pending decision."""
        rule = self.rules.get(event.event_type)
        if rule is None:
            return

        player = getattr(event, rule.who, None)
        if player is None:
            return

        mind = self.minds.mind_for(player)
        pending = mind.take_pending()
        if pending is None:
            return

        situation, action = pending
        mind.bank.learn(situation, action, rule.valence, rule.significance)

    def decay(self, minutes_elapsed: float) -> None:
        """Fade all learned memories with time."""
        factor = DECAY_PER_MINUTE ** minutes_elapsed
        for mind in self.minds:
            mind.bank.decay(factor)
