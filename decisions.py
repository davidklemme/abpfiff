"""
Decision layer: what the player on the ball chooses to do.

Implements the dual-process model from psychological-engine.md section
5.1 (Phase 2 slice) behind a DecisionModel Protocol, so the resolver's
decision-making is swappable:

  System 2 (analysis)  - utility scores per action from the tactical
                         situation, warped by personality (aggression,
                         vision) and confidence
  System 1 (instinct)  - the player's InstinctBank queried by situation
                         similarity, tilted by confidence
  Blend                - psychology.system1_weight: pressure against
                         effective composure decides how much instinct
                         overrides analysis

Action vocabulary: shoot | dribble | pass_forward | pass_safe.
The resolver maps pass_forward / pass_safe onto pass-target scoring bias.
"""
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from models import MatchState, Player, Team
from situation import SituationEmbedding
from instincts import InstinctBank, ALL_ACTIONS, default_bank_for
import psychology


@dataclass
class DecisionContext:
    """Everything the decision layer knows about the moment."""
    holder: Player
    attacking_team: Team
    defending_team: Team
    state: MatchState
    situation: SituationEmbedding
    in_shooting_range: bool
    space_ahead: float
    nearest_defender_dist: float
    lanes: List[Tuple[Player, float]]
    best_forward_lane: float   # quality of best forward option (0 if none)
    best_safe_lane: float      # quality of best lateral/backward option


class DecisionModel(Protocol):
    """Chooses one of ALL_ACTIONS for the ball holder."""

    def decide(self, context: DecisionContext) -> str:
        ...


class DualProcessDecisionModel:
    """System 1/2 blended action selection."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self._banks: Dict[int, InstinctBank] = {}

    def _bank_for(self, player: Player) -> InstinctBank:
        key = id(player)
        bank = self._banks.get(key)
        if bank is None:
            bank = default_bank_for(player)
            self._banks[key] = bank
        return bank

    # -- System 2: deliberate utility scoring --------------------------------

    def utilities(self, ctx: DecisionContext) -> Dict[str, float]:
        """Tactical utility per action, personality-warped.

        The numbers port the tuning of the old probability tree so the
        aggregate action mix stays calibrated; the structure (explicit
        per-action utilities) is what Phase 2 adds.
        """
        holder = ctx.holder
        dribbling = holder.effective_attribute('dribbling')
        confidence = holder.confidence
        aggression_tilt = (holder.aggression - 50) / 100.0  # -0.5 .. +0.5
        vision_factor = 0.5 + holder.effective_attribute('vision') / 200.0  # ~0.55-1.0

        # SHOOT: only a real option in range
        shoot = 0.0
        if ctx.in_shooting_range:
            shoot = 0.42
            if 30 < holder.position.x < 70:
                shoot += 0.18
            if holder.effective_attribute('composure') > 70:
                shoot += 0.10
            # Confidence pulls the trigger - the strongest single state
            # effect on shooting (a rattled striker snatches at chances)
            shoot += confidence * 0.30

        # DRIBBLE: space and skill make carrying attractive
        dribble = 0.12
        if ctx.space_ahead > 20:
            dribble += 0.20
        elif ctx.space_ahead > 10:
            dribble += 0.10
        if dribbling > 75:
            dribble += 0.15
        elif dribbling > 60:
            dribble += 0.08
        if ctx.nearest_defender_dist > 15:
            dribble += 0.10
        dribble += confidence * 0.08

        # PASS FORWARD: needs a forward option worth playing; vision governs
        # how well the player perceives it
        directness = (ctx.attacking_team.tactics.directness
                      if ctx.attacking_team.tactics else 50)
        pass_forward = (0.15 + ctx.best_forward_lane * 0.9) * vision_factor
        pass_forward *= 1.0 + (directness - 50) / 150.0
        pass_forward += confidence * 0.05

        # PASS SAFE: always available; grows attractive under pressure
        pass_safe = 0.22 + ctx.best_safe_lane * 0.5
        pass_safe += ctx.situation.pressure * 0.25

        # Personality warp: aggression trades safety for boldness
        utilities = {
            "shoot": shoot * (1.0 + aggression_tilt * 0.5),
            "dribble": dribble * (1.0 + aggression_tilt * 0.4),
            "pass_forward": pass_forward * (1.0 + aggression_tilt * 0.3),
            "pass_safe": pass_safe * (1.0 - aggression_tilt * 0.4),
        }
        return {a: max(0.0, u) for a, u in utilities.items()}

    # -- blend and select ----------------------------------------------------

    def decide(self, context: DecisionContext) -> str:
        holder = context.holder

        # System weights: pressure vs effective composure
        sys1 = psychology.system1_weight(context.situation.pressure, holder)

        analysis = _normalize(self.utilities(context))
        instinct = self._bank_for(holder).query(context.situation,
                                                confidence=holder.confidence)

        blended = {
            action: (1.0 - sys1) * analysis.get(action, 0.0)
                    + sys1 * instinct.get(action, 0.0)
            for action in ALL_ACTIONS
        }

        # Impossible actions stay impossible whatever instinct says
        if not context.in_shooting_range:
            blended["shoot"] = 0.0
        if not context.lanes:
            blended["pass_forward"] = 0.0
            blended["pass_safe"] = 0.0

        return _weighted_choice(blended, self.rng, fallback="dribble")


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {a: 1.0 / len(weights) for a in weights}
    return {a: w / total for a, w in weights.items()}


def _weighted_choice(weights: Dict[str, float], rng: random.Random,
                     fallback: str) -> str:
    total = sum(weights.values())
    if total <= 0:
        return fallback
    r = rng.random() * total
    cumulative = 0.0
    for action, weight in weights.items():
        cumulative += weight
        if r <= cumulative:
            return action
    return fallback
