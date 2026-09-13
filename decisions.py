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
from instincts import ALL_ACTIONS, SAFE_ACTIONS, aggression_tilted
from minds import MindRegistry
import psychology

# Cognitive load is a core mechanism: an unfamiliar situation costs
# capacity on top of match pressure. Familiarity (instincts.familiarity -
# role schooling plus lived experience) buys relief: veterans of fifty
# big nights process the moment cheaply, debutants pay full price.
NOVELTY_LOAD = 0.3

# What System 1 offers when the bank has nothing for this situation:
# the safe ball, mostly - but an untrained instinct is still THAT
# player's instinct, so aggression tilts even the fallback (a reckless
# player's panic is rasher than a cautious one's).
SAFE_FALLBACK = {
    action: (0.55 if action in SAFE_ACTIONS
             else 0.45 / (len(ALL_ACTIONS) - len(SAFE_ACTIONS)))
    for action in ALL_ACTIONS
}


def _fallback_for(player: Player) -> dict:
    tilted = aggression_tilted(SAFE_FALLBACK, player.aggression)
    total = sum(tilted.values())
    return {action: weight / total for action, weight in tilted.items()}


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
    box_targets: int = 0  # teammates in the box a cross could find


class DecisionModel(Protocol):
    """Chooses one of ALL_ACTIONS for the ball holder."""

    def decide(self, context: DecisionContext) -> str:
        ...


def tiered_bonus(value: float, tiers: Tuple[Tuple[float, float], ...]) -> float:
    """First-match threshold table: tiers are (min_value, bonus) pairs in
    descending order. Replaces stacked if/elif bonus chains."""
    for threshold, bonus in tiers:
        if value > threshold:
            return bonus
    return 0.0


# Dribble utility inputs as threshold tables
SPACE_AHEAD_TIERS = ((20.0, 0.20), (10.0, 0.10))
DRIBBLE_SKILL_TIERS = ((75.0, 0.15), (60.0, 0.08))
DEFENDER_DISTANCE_TIERS = ((15.0, 0.10),)


class DualProcessDecisionModel:
    """System 1/2 blended action selection."""

    def __init__(self, rng: Optional[random.Random] = None,
                 minds: Optional[MindRegistry] = None):
        self.rng = rng or random.Random()
        # Shared with the learning layer: decisions read the banks that
        # outcomes write (learning.py)
        # `is not None`, not `or`: an empty registry is falsy (len 0) but
        # still the caller's shared substrate
        self.minds = minds if minds is not None else MindRegistry()

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
        dribble = (0.12
                   + tiered_bonus(ctx.space_ahead, SPACE_AHEAD_TIERS)
                   + tiered_bonus(dribbling, DRIBBLE_SKILL_TIERS)
                   + tiered_bonus(ctx.nearest_defender_dist, DEFENDER_DISTANCE_TIERS)
                   + confidence * 0.08)

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

        # CROSS: continuous in the situation vector - the wider and more
        # advanced the holder, the more the delivery suggests itself;
        # vision governs spotting the runs. Needs someone to aim at.
        cross = 0.0
        if ctx.box_targets > 0:
            situation = ctx.situation
            cross = (situation.width * situation.progression * 0.9
                     + 0.06 * min(3, ctx.box_targets)) * vision_factor
            cross += confidence * 0.05

        # Personality warp: aggression trades safety for boldness
        utilities = {
            "shoot": shoot * (1.0 + aggression_tilt * 0.5),
            "dribble": dribble * (1.0 + aggression_tilt * 0.4),
            "pass_forward": pass_forward * (1.0 + aggression_tilt * 0.3),
            "cross": cross * (1.0 + aggression_tilt * 0.2),
            "pass_safe": pass_safe * (1.0 - aggression_tilt * 0.4),
        }
        return {a: max(0.0, u) for a, u in utilities.items()}

    # -- blend and select ----------------------------------------------------

    def decide(self, context: DecisionContext) -> str:
        holder = context.holder
        mind = self.minds.mind_for(holder)

        # One pass over the bank yields both what System 1 has to say and
        # how well the player RECOGNIZES this situation (pre-exposure).
        instinct_raw, familiarity = mind.bank.recall(
            context.situation, confidence=holder.confidence)

        # Cognitive load: match pressure (which already carries the
        # environment through sensitivity) plus the cost of novelty.
        # Pre-exposure is load relief; the same lights weigh less the
        # fiftieth time.
        load = min(1.0, context.situation.pressure
                   + NOVELTY_LOAD * (1.0 - familiarity))

        # System weights: load vs effective composure
        sys1 = psychology.system1_weight(load, holder)

        analysis = _normalize(self.utilities(context))

        # System 1 can only offer what pre-exposure put there: in novel
        # territory the instinct flattens toward the safe default -
        # high load with no familiar patterns is the debutant freeze,
        # not sudden boldness.
        fallback = _fallback_for(holder)
        instinct = {
            action: familiarity * instinct_raw.get(action, 0.0)
                    + (1.0 - familiarity) * fallback[action]
            for action in ALL_ACTIONS
        }

        # Overload degrades System 1 itself: load beyond what composure
        # absorbs blurs even trained automatisms toward indiscriminate
        # noise. (Novelty above = instinct has nothing to say; overload
        # here = instinct can no longer say it clearly.)
        integrity = psychology.system1_integrity(load, holder)
        uniform = 1.0 / len(ALL_ACTIONS)
        instinct = {
            action: integrity * weight + (1.0 - integrity) * uniform
            for action, weight in instinct.items()
        }

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
        if context.box_targets == 0:
            blended["cross"] = 0.0  # nobody to aim at

        action = _weighted_choice(blended, self.rng, fallback="dribble")

        # Record for the learning layer: the outcome event that follows
        # will be paired with this decision (learning.py)
        mind.remember_decision(context.situation, action)
        return action


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
