"""
Execution quality: how well a player performs a technical action right now.

Every contested action in the engine (pass, first touch, interception,
dribble duel, shot, save) is resolved from the same factor stack instead
of flat per-action constants:

    skill       - the relevant attribute, already fatigue-adjusted
                  (Player.effective_attribute)
    pressure    - spatial/tactical pressure blended against composure,
                  confidence and fatigue (psychology.system1_weight):
                  players leaning on instinct execute worse
    confidence  - psychological momentum of the individual (-1..1)
    momentum    - momentum of the team (0..100), a smaller ambient effect

This is the source of outcome variance by design: two players in the same
spot with the same attribute diverge because of their physical and mental
state, and the same player diverges across the match as those states move.
"""
from models import MatchState, Player, Team
import psychology

# Factor weights: how strongly each state factor bends execution.
PRESSURE_PENALTY = 0.30      # at full instinct-mode, lose up to 30%
CONFIDENCE_SWING = 0.12      # +/- 12% across the confidence range
MOMENTUM_SWING = 0.06        # +/- 6% across the team momentum range

QUALITY_FLOOR = 0.05
QUALITY_CEIL = 1.0


def execution_quality(player: Player, attribute: str, state: MatchState,
                      opponents: Team, momentum: float = None) -> float:
    """0-1 quality of executing `attribute` in the current situation."""
    # Skill, degraded by physical fatigue
    quality = player.effective_attribute(attribute) / 100.0

    # Pressure vs. composure/confidence/fatigue: instinct-dominant execution
    # is worse execution
    pressure = psychology.calculate_pressure(player, state, opponents)
    sys1 = psychology.system1_weight(pressure.total, player)
    quality *= 1.0 - PRESSURE_PENALTY * sys1

    # Individual psychological momentum
    quality *= 1.0 + CONFIDENCE_SWING * player.confidence

    # Team momentum (0-100, 50 = neutral)
    if momentum is not None:
        quality *= 1.0 + MOMENTUM_SWING * (momentum - 50.0) / 50.0

    return max(QUALITY_FLOOR, min(QUALITY_CEIL, quality))


def contest(quality_a: float, quality_b: float) -> float:
    """Probability that side A wins a direct duel of two qualities."""
    total = quality_a + quality_b
    if total <= 0:
        return 0.5
    return quality_a / total
