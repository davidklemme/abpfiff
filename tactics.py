"""
Tactical principles engine.
Defines how teams behave based on tactical instructions.
This is where Guardiola, Klopp, Flick etc. styles emerge from.
"""
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict
from enum import Enum
from models import Player, Team, Position, Phase, MatchState, Ball
import random
import math


class TriggerType(Enum):
    """When does a tactical principle activate?"""
    IN_POSSESSION = "in_possession"
    OUT_OF_POSSESSION = "out_of_possession"
    BALL_LOST = "ball_lost"
    BALL_WON = "ball_won"
    BUILDUP = "buildup"
    ATTACKING_THIRD = "attacking_third"
    DEFENDING_THIRD = "defending_third"
    ALWAYS = "always"


@dataclass
class MovementInstruction:
    """Where should a player move?

    All y values (target_y, relative_y) are expressed in the team's attacking
    frame: the team always attacks toward y=100. The engine converts to
    absolute pitch coordinates via Team.frame_y, so the same instruction
    works for both the home and the away team.
    """
    target_x: Optional[float] = None  # Absolute or None for relative
    target_y: Optional[float] = None  # In team attacking frame
    relative_x: float = 0  # Relative to current position
    relative_y: float = 0  # In team attacking frame (positive = forward)
    towards_ball: float = 0  # 0-1, how much to move toward ball
    towards_space: float = 0  # 0-1, how much to find open space
    maintain_shape: float = 0  # 0-1, how much to maintain base position
    # Signed lateral movement relative to the player's own side of the pitch:
    # positive = tuck toward the center (x=50), negative = push toward the
    # nearer touchline. Replaces per-side relative_x duplication.
    towards_center_x: float = 0


@dataclass
class TacticalPrinciple:
    """
    A single tactical instruction that can activate.
    The building blocks of a tactical system.
    """
    name: str
    trigger: TriggerType
    applies_to_roles: List[str]  # Which roles this affects
    movement: MovementInstruction
    priority: int = 5  # Higher = more important (1-10)

    # Conditions for activation
    ball_zone_y_min: float = 0  # Ball must be in this y range
    ball_zone_y_max: float = 100
    team_has_ball: Optional[bool] = None  # None = don't care

    # Requirements
    min_stamina: float = 0
    required_attributes: Dict[str, int] = field(default_factory=dict)

    def check_conditions(self, player: Player, state: MatchState, team_attacking: bool,
                         team: Optional[Team] = None) -> bool:
        """Check if this principle should activate for this player.

        Ball-zone conditions are expressed in the team's attacking frame
        (y=100 is always the opponent goal); pass `team` so away-team
        principles evaluate against the correct half of the pitch.
        """
        # Check role
        if self.applies_to_roles and player.role not in self.applies_to_roles:
            return False

        # Ball position in the team's attacking frame
        ball_y = state.ball.position.y
        if team is not None:
            ball_y = team.frame_y(ball_y)

        # Check trigger type
        if not self._check_trigger(state, team_attacking, ball_y):
            return False

        # Check ball position
        if not (self.ball_zone_y_min <= ball_y <= self.ball_zone_y_max):
            return False

        # Check possession
        if self.team_has_ball is not None:
            if self.team_has_ball != team_attacking:
                return False

        # Check stamina
        if player.fatigue > (100 - self.min_stamina):
            return False

        # Check attributes
        for attr, min_val in self.required_attributes.items():
            if player.effective_attribute(attr) < min_val:
                return False

        return True

    def _check_trigger(self, state: MatchState, team_attacking: bool,
                       ball_y: float) -> bool:
        """Check if the trigger condition is met.

        `ball_y` is the ball's y in the team's attacking frame."""
        if self.trigger == TriggerType.ALWAYS:
            return True
        elif self.trigger == TriggerType.IN_POSSESSION:
            return team_attacking
        elif self.trigger == TriggerType.OUT_OF_POSSESSION:
            return not team_attacking
        elif self.trigger == TriggerType.BALL_LOST:
            # Check if we just lost the ball (within last few ticks)
            return not team_attacking and state.ticks_since_possession_change < 6
        elif self.trigger == TriggerType.BALL_WON:
            # Check if we just won the ball
            return team_attacking and state.ticks_since_possession_change < 6
        elif self.trigger == TriggerType.BUILDUP:
            # In possession, ball in own half
            return team_attacking and ball_y < 50
        elif self.trigger == TriggerType.ATTACKING_THIRD:
            # In possession, ball in attacking third
            return team_attacking and ball_y > 66
        elif self.trigger == TriggerType.DEFENDING_THIRD:
            # Out of possession, ball in our defensive third
            return not team_attacking and ball_y < 33
        return True


@dataclass
class TacticalSetup:
    """
    Complete tactical setup for a team.
    Combination of principles creates emergent tactical behavior.
    """
    name: str
    principles: List[TacticalPrinciple]

    # Global settings
    defensive_line_height: float = 40  # 0-100, higher = more aggressive
    pressing_intensity: float = 50  # 0-100
    pressing_trigger_zone: float = 70  # Y position where pressing activates
    counter_press_duration: float = 6  # Seconds to counter-press after losing ball
    compactness_target: float = 35  # Target vertical compactness
    width_in_possession: float = 70  # How wide to stretch
    width_out_of_possession: float = 50  # How narrow to be
    tempo: float = 50  # 0-100, affects passing speed/risk
    directness: float = 50  # 0-100, higher = more direct/vertical

    def get_active_principles(self, player: Player, state: MatchState,
                               team_attacking: bool,
                               team: Optional[Team] = None) -> List[TacticalPrinciple]:
        """Get all principles that should be active for this player"""
        active = []
        for principle in self.principles:
            if principle.check_conditions(player, state, team_attacking, team):
                active.append(principle)
        return sorted(active, key=lambda p: p.priority, reverse=True)


# =============================================================================
# PRE-BUILT TACTICAL SYSTEMS
# =============================================================================

def create_guardiola_positional_play() -> TacticalSetup:
    """
    Guardiola's positional play system.
    Key concepts: occupation of half-spaces, third-man combinations,
    patient buildup, inverted fullbacks.
    """
    principles = [
        # Buildup: GK and CBs play out
        TacticalPrinciple(
            name="play_out_from_back",
            trigger=TriggerType.BUILDUP,
            applies_to_roles=["gk", "cb", "cb_l", "cb_r"],
            movement=MovementInstruction(maintain_shape=0.4, towards_ball=0.1, towards_space=0.3),
            priority=6,
            ball_zone_y_max=30,
            team_has_ball=True
        ),

        # DM drops between CBs
        TacticalPrinciple(
            name="dm_drops_to_build",
            trigger=TriggerType.BUILDUP,
            applies_to_roles=["dm", "cdm"],
            movement=MovementInstruction(relative_y=-10, maintain_shape=0.2, towards_space=0.3),
            priority=7,
            ball_zone_y_max=35,
            team_has_ball=True,
            required_attributes={"composure": 60, "passing": 65}
        ),

        # Fullbacks invert into midfield
        TacticalPrinciple(
            name="inverted_fullback",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["lb", "rb", "lwb", "rwb"],
            movement=MovementInstruction(
                towards_center_x=15,  # Move inside, whichever side they play
                relative_y=10,
                maintain_shape=0.2
            ),
            priority=6,
            ball_zone_y_min=30,
            team_has_ball=True,
            required_attributes={"passing": 60, "positioning": 55}
        ),

        # CMs occupy half-spaces
        TacticalPrinciple(
            name="occupy_half_space",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["cm", "cm_l", "cm_r", "am"],
            movement=MovementInstruction(
                towards_space=0.6,
                relative_y=5,
                maintain_shape=0.3
            ),
            priority=7,
            ball_zone_y_min=35,
            team_has_ball=True,
            required_attributes={"positioning": 60, "first_touch": 55}
        ),

        # Wingers stay wide and high
        TacticalPrinciple(
            name="winger_width",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["lw", "rw", "lm", "rm"],
            movement=MovementInstruction(
                towards_center_x=-20,  # Push to the nearer touchline
                relative_y=5,
                maintain_shape=0.4
            ),
            priority=5,
            team_has_ball=True
        ),

        # False 9 drops to link
        TacticalPrinciple(
            name="false_nine_drop",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["st", "cf"],
            movement=MovementInstruction(
                relative_y=-15,
                towards_ball=0.3,
                towards_space=0.4
            ),
            priority=6,
            ball_zone_y_min=40,
            ball_zone_y_max=75,
            team_has_ball=True,
            required_attributes={"first_touch": 65, "passing": 60}
        ),

        # High press when losing ball
        TacticalPrinciple(
            name="immediate_press",
            trigger=TriggerType.BALL_LOST,
            applies_to_roles=["st", "cf", "lw", "rw", "am"],
            movement=MovementInstruction(towards_ball=0.7),
            priority=9,
            ball_zone_y_min=60,
            team_has_ball=False,
            min_stamina=30
        ),

        # Defensive shape
        TacticalPrinciple(
            name="compact_defense",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["cb", "cb_l", "cb_r", "lb", "rb", "dm"],
            movement=MovementInstruction(
                maintain_shape=0.4,
                towards_ball=0.4
            ),
            priority=8,
            team_has_ball=False
        ),
    ]

    return TacticalSetup(
        name="Positional Play (Guardiola)",
        principles=principles,
        defensive_line_height=55,
        pressing_intensity=70,
        pressing_trigger_zone=65,
        counter_press_duration=8,
        compactness_target=30,
        width_in_possession=75,
        width_out_of_possession=45,
        tempo=40,  # Patient
        directness=30  # Possession-based
    )


def create_gegenpressing() -> TacticalSetup:
    """
    Klopp/Rangnick Gegenpressing system.
    Key concepts: immediate counter-press, vertical transitions,
    high defensive line, intense pressing triggers.
    """
    principles = [
        # COUNTER-PRESS: The signature move
        TacticalPrinciple(
            name="gegenpress_immediate",
            trigger=TriggerType.BALL_LOST,
            applies_to_roles=["st", "cf", "lw", "rw", "am", "cm", "cm_l", "cm_r"],
            movement=MovementInstruction(towards_ball=0.9),
            priority=10,
            ball_zone_y_min=50,  # Only in opponent's half
            team_has_ball=False,
            min_stamina=40,
            required_attributes={"workrate": 60, "stamina": 60}
        ),

        # Rest defense covers
        TacticalPrinciple(
            name="rest_defense_cover",
            trigger=TriggerType.BALL_LOST,
            applies_to_roles=["cb", "cb_l", "cb_r", "dm"],
            movement=MovementInstruction(
                relative_y=-8,
                maintain_shape=0.3,
                towards_ball=0.2
            ),
            priority=8,
            team_has_ball=False
        ),

        # High pressing trigger
        TacticalPrinciple(
            name="high_press_trigger",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["st", "cf", "lw", "rw"],
            movement=MovementInstruction(towards_ball=0.7),
            priority=8,
            ball_zone_y_min=70,
            team_has_ball=False,
            min_stamina=30
        ),

        # Quick vertical progression
        TacticalPrinciple(
            name="quick_vertical",
            trigger=TriggerType.BALL_WON,
            applies_to_roles=["cm", "cm_l", "cm_r", "am"],
            movement=MovementInstruction(relative_y=15, towards_space=0.5),
            priority=9,
            team_has_ball=True
        ),

        # Wingers ready to run in behind
        TacticalPrinciple(
            name="run_in_behind",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["lw", "rw"],
            movement=MovementInstruction(
                relative_y=15,
                towards_space=0.4,
                maintain_shape=0.2
            ),
            priority=7,
            ball_zone_y_min=50,
            team_has_ball=True,
            required_attributes={"pace": 70}
        ),

        # Fullbacks overlap
        TacticalPrinciple(
            name="fullback_overlap",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["lb", "rb"],
            movement=MovementInstruction(
                relative_y=20,
                towards_center_x=-10,  # Overlap wide
                maintain_shape=0.2
            ),
            priority=6,
            ball_zone_y_min=50,
            team_has_ball=True,
            required_attributes={"stamina": 65, "pace": 60}
        ),

        # High defensive line
        TacticalPrinciple(
            name="high_line",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["cb", "cb_l", "cb_r"],
            movement=MovementInstruction(
                target_y=50,  # Push up
                maintain_shape=0.8
            ),
            priority=7,
            ball_zone_y_min=40,
            team_has_ball=False
        ),
    ]

    return TacticalSetup(
        name="Gegenpressing (Klopp/Rangnick)",
        principles=principles,
        defensive_line_height=60,
        pressing_intensity=90,
        pressing_trigger_zone=70,
        counter_press_duration=6,
        compactness_target=25,
        width_in_possession=65,
        width_out_of_possession=40,
        tempo=80,  # Fast
        directness=70  # Vertical
    )


def create_low_block_counter() -> TacticalSetup:
    """
    Deep defensive block with quick counter-attacks.
    Mourinho, Simeone style (at times).
    """
    principles = [
        # Deep defensive line
        TacticalPrinciple(
            name="deep_block",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["cb", "cb_l", "cb_r", "lb", "rb"],
            movement=MovementInstruction(
                target_y=25,
                maintain_shape=0.5,
                towards_ball=0.3
            ),
            priority=9,
            team_has_ball=False
        ),

        # Midfield shields defense
        TacticalPrinciple(
            name="midfield_shield",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["dm", "cdm", "cm", "cm_l", "cm_r"],
            movement=MovementInstruction(
                target_y=35,
                maintain_shape=0.4,
                towards_ball=0.4
            ),
            priority=8,
            team_has_ball=False
        ),

        # Forwards stay high for counter
        TacticalPrinciple(
            name="stay_high_for_counter",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["st", "cf"],
            movement=MovementInstruction(
                target_y=65,
                maintain_shape=0.3
            ),
            priority=6,
            team_has_ball=False
        ),

        # Quick transition forward
        TacticalPrinciple(
            name="counter_attack_burst",
            trigger=TriggerType.BALL_WON,
            applies_to_roles=["st", "cf", "lw", "rw"],
            movement=MovementInstruction(
                relative_y=25,
                towards_space=0.6
            ),
            priority=10,
            team_has_ball=True,
            required_attributes={"pace": 65}
        ),

        # Wingers counter wide
        TacticalPrinciple(
            name="wide_counter_run",
            trigger=TriggerType.BALL_WON,
            applies_to_roles=["lw", "rw", "lm", "rm"],
            movement=MovementInstruction(
                relative_y=20,
                towards_center_x=-15,  # Break wide on the counter
                towards_space=0.5
            ),
            priority=9,
            team_has_ball=True,
            required_attributes={"pace": 70}
        ),

        # Narrow in defense
        TacticalPrinciple(
            name="narrow_shape",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["lw", "rw", "lm", "rm"],
            movement=MovementInstruction(
                towards_center_x=10,  # Tuck in
                relative_y=-10,
                maintain_shape=0.5
            ),
            priority=7,
            team_has_ball=False
        ),
    ]

    return TacticalSetup(
        name="Low Block Counter",
        principles=principles,
        defensive_line_height=25,
        pressing_intensity=30,
        pressing_trigger_zone=90,  # Only press very high
        counter_press_duration=2,  # Don't counter-press, retreat
        compactness_target=20,
        width_in_possession=60,
        width_out_of_possession=35,
        tempo=70,
        directness=85  # Very direct on counter
    )


def create_fluid_4231() -> TacticalSetup:
    """
    Balanced 4-2-3-1 with fluid movement.
    A good default / baseline tactical setup.
    """
    principles = [
        TacticalPrinciple(
            name="balanced_possession",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["cm", "cm_l", "cm_r", "dm"],
            movement=MovementInstruction(
                maintain_shape=0.5,
                towards_ball=0.2,
                towards_space=0.3
            ),
            priority=5,
            team_has_ball=True
        ),

        TacticalPrinciple(
            name="balanced_defense",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["cb", "cb_l", "cb_r", "lb", "rb", "dm"],
            movement=MovementInstruction(
                maintain_shape=0.7,
                towards_ball=0.2
            ),
            priority=6,
            team_has_ball=False
        ),

        TacticalPrinciple(
            name="forward_runs",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["am", "lw", "rw"],
            movement=MovementInstruction(
                relative_y=10,
                towards_space=0.4,
                maintain_shape=0.3
            ),
            priority=5,
            ball_zone_y_min=40,
            team_has_ball=True
        ),

        TacticalPrinciple(
            name="striker_movement",
            trigger=TriggerType.IN_POSSESSION,
            applies_to_roles=["st", "cf"],
            movement=MovementInstruction(
                towards_space=0.5,
                maintain_shape=0.3
            ),
            priority=5,
            team_has_ball=True
        ),

        TacticalPrinciple(
            name="press_high",
            trigger=TriggerType.OUT_OF_POSSESSION,
            applies_to_roles=["st", "cf", "am"],
            movement=MovementInstruction(towards_ball=0.5),
            priority=6,
            ball_zone_y_min=65,
            team_has_ball=False
        ),
    ]

    return TacticalSetup(
        name="Balanced 4-2-3-1",
        principles=principles,
        defensive_line_height=40,
        pressing_intensity=50,
        pressing_trigger_zone=70,
        counter_press_duration=4,
        compactness_target=35,
        width_in_possession=65,
        width_out_of_possession=50,
        tempo=50,
        directness=50
    )


# Factory function
TACTICAL_PRESETS = {
    "positional_play": create_guardiola_positional_play,
    "gegenpressing": create_gegenpressing,
    "low_block_counter": create_low_block_counter,
    "balanced": create_fluid_4231,
}


def get_tactical_setup(name: str) -> TacticalSetup:
    """Get a pre-built tactical setup by name"""
    if name in TACTICAL_PRESETS:
        return TACTICAL_PRESETS[name]()
    return create_fluid_4231()  # Default
