"""
Core data models for the football match engine.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import math


class Trait(Enum):
    """Player traits that enable certain tactical roles"""
    PACE = "pace"
    STAMINA = "stamina"
    PASSING = "passing"
    SHOOTING = "shooting"
    DEFENDING = "defending"
    DRIBBLING = "dribbling"
    POSITIONING = "positioning"
    COMPOSURE = "composure"
    WORKRATE = "workrate"
    AGGRESSION = "aggression"
    AERIAL = "aerial"
    FIRST_TOUCH = "first_touch"
    VISION = "vision"
    LEADERSHIP = "leadership"


class Phase(Enum):
    """Match phases"""
    BUILDUP = "buildup"
    PROGRESSION = "progression"
    ATTACKING = "attacking"
    DEFENDING = "defending"
    TRANSITION_ATK = "transition_attack"
    TRANSITION_DEF = "transition_defense"
    SET_PIECE = "set_piece"


class Zone(Enum):
    """Pitch zones"""
    GK = "goalkeeper"
    DEF_LEFT = "defense_left"
    DEF_CENTER = "defense_center"
    DEF_RIGHT = "defense_right"
    MID_LEFT = "midfield_left"
    MID_CENTER = "midfield_center"
    MID_RIGHT = "midfield_right"
    ATK_LEFT = "attack_left"
    ATK_CENTER = "attack_center"
    ATK_RIGHT = "attack_right"


@dataclass
class Position:
    """2D position on the pitch (0-100 x 0-100)"""
    x: float  # 0 = left touchline, 100 = right touchline
    y: float  # 0 = own goal line, 100 = opponent goal line

    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def move_towards(self, target: 'Position', distance: float) -> 'Position':
        """Move towards target by given distance"""
        d = self.distance_to(target)
        if d == 0:
            return Position(self.x, self.y)
        ratio = min(distance / d, 1.0)
        return Position(
            self.x + (target.x - self.x) * ratio,
            self.y + (target.y - self.y) * ratio
        )

    def clamp(self) -> 'Position':
        """Keep position within pitch bounds"""
        return Position(
            max(0, min(100, self.x)),
            max(0, min(100, self.y))
        )

    def __repr__(self):
        return f"({self.x:.1f}, {self.y:.1f})"


@dataclass
class Player:
    """Individual player with attributes and state"""
    name: str
    number: int

    # Core attributes (0-100)
    pace: int = 50
    stamina: int = 50
    passing: int = 50
    shooting: int = 50
    defending: int = 50
    dribbling: int = 50
    positioning: int = 50
    composure: int = 50
    workrate: int = 50
    aggression: int = 50
    aerial: int = 50
    first_touch: int = 50
    vision: int = 50

    # Current state
    position: Position = field(default_factory=lambda: Position(50, 50))
    base_position: Position = field(default_factory=lambda: Position(50, 50))
    fatigue: float = 0.0  # 0-100, higher = more tired
    has_ball: bool = False
    confidence: float = 0.0  # -1 to 1: current psychological momentum/form

    # Sensitivity to the environment (0-100): stakes, crowd, limelight.
    # Converts match conditions into personal mental load - which consumes
    # cognitive capacity (psychology.calculate_pressure). A trait, not a
    # skill: it barely moves within a match.
    sensitivity: int = 50

    # Motion state (derived each tick by the engine, smoothed): the basis
    # for facing/orientation. Not an attribute - orientation is state.
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    # Discipline
    yellow_cards: int = 0
    sent_off: bool = False

    # Role assignment
    role: str = "default"

    # Stable identity: minds, memories and (later) career persistence key
    # off this, never off object identity. Derived from name+number unless
    # provided explicitly.
    player_id: str = ""

    def __post_init__(self):
        if not self.player_id:
            self.player_id = f"{self.name}#{self.number}"

    def influence_at(self, pos: Position) -> float:
        """Calculate player's defensive/control influence at a position"""
        distance = self.position.distance_to(pos)
        if distance == 0:
            return 1.0

        # Base influence decreases with distance
        base = 1.0 / (1.0 + distance * 0.1)

        # Modifiers based on attributes
        pace_bonus = 1.0 + (self.pace - 50) * 0.005
        positioning_bonus = 1.0 + (self.positioning - 50) * 0.005
        fatigue_penalty = 1.0 - (self.fatigue * 0.003)

        return base * pace_bonus * positioning_bonus * fatigue_penalty

    def effective_attribute(self, attr: str) -> float:
        """Get attribute value adjusted for fatigue"""
        base = getattr(self, attr, 50)
        fatigue_penalty = self.fatigue * 0.3  # Lose up to 30% at max fatigue
        return max(10, base - fatigue_penalty)

    def __repr__(self):
        return f"{self.name}({self.number})"


@dataclass
class Team:
    """Team with players and tactical setup"""
    name: str
    players: List[Player]
    tactics: 'TacticalSetup' = None

    # Attack direction: True = attacks toward y=100, False = toward y=0.
    # Away teams get False via teams.flip_team_positions().
    attacks_up: bool = True

    # Calculated metrics
    possession: float = 50.0
    momentum: float = 50.0

    def frame_y(self, y: float) -> float:
        """Map an absolute pitch y into this team's attacking frame.

        In the frame the team always attacks toward y=100, so tactical logic
        can be written once for both teams. The mapping is its own inverse:
        frame_y(frame_y(y)) == y, so it also converts frame targets back to
        absolute coordinates.
        """
        return y if self.attacks_up else 100.0 - y

    @property
    def attack_sign(self) -> float:
        """+1 if this team attacks toward increasing y, else -1."""
        return 1.0 if self.attacks_up else -1.0

    @property
    def attacking_goal(self) -> Position:
        """The goal this team is shooting at."""
        return Position(50.0, 100.0 if self.attacks_up else 0.0)

    @property
    def own_goal(self) -> Position:
        """The goal this team is defending."""
        return Position(50.0, 0.0 if self.attacks_up else 100.0)

    @property
    def outfield_players(self) -> List[Player]:
        return [p for p in self.players if p.role != "gk"]

    @property
    def goalkeeper(self) -> Optional[Player]:
        for p in self.players:
            if p.role == "gk":
                return p
        return None

    def get_player_with_ball(self) -> Optional[Player]:
        for p in self.players:
            if p.has_ball:
                return p
        return None

    def avg_position(self) -> Position:
        """Average position of outfield players"""
        outfield = self.outfield_players
        if not outfield:
            return Position(50, 50)
        return Position(
            sum(p.position.x for p in outfield) / len(outfield),
            sum(p.position.y for p in outfield) / len(outfield)
        )

    def compactness_vertical(self) -> float:
        """Distance between defensive and attacking lines"""
        outfield = self.outfield_players
        if len(outfield) < 2:
            return 0
        y_positions = [p.position.y for p in outfield]
        return max(y_positions) - min(y_positions)

    def compactness_horizontal(self) -> float:
        """Width of the team"""
        outfield = self.outfield_players
        if len(outfield) < 2:
            return 0
        x_positions = [p.position.x for p in outfield]
        return max(x_positions) - min(x_positions)

    def defensive_line_height(self) -> float:
        """Y position of the defensive line"""
        defenders = [p for p in self.players if "def" in p.role.lower() or p.role == "cb" or p.role == "lb" or p.role == "rb"]
        if not defenders:
            defenders = sorted(self.outfield_players, key=lambda p: p.position.y)[:4]
        if not defenders:
            return 25
        return sum(p.position.y for p in defenders) / len(defenders)


class BallState(Enum):
    """Ball state"""
    HELD = "held"           # Player has the ball at feet
    GROUND_PASS = "ground_pass"  # Rolling on ground
    AIR_PASS = "air_pass"    # In the air (lofted pass, cross)
    SHOT = "shot"           # Shot toward goal
    LOOSE = "loose"         # No one has it


@dataclass
class Ball:
    """The ball with physics and flight model"""
    position: Position = field(default_factory=lambda: Position(50, 50))
    holder: Optional[Player] = None
    in_play: bool = True

    # Flight state
    state: BallState = BallState.HELD
    target_position: Optional[Position] = None
    target_player: Optional[Player] = None  # Intended recipient
    flight_ticks_remaining: int = 0
    flight_speed: float = 0  # Units per tick
    passer: Optional[Player] = None  # Who kicked it

    def give_to(self, player: Player):
        """Instantly give ball to player (for tackles, loose ball wins)"""
        if self.holder:
            self.holder.has_ball = False
        self.holder = player
        player.has_ball = True
        self.position = Position(player.position.x, player.position.y)
        self.state = BallState.HELD
        self.target_position = None
        self.target_player = None
        self.flight_ticks_remaining = 0
        self.passer = None

    def start_pass(self, passer: Player, target: Player, is_lofted: bool = False,
                   lead_position: Optional[Position] = None):
        """Start a pass - ball will travel over time.

        `lead_position` aims the ball ahead of the receiver (a lead pass
        into space); the receiver is expected to move to meet it. Without
        it the ball is played to the receiver's feet at kick time.
        """
        if self.holder:
            self.holder.has_ball = False
        self.holder = None
        self.passer = passer
        self.target_player = target
        if lead_position is not None:
            self.target_position = Position(lead_position.x, lead_position.y)
        else:
            self.target_position = Position(target.position.x, target.position.y)

        # Calculate flight time based on distance
        distance = passer.position.distance_to(self.target_position)

        if is_lofted:
            self.state = BallState.AIR_PASS
            self.flight_speed = 12  # Slower in air
        else:
            self.state = BallState.GROUND_PASS
            self.flight_speed = 18  # Faster on ground
        # ceil, not int: the counter must outlast the full distance so the
        # ball really reaches the aim point (arrival is positional)
        self.flight_ticks_remaining = max(1, math.ceil(distance / self.flight_speed))

    def launch_clear(self, kicker: Player, target_pos: Position):
        """Hoof the ball toward an area with no intended receiver - it
        travels as a lofted ball and runs loose where it lands."""
        if self.holder:
            self.holder.has_ball = False
        self.holder = None
        self.passer = kicker
        self.target_player = None
        self.target_position = Position(target_pos.x, target_pos.y)
        self.state = BallState.AIR_PASS
        self.flight_speed = 14
        distance = kicker.position.distance_to(self.target_position)
        self.flight_ticks_remaining = max(1, math.ceil(distance / self.flight_speed))

    def start_shot(self, shooter: Player, target_pos: Position):
        """Start a shot toward goal"""
        if self.holder:
            self.holder.has_ball = False
        self.holder = None
        self.passer = shooter
        self.target_position = target_pos
        self.target_player = None
        self.state = BallState.SHOT

        distance = shooter.position.distance_to(target_pos)
        self.flight_speed = 25  # Shots are fast
        self.flight_ticks_remaining = max(1, math.ceil(distance / self.flight_speed))

    def update_flight(self) -> bool:
        """Update ball position during flight. Returns True if ball arrived.

        Arrival is positional - the ball has actually reached its aim point
        - with the tick counter only as a safety cap. (A counter-only check
        used to land balls up to one flight-speed short of the target.)
        """
        if self.state == BallState.HELD or self.target_position is None:
            return False

        if self.flight_ticks_remaining <= 0:
            return True

        # Move ball toward target
        self.position = self.position.move_towards(self.target_position, self.flight_speed)
        self.flight_ticks_remaining -= 1

        arrived_at_target = self.position.distance_to(self.target_position) < 1e-6
        return arrived_at_target or self.flight_ticks_remaining <= 0

    def is_in_flight(self) -> bool:
        """Check if ball is traveling"""
        return self.state in [BallState.GROUND_PASS, BallState.AIR_PASS, BallState.SHOT]

    def make_loose(self):
        """Ball becomes loose (no one has it, not traveling to anyone)"""
        if self.holder:
            self.holder.has_ball = False
        self.holder = None
        self.state = BallState.LOOSE
        self.target_position = None
        self.target_player = None
        self.flight_ticks_remaining = 0


@dataclass
class MatchEvent:
    """An event during the match"""
    minute: int
    event_type: str  # "pass", "shot", "tackle", "goal", "save", etc.
    player: Player
    target_player: Optional[Player] = None
    position: Position = None
    success: bool = True
    description: str = ""


@dataclass
class Environment:
    """The conditions a match is played under - one vector per match.

    Defaults are neutral (a plain training-ground afternoon), so an
    unconfigured match behaves exactly as before. Each dimension is a
    factor input: players convert it to personal mental load through
    their sensitivity, never through special-cased branches.
    """
    visibility: float = 1.0       # 0-1: floodlit night fog -> clear day
    stakes: float = 0.0           # 0-1: friendly -> cup final
    crowd_intensity: float = 0.0  # 0-1: empty ground -> cauldron

    def psychological_load(self, sensitivity: float) -> float:
        """How much mental load this environment puts on a player with
        the given sensitivity (0-100). Insensitive veterans shrug off
        the limelight; sensitive players feel every eye."""
        exposure = self.stakes * 0.6 + self.crowd_intensity * 0.4
        susceptibility = 0.3 + 1.4 * (sensitivity / 100.0)
        return min(1.0, exposure * susceptibility)


@dataclass
class MatchState:
    """Current state of the match"""
    home_team: Team
    away_team: Team
    ball: Ball
    environment: Environment = field(default_factory=Environment)
    minute: int = 0
    phase: Phase = Phase.BUILDUP
    events: List[MatchEvent] = field(default_factory=list)
    home_score: int = 0
    away_score: int = 0

    # Track who's attacking (True = home, False = away)
    home_attacking: bool = True

    # Track possession changes for triggers like BALL_LOST, BALL_WON
    ticks_since_possession_change: int = 0

    @property
    def attacking_team(self) -> Team:
        return self.home_team if self.home_attacking else self.away_team

    @property
    def defending_team(self) -> Team:
        return self.away_team if self.home_attacking else self.home_team

    def switch_possession(self):
        self.home_attacking = not self.home_attacking
        self.phase = Phase.TRANSITION_ATK if self.ball.holder else Phase.TRANSITION_DEF
        self.ticks_since_possession_change = 0  # Reset counter on possession change

    def tick(self):
        """Called each simulation tick to update counters"""
        self.ticks_since_possession_change += 1
