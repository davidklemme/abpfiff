"""
Movement model: where every player without the ball moves each tick.

Combines explicit tactical principles (tactics.py) with role-based default
behavior. All y logic runs in the team's attacking frame (Team.frame_y),
so the same rules serve both directions of play.
"""
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from models import BallState, MatchState, Player, Position, Team
from spatial import SpaceControl
from tactics import MovementInstruction


@dataclass
class RoleMoveContext:
    """Frame-space inputs for a role movement handler.

    All y values are in the team's attacking frame (attack toward y=100);
    handlers return (target_x, target_y) in the same frame.
    """
    role: str
    rng: random.Random
    ball_x: float
    ball_y: float
    base_x: float
    base_y: float
    pos_x: float
    pos_y: float
    distance_to_ball: float
    workrate: float
    get_def_line: Callable[[], float]  # opposing defensive line, lazily

    @property
    def is_left_side(self) -> bool:
        return 'l' in self.role

    @property
    def same_side_as_ball(self) -> bool:
        return (self.pos_x < 50) == (self.ball_x < 50)


# -- Attacking role behaviors (one small handler per role group) -------------

def _atk_gk(c: RoleMoveContext) -> Tuple[float, float]:
    # GK stays back but supports buildup
    return c.base_x, c.base_y + 5


def _atk_cb(c: RoleMoveContext) -> Tuple[float, float]:
    # CBs provide safety, spread for buildup
    spread = 10 if c.pos_x < 50 else -10
    return c.base_x + spread, min(c.base_y + 15, c.ball_y - 20)


def _atk_fullback(c: RoleMoveContext) -> Tuple[float, float]:
    # Fullbacks push forward and wide; higher when the ball is on their side
    target_y = min(c.ball_y + 10, 85)
    if c.same_side_as_ball:
        target_y = min(target_y + 10, 90)
    return (15 if c.is_left_side else 85), target_y


def _atk_dm(c: RoleMoveContext) -> Tuple[float, float]:
    # DM links defense and midfield
    return 50 + (c.ball_x - 50) * 0.3, max(c.ball_y - 15, c.base_y)


def _atk_cm(c: RoleMoveContext) -> Tuple[float, float]:
    # CMs support ball, find pockets in the half-spaces
    target_y = c.ball_y + c.rng.uniform(-5, 10)
    target_x = max(25, c.ball_x - 15) if c.pos_x < 50 else min(75, c.ball_x + 15)
    return target_x, target_y


def _atk_am(c: RoleMoveContext) -> Tuple[float, float]:
    # AM finds space between the lines
    return 50 + (c.ball_x - 50) * 0.5, c.ball_y + c.rng.uniform(5, 20)


def _atk_winger(c: RoleMoveContext) -> Tuple[float, float]:
    # Wingers hold width and depth; come narrow for the cutback when the
    # ball is on the far side
    if not c.same_side_as_ball:
        return (30 if c.is_left_side else 70), c.ball_y + 15
    return (10 if c.is_left_side else 90), max(c.ball_y, 60)


def _atk_striker(c: RoleMoveContext) -> Tuple[float, float]:
    # Strikers stretch the defense but stay onside (simplified)
    target_y = min(c.ball_y + 25, 95)
    target_x = 50 + c.rng.uniform(-20, 20)
    return target_x, min(target_y, c.get_def_line() + 5)


# -- Defending role behaviors ------------------------------------------------

def _def_gk(c: RoleMoveContext) -> Tuple[float, float]:
    return 50 + (c.ball_x - 50) * 0.15, 5


def _def_cb(c: RoleMoveContext) -> Tuple[float, float]:
    # Maintain the line, shift toward the ball side
    return c.base_x + (c.ball_x - 50) * 0.15, min(35, c.ball_y - 10)


def _def_fullback(c: RoleMoveContext) -> Tuple[float, float]:
    # Track wingers; tuck in when the ball is on the far side
    target_x = c.base_x if c.same_side_as_ball else (30 if c.is_left_side else 70)
    return target_x, min(40, c.ball_y - 5)


def _def_dm(c: RoleMoveContext) -> Tuple[float, float]:
    # Screen the defense
    return 50 + (c.ball_x - 50) * 0.4, min(45, c.ball_y - 5)


def _def_cm(c: RoleMoveContext) -> Tuple[float, float]:
    # Press when close and willing, otherwise cover passing lanes
    if c.distance_to_ball < 25 and c.workrate > 60:
        return c.ball_x, c.ball_y - 5
    return c.base_x + (c.ball_x - 50) * 0.3, max(35, c.ball_y - 15)


def _def_am(c: RoleMoveContext) -> Tuple[float, float]:
    # Drop into midfield
    return 50 + (c.ball_x - 50) * 0.3, max(45, c.ball_y - 10)


def _def_winger(c: RoleMoveContext) -> Tuple[float, float]:
    # Press high when the ball is advanced, otherwise track back
    if c.ball_y > 60 and c.workrate > 55:
        return c.ball_x + (10 if c.is_left_side else -10), c.ball_y + 5
    return (25 if c.is_left_side else 75), max(40, c.ball_y - 10)


def _def_striker(c: RoleMoveContext) -> Tuple[float, float]:
    # Light press from the front, or stay high for the counter
    if c.ball_y > 50:
        return c.ball_x + c.rng.uniform(-10, 10), c.ball_y + 8
    return 50, 65


def _role_table(*groups) -> Dict[str, Callable]:
    """Expand ((roles...), handler) pairs into a role -> handler map."""
    table = {}
    for roles, handler in groups:
        for role in roles:
            table[role] = handler
    return table


ROLE_GROUPS = {
    "gk": ('gk',),
    "cb": ('cb', 'cb_l', 'cb_r'),
    "fullback": ('lb', 'rb', 'lwb', 'rwb'),
    "dm": ('dm', 'cdm'),
    "cm": ('cm', 'cm_l', 'cm_r'),
    "am": ('am', 'cam'),
    "winger": ('lw', 'rw', 'lm', 'rm'),
    "striker": ('st', 'cf'),
}

ATTACK_ROLE_HANDLERS = _role_table(
    (ROLE_GROUPS["gk"], _atk_gk),
    (ROLE_GROUPS["cb"], _atk_cb),
    (ROLE_GROUPS["fullback"], _atk_fullback),
    (ROLE_GROUPS["dm"], _atk_dm),
    (ROLE_GROUPS["cm"], _atk_cm),
    (ROLE_GROUPS["am"], _atk_am),
    (ROLE_GROUPS["winger"], _atk_winger),
    (ROLE_GROUPS["striker"], _atk_striker),
)

DEFEND_ROLE_HANDLERS = _role_table(
    (ROLE_GROUPS["gk"], _def_gk),
    (ROLE_GROUPS["cb"], _def_cb),
    (ROLE_GROUPS["fullback"], _def_fullback),
    (ROLE_GROUPS["dm"], _def_dm),
    (ROLE_GROUPS["cm"], _def_cm),
    (ROLE_GROUPS["am"], _def_am),
    (ROLE_GROUPS["winger"], _def_winger),
    (ROLE_GROUPS["striker"], _def_striker),
)


class RoleMovementModel:
    """Tactical-principle plus role-default movement."""

    def __init__(self, space_control: Optional[SpaceControl] = None,
                 rng: Optional[random.Random] = None):
        self.space_control = space_control or SpaceControl(resolution=10)
        self.rng = rng or random.Random()

    def update_positions(self, state: MatchState) -> None:
        """Move players of both teams according to tactical principles"""
        self._apply_team_movements(state.home_team, state, state.home_attacking)
        self._apply_team_movements(state.away_team, state, not state.home_attacking)

    def _apply_team_movements(self, team: Team, state: MatchState,
                              team_attacking: bool) -> None:
        """Apply tactical movements for one team"""
        tactics = team.tactics
        other_team = state.away_team if team is state.home_team else state.home_team
        ball = state.ball

        # The intended receiver of an in-flight pass breaks from role
        # movement and runs to meet the ball at its arrival point
        receiver = None
        if (ball.is_in_flight() and ball.state != BallState.SHOT
                and ball.target_player in team.players):
            receiver = ball.target_player

        # A loose ball gets run down by the nearest outfield player
        chaser = None
        if ball.state == BallState.LOOSE:
            candidates = team.outfield_players or team.players
            chaser = min(candidates,
                         key=lambda p: p.position.distance_to(ball.position))

        for player in team.players:
            # Skip ball holder - they move via dribble
            if player.has_ball:
                continue

            if player is receiver:
                self._sprint_towards(player, ball.target_position or ball.position)
                continue

            if player is chaser:
                self._sprint_towards(player, ball.position)
                continue

            # Get active principles for this player
            principles = []
            if tactics:
                principles = tactics.get_active_principles(player, state, team_attacking, team)

            if principles:
                # Apply highest priority principle
                principle = principles[0]
                self.apply_movement(player, principle.movement, state, team)
            else:
                # Smart default movement based on role and game state
                self._apply_default_movement(player, state, team, team_attacking, other_team)

    def _sprint_towards(self, player: Player, target: Position) -> None:
        """Full-effort run to a spot (meeting a pass, chasing a loose ball)."""
        max_speed = (player.effective_attribute('pace') / 100) * 3
        player.position = player.position.move_towards(target, max_speed).clamp()

    def apply_movement(self, player: Player, movement: MovementInstruction,
                       state: MatchState, team: Team) -> None:
        """Apply a movement instruction to a player.

        Instruction y values are in the team's attacking frame (the team
        always attacks toward frame y=100); we compute the target in that
        frame and convert back to absolute pitch coordinates at the end.
        """
        f = team.frame_y
        target_x = player.position.x
        target_y = f(player.position.y)

        # Absolute target (frame coordinates)
        if movement.target_x is not None:
            target_x = movement.target_x
        if movement.target_y is not None:
            target_y = movement.target_y

        # Relative movement (frame coordinates)
        target_x += movement.relative_x
        target_y += movement.relative_y

        # Lateral movement relative to the player's side of the pitch:
        # positive tucks toward the center, negative pushes to the touchline
        if movement.towards_center_x:
            side = 1.0 if player.base_position.x < 50 else -1.0
            target_x += movement.towards_center_x * side

        # Move toward ball
        if movement.towards_ball > 0:
            ball_pos = state.ball.position
            target_x += (ball_pos.x - player.position.x) * movement.towards_ball
            target_y += (f(ball_pos.y) - f(player.position.y)) * movement.towards_ball

        # Move toward open space
        if movement.towards_space > 0:
            opponents = state.away_team if team is state.home_team else state.home_team
            open_spaces = self.space_control.find_open_spaces(team, opponents)
            if open_spaces:
                # Find nearest advantageous space
                best_space = min(
                    open_spaces,
                    key=lambda s: player.position.distance_to(s)
                )
                target_x += (best_space.x - player.position.x) * movement.towards_space * 0.3
                target_y += (f(best_space.y) - f(player.position.y)) * movement.towards_space * 0.3

        # Maintain shape (blend toward base position)
        if movement.maintain_shape > 0:
            target_x = target_x * (1 - movement.maintain_shape) + player.base_position.x * movement.maintain_shape
            target_y = target_y * (1 - movement.maintain_shape) + f(player.base_position.y) * movement.maintain_shape

        # Calculate movement speed based on pace and fatigue
        max_speed = (player.effective_attribute('pace') / 100) * 3  # Max 3 units per tick
        target_pos = Position(target_x, f(target_y)).clamp()

        # Move toward target
        new_pos = player.position.move_towards(target_pos, max_speed)
        player.position = new_pos.clamp()

    def _apply_default_movement(self, player: Player, state: MatchState,
                                team: Team, team_attacking: bool,
                                other_team: Team) -> None:
        """
        Smart default movement when no tactical principle applies.
        This is the heart of emergent tactical behavior.
        """
        ball_pos = state.ball.position
        ball_holder = state.ball.holder

        if team_attacking:
            # ATTACKING: Support play, find space, make runs
            target_x, target_y = self.attacking_movement(
                player, ball_pos, ball_holder, team, other_team, state
            )
        else:
            # DEFENDING: Track runners, maintain shape, close space
            target_x, target_y = self.defending_movement(
                player, ball_pos, ball_holder, team, other_team, state
            )

        # Movement speed based on player attributes and urgency
        pace = player.effective_attribute('pace')
        workrate = player.effective_attribute('workrate')
        urgency = self._calculate_urgency(player, ball_pos, team_attacking)

        max_speed = (pace / 100) * 2.5 * (0.5 + workrate / 200) * urgency

        # Apply movement
        target_pos = Position(target_x, target_y).clamp()
        new_pos = player.position.move_towards(target_pos, max_speed)
        player.position = new_pos.clamp()

    def _role_context(self, player: Player, ball_pos: Position, team: Team,
                      other_team: Team) -> RoleMoveContext:
        f = team.frame_y
        return RoleMoveContext(
            role=player.role.lower(),
            rng=self.rng,
            ball_x=ball_pos.x, ball_y=f(ball_pos.y),
            base_x=player.base_position.x, base_y=f(player.base_position.y),
            pos_x=player.position.x, pos_y=f(player.position.y),
            distance_to_ball=player.position.distance_to(ball_pos),
            workrate=player.effective_attribute('workrate'),
            get_def_line=lambda: self.get_defensive_line(other_team, f),
        )

    def attacking_movement(self, player: Player, ball_pos: Position,
                           ball_holder: Optional[Player], team: Team,
                           other_team: Team, state: MatchState) -> tuple:
        """Calculate attacking movement: the role handler proposes a frame
        target, then shared attacking adjustments (jitter, confidence-driven
        ball seeking, forward push) shape it. Returns absolute coordinates.
        """
        context = self._role_context(player, ball_pos, team, other_team)
        handler = ATTACK_ROLE_HANDLERS.get(context.role)
        if handler:
            target_x, target_y = handler(context)
        else:
            target_x, target_y = context.pos_x, context.pos_y

        # Add some unpredictability
        target_x += self.rng.uniform(-3, 3)
        target_y += self.rng.uniform(-2, 2)

        # When attacking, prioritize calculated position over base (more fluid).
        # Confidence drives how much a player commits to the advanced, ball-
        # seeking position vs. retreating toward their safe base position -
        # a rattled player hides, a confident one demands involvement.
        ball_seeking = max(0.5, min(0.95, 0.8 + player.confidence * 0.15))
        target_x = target_x * ball_seeking + context.base_x * (1 - ball_seeking)
        target_y = target_y * ball_seeking + context.base_y * (1 - ball_seeking)

        # Push all outfield players forward when team has ball
        if ball_holder and ball_holder in team.players:
            target_y = min(target_y + 5, 95)

        return target_x, team.frame_y(target_y)

    def defending_movement(self, player: Player, ball_pos: Position,
                           ball_holder: Optional[Player], team: Team,
                           other_team: Team, state: MatchState) -> tuple:
        """Calculate defending movement: role handler target in frame space,
        blended toward the base position for shape. Returns absolute
        coordinates."""
        context = self._role_context(player, ball_pos, team, other_team)
        handler = DEFEND_ROLE_HANDLERS.get(context.role)
        if handler:
            target_x, target_y = handler(context)
        else:
            target_x, target_y = context.pos_x, context.pos_y

        # Blend with base (more defensive = more base-weighted)
        target_x = target_x * 0.5 + context.base_x * 0.5
        target_y = target_y * 0.5 + context.base_y * 0.5

        return target_x, team.frame_y(target_y)

    def get_defensive_line(self, team: Team,
                           frame_y: Optional[Callable] = None) -> float:
        """Get the y-position of a team's defensive line.

        `frame_y` maps absolute y into the frame the caller works in
        (e.g. the attacking team's frame when checking offside-ish lines);
        defaults to absolute coordinates. The line is the deepest defender
        relative to the goal the team defends, i.e. the max in the frame of
        the team attacking them.
        """
        f = frame_y or (lambda y: y)
        defenders = [p for p in team.players if p.role in ['cb', 'cb_l', 'cb_r', 'lb', 'rb']]
        if defenders:
            return max(f(p.position.y) for p in defenders)
        return 70  # No recognized defenders: assume a deep line

    def _calculate_urgency(self, player: Player, ball_pos: Position,
                           team_attacking: bool) -> float:
        """Calculate movement urgency (affects speed)"""
        distance = player.position.distance_to(ball_pos)

        if distance < 20:
            base = 1.2  # Close to ball = urgent
        elif distance < 40:
            base = 1.0
        else:
            base = 0.7  # Far from ball = less urgent

        # Confident players engage more urgently, rattled players drag their feet
        return max(0.3, base + player.confidence * 0.15)
