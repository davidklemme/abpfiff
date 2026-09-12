"""
Movement model: where every player without the ball moves each tick.

Combines explicit tactical principles (tactics.py) with role-based default
behavior. All y logic runs in the team's attacking frame (Team.frame_y),
so the same rules serve both directions of play.
"""
import random
from typing import Callable, Optional

from models import MatchState, Player, Position, Team
from spatial import SpaceControl
from tactics import MovementInstruction


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

        for player in team.players:
            # Skip ball holder - they move via dribble
            if player.has_ball:
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

    def attacking_movement(self, player: Player, ball_pos: Position,
                           ball_holder: Optional[Player], team: Team,
                           other_team: Team, state: MatchState) -> tuple:
        """Calculate attacking movement for player.

        All y values inside this function are in the team's attacking frame
        (the team always attacks toward frame y=100). The returned target is
        converted back to absolute pitch coordinates.
        """
        f = team.frame_y
        rng = self.rng
        ball_x, ball_y = ball_pos.x, f(ball_pos.y)
        base_x, base_y = player.base_position.x, f(player.base_position.y)
        pos_x, pos_y = player.position.x, f(player.position.y)

        target_x = pos_x
        target_y = pos_y

        # Role-based attacking behavior
        role = player.role.lower()

        if role in ['gk']:
            # GK stays back but supports buildup
            target_y = base_y + 5
            target_x = base_x

        elif role in ['cb', 'cb_l', 'cb_r']:
            # CBs provide safety, spread for buildup
            target_y = min(base_y + 15, ball_y - 20)
            # Spread horizontally when team has ball
            spread = 10 if pos_x < 50 else -10
            target_x = base_x + spread

        elif role in ['lb', 'rb', 'lwb', 'rwb']:
            # Fullbacks push forward and wide to stretch play
            target_y = min(ball_y + 10, 85)
            target_x = 15 if 'l' in role else 85
            # If ball is on their side, get even higher
            if (pos_x < 50) == (ball_x < 50):
                target_y = min(target_y + 10, 90)

        elif role in ['dm', 'cdm']:
            # DM links defense and midfield
            target_y = max(ball_y - 15, base_y)
            target_x = 50 + (ball_x - 50) * 0.3

        elif role in ['cm', 'cm_l', 'cm_r']:
            # CMs support ball, find pockets of space
            target_y = ball_y + rng.uniform(-5, 10)
            # Move to half-spaces
            if pos_x < 50:
                target_x = max(25, ball_x - 15)
            else:
                target_x = min(75, ball_x + 15)

        elif role in ['am', 'cam']:
            # AM finds space between lines
            target_y = ball_y + rng.uniform(5, 20)
            target_x = 50 + (ball_x - 50) * 0.5

        elif role in ['lw', 'rw', 'lm', 'rm']:
            # Wingers: width and depth
            target_x = 10 if 'l' in role else 90
            target_y = max(ball_y, 60)
            # If ball is on opposite side, come narrower for cutback
            if (pos_x < 50) != (ball_x < 50):
                target_x = 30 if 'l' in role else 70
                target_y = ball_y + 15

        elif role in ['st', 'cf']:
            # Strikers: stretch defense, make runs
            target_y = min(ball_y + 25, 95)
            # Drift across to find space
            target_x = 50 + rng.uniform(-20, 20)
            # Stay onside (simplified) - defensive line measured in our frame
            def_line = self.get_defensive_line(other_team, f)
            target_y = min(target_y, def_line + 5)

        # Add some unpredictability
        target_x += rng.uniform(-3, 3)
        target_y += rng.uniform(-2, 2)

        # When attacking, prioritize calculated position over base (more fluid).
        # Confidence drives how much a player commits to the advanced, ball-
        # seeking position vs. retreating toward their safe base position -
        # a rattled player hides, a confident one demands involvement.
        ball_seeking = max(0.5, min(0.95, 0.8 + player.confidence * 0.15))
        target_x = target_x * ball_seeking + base_x * (1 - ball_seeking)
        target_y = target_y * ball_seeking + base_y * (1 - ball_seeking)

        # Push all outfield players forward when team has ball
        if ball_holder and ball_holder in team.players:
            target_y = min(target_y + 5, 95)

        return target_x, f(target_y)

    def defending_movement(self, player: Player, ball_pos: Position,
                           ball_holder: Optional[Player], team: Team,
                           other_team: Team, state: MatchState) -> tuple:
        """Calculate defending movement for player.

        All y values inside this function are in the team's attacking frame
        (own goal at frame y=0). The returned target is converted back to
        absolute pitch coordinates.
        """
        f = team.frame_y
        rng = self.rng
        ball_x, ball_y = ball_pos.x, f(ball_pos.y)
        base_x, base_y = player.base_position.x, f(player.base_position.y)
        pos_x = player.position.x

        target_x = pos_x
        target_y = f(player.position.y)

        role = player.role.lower()
        distance_to_ball = player.position.distance_to(ball_pos)

        if role in ['gk']:
            # GK adjusts position based on ball
            target_y = 5
            target_x = 50 + (ball_x - 50) * 0.15

        elif role in ['cb', 'cb_l', 'cb_r']:
            # CBs: maintain line, cover central areas
            target_y = min(35, ball_y - 10)
            # Shift toward ball side
            target_x = base_x + (ball_x - 50) * 0.15

        elif role in ['lb', 'rb', 'lwb', 'rwb']:
            # Fullbacks: track wingers, stay compact
            target_y = min(40, ball_y - 5)
            # Tuck in if ball is on far side
            if (pos_x < 50) != (ball_x < 50):
                target_x = 30 if 'l' in role else 70
            else:
                target_x = base_x

        elif role in ['dm', 'cdm']:
            # DM screens defense
            target_y = min(45, ball_y - 5)
            target_x = 50 + (ball_x - 50) * 0.4

        elif role in ['cm', 'cm_l', 'cm_r']:
            # CMs: press or cover
            if distance_to_ball < 25 and player.effective_attribute('workrate') > 60:
                # Press
                target_x = ball_x
                target_y = ball_y - 5
            else:
                # Cover passing lanes
                target_y = max(35, ball_y - 15)
                target_x = base_x + (ball_x - 50) * 0.3

        elif role in ['am', 'cam']:
            # AM drops into midfield when defending
            target_y = max(45, ball_y - 10)
            target_x = 50 + (ball_x - 50) * 0.3

        elif role in ['lw', 'rw', 'lm', 'rm']:
            # Wingers: track back or press
            if ball_y > 60 and player.effective_attribute('workrate') > 55:
                # Press high
                target_y = ball_y + 5
                target_x = ball_x + (10 if 'l' in role else -10)
            else:
                # Track back
                target_y = max(40, ball_y - 10)
                target_x = 25 if 'l' in role else 75

        elif role in ['st', 'cf']:
            # Strikers: light press or stay high for counter
            if ball_y > 50:
                # Press from front
                target_y = ball_y + 8
                target_x = ball_x + rng.uniform(-10, 10)
            else:
                # Stay high for counter-attack
                target_y = 65
                target_x = 50

        # Blend with base (more defensive = more base-weighted)
        target_x = target_x * 0.5 + base_x * 0.5
        target_y = target_y * 0.5 + base_y * 0.5

        return target_x, f(target_y)

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
