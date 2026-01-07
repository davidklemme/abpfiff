"""
ASCII visualization for the match engine.
Renders the pitch, players, ball, and space control.
"""
from typing import List, Optional
from models import Position, Player, Team, Ball, MatchState
from spatial import SpaceControl
import os


class ASCIIVisualizer:
    """
    Renders the match state as ASCII art.
    """

    # Pitch dimensions in characters
    WIDTH = 60
    HEIGHT = 40

    # Characters
    EMPTY = '·'
    BALL = '●'
    HOME_PLAYER = 'H'
    AWAY_PLAYER = 'A'
    HOME_PLAYER_BALL = 'Ⓗ'
    AWAY_PLAYER_BALL = 'Ⓐ'
    GOAL_POST = '║'
    GOAL_LINE = '═'
    SIDELINE = '│'
    TOPLINE = '─'
    CORNER = '┼'

    # Space control visualization
    CONTROL_HOME_STRONG = '▓'
    CONTROL_HOME_WEAK = '▒'
    CONTROL_CONTESTED = '░'
    CONTROL_AWAY_WEAK = '▒'
    CONTROL_AWAY_STRONG = '▓'

    def __init__(self, show_space_control: bool = False):
        self.show_space_control = show_space_control
        self.space_control = SpaceControl(resolution=20)

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self, state: MatchState, clear: bool = False) -> str:
        """Render the current match state"""
        if clear:
            self.clear_screen()

        # Initialize empty pitch
        pitch = self._create_empty_pitch()

        # Add space control if enabled
        if self.show_space_control:
            self._add_space_control(pitch, state)

        # Add players
        self._add_players(pitch, state.home_team, is_home=True)
        self._add_players(pitch, state.away_team, is_home=False)

        # Add ball
        self._add_ball(pitch, state.ball)

        # Convert to string
        output = self._pitch_to_string(pitch, state)

        return output

    def _create_empty_pitch(self) -> List[List[str]]:
        """Create empty pitch grid"""
        pitch = [[self.EMPTY for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]

        # Draw pitch markings
        # Top and bottom lines
        for x in range(self.WIDTH):
            pitch[0][x] = self.TOPLINE
            pitch[self.HEIGHT - 1][x] = self.TOPLINE

        # Side lines
        for y in range(self.HEIGHT):
            pitch[y][0] = self.SIDELINE
            pitch[y][self.WIDTH - 1] = self.SIDELINE

        # Corners
        pitch[0][0] = '┌'
        pitch[0][self.WIDTH - 1] = '┐'
        pitch[self.HEIGHT - 1][0] = '└'
        pitch[self.HEIGHT - 1][self.WIDTH - 1] = '┘'

        # Center line
        center_y = self.HEIGHT // 2
        for x in range(1, self.WIDTH - 1):
            pitch[center_y][x] = '─'

        # Center circle (simplified)
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        for dx, dy in [(-3, 0), (3, 0), (0, -2), (0, 2), (-2, -1), (2, -1), (-2, 1), (2, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < self.WIDTH - 1 and 0 < ny < self.HEIGHT - 1:
                pitch[ny][nx] = '○'

        # Center spot
        pitch[cy][cx] = '◎'

        # Goals
        goal_width = 8
        goal_start = (self.WIDTH - goal_width) // 2
        goal_end = goal_start + goal_width

        # Top goal (away)
        for x in range(goal_start, goal_end):
            pitch[0][x] = self.GOAL_LINE
        pitch[0][goal_start] = '╔'
        pitch[0][goal_end - 1] = '╗'

        # Bottom goal (home)
        for x in range(goal_start, goal_end):
            pitch[self.HEIGHT - 1][x] = self.GOAL_LINE
        pitch[self.HEIGHT - 1][goal_start] = '╚'
        pitch[self.HEIGHT - 1][goal_end - 1] = '╝'

        # Penalty areas (simplified)
        pen_width = 20
        pen_height = 6
        pen_start = (self.WIDTH - pen_width) // 2

        # Top penalty area
        for y in range(1, pen_height):
            pitch[y][pen_start] = '│'
            pitch[y][pen_start + pen_width - 1] = '│'
        for x in range(pen_start, pen_start + pen_width):
            pitch[pen_height - 1][x] = '─'

        # Bottom penalty area
        for y in range(self.HEIGHT - pen_height, self.HEIGHT - 1):
            pitch[y][pen_start] = '│'
            pitch[y][pen_start + pen_width - 1] = '│'
        for x in range(pen_start, pen_start + pen_width):
            pitch[self.HEIGHT - pen_height][x] = '─'

        return pitch

    def _add_space_control(self, pitch: List[List[str]], state: MatchState):
        """Add space control shading to pitch"""
        grid = self.space_control.calculate_control_grid(state.home_team, state.away_team)

        for gy, row in enumerate(grid):
            for gx, control in enumerate(row):
                # Map grid position to pitch position
                px = int((gx / len(row)) * (self.WIDTH - 2)) + 1
                py = int((gy / len(grid)) * (self.HEIGHT - 2)) + 1

                if py < 1 or py >= self.HEIGHT - 1:
                    continue
                if px < 1 or px >= self.WIDTH - 1:
                    continue

                # Only shade empty cells
                if pitch[py][px] == self.EMPTY:
                    if control > 0.65:
                        pitch[py][px] = '█'  # Strong home control
                    elif control > 0.55:
                        pitch[py][px] = '▓'  # Weak home control
                    elif control < 0.35:
                        pitch[py][px] = '░'  # Strong away control
                    elif control < 0.45:
                        pitch[py][px] = '▒'  # Weak away control

    def _add_players(self, pitch: List[List[str]], team: Team, is_home: bool):
        """Add team's players to pitch"""
        for player in team.players:
            # Convert 0-100 position to pitch coordinates
            px = int((player.position.x / 100) * (self.WIDTH - 2)) + 1
            # Flip Y for away team (they attack downward in our view)
            if is_home:
                py = self.HEIGHT - 2 - int((player.position.y / 100) * (self.HEIGHT - 2))
            else:
                py = int(((100 - player.position.y) / 100) * (self.HEIGHT - 2)) + 1

            # Clamp to valid range
            px = max(1, min(self.WIDTH - 2, px))
            py = max(1, min(self.HEIGHT - 2, py))

            # Determine character
            if player.has_ball:
                char = self.HOME_PLAYER_BALL if is_home else self.AWAY_PLAYER_BALL
            else:
                char = self.HOME_PLAYER if is_home else self.AWAY_PLAYER

            pitch[py][px] = char

    def _add_ball(self, pitch: List[List[str]], ball: Ball):
        """Add ball to pitch (if not held by player)"""
        if ball.holder is None:
            px = int((ball.position.x / 100) * (self.WIDTH - 2)) + 1
            py = self.HEIGHT - 2 - int((ball.position.y / 100) * (self.HEIGHT - 2))
            px = max(1, min(self.WIDTH - 2, px))
            py = max(1, min(self.HEIGHT - 2, py))
            pitch[py][px] = self.BALL

    def _pitch_to_string(self, pitch: List[List[str]], state: MatchState) -> str:
        """Convert pitch grid to string with header info"""
        lines = []

        # Header
        lines.append(f"{'═' * self.WIDTH}")
        lines.append(f"  {state.home_team.name} {state.home_score} - {state.away_score} {state.away_team.name}  │  {state.minute}'")
        lines.append(f"{'═' * self.WIDTH}")

        # Pitch
        for row in pitch:
            lines.append(''.join(row))

        # Legend
        lines.append(f"{'─' * self.WIDTH}")
        lines.append(f"  H = {state.home_team.name}  │  A = {state.away_team.name}  │  ● = Ball")

        # Tactical info
        lines.append(f"  Phase: {state.phase.value}  │  Possession: {'Home' if state.home_attacking else 'Away'}")

        if self.show_space_control:
            lines.append(f"  Space: █▓ = Home control  │  ░▒ = Away control")

        return '\n'.join(lines)


class CompactVisualizer:
    """
    More compact visualization showing just key info.
    Good for rapid simulation display.
    """

    WIDTH = 40
    HEIGHT = 25

    def render(self, state: MatchState) -> str:
        """Render compact view"""
        lines = []

        # Score line
        lines.append(f"┌{'─' * (self.WIDTH - 2)}┐")
        score = f"{state.home_team.name} {state.home_score}-{state.away_score} {state.away_team.name}"
        time = f"{state.minute}'"
        header = f"│ {score:^{self.WIDTH - 10}} {time:>5} │"
        lines.append(header)
        lines.append(f"├{'─' * (self.WIDTH - 2)}┤")

        # Mini pitch
        pitch = [[' ' for _ in range(self.WIDTH - 2)] for _ in range(self.HEIGHT - 6)]

        # Add players as numbers
        for player in state.home_team.players:
            px = int((player.position.x / 100) * (self.WIDTH - 4)) + 1
            py = (self.HEIGHT - 7) - int((player.position.y / 100) * (self.HEIGHT - 7))
            px = max(0, min(self.WIDTH - 3, px))
            py = max(0, min(self.HEIGHT - 7, py))
            char = 'Ⓗ' if player.has_ball else 'h'
            pitch[py][px] = char

        for player in state.away_team.players:
            px = int((player.position.x / 100) * (self.WIDTH - 4)) + 1
            py = int(((100 - player.position.y) / 100) * (self.HEIGHT - 7))
            px = max(0, min(self.WIDTH - 3, px))
            py = max(0, min(self.HEIGHT - 7, py))
            char = 'Ⓐ' if player.has_ball else 'a'
            pitch[py][px] = char

        # Ball
        if state.ball.holder is None:
            px = int((state.ball.position.x / 100) * (self.WIDTH - 4)) + 1
            py = (self.HEIGHT - 7) - int((state.ball.position.y / 100) * (self.HEIGHT - 7))
            px = max(0, min(self.WIDTH - 3, px))
            py = max(0, min(self.HEIGHT - 7, py))
            pitch[py][px] = '●'

        for row in pitch:
            lines.append(f"│{''.join(row)}│")

        lines.append(f"└{'─' * (self.WIDTH - 2)}┘")

        return '\n'.join(lines)


class EventLogVisualizer:
    """
    Text-based event log visualization.
    Shows match events in a ticker-style format.
    """

    def __init__(self, max_events: int = 10):
        self.max_events = max_events
        self.events = []

    def add_event(self, minute: int, text: str, is_important: bool = False):
        """Add an event to the log"""
        prefix = "⚡" if is_important else "  "
        self.events.append(f"{prefix} {minute:3}' │ {text}")
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def render(self) -> str:
        """Render the event log"""
        lines = ["┌─ Match Events ─────────────────────────────┐"]
        for event in self.events:
            lines.append(f"│ {event:<43} │")
        while len(lines) < self.max_events + 1:
            lines.append(f"│{' ' * 45}│")
        lines.append("└─────────────────────────────────────────────┘")
        return '\n'.join(lines)


class TacticalVisualizer:
    """
    Visualizes tactical shapes and movements.
    """

    def render_formation(self, team: Team, width: int = 30, height: int = 20) -> str:
        """Render team's current shape"""
        pitch = [[' ' for _ in range(width)] for _ in range(height)]

        # Add players
        for player in team.players:
            px = int((player.position.x / 100) * (width - 2)) + 1
            py = height - 2 - int((player.position.y / 100) * (height - 2))
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))

            # Use role abbreviation
            role_abbrev = player.role[:2].upper() if player.role else str(player.number)
            if len(role_abbrev) == 1:
                pitch[py][px] = role_abbrev
            else:
                if px < width - 1:
                    pitch[py][px] = role_abbrev[0]
                    pitch[py][px + 1] = role_abbrev[1]

        lines = [f"┌{'─' * width}┐"]
        lines.append(f"│ {team.name:^{width - 2}} │")
        lines.append(f"├{'─' * width}┤")
        for row in pitch:
            lines.append(f"│{''.join(row)}│")
        lines.append(f"└{'─' * width}┘")

        # Stats
        lines.append(f"  Compactness: {team.compactness_vertical():.1f}m vert, {team.compactness_horizontal():.1f}m horiz")
        lines.append(f"  Def Line: {team.defensive_line_height():.1f}m")

        return '\n'.join(lines)


def demo_visualization():
    """Demo the visualization system"""
    from teams import create_demo_teams

    home, away = create_demo_teams()
    ball = Ball()
    ball.give_to(home.players[5])  # Give ball to a midfielder

    state = MatchState(
        home_team=home,
        away_team=away,
        ball=ball,
        minute=23,
        home_score=1,
        away_score=0
    )

    # Standard visualization
    viz = ASCIIVisualizer(show_space_control=False)
    print(viz.render(state))
    print()

    # With space control
    viz_control = ASCIIVisualizer(show_space_control=True)
    print(viz_control.render(state))


if __name__ == "__main__":
    demo_visualization()
