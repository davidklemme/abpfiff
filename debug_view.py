#!/usr/bin/env python3
"""
Debug visualizer with colors and step-by-step simulation.
Run with: python3 debug_view.py
"""
import time
import sys
from models import Ball, MatchState, MatchEvent, Position
from engine import MatchEngine, SimulationConfig
from teams import create_pep_city, create_klopp_liverpool, flip_team_positions
from spatial import SpaceControl


# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Home team = Blue
    HOME = '\033[94m'        # Bright blue
    HOME_BOLD = '\033[1;94m'

    # Away team = Red
    AWAY = '\033[91m'        # Bright red
    AWAY_BOLD = '\033[1;91m'

    # Ball = Yellow
    BALL = '\033[93m'
    BALL_BOLD = '\033[1;93m'

    # Pitch = Green
    PITCH = '\033[32m'
    PITCH_DIM = '\033[2;32m'

    # Info
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'


class DebugVisualizer:
    """
    Color-coded debug visualizer with player numbers.
    """

    WIDTH = 70
    HEIGHT = 35

    def __init__(self):
        self.space_control = SpaceControl(resolution=10)

    def render(self, state: MatchState, last_event: MatchEvent = None) -> str:
        """Render with colors and player numbers"""
        lines = []

        # Header
        lines.append(f"{Colors.WHITE}{'═' * self.WIDTH}{Colors.RESET}")
        home_str = f"{Colors.HOME_BOLD}{state.home_team.name}{Colors.RESET}"
        away_str = f"{Colors.AWAY_BOLD}{state.away_team.name}{Colors.RESET}"
        score = f"{home_str} {Colors.WHITE}{state.home_score} - {state.away_score}{Colors.RESET} {away_str}"
        lines.append(f"  {score}  │  {Colors.CYAN}{state.minute}'{Colors.RESET}")
        lines.append(f"{Colors.WHITE}{'═' * self.WIDTH}{Colors.RESET}")

        # Create pitch grid (stores tuples of (char, color))
        pitch = [[(Colors.PITCH_DIM + '·' + Colors.RESET) for _ in range(self.WIDTH)]
                 for _ in range(self.HEIGHT)]

        # Draw pitch markings
        self._draw_pitch_markings(pitch)

        # Add home players (BLUE) - they attack upward (toward y=0 on screen)
        for player in state.home_team.players:
            px = int((player.position.x / 100) * (self.WIDTH - 2)) + 1
            py = self.HEIGHT - 2 - int((player.position.y / 100) * (self.HEIGHT - 2))
            px = max(1, min(self.WIDTH - 2, px))
            py = max(1, min(self.HEIGHT - 2, py))

            num = str(player.number) if player.number < 10 else str(player.number)
            if player.has_ball:
                pitch[py][px] = f"{Colors.BALL_BOLD}⬤{Colors.RESET}"
                # Show number next to ball holder
                if px + 1 < self.WIDTH - 1:
                    pitch[py][px + 1] = f"{Colors.HOME_BOLD}{num}{Colors.RESET}"
            else:
                pitch[py][px] = f"{Colors.HOME_BOLD}{num}{Colors.RESET}"

        # Add away players (RED) - they attack downward (toward y=HEIGHT on screen)
        # Away team's goal is at screen TOP (row 0), so their y=5 maps to top
        for player in state.away_team.players:
            px = int((player.position.x / 100) * (self.WIDTH - 2)) + 1
            py = int((player.position.y / 100) * (self.HEIGHT - 2)) + 1
            px = max(1, min(self.WIDTH - 2, px))
            py = max(1, min(self.HEIGHT - 2, py))

            num = str(player.number) if player.number < 10 else str(player.number)
            if player.has_ball:
                pitch[py][px] = f"{Colors.BALL_BOLD}⬤{Colors.RESET}"
                if px + 1 < self.WIDTH - 1:
                    pitch[py][px + 1] = f"{Colors.AWAY_BOLD}{num}{Colors.RESET}"
            else:
                pitch[py][px] = f"{Colors.AWAY_BOLD}{num}{Colors.RESET}"

        # Add loose ball
        if state.ball.holder is None:
            px = int((state.ball.position.x / 100) * (self.WIDTH - 2)) + 1
            py = self.HEIGHT - 2 - int((state.ball.position.y / 100) * (self.HEIGHT - 2))
            px = max(1, min(self.WIDTH - 2, px))
            py = max(1, min(self.HEIGHT - 2, py))
            pitch[py][px] = f"{Colors.BALL_BOLD}●{Colors.RESET}"

        # Render pitch
        for row in pitch:
            lines.append(''.join(row))

        # Legend
        lines.append(f"{Colors.GRAY}{'─' * self.WIDTH}{Colors.RESET}")
        lines.append(
            f"  {Colors.HOME_BOLD}■ {state.home_team.name}{Colors.RESET}  │  "
            f"{Colors.AWAY_BOLD}■ {state.away_team.name}{Colors.RESET}  │  "
            f"{Colors.BALL_BOLD}⬤ Ball{Colors.RESET}"
        )

        # Ball holder info
        holder = state.ball.holder
        if holder:
            team_color = Colors.HOME_BOLD if holder in state.home_team.players else Colors.AWAY_BOLD
            lines.append(
                f"  {Colors.WHITE}Ball:{Colors.RESET} {team_color}#{holder.number} {holder.name}{Colors.RESET} "
                f"at ({holder.position.x:.0f}, {holder.position.y:.0f})"
            )

        # Last event
        if last_event:
            event_color = Colors.YELLOW if last_event.event_type in ['goal', 'shot', 'save'] else Colors.WHITE
            lines.append(f"  {event_color}► {last_event.description}{Colors.RESET}")

        return '\n'.join(lines)

    def _draw_pitch_markings(self, pitch):
        """Draw the pitch lines"""
        # Border
        for x in range(self.WIDTH):
            pitch[0][x] = f"{Colors.PITCH}─{Colors.RESET}"
            pitch[self.HEIGHT - 1][x] = f"{Colors.PITCH}─{Colors.RESET}"
        for y in range(self.HEIGHT):
            pitch[y][0] = f"{Colors.PITCH}│{Colors.RESET}"
            pitch[y][self.WIDTH - 1] = f"{Colors.PITCH}│{Colors.RESET}"

        # Corners
        pitch[0][0] = f"{Colors.PITCH}┌{Colors.RESET}"
        pitch[0][self.WIDTH - 1] = f"{Colors.PITCH}┐{Colors.RESET}"
        pitch[self.HEIGHT - 1][0] = f"{Colors.PITCH}└{Colors.RESET}"
        pitch[self.HEIGHT - 1][self.WIDTH - 1] = f"{Colors.PITCH}┘{Colors.RESET}"

        # Center line
        cy = self.HEIGHT // 2
        for x in range(1, self.WIDTH - 1):
            pitch[cy][x] = f"{Colors.PITCH}─{Colors.RESET}"

        # Center circle
        cx = self.WIDTH // 2
        pitch[cy][cx] = f"{Colors.PITCH}◎{Colors.RESET}"
        for dx, dy in [(-4, 0), (4, 0), (0, -2), (0, 2), (-3, -1), (3, -1), (-3, 1), (3, 1)]:
            nx, ny = cx + dx, cy + dy
            if 1 < nx < self.WIDTH - 2 and 1 < ny < self.HEIGHT - 2:
                pitch[ny][nx] = f"{Colors.PITCH}○{Colors.RESET}"

        # Goals
        goal_w = 10
        gs = (self.WIDTH - goal_w) // 2
        ge = gs + goal_w
        for x in range(gs, ge):
            pitch[0][x] = f"{Colors.WHITE}═{Colors.RESET}"
            pitch[self.HEIGHT - 1][x] = f"{Colors.WHITE}═{Colors.RESET}"


def run_debug_simulation():
    """Run step-by-step simulation with debug output"""

    print(f"\n{Colors.BOLD}=== ANSTOSS ENGINE DEBUG MODE ==={Colors.RESET}\n")
    print(f"Controls:")
    print(f"  {Colors.CYAN}ENTER{Colors.RESET} = Next tick")
    print(f"  {Colors.CYAN}m{Colors.RESET}     = Skip to next minute")
    print(f"  {Colors.CYAN}q{Colors.RESET}     = Quit")
    print()

    # Create teams
    home = create_pep_city()
    away = create_klopp_liverpool()
    away.name = "FC Klopp"
    flip_team_positions(away)  # Flip so away team faces opposite direction

    print(f"{Colors.HOME_BOLD}{home.name}{Colors.RESET} ({home.tactics.name})")
    print(f"  vs")
    print(f"{Colors.AWAY_BOLD}{away.name}{Colors.RESET} ({away.tactics.name})")
    print()
    input("Press ENTER to start...")

    # Initialize
    ball = Ball()
    state = MatchState(home_team=home, away_team=away, ball=ball, minute=0)

    # Give ball to home striker at center
    striker = next((p for p in home.players if p.role == 'st'), home.players[-1])
    striker.position = Position(50, 50)
    ball.give_to(striker)
    ball.position = Position(50, 50)

    # Engine with slow ticks
    config = SimulationConfig(ticks_per_minute=6, randomness=0.25)
    engine = MatchEngine(config)
    viz = DebugVisualizer()

    last_event = None
    tick = 0
    minute = 0

    try:
        while minute < 90:
            # Clear and render
            print("\033[2J\033[H", end="")  # Clear screen
            print(viz.render(state, last_event))
            print()
            print(f"{Colors.GRAY}Minute {minute} │ Tick {tick % config.ticks_per_minute + 1}/{config.ticks_per_minute}{Colors.RESET}")
            print(f"{Colors.GRAY}[ENTER]=tick  [m]=minute  [q]=quit{Colors.RESET}")

            # Wait for input
            try:
                cmd = input().strip().lower()
            except EOFError:
                break

            if cmd == 'q':
                break
            elif cmd == 'm':
                # Skip to next minute
                remaining = config.ticks_per_minute - (tick % config.ticks_per_minute)
                for _ in range(remaining):
                    events = engine.simulate_tick(state)
                    if events:
                        last_event = events[-1]
                    tick += 1
                minute += 1
                state.minute = minute
            else:
                # Single tick
                events = engine.simulate_tick(state)
                if events:
                    last_event = events[-1]
                tick += 1
                if tick % config.ticks_per_minute == 0:
                    minute += 1
                    state.minute = minute

    except KeyboardInterrupt:
        pass

    # Final
    print("\033[2J\033[H", end="")
    print(f"\n{Colors.BOLD}=== FINAL RESULT ==={Colors.RESET}")
    print(f"{Colors.HOME_BOLD}{home.name}{Colors.RESET} {state.home_score} - {state.away_score} {Colors.AWAY_BOLD}{away.name}{Colors.RESET}")
    print()


if __name__ == "__main__":
    run_debug_simulation()
