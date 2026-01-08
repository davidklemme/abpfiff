#!/usr/bin/env python3
"""
Demo script for the Anstoss-style match engine.
Shows tactical football simulation with ASCII visualization.
"""
import time
import sys
import argparse
from typing import List

from models import Ball, MatchState, MatchEvent, Phase
from engine import MatchEngine, SimulationConfig
from teams import (
    create_demo_teams,
    create_tactical_matchup,
    create_pep_city,
    create_klopp_liverpool,
    create_mourinho_bus,
    flip_team_positions
)
from visualizer import ASCIIVisualizer, CompactVisualizer, EventLogVisualizer


def run_visual_match(home_style: str = "positional_play",
                     away_style: str = "gegenpressing",
                     speed: float = 0.3,
                     show_space_control: bool = False):
    """Run a match with live ASCII visualization"""

    print("\n" + "=" * 60)
    print("  ANSTOSS ENGINE - Tactical Football Simulation")
    print("=" * 60 + "\n")

    # Create teams
    if home_style == "pep":
        home = create_pep_city()
    elif home_style == "klopp":
        home = create_klopp_liverpool()
    elif home_style == "mourinho":
        home = create_mourinho_bus()
    else:
        home, _ = create_tactical_matchup(home_style, "balanced")

    if away_style == "pep":
        away = create_pep_city()
        away.name = "FC Guardiola B"
        flip_team_positions(away)
    elif away_style == "klopp":
        away = create_klopp_liverpool()
        away.name = "FC Gegenpresser"
        flip_team_positions(away)
    elif away_style == "mourinho":
        away = create_mourinho_bus()
        away.name = "FC Bus"
        flip_team_positions(away)
    else:
        _, away = create_tactical_matchup("balanced", away_style)  # Already flipped

    print(f"  {home.name} ({home.tactics.name if home.tactics else 'Default'})")
    print(f"  vs")
    print(f"  {away.name} ({away.tactics.name if away.tactics else 'Default'})")
    print("\n" + "-" * 60)
    input("Press Enter to kick off...")

    # Initialize match
    ball = Ball()
    state = MatchState(
        home_team=home,
        away_team=away,
        ball=ball,
        minute=0
    )

    # Give ball to home team striker
    striker = next((p for p in home.players if p.role == 'st'), home.players[-1])
    ball.give_to(striker)
    ball.position = striker.position

    # Setup engine and visualization
    config = SimulationConfig(ticks_per_minute=6, randomness=0.25)
    engine = MatchEngine(config)
    visualizer = ASCIIVisualizer(show_space_control=show_space_control)
    event_log = EventLogVisualizer(max_events=8)

    # Event handler
    def handle_event(event: MatchEvent):
        is_important = event.event_type in ['goal', 'save', 'tackle']
        event_log.add_event(event.minute, event.description, is_important)

    engine.on_event(handle_event)

    # Run match
    try:
        for minute in range(1, 91):
            state.minute = minute

            # Simulate this minute
            events = engine._simulate_minute(state)

            # Render
            print("\033[2J\033[H")  # Clear screen
            print(visualizer.render(state))
            print()
            print(event_log.render())
            print()
            print(f"  Tactics: {home.tactics.name if home.tactics else 'N/A'} vs {away.tactics.name if away.tactics else 'N/A'}")
            print(f"  Press Ctrl+C to end match early")

            # Half time pause
            if minute == 45:
                print("\n  === HALF TIME ===")
                engine._half_time(state)
                time.sleep(2)

            time.sleep(speed)

    except KeyboardInterrupt:
        print("\n\nMatch interrupted!")

    # Final result
    print("\n" + "=" * 60)
    print("  FULL TIME")
    print("=" * 60)
    print(f"\n  {home.name} {state.home_score} - {state.away_score} {away.name}\n")

    return state


def run_quick_simulation(home_style: str, away_style: str, num_matches: int = 100):
    """Run multiple matches quickly to test tactical effectiveness"""

    print(f"\nSimulating {num_matches} matches: {home_style} vs {away_style}")
    print("-" * 50)

    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals = 0
    away_goals = 0

    config = SimulationConfig(ticks_per_minute=8, randomness=0.3)
    engine = MatchEngine(config)

    for i in range(num_matches):
        # Create fresh teams each match
        home, away = create_tactical_matchup(home_style, away_style)

        ball = Ball()
        state = MatchState(home_team=home, away_team=away, ball=ball)

        # Kickoff
        striker = next((p for p in home.players if p.role == 'st'), home.players[-1])
        ball.give_to(striker)

        # Simulate
        state = engine.simulate_match(state, minutes=90)

        # Record results
        home_goals += state.home_score
        away_goals += state.away_score

        if state.home_score > state.away_score:
            home_wins += 1
        elif state.away_score > state.home_score:
            away_wins += 1
        else:
            draws += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_matches} matches...")

    print("\n" + "=" * 50)
    print("  SIMULATION RESULTS")
    print("=" * 50)
    print(f"\n  {home_style.upper()}:")
    print(f"    Wins: {home_wins} ({home_wins/num_matches*100:.1f}%)")
    print(f"    Goals: {home_goals} ({home_goals/num_matches:.2f} per match)")

    print(f"\n  {away_style.upper()}:")
    print(f"    Wins: {away_wins} ({away_wins/num_matches*100:.1f}%)")
    print(f"    Goals: {away_goals} ({away_goals/num_matches:.2f} per match)")

    print(f"\n  Draws: {draws} ({draws/num_matches*100:.1f}%)")
    print()


def show_tactical_comparison():
    """Show how different tactics compare against each other"""

    tactics = ["positional_play", "gegenpressing", "low_block_counter", "balanced"]

    print("\n" + "=" * 70)
    print("  TACTICAL EFFECTIVENESS MATRIX")
    print("  (Win % for row tactic vs column tactic, 50 matches each)")
    print("=" * 70 + "\n")

    results = {}

    for t1 in tactics:
        results[t1] = {}
        for t2 in tactics:
            if t1 == t2:
                results[t1][t2] = 50.0  # vs self = 50%
                continue

            # Quick sim
            config = SimulationConfig(ticks_per_minute=6, randomness=0.3)
            engine = MatchEngine(config)

            wins = 0
            for _ in range(50):
                home, away = create_tactical_matchup(t1, t2)
                ball = Ball()
                state = MatchState(home_team=home, away_team=away, ball=ball)
                striker = next((p for p in home.players if p.role == 'st'), home.players[-1])
                ball.give_to(striker)
                state = engine.simulate_match(state, minutes=90)
                if state.home_score > state.away_score:
                    wins += 1

            results[t1][t2] = wins * 2  # Convert to percentage

    # Print matrix
    header = "                    " + "  ".join(f"{t[:12]:>12}" for t in tactics)
    print(header)
    print("-" * len(header))

    for t1 in tactics:
        row = f"{t1[:18]:<18}"
        for t2 in tactics:
            val = results[t1][t2]
            row += f"  {val:>10.0f}%"
        print(row)

    print("\n  (Higher % = row tactic beats column tactic more often)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Anstoss Engine Demo")
    parser.add_argument('--mode', choices=['visual', 'quick', 'matrix'],
                        default='visual', help='Simulation mode')
    parser.add_argument('--home', default='positional_play',
                        choices=['positional_play', 'gegenpressing', 'low_block_counter',
                                 'balanced', 'pep', 'klopp', 'mourinho'],
                        help='Home team tactical style')
    parser.add_argument('--away', default='gegenpressing',
                        choices=['positional_play', 'gegenpressing', 'low_block_counter',
                                 'balanced', 'pep', 'klopp', 'mourinho'],
                        help='Away team tactical style')
    parser.add_argument('--speed', type=float, default=0.3,
                        help='Visualization speed (seconds per minute)')
    parser.add_argument('--matches', type=int, default=100,
                        help='Number of matches for quick simulation')
    parser.add_argument('--space-control', action='store_true',
                        help='Show space control visualization')

    args = parser.parse_args()

    if args.mode == 'visual':
        run_visual_match(args.home, args.away, args.speed, args.space_control)
    elif args.mode == 'quick':
        run_quick_simulation(args.home, args.away, args.matches)
    elif args.mode == 'matrix':
        show_tactical_comparison()


if __name__ == "__main__":
    main()
