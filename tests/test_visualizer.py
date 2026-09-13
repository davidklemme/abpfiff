#!/usr/bin/env python3
"""
Tests for the ASCII visualizer: color output, and that both teams render
from the same absolute coordinate mapping.

No test framework dependency - plain asserts, run directly:
    python3 tests/test_visualizer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchState, Player, Position, Team
from visualizer import ASCIIVisualizer, CompactVisualizer, colorize, COLOR_CODES


def make_player(name, role, x, y) -> Player:
    p = Player(name=name, number=1, role=role)
    p.position = Position(x, y)
    p.base_position = Position(x, y)
    return p


def make_state():
    home = Team(name="Home", players=[make_player("HGK", "gk", 50, 5)])
    away = Team(name="Away", players=[make_player("AGK", "gk", 50, 95)],
                attacks_up=False)
    return MatchState(home_team=home, away_team=away, ball=Ball())


def test_colorize_wraps_and_resets():
    out = colorize("H", "home", enabled=True)
    assert out.startswith("\033[") and out.endswith("\033[0m")
    assert f"38;5;{COLOR_CODES['home']}m" in out
    assert colorize("H", "home", enabled=False) == "H"


def test_render_without_color_has_no_ansi():
    viz = ASCIIVisualizer(use_color=False)
    out = viz.render(make_state())
    assert "\033[" not in out
    assert "H" in out and "A" in out


def test_render_with_color_paints_teams_differently():
    viz = ASCIIVisualizer(use_color=True)
    out = viz.render(make_state())
    assert f"38;5;{COLOR_CODES['home']}m" in out
    assert f"38;5;{COLOR_CODES['away']}m" in out
    assert f"38;5;{COLOR_CODES['field']}m" in out  # light gray pitch


def test_keepers_render_at_opposite_ends_with_one_mapping():
    """Home GK (y=5) must be near the bottom of the grid, away GK (y=95)
    near the top - both through the same absolute mapping."""
    viz = ASCIIVisualizer(use_color=False)
    state = make_state()
    _, py_home = viz._to_cell(state.home_team.players[0].position)
    _, py_away = viz._to_cell(state.away_team.players[0].position)
    assert py_home > ASCIIVisualizer.HEIGHT * 0.75
    assert py_away < ASCIIVisualizer.HEIGHT * 0.25


def test_compact_visualizer_renders_both_teams():
    viz = CompactVisualizer(use_color=False)
    out = viz.render(make_state())
    assert "h" in out and "a" in out
    assert "\033[" not in out


TESTS = [obj for name, obj in list(globals().items())
         if name.startswith("test_") and callable(obj)]


def main():
    failures = []
    for test in TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            failures.append(test.__name__)
            print(f"  FAIL  {test.__name__}: {e}")
        except Exception as e:
            failures.append(test.__name__)
            print(f"  ERROR {test.__name__}: {e!r}")

    print()
    print(f"{len(TESTS) - len(failures)}/{len(TESTS)} passed")
    if failures:
        print("Failed:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
