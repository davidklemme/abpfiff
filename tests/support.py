"""
Shared test utilities: the runner and common fixtures.

Every test file stays independently runnable (python3 tests/test_x.py);
this module removes the per-file copies of the runner loop and the
player/team factories.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Ball, MatchState, Player, Position, Team  # noqa: E402


def make_player(name="P", role="cm", x=50.0, y=50.0, **attrs) -> Player:
    player = Player(name=name, number=1, role=role, **attrs)
    player.position = Position(x, y)
    player.base_position = Position(x, y)
    return player


def make_teams(home_players, away_players, away_flipped=True):
    home = Team(name="Home", players=home_players)
    away = Team(name="Away", players=away_players, attacks_up=not away_flipped)
    return home, away


def make_match(home_players, away_players, away_flipped=True) -> MatchState:
    home, away = make_teams(home_players, away_players, away_flipped)
    return MatchState(home_team=home, away_team=away, ball=Ball())


def run_tests(module_globals) -> None:
    """Discover test_* callables in the calling module and run them."""
    tests = [obj for name, obj in list(module_globals.items())
             if name.startswith("test_") and callable(obj)]

    failures = []
    for test in tests:
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
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("Failed:", ", ".join(failures))
        sys.exit(1)
