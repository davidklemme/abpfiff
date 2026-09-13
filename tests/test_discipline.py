#!/usr/bin/env python3
"""
Tests for discipline: fouls (intent x mistimed execution), cards,
sending-off, and free-kick restarts - plus stable player identity.

Run directly: python3 tests/test_discipline.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, make_match, run_tests
from models import Ball, MatchState, Player, Position
from engine import MatchEngine, SimulationConfig
from minds import MindRegistry
from teams import create_tactical_matchup


def resolver(seed=1):
    return MatchEngine(SimulationConfig(seed=seed)).actions


# ---------------------------------------------------------------------------
# Foul model: intent x mistimed execution
# ---------------------------------------------------------------------------

def test_aggression_raises_foul_chance():
    actions = resolver()
    calm = make_player("Calm", aggression=20)
    nasty = make_player("Nasty", aggression=90)
    assert actions._foul_chance(nasty, 0.5) > actions._foul_chance(calm, 0.5)


def test_poor_execution_raises_foul_chance():
    """The factor stack does the work: a low execution quality (tired,
    pressured, rattled) means more mistimed challenges."""
    actions = resolver()
    defender = make_player("D", aggression=50)
    assert actions._foul_chance(defender, 0.2) > actions._foul_chance(defender, 0.8)


def test_foul_awards_free_kick_to_the_fouled_team():
    engine = MatchEngine(SimulationConfig(seed=2))
    attacker = make_player("Attacker", role="st", x=60, y=70)
    defender = make_player("Defender", role="cb", x=61, y=70)
    state = make_match([attacker], [defender])
    state.ball.give_to(attacker)

    event = engine.actions._resolve_foul(state, attacker, defender)

    assert event.event_type == "free_kick"
    assert state.ball.holder in state.home_team.players
    assert state.home_attacking


def test_second_yellow_sends_the_player_off():
    engine = MatchEngine(SimulationConfig(seed=3))
    hothead = make_player("Hothead", role="cb", aggression=95)
    hothead.yellow_cards = 1
    state = make_match([make_player("St")], [hothead])

    cards = []
    engine.on_event(lambda ev: cards.append(ev.event_type)
                    if ev.event_type in ("yellow_card", "red_card") else None)

    # Force the booking path until a second yellow lands
    for _ in range(200):
        if hothead.sent_off:
            break
        engine.actions._book_if_warranted(state, hothead)

    assert hothead.sent_off
    assert "red_card" in cards
    assert hothead not in state.away_team.players  # plays on short-handed


def test_cards_dent_confidence():
    import psychology
    from models import MatchEvent
    player = make_player("Booked")
    state = make_match([player], [make_player("Opp")])

    psychology.process_feedback(MatchEvent(
        minute=30, event_type="yellow_card", player=player), state)
    after_yellow = player.confidence
    psychology.process_feedback(MatchEvent(
        minute=60, event_type="red_card", player=player), state)

    assert after_yellow < 0
    assert player.confidence < after_yellow


def test_matches_produce_fouls_and_free_kicks():
    counts = {"foul": 0, "free_kick": 0, "yellow_card": 0}
    engine = MatchEngine(SimulationConfig(ticks_per_minute=6, seed=44))
    engine.on_event(lambda ev: counts.__setitem__(
        ev.event_type, counts[ev.event_type] + 1)
        if ev.event_type in counts else None)

    for _ in range(4):
        home, away = create_tactical_matchup("balanced", "balanced")
        state = MatchState(home_team=home, away_team=away, ball=Ball())
        engine.simulate_match(state, minutes=90)

    assert counts["foul"] > 5, counts
    assert counts["free_kick"] == counts["foul"], counts  # every foul restarts
    assert counts["yellow_card"] > 0, counts


# ---------------------------------------------------------------------------
# Stable identity
# ---------------------------------------------------------------------------

def test_player_id_is_stable_across_recreation():
    first = Player(name="Erika Musterfrau", number=9)
    second = Player(name="Erika Musterfrau", number=9)
    assert first.player_id == second.player_id
    assert first is not second


def test_minds_follow_identity_not_object():
    registry = MindRegistry()
    first = make_player("Keeper", role="gk")
    mind = registry.mind_for(first)
    mind.bank.instincts.append("marker")  # tag the bank

    reincarnation = make_player("Keeper", role="gk")
    assert registry.mind_for(reincarnation) is mind
    assert registry.mind_for(reincarnation).player is reincarnation


def test_mirror_matchups_have_distinct_identities():
    home, away = create_tactical_matchup("balanced", "balanced")
    home_ids = {p.player_id for p in home.players}
    away_ids = {p.player_id for p in away.players}
    assert not home_ids & away_ids, "opponents must never share identities"


if __name__ == "__main__":
    run_tests(globals())
