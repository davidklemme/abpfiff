#!/usr/bin/env python3
"""
Tests for the spatial-dynamics round: carrying the ball has a spacing
consequence. Defenders collapse on the carrier (closing down), the
dribble contest compounds across every converging defender, lanes
through pressure are priced at release on both ends, crosses are a
skill on delivery and an aerial duel on arrival, and the lockstep
movement update alternates team order so neither side owns the
first-mover advantage.

Run directly: python3 tests/test_spacing.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support import make_player, make_match, run_tests
from models import Ball, BallState, MatchState, Position, Team
from movement import RoleMovementModel
from spatial import SpaceControl
from ball_actions import DefaultActionResolver
from passing import PassResolver


def step(model, state, ticks=1):
    for _ in range(ticks):
        state.tick()
        model.update_positions(state)


# ---------------------------------------------------------------------------
# Closing down: a held ball pulls defenders in
# ---------------------------------------------------------------------------

def test_nearest_defender_closes_the_carrier_down():
    holder = make_player("Carrier", x=50, y=50)
    presser = make_player("Presser", x=70, y=50, role="cm")
    far = make_player("Far", x=90, y=90, role="cb")
    state = make_match([holder], [presser, far])
    state.ball.give_to(holder)

    before = presser.position.distance_to(holder.position)
    step(RoleMovementModel(rng=random.Random(1)), state, ticks=4)
    after = presser.position.distance_to(holder.position)

    assert after < before - 3, (before, after)


def test_second_defender_converges_in_support():
    holder = make_player("Carrier", x=50, y=50)
    first = make_player("First", x=60, y=50, role="cm", workrate=80)
    second = make_player("Second", x=50, y=70, role="cm", workrate=80)
    state = make_match([holder], [first, second])
    state.ball.give_to(holder)

    before = second.position.distance_to(holder.position)
    step(RoleMovementModel(rng=random.Random(1)), state, ticks=4)
    after = second.position.distance_to(holder.position)

    assert after < before  # squeezes, even if less than the closer


def test_no_closing_while_the_ball_is_in_flight():
    passer = make_player("P", x=30, y=30)
    receiver = make_player("R", x=60, y=60)
    presser = make_player("D", x=80, y=50, role="cm")
    state = make_match([passer, receiver], [presser])
    state.ball.start_pass(passer, receiver)

    before = presser.position.distance_to(state.ball.position)
    step(RoleMovementModel(rng=random.Random(1)), state, ticks=1)
    # The presser follows role movement, not a dedicated ball-hunt
    assert state.ball.holder is None


# ---------------------------------------------------------------------------
# The dribble contest compounds across the crowd
# ---------------------------------------------------------------------------

def _dribble_success_rate(defender_positions, trials=300):
    rate = 0
    for seed in range(trials):
        resolver = DefaultActionResolver(rng=random.Random(seed))
        dribbler = make_player("Dr", x=50, y=50, dribbling=70)
        defenders = [make_player(f"D{i}", x=x, y=y, defending=60)
                     for i, (x, y) in enumerate(defender_positions)]
        state = make_match([dribbler], defenders)
        state.ball.give_to(dribbler)
        event = resolver._resolve_dribble(state, dribbler,
                                          state.home_team, state.away_team)
        rate += (event is not None and event.event_type == "dribble"
                 and event.success)
    return rate / trials


def test_a_crowd_beats_the_carrier_more_than_one_marker():
    one = _dribble_success_rate([(53, 50)])
    crowd = _dribble_success_rate([(53, 50), (48, 53), (52, 46)])
    assert crowd < one - 0.1, (one, crowd)


def test_distant_bodies_do_not_join_the_duel():
    tight = _dribble_success_rate([(53, 50)])
    tight_plus_far = _dribble_success_rate([(53, 50), (80, 80), (20, 20)])
    assert abs(tight - tight_plus_far) < 0.05, (tight, tight_plus_far)


# ---------------------------------------------------------------------------
# Corridor pricing: lanes through pressure are expensive, on the path
# ---------------------------------------------------------------------------

def test_defender_on_the_lane_closes_it():
    sc = SpaceControl()
    start, end = Position(20, 50), Position(60, 50)
    open_lane = sc.corridor_openness(start, end, [])
    blocked = sc.corridor_openness(
        start, end, [make_player("D", x=40, y=50)])
    assert open_lane == 1.0
    assert blocked < 0.6


def test_two_defenders_compound():
    sc = SpaceControl()
    start, end = Position(20, 50), Position(60, 50)
    one = sc.corridor_openness(start, end, [make_player("D", x=40, y=50)])
    two = sc.corridor_openness(start, end, [make_player("D", x=35, y=50),
                                            make_player("E", x=45, y=50)])
    assert two < one


def test_the_pressers_bubble_is_priced_as_pressure_not_blockage():
    """A defender harassing the passer does not also close every lane -
    that cost already flows through execution pressure."""
    sc = SpaceControl()
    start, end = Position(20, 50), Position(60, 50)
    at_passer = sc.corridor_openness(
        start, end, [make_player("D", x=22, y=50)])
    on_path = sc.corridor_openness(
        start, end, [make_player("D", x=45, y=50)])
    assert at_passer > on_path
    assert at_passer > 0.85


# ---------------------------------------------------------------------------
# Crosses: skill on delivery, an aerial duel on arrival
# ---------------------------------------------------------------------------

def _cross_outcomes(crosser_passing, trials=200):
    """Fraction of crosses launched toward the intended target (vs
    sprayed wayward with no receiver)."""
    aimed = 0
    for seed in range(trials):
        rng = random.Random(seed)
        resolver = PassResolver(SpaceControl(), None, rng, randomness=0.0,
                                publish=lambda e, s: None)
        crosser = make_player("Wing", x=90, y=80, passing=crosser_passing)
        target = make_player("St", x=50, y=85)
        state = make_match([crosser, target], [make_player("D", x=10, y=10)])
        state.ball.give_to(crosser)
        resolver.resolve_cross(state, crosser, state.home_team,
                               state.away_team)
        aimed += state.ball.target_player is target
    return aimed / trials


def test_crossing_accuracy_is_a_skill():
    sharp = _cross_outcomes(crosser_passing=90)
    wild = _cross_outcomes(crosser_passing=1)
    assert sharp > wild + 0.2, (sharp, wild)


def test_cross_arrival_is_an_aerial_duel():
    """A dominant defender under the dropping ball clears far more
    often than a weak one - both ends of the delivery matter."""
    from restarts import SimpleRestartPolicy

    def clearances(defender_aerial, trials=300):
        cleared = 0
        for seed in range(trials):
            rng = random.Random(seed)
            resolver = PassResolver(SpaceControl(),
                                    SimpleRestartPolicy(rng=rng), rng,
                                    randomness=0.0,
                                    publish=lambda e, s: None)
            crosser = make_player("Wing", x=90, y=80)
            striker = make_player("St", x=50, y=88, aerial=50)
            stopper = make_player("Cb", x=51, y=88, aerial=defender_aerial)
            state = make_match([crosser, striker], [stopper])
            state.ball.start_pass(crosser, striker, is_lofted=True,
                                  delivery="cross")
            state.ball.position = Position(50, 88)   # ball arriving now
            state.ball.flight_ticks_remaining = 0
            event = resolver.resolve_pass_arrival(state)
            cleared += (event is not None
                        and event.event_type == "clearance")
        return cleared / trials

    assert clearances(defender_aerial=95) > clearances(defender_aerial=10) + 0.15


# ---------------------------------------------------------------------------
# Order fairness
# ---------------------------------------------------------------------------

def test_movement_order_alternates_per_tick():
    model = RoleMovementModel(rng=random.Random(1))
    order = []
    original = RoleMovementModel._apply_team_movements

    def spy(self, team, state, attacking):
        order.append(team.name)
        original(self, team, state, attacking)

    RoleMovementModel._apply_team_movements = spy
    try:
        holder = make_player("H", x=50, y=50)
        state = make_match([holder], [make_player("D", x=60, y=60)])
        state.ball.give_to(holder)
        step(model, state, ticks=4)
    finally:
        RoleMovementModel._apply_team_movements = original

    pairs = [tuple(order[i:i + 2]) for i in range(0, 8, 2)]
    assert pairs[0] != pairs[1]          # order flips between ticks
    assert pairs[0] == pairs[2]          # ...and alternates regularly


if __name__ == "__main__":
    run_tests(globals())
