"""
Ball action coordination: what the player in possession does.

DefaultActionResolver is a coordinator, not a god object: the decision is
delegated to an injected DecisionModel (decisions.py), pass mechanics to
PassResolver (passing.py), shooting to ShotResolver (shooting.py), and
out-of-play situations to the RestartPolicy (restarts.py). What remains
here is the dispatch itself plus the small duel/loose-ball resolutions.
"""
import random
from typing import Callable, Optional

from models import BallState, MatchEvent, MatchState, Player, Position, Team
from spatial import SpaceControl
from restarts import SimpleRestartPolicy
from execution import execution_quality, contest
from situation import situation_for
from decisions import DecisionContext, DecisionModel, DualProcessDecisionModel
from passing import (  # noqa: F401 (CONTROL_RADIUS re-exported for tests)
    PassResolver, CONTROL_RADIUS, box_targets
)
from shooting import ShotResolver

# Chance a won tackle deflects the ball over the nearest touchline
TACKLE_DEFLECTION_CHANCE = 0.3

# A loose ball must be run down: only claimable within this radius.
LOOSE_BALL_CLAIM_RADIUS = 4.0

# Decision -> resolution intent: pass decisions carry a bias into
# pass-target scoring instead of branching per action downstream.
PASS_BIAS_BY_ACTION = {"pass_forward": 0.5, "pass_safe": -0.5}

# Defenders under heavy pressure near their own goal clear their lines
HOLDER_CLEARANCE_FRAME_Y = 22.0
HOLDER_CLEARANCE_CHANCE = 0.35

# Discipline: a foul is aggressive INTENT x mistimed EXECUTION. Intent
# comes from the aggression trait; mistiming is 1 - execution quality, the
# same factor stack (skill, fatigue, pressure, confidence, momentum) that
# resolves every other contested action - so tired, rattled, pressured
# challengers foul more without any bespoke formula.
FOUL_INTENT_BASE = 0.5
FOUL_INTENT_AGGRESSION = 0.6

# A defender close enough to the holder engages in a challenge: the duel
# ends in a foul, a dispossession, or the holder riding it and playing on
CHALLENGE_RADIUS = 4.0
CHALLENGE_CHANCE = 0.5
YELLOW_CARD_CHANCE = 0.22
YELLOW_AGGRESSION_WEIGHT = 0.10
STRAIGHT_RED_CHANCE = 0.02


class DefaultActionResolver:
    """Resolves the ball each tick; returns at most one primary MatchEvent.

    Secondary events that precede a restart (e.g. the miscontrol that put
    the ball out of play) are reported through the `publish` callback so the
    engine can emit them and feed psychology before the restart event.
    """

    def __init__(self, space_control: Optional[SpaceControl] = None,
                 restart_policy: Optional[SimpleRestartPolicy] = None,
                 rng: Optional[random.Random] = None,
                 randomness: float = 0.3,
                 publish: Optional[Callable[[MatchEvent, MatchState], None]] = None,
                 decision_model: Optional[DecisionModel] = None):
        self.space_control = space_control or SpaceControl(resolution=10)
        self.restarts = restart_policy or SimpleRestartPolicy(rng=rng)
        self.rng = rng or random.Random()
        self.randomness = randomness
        self.publish = publish or (lambda event, state: None)
        self.decision_model = decision_model or DualProcessDecisionModel(rng=self.rng)
        self.passes = PassResolver(self.space_control, self.restarts,
                                   self.rng, randomness, self.publish)
        self.shots = ShotResolver(self.restarts, self.rng, randomness,
                                  self.publish)

    # -- main entry ---------------------------------------------------------

    def resolve(self, state: MatchState) -> Optional[MatchEvent]:
        """Resolve what happens with the ball"""
        ball = state.ball

        # Handle ball in flight (passes, shots traveling)
        if ball.is_in_flight():
            return self._resolve_ball_in_flight(state)

        holder = ball.holder

        # Handle loose ball
        if holder is None:
            return self._resolve_loose_ball(state)

        # IMPORTANT: Determine attacking/defending based on WHO HAS THE BALL
        holder_is_home = holder in state.home_team.players
        if holder_is_home:
            attacking_team = state.home_team
            defending_team = state.away_team
        else:
            attacking_team = state.away_team
            defending_team = state.home_team

        # Sync the home_attacking flag with reality
        if holder_is_home != state.home_attacking:
            state.home_attacking = holder_is_home

        # Ball position safety net
        ball.position = ball.position.clamp()

        # A defender in touching distance engages the holder: duel first,
        # the chosen action only if the holder rides the challenge
        challenger = min(
            defending_team.players,
            key=lambda p: p.position.distance_to(holder.position),
            default=None
        ) if defending_team.players else None
        challenger_dist = (challenger.position.distance_to(holder.position)
                           if challenger else 50.0)

        if (challenger_dist < CHALLENGE_RADIUS
                and self.rng.random() < CHALLENGE_CHANCE):
            duel_event = self._resolve_duel(state, holder, challenger,
                                            attacking_team, defending_team)
            if duel_event is not None:
                return duel_event
            # Rode the challenge: play continues this tick

        # Pressed deep in their own end, players clear their lines instead
        # of playing through it
        holder_frame_y = attacking_team.frame_y(holder.position.y)
        if (holder_frame_y < HOLDER_CLEARANCE_FRAME_Y
                and challenger_dist < CHALLENGE_RADIUS + 2.0
                and self.rng.random() < HOLDER_CLEARANCE_CHANCE):
            return self.passes.clear_danger(state, holder, attacking_team)

        # Decision: shoot, dribble, cross or pass (dual-process model)
        action = self.decide_action(holder, attacking_team, defending_team, state)

        if action == "shoot" and self.in_shooting_range(holder.position, attacking_team):
            return self.shots.resolve_shot(state, holder, attacking_team,
                                           defending_team)
        if action == "dribble":
            return self._resolve_dribble(state, holder, attacking_team,
                                         defending_team)
        if action == "cross":
            cross_event = self.passes.resolve_cross(state, holder,
                                                    attacking_team, defending_team)
            if cross_event is not None:
                return cross_event
            # Nobody in the box after all: fall through to a normal pass

        # pass_forward / pass_safe bias the pass-target scoring; anything
        # else (a stub model's plain "pass") is neutral
        bias = PASS_BIAS_BY_ACTION.get(action, 0.0)
        return self.passes.resolve_pass(state, holder, attacking_team,
                                        defending_team, forward_bias=bias,
                                        turnover=self._resolve_turnover)

    def _resolve_ball_in_flight(self, state: MatchState) -> Optional[MatchEvent]:
        """Handle ball traveling through the air/ground"""
        ball = state.ball

        # Check for interceptions before ball moves
        interception_event = self.passes.check_interception(state)
        if interception_event:
            return interception_event

        # Move ball
        if not ball.update_flight():
            return None  # Ball still traveling - no event yet

        # Ball arrived at destination
        if ball.state == BallState.SHOT:
            return self.shots.resolve_shot_arrival(state)
        return self.passes.resolve_pass_arrival(state)

    # -- decision -----------------------------------------------------------

    def decide_action(self, holder: Player, attacking_team: Team,
                      defending_team: Team, state: MatchState) -> str:
        """Choose the holder's action via the injected DecisionModel.
        Returns 'shoot', 'dribble', 'pass_forward' or 'pass_safe'."""
        context = self.build_decision_context(holder, attacking_team,
                                              defending_team, state)
        return self.decision_model.decide(context)

    def build_decision_context(self, holder: Player, attacking_team: Team,
                               defending_team: Team,
                               state: MatchState) -> DecisionContext:
        """Assemble everything the decision layer needs about this moment."""
        lanes = self.space_control.find_passing_lanes(
            holder, attacking_team.players, defending_team.players
        )

        f = attacking_team.frame_y
        holder_y = f(holder.position.y)
        best_forward = max((q for t, q in lanes
                            if f(t.position.y) - holder_y > 5), default=0.0)
        best_safe = max((q for t, q in lanes
                         if f(t.position.y) - holder_y <= 5), default=0.0)

        nearest_def_dist = min(
            (p.position.distance_to(holder.position) for p in defending_team.players),
            default=50
        )

        return DecisionContext(
            holder=holder,
            attacking_team=attacking_team,
            defending_team=defending_team,
            state=state,
            situation=situation_for(holder, attacking_team, defending_team,
                                    state, lanes),
            in_shooting_range=self.in_shooting_range(holder.position, attacking_team),
            space_ahead=self._space_ahead(holder, attacking_team, defending_team),
            nearest_defender_dist=nearest_def_dist,
            lanes=lanes,
            best_forward_lane=best_forward,
            best_safe_lane=best_safe,
            box_targets=len(box_targets(holder, attacking_team)),
        )

    def in_shooting_range(self, pos: Position, attacking_team: Team) -> bool:
        """Check if position is in shooting range of the attacked goal"""
        goal = attacking_team.attacking_goal
        distance = pos.distance_to(goal)
        return distance < 30 and attacking_team.frame_y(pos.y) > 70

    def _space_ahead(self, player: Player, attacking_team: Team,
                     defending_team: Team) -> float:
        """Calculate how much space a player has ahead of them"""
        # Check area in front of player (toward the goal they attack)
        f = attacking_team.frame_y
        check_y_frame = min(f(player.position.y) + 20, 100)
        check_pos = Position(player.position.x, f(check_y_frame))

        return min(
            (p.position.distance_to(check_pos) for p in defending_team.players),
            default=50
        )

    # -- loose ball ---------------------------------------------------------

    def _resolve_loose_ball(self, state: MatchState) -> Optional[MatchEvent]:
        """Resolve who gets a loose ball"""
        ball_pos = state.ball.position

        # Find nearest player from each team
        all_players = state.home_team.players + state.away_team.players
        nearest = min(all_players, key=lambda p: p.position.distance_to(ball_pos))

        # Nobody close enough yet: the ball stays loose and players run it
        # down through the movement model (loose-ball chase)
        if nearest.position.distance_to(ball_pos) > LOOSE_BALL_CLAIM_RADIUS:
            return None

        state.ball.give_to(nearest)

        # Update possession
        is_home_player = nearest in state.home_team.players
        if is_home_player != state.home_attacking:
            state.switch_possession()

        return MatchEvent(
            minute=state.minute,
            event_type="ball_won",
            player=nearest,
            position=ball_pos,
            success=True,
            description=f"{nearest.name} wins the ball"
        )

    # -- dribbling / duels --------------------------------------------------

    def _resolve_dribble(self, state: MatchState, dribbler: Player,
                         attacking_team: Team,
                         defending_team: Team) -> Optional[MatchEvent]:
        """Resolve a dribble attempt"""
        # Find nearest defender
        defenders = [p for p in defending_team.players
                     if p.position.distance_to(dribbler.position) < 15]

        if not defenders:
            # Space to dribble
            self.move_dribbler(dribbler, state, attacking_team)
            return MatchEvent(
                minute=state.minute,
                event_type="dribble",
                player=dribbler,
                position=dribbler.position,
                success=True,
                description=f"{dribbler.name} carries the ball forward"
            )

        # Contested dribble: a duel of two full factor stacks - skill,
        # fatigue, pressure, confidence and momentum on BOTH sides
        nearest_def = min(defenders, key=lambda p: p.position.distance_to(dribbler.position))

        q_attacker = execution_quality(dribbler, 'dribbling', state,
                                       defending_team,
                                       momentum=attacking_team.momentum)
        q_defender = execution_quality(nearest_def, 'defending', state,
                                       attacking_team,
                                       momentum=defending_team.momentum)

        success_prob = contest(q_attacker, q_defender)
        success_prob += (self.rng.random() - 0.5) * self.randomness

        if self.rng.random() < success_prob:
            self.move_dribbler(dribbler, state, attacking_team)
            return MatchEvent(
                minute=state.minute,
                event_type="dribble",
                player=dribbler,
                position=dribbler.position,
                success=True,
                description=f"{dribbler.name} beats {nearest_def.name}"
            )
        return self._resolve_tackle(state, dribbler, nearest_def, q_defender)

    def move_dribbler(self, dribbler: Player, state: MatchState,
                      attacking_team: Team):
        """Move dribbler forward toward the goal their team attacks"""
        target = attacking_team.attacking_goal
        new_pos = dribbler.position.move_towards(target, 2)
        dribbler.position = new_pos.clamp()
        state.ball.position = dribbler.position

    def _resolve_duel(self, state: MatchState, holder: Player,
                      challenger: Player, attacking_team: Team,
                      defending_team: Team) -> Optional[MatchEvent]:
        """A defender engages the holder. Outcomes: foul (free kick),
        dispossession (via the tackle path), or None - the holder rides
        the challenge and play continues."""
        q_holder = execution_quality(holder, 'dribbling', state,
                                     defending_team,
                                     momentum=attacking_team.momentum)
        q_challenger = execution_quality(challenger, 'defending', state,
                                         attacking_team,
                                         momentum=defending_team.momentum)

        if self.rng.random() < self._foul_chance(challenger, q_challenger):
            return self._resolve_foul(state, holder, challenger)

        if self.rng.random() < contest(q_challenger, q_holder):
            return self._resolve_tackle(state, holder, challenger, q_challenger)

        return None  # held them off

    def _resolve_tackle(self, state: MatchState, attacker: Player,
                        defender: Player,
                        defender_quality: float = 0.5) -> MatchEvent:
        """Resolve a tackle attempt: clean win, deflection out, or a foul"""
        # Mistimed challenge: foul, possible card, free kick
        if self.rng.random() < self._foul_chance(defender, defender_quality):
            return self._resolve_foul(state, attacker, defender)

        tackle_event = MatchEvent(
            minute=state.minute,
            event_type="tackle",
            player=defender,
            target_player=attacker,
            position=defender.position,
            success=True,
            description=f"{defender.name} tackles {attacker.name}"
        )

        # Some tackles deflect the ball over the nearest touchline:
        # throw-in for the attacker's team (defender touched it last)
        if self.rng.random() < TACKLE_DEFLECTION_CHANCE:
            self.publish(tackle_event, state)
            raw = Position(
                -1.0 if attacker.position.x < 50 else 101.0,
                attacker.position.y
            )
            return self.restarts.resolve_out_of_bounds(state, raw, defender)

        state.ball.give_to(defender)
        state.switch_possession()
        return tackle_event

    def _foul_chance(self, defender: Player, execution_q: float) -> float:
        """Foul = intent (aggression) x mistiming (1 - execution quality).
        The execution quality already carries fatigue, pressure, confidence
        and momentum, so all those factors flow into fouling for free."""
        intent = FOUL_INTENT_BASE + (defender.aggression - 50) / 100.0 * FOUL_INTENT_AGGRESSION
        return max(0.02, intent * (1.0 - execution_q))

    def _resolve_foul(self, state: MatchState, attacker: Player,
                      defender: Player) -> MatchEvent:
        """A mistimed challenge: publish the foul (and any card), then
        restart with a free kick for the fouled player's team."""
        foul_event = MatchEvent(
            minute=state.minute,
            event_type="foul",
            player=defender,
            target_player=attacker,
            position=attacker.position,
            success=False,
            description=f"{defender.name} fouls {attacker.name}"
        )
        self.publish(foul_event, state)
        self._book_if_warranted(state, defender)

        attacked_team = (state.home_team
                         if attacker in state.home_team.players
                         else state.away_team)
        return self.restarts.free_kick(state, attacker, attacked_team)

    def _book_if_warranted(self, state: MatchState, defender: Player) -> None:
        """Card decision for a foul: straight red (rare), yellow (common
        for aggressive players), second yellow = red = sent off."""
        aggression_tilt = (defender.aggression - 50) / 100.0
        roll = self.rng.random()

        if roll < STRAIGHT_RED_CHANCE:
            self._send_off(state, defender)
            return

        yellow_chance = YELLOW_CARD_CHANCE + aggression_tilt * YELLOW_AGGRESSION_WEIGHT
        if roll < STRAIGHT_RED_CHANCE + yellow_chance:
            defender.yellow_cards += 1
            self.publish(MatchEvent(
                minute=state.minute, event_type="yellow_card",
                player=defender, position=defender.position, success=False,
                description=f"Yellow card for {defender.name}"), state)
            if defender.yellow_cards >= 2:
                self._send_off(state, defender)

    def _send_off(self, state: MatchState, player: Player) -> None:
        """Red card: the player leaves the pitch; their team plays on
        short-handed."""
        player.sent_off = True
        self.publish(MatchEvent(
            minute=state.minute, event_type="red_card",
            player=player, position=player.position, success=False,
            description=f"RED CARD! {player.name} is sent off"), state)
        for team in (state.home_team, state.away_team):
            if player in team.players:
                team.players.remove(player)

    def _resolve_turnover(self, state: MatchState, loser: Player,
                          reason: str) -> MatchEvent:
        """Resolve a turnover"""
        # Find nearest opponent
        defending_team = state.defending_team
        nearest_opponent = min(
            defending_team.players,
            key=lambda p: p.position.distance_to(loser.position)
        )

        state.ball.give_to(nearest_opponent)
        state.switch_possession()

        return MatchEvent(
            minute=state.minute,
            event_type="turnover",
            player=loser,
            target_player=nearest_opponent,
            position=loser.position,
            success=False,
            description=f"{loser.name} loses the ball ({reason})"
        )

    # -- facades (stable API for tests, engine delegators, scripts) ----------

    def lead_position(self, passer: Player, receiver: Player,
                      attacking_team: Team) -> Position:
        return self.passes.lead_position(passer, receiver, attacking_team)

    def resolve_shot(self, state: MatchState, shooter: Player,
                     attacking_team: Team, defending_team: Team) -> MatchEvent:
        return self.shots.resolve_shot(state, shooter, attacking_team,
                                       defending_team)

    def _resolve_pass(self, state: MatchState, passer: Player,
                      attacking_team: Team, defending_team: Team,
                      forward_bias: float = 0.0) -> Optional[MatchEvent]:
        return self.passes.resolve_pass(state, passer, attacking_team,
                                        defending_team, forward_bias,
                                        turnover=self._resolve_turnover)

    def _resolve_pass_arrival(self, state: MatchState) -> Optional[MatchEvent]:
        return self.passes.resolve_pass_arrival(state)

    def _check_interception(self, state: MatchState) -> Optional[MatchEvent]:
        return self.passes.check_interception(state)
