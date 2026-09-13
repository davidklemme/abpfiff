"""
Player minds: one owner for each player's decision-psychology internals.

A PlayerMind bundles the instinct bank (System 1 content, seeded by role
and then written to by experience) and the pending decision awaiting an
outcome, so the decision layer (reads) and the learning layer (writes)
share one substrate instead of keeping private copies.

Minds live as long as the registry does: reuse the same Player objects
across matches (a season loop) and their learned instincts and pending
psychology carry over.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from models import Player
from instincts import InstinctBank, default_bank_for
from situation import SituationEmbedding


@dataclass
class PlayerMind:
    player: Player
    bank: InstinctBank
    # The last decision made on the ball, awaiting its outcome:
    # (situation it was made in, chosen action)
    pending_decision: Optional[Tuple[SituationEmbedding, str]] = None

    def remember_decision(self, situation: SituationEmbedding, action: str) -> None:
        self.pending_decision = (situation, action)

    def take_pending(self) -> Optional[Tuple[SituationEmbedding, str]]:
        """Return and clear the pending decision (one outcome per decision)."""
        pending = self.pending_decision
        self.pending_decision = None
        return pending


class MindRegistry:
    """Lazily creates and stores one PlayerMind per player identity.

    Keyed by Player.player_id (stable identity), not object identity, so
    a recreated squad with the same identities finds its accumulated
    minds - the substrate for match-to-match and career continuity."""

    def __init__(self):
        self._minds: Dict[str, PlayerMind] = {}

    def mind_for(self, player: Player) -> PlayerMind:
        mind = self._minds.get(player.player_id)
        if mind is None:
            mind = PlayerMind(player=player, bank=default_bank_for(player))
            self._minds[player.player_id] = mind
        else:
            mind.player = player  # rebind to the current incarnation
        return mind
