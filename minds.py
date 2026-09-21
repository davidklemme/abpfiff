"""
Player minds: one owner for each player's decision-psychology internals.

A PlayerMind bundles the instinct bank (System 1 content, seeded by role
and then written to by experience) and the pending decision awaiting an
outcome, so the decision layer (reads) and the learning layer (writes)
share one substrate instead of keeping private copies.

Minds live as long as the registry does: reuse the same Player objects
across matches (a season loop) and their learned instincts and pending
psychology carry over. Temporal continuity (docs/specs/
temporal-continuity.md) builds on this: `close_match` is the match
boundary protocol, `to_dict`/`from_dict` persist minds between
processes. Identity is the record, not the current state - everything
here is keyed by the stable `player_id`.
"""
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from models import Player
from instincts import (Instinct, InstinctBank, default_bank_for, DIMENSIONS,
                       LIVED_WIDTH_PLASTIC)
from situation import SituationEmbedding

# Serialization schema tag. Embedding prototypes are stored as plain
# value lists, so a mind saved at fewer dimensions restores cleanly into
# a grown embedding: missing trailing dimensions take their neutral
# dataclass defaults, which the fixed-scale similarity kernel prices at
# zero only where live situations sit at those defaults too.
SCHEMA_VERSION = 2

# Match-boundary rates for transient player state. Memory uses a separate
# match-count power-law clock; confidence still fades continuously in play and
# must not be re-applied here as minutes.
CONFIDENCE_RETENTION_PER_REST_DAY = 0.9  # form fades toward neutral
FATIGUE_REMAINING_PER_REST_DAY = 0.5     # legs recover quickly

DEFAULT_REST_DAYS = 7.0


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
        # Serialized records for identities not (yet) seen this session;
        # revived on first mind_for, re-emitted verbatim by to_dict so a
        # partial squad never loses another identity's history.
        self._dormant: Dict[str, dict] = {}

    def mind_for(self, player: Player) -> PlayerMind:
        mind = self._minds.get(player.player_id)
        if mind is None:
            mind = PlayerMind(player=player, bank=default_bank_for(player))
            record = self._dormant.pop(player.player_id, None)
            if record is not None:
                self._restore_into(mind, record)
            self._minds[player.player_id] = mind
        else:
            mind.player = player  # rebind to the current incarnation
        return mind

    def get(self, player_id: str) -> Optional[PlayerMind]:
        """The mind for an identity, if one exists (no lazy creation)."""
        return self._minds.get(player_id)

    def __iter__(self) -> Iterator[PlayerMind]:
        return iter(self._minds.values())

    def __len__(self) -> int:
        return len(self._minds)

    # -- match boundary protocol (docs/specs/temporal-continuity.md) --------

    def close_match(self, players: Iterable[Player],
                    rest_days: float = DEFAULT_REST_DAYS) -> None:
        """One explicit call at match end.

        Flushes pending decisions (an outcome that never arrived is
        never learned from), advances the unified memory clock by one match,
        and mean-reverts psych/physical state over the passed rest days:
        confidence fades toward neutral, fatigue recovers. Traumas do
        NOT revert - that is what makes them traumas.

        `players` should be the full rosters (confidence and fatigue
        live on players who may never have made an on-ball decision)."""
        for mind in self._minds.values():
            mind.take_pending()
            mind.bank.decay(1.0)

        confidence_factor = CONFIDENCE_RETENTION_PER_REST_DAY ** rest_days
        fatigue_factor = FATIGUE_REMAINING_PER_REST_DAY ** rest_days
        for player in players:
            player.confidence *= confidence_factor
            player.fatigue *= fatigue_factor

    # -- serialization (docs/specs/temporal-continuity.md) -------------------

    def to_dict(self) -> dict:
        """Plain-JSON-able snapshot of complete, evolving trace banks."""
        minds: Dict[str, dict] = {}
        for player_id, mind in self._minds.items():
            minds[player_id] = {
                "confidence": mind.player.confidence,
                "accumulated_evidence": mind.bank.accumulated_evidence,
                "instincts": [
                    {
                        "name": instinct.name,
                        "prototype": list(instinct.prototype.as_tuple()),
                        "weights": dict(instinct.action_weights),
                        "mass": instinct.mass,
                        "source": instinct.source,
                        "widths": list(instinct.widths),
                        "repetitions": instinct.repetitions,
                        "spacing": instinct.spacing,
                        "age": instinct.age,
                        "birth_width": instinct.birth_width,
                    }
                    for instinct in mind.bank.instincts
                    if isinstance(instinct, Instinct)
                ],
            }
        for player_id, record in self._dormant.items():
            minds.setdefault(player_id, record)
        return {"schema": SCHEMA_VERSION, "minds": minds}

    @classmethod
    def from_dict(cls, data: dict,
                  players: Iterable[Player] = ()) -> "MindRegistry":
        """Restore a registry from `to_dict` output. Records for the
        given players are revived immediately (role banks re-seeded,
        learned memories re-attached, confidence restored); records for
        identities not in `players` stay dormant until first seen."""
        schema = data.get("schema")
        if schema not in (1, SCHEMA_VERSION):
            raise ValueError(f"unsupported mind schema: {schema!r}")
        registry = cls()
        if schema == 1:
            data = cls._migrate_v1(data)
        registry._dormant = {player_id: record for player_id, record
                             in data.get("minds", {}).items()}
        for player in players:
            if player.player_id in registry._dormant:
                registry.mind_for(player)
        return registry

    @staticmethod
    def _restore_into(mind: PlayerMind, record: dict) -> None:
        mind.player.confidence = record["confidence"]
        if not record.get("migrated_v1"):
            mind.bank.instincts = []
        for entry in record["instincts"]:
            mind.bank.instincts.append(Instinct(
                name=entry["name"],
                prototype=SituationEmbedding(*entry["prototype"]),
                action_weights=dict(entry["weights"]),
                mass=entry["mass"],
                source=entry["source"],
                widths=tuple(entry["widths"]),
                repetitions=entry["repetitions"],
                spacing=entry["spacing"],
                age=entry["age"],
                birth_width=entry["birth_width"],
            ))
        mind.bank.accumulated_evidence = record.get(
            "accumulated_evidence",
            sum(trace.repetitions for trace in mind.bank.learned_memories()))

    @staticmethod
    def _migrate_v1(data: dict) -> dict:
        """Translate scalar learned records; role traces are freshly derived.

        Version 1 never persisted role seeds, so migration materializes current
        role traces when the player is revived and appends these migrated
        learned entries in ``_restore_into`` via the marker below.
        """
        minds = {}
        for player_id, record in data.get("minds", {}).items():
            entries = []
            for entry in record.get("instincts", []):
                strength = max(0.0, float(entry.get("strength", 0.0)))
                entries.append({
                    "name": entry["name"], "prototype": entry["prototype"],
                    "weights": entry["weights"], "mass": strength,
                    "source": entry["source"],
                    "widths": [LIVED_WIDTH_PLASTIC] * DIMENSIONS,
                    "repetitions": max(0.0, math.expm1(strength)),
                    "spacing": max(0.0, math.expm1(strength)),
                    "age": 0.0, "birth_width": LIVED_WIDTH_PLASTIC,
                })
            minds[player_id] = {"confidence": record["confidence"],
                                "accumulated_evidence": sum(
                                    entry["repetitions"] for entry in entries),
                                "instincts": entries, "migrated_v1": True}
        return {"schema": SCHEMA_VERSION, "minds": minds}
