# Temporal Continuity — Little Spec

Status: first slice IMPLEMENTED 2026-09-13 (minds.close_match,
minds.to_dict/from_dict, series.py runner + occasion schedule +
trajectory gates, tests/test_continuity.py; CI runs the series gate
on seeds 42 and 99). Leagues/teams still deliberately deferred.

For an action to have an effect outside the game it needs continuation.
This spec defines the smallest mechanism that makes match outcomes
*matter beyond the match* — and deliberately defers the league/season
framework around it.

## Principles (prior art)

Pulled from the knowledge graph before writing this:

1. **Identity is the record, not the current state** (MCube belt-identity
   decision): a belt stays the same belt through splices and cutouts —
   the asset tracks physical changes, the record continues. Applied
   here: `player_id` is the continuous thread; attributes, confidence,
   and memories are state that changes *on* that thread. Development
   (attribute change over time) never needs a new identity.
2. **Ship the smallest persistent memory level first, defer the
   scaffolding** (Komplyzen agent/memory-model decision): persistent
   state + memory was shipped as thin levels on existing infrastructure,
   cross-context memory explicitly deferred. Applied here: minds already
   persist in-process (`MindRegistry`, keyed by `player_id`); the next
   level is serialize/restore + a boundary protocol — not leagues.
3. **A wrong merge is far more damaging than two separate records**
   (KG dedup calibration): consolidation at boundaries must be
   conservative. Applied here: the match boundary applies decay and
   pruning, but never merges memories more aggressively than the
   in-match `MEMORY_MERGE_SIMILARITY` rule.
4. **This repo's own precedent**: every phase shipped as a thin,
   validation-gated slice against the real engine, no framework
   ceremony first. Temporal continuity follows the same shape.

## Q: Is this sensible without leagues and teams?

**Yes — stable IDs first, watch the evolution.** Decision and rationale:

- The interesting open question is whether experience accumulation
  produces *sensible trajectories* — confidence arcs, trauma
  accumulation, veterans diverging from debutants, memory banks that
  don't degenerate. All of that is observable with nothing more than
  persistent squads playing repeated matches. A league adds narrative,
  not signal.
- The one thing a league genuinely provides — **occasion variance**
  (stakes, crowds: the load dimension needs big nights and quiet
  Tuesdays to matter over a career) — is cheaply simulated by an
  **occasion schedule**: a seeded generator of `Environment` values per
  match. That is the league's *interface* without its machinery.
- Building league scaffolding before we know the evolution dynamics are
  healthy repeats the original sin this project was reviewed for
  (design ahead of validation, finding F9). Leagues become worth
  building exactly when trajectories look sane and we want standings,
  promotion pressure, and derby stakes to *generate* the occasion
  schedule instead of sampling it.

## The mechanism

### 1. Match boundary protocol

`MindRegistry.close_match(players, rest_days)` — one explicit call at
match end (the series runner invokes it; `players` is the full rosters,
because confidence and fatigue live on players who may never have made
an on-ball decision):

- **Flush pending decisions**: a decision whose outcome never arrived
  is dropped, never learned from.
- **Consolidate**: apply *between-match* decay for the rest days
  (`bank.decay` at a calmer rate than the in-match `DECAY_PER_MINUTE` -
  match-time decay already runs continuously per tick, engine.py, so
  the boundary must not re-apply it), prune to `MAX_LEARNED_MEMORIES`,
  forget below `MIN_MEMORY_STRENGTH`. No boundary-time merging beyond
  the in-match rule (principle 3).
- **Psych state reversion**: confidence mean-reverts toward baseline
  (a night's sleep) and fatigue recovers, both as a function of rest
  days. Traumas do NOT revert — that is what makes them traumas.
  (Match-scoped state — team momentum, cards, sent-off removals,
  positions — is the series runner's reset, not the mind's.)
- Role-seeded schooling is never pruned or decayed away: it is the
  floor lived experience is written over.

### 2. Serialization (mind persistence)

`minds.py` gains `to_dict()/from_dict()` (plain JSON): per player_id,
the learned instincts (prototype tuple, action weights, strength,
source) and confidence. Versioned with a schema tag; embeddings are
tuples so dimension growth is detectable (a stored 7-tuple read by an
8-dimension engine pads the missing dimension with its neutral default
— the fixed-scale similarity kernel makes that geometrically sound).
Role seeds are NOT serialized — they are derived from the player and
re-seeded on load (keeps files small and lets schooling tuning apply
retroactively).

### 3. Series runner

`series.py` (or a `validate.py` mode): N matches between persistent
squads, minds carried across, an **occasion schedule** (seeded
`Environment` per match: mostly routine, some big nights), boundary
protocol between matches. This is the whole "framework" for now.

### 4. Observability — watch the evolution

Extend the validation harness with per-series trajectory metrics, each
with a gate band (degenerate = CI fail) and a target band:

- bank composition over time: learned-memory count, mean strength,
  anchor:trauma ratio (gate: does not collapse to all-trauma or
  saturate at the prune cap immediately)
- confidence trajectory: distribution across the series (gate: no
  permanent pinning at ±1)
- behavioral drift: bold-action share match 1 vs match N for the same
  archetypes (gate: separation between archetypes survives learning)
- veteran effect: familiarity on big nights, experienced vs fresh squad
  (target: grows with lived big nights — the load dimension's promise)
- round-trip determinism: serialize → restore → identical next match
  under the same seed (gate)

### Acceptance gates for the first slice (all met)

1. A 10-match series with persistent minds runs deterministically under
   a seed, and `close_match` is covered by tests (flush, reversion,
   consolidation, schooling floor).
2. Serialization round-trips byte-stable and dimension-forward.
3. Trajectory metrics exist in the harness with at least the degenerate
   gates armed on 3 seeds.

## Q: Break the vector approach out sport-independent?

**Yes in principle, not yet in packaging.** The cognition core is
already *nearly* sport-agnostic — and should be kept that way — but
extraction now would be abstracting from one example.

- **Sport-agnostic today (the "cognition core")**: the embedding
  container + fixed-scale similarity kernel (`situation.py`'s data
  half), `InstinctBank` (prototypes, recall, learn/merge/prune/decay),
  `PlayerMind`/`MindRegistry`, the dual-process blend, load/overload/
  composure math (`psychology`'s capacity half), outcome→valence
  learning rules as data. None of this knows what football is.
- **Sport adapter (football)**: dimension *semantics* and extraction
  (`situation_for`), the action vocabulary, System 2 utilities,
  execution/contest resolution, laws of the game, roles/seeding
  content.
- **Decision**: enforce the boundary in-repo now — core modules must
  not grow imports of match semantics; the action vocabulary and role
  seeds are adapter *content* fed into core *machinery*. (Known smells
  to clean opportunistically, not as a project: `situation_for` — pure
  adapter code — lives in the core's module; `default_bank_for` bakes
  football roles into `instincts.py`.) Extract to a separate package
  only when a second sport exists to validate the cut — a deliberately
  tiny one (futsal or handball) would be the honest test that the core
  carries over and only adapters change.

## Order when we build

Identity (done) → boundary protocol → serialization → series runner +
occasion schedule → trajectory metrics/gates → *then* leagues/teams as
the generator of stakes, standings and schedules → careers / Phase 4
social systems.
