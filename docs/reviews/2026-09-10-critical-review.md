# Critical Review & Next Steps — September 2026

Scope: full read of `models.py`, `spatial.py`, `tactics.py`, `engine.py`, `teams.py`,
`psychology.py`, `tests/test_psychology.py`, and the psychological-engine
architecture doc, plus empirical simulation runs to verify suspected issues.

## Verdict

The project's *ideas* are sound and unusually well articulated: pitch-control as a
first-class model, tactics as composable behavioral principles, and a phased
psychological layer with a genuinely good Phase-1 slice (small, tested, integrated).

The *simulation core is not yet valid*, however. The engine has no concept of
per-team attack direction: every piece of spatial logic — shooting range, shot
targets, dribble direction, role-based movement, forward-pass scoring, tactical
zone triggers — is hardcoded to the home team's frame (attack toward y=100).
Everything built on top of match outcomes (the tactical effectiveness matrix,
psychology feedback loops, any Guardiola-vs-Klopp claims) currently measures an
artifact, not tactics. Fixing this must come before any further feature work.

## Evidence

60 matches, identical `balanced` tactics, identical skill 70, both sides:

```
Home: 21 W, 93 goals (1.55/match), 589 shots
Away: 18 W, 78 goals (1.30/match), 524 shots
Away shot positions: avg y = 80.3, range 70.2–100.0
```

The away team's goalkeeper starts at y=95 (after `flip_team_positions`), yet every
away shot is taken at y>70 and targeted at `Position(50, 100)` — the away team
attacks its **own** goal end. The scoreline only looks plausible because of
compensating bugs (see below).

## Critical findings

### F1 — No per-team attack direction (blocker)

`engine.py` hardcodes the home frame everywhere:

- `_in_shooting_range` requires `pos.y > 70` and distance to `(50, 100)` for both teams.
- `_resolve_shot` and `_move_dribbler` always target `Position(50, 100)`.
- `_attacking_movement` / `_defending_movement` role logic pushes every team toward
  high y when attacking, low y when defending — for the away team this is backwards.
- `_resolve_pass` scores `y_diff = target.y - passer.y` as "forward" for both teams,
  so the away team is *rewarded* for passing toward its own goal.
- `psychology._feedback_miss` measures "clear chance" distance to `(50, 100)` for both teams.

The half-time flip in `_half_time` (mirror all y positions, keep the fixed goal)
means each team spends one half as "the broken team", which is why aggregate stats
come out near-symmetric. That symmetry is a coincidence of two bugs, not correctness.

### F2 — Save resolution is geometrically wrong

`_resolve_shot_arrival` picks the correct keeper (`state.defending_team.goalkeeper`)
but computes save chance from that keeper's distance to `(50, 100)`. In the first
half, the home keeper "saves" away shots from 95 units away (losing the +0.2
proximity bonus), while the away keeper — physically standing in the goalmouth
being shot at — is not involved at all.

### F3 — Missed shots teleport the ball

A miss always places the ball at `(50, 5)` — the home goal-kick area — regardless
of which end the shot was taken at, then relies on the loose-ball scramble.

### F4 — Dead conditionals in `tactics.py`

`relative_x=-15 if True else 15` (`inverted_fullback`) and `20 if True else -20`
(`winger_width`) are constant expressions: both fullbacks shift left, both wingers
shift right, whatever side they play. The intended left/right symmetry never runs.

### F5 — Tactical zone conditions are frame-dependent

`TacticalPrinciple.ball_zone_y_min/max` and the `BUILDUP` / `ATTACKING_THIRD` /
`DEFENDING_THIRD` triggers compare against absolute ball y. For the away team,
"gegenpress only in the opponent's half" (`ball_zone_y_min=50`) actually fires in
their **own** half. `MovementInstruction.target_y` / `relative_y` are likewise
interpreted in home frame (`high_line` `target_y=50` pushes away CBs the wrong way).
Consequence: the tactical matrix in `demo.py --mode matrix` does not measure what it
claims to measure.

### F6 — Nondeterminism and no seed control

`create_442_team` uses `hash(name)` for attribute variation → team strength varies
per Python process (PYTHONHASHSEED). The engine uses the global `random` module
with no injectable seed, so no run is reproducible. For an engine whose selling
point is *testable emergent behavior*, reproducibility is a requirement, not a nicety.

### F7 — Pass model teleports possession

`Ball.start_pass` locks the target's position at kick time, and `_resolve_pass_arrival`
hands the ball to the target player wherever they now are — receiver and ball can be
10+ units apart. Acceptable for now, but it undermines the spatial-control premise
(passing lanes are evaluated against geometry the resolution then ignores).

### F8 — Missing minimum rules of the game

No out-of-bounds, throw-ins, corners, goal kicks (beyond the miss teleport),
offside (a one-line clamp exists for strikers only), or fouls. These are listed in
the README TODO — fine — but the absence of *any* boundary handling means loose
balls near the touchline produce silent nonsense rather than restarts.

### F9 — Process: design far ahead of validation

The psychological-engine doc is ~1,900 lines specifying Phases 2–4 (instinct banks,
trauma memory, mentorship, serialization schemas, performance budgets in μs) on top
of an engine whose core spatial loop was never validated. The Phase-1 psychology
implementation itself is the best-engineered part of the repo — small, documented,
18 passing tests — which proves the team *can* work incrementally. The lesson: the
same "thin slice + tests" discipline must be applied to the match engine before
Phase 2 psychology, or every phase inherits an invalid substrate. Separately, the
vendored BMAD framework (~480 markdown files under `.claude/` and `_bmad/`) dwarfs
the actual project and should not live in the product repo unfiltered.

Also worth noting: only psychology has tests. `engine.py` (1,050 lines, the actual
product) has zero direct coverage; nothing would have caught F1–F5.

## Assessment of the approach

- **Spatial-control model**: right idea, reasonable math, cheap enough. Keep.
- **Principles-based tactics**: good abstraction; conditions/movements just need to
  be defined in the *team's* frame. Keep, fix the frame.
- **Psychology Phase 1**: well-scoped, correctly deferred complexity, real tests. Keep.
- **Phases 2–4 psychology doc**: keep as vision, but treat as unvalidated
  hypotheses; do not implement until the engine produces trustworthy matches.

## Plan

### Phase 0 — Correctness sprint (this branch, implemented)

1. Per-team attack direction: `Team.attacks_up` flag plus `to_frame`/`from_frame`
   transforms; all engine spatial logic (shooting, dribbling, role movement,
   pass scoring, defensive line, urgency) computed in team frame.
2. Tactical conditions and movement instructions interpreted in team frame
   (`get_active_principles` passes the team through; `target_y`/`relative_y`
   transformed in `_apply_movement`).
3. Shot resolution: shots target the correct goal; save chance uses the actual
   defending keeper's distance to *that* goal; misses restart from the defending
   keeper (goal kick), not `(50, 5)`.
4. `psychology._feedback_miss` measures clear-chance distance to the shooter's
   actual target goal.
5. Fix F4 dead conditionals (side-specific principles).
6. Determinism: replace `hash(name)` with a stable hash; `SimulationConfig.seed`.
7. Half-time flips `attacks_up` along with positions, so both halves are correct.
8. New test suite `tests/test_direction.py`: unit tests for transforms, shooting
   range, shot targets, pass-direction scoring in both frames, plus statistical
   acceptance tests:
   - every shot is taken in the shooter's attacking third *in the shooter's frame*;
   - both teams score over a multi-match sample;
   - identical mirrored teams produce roughly symmetric outcomes.

### Phase 0.5 — Rules-of-the-game minimum (this branch, implemented)

Ball out of play → possession restart (throw-in/goal-kick/corner as simple
possession events, no set-piece simulation yet): deflected tackles and
miscontrols can put the ball over a line, wide shots restart as goal kicks,
and parried saves can go behind for corners. Kickoffs happen automatically
at match start, after goals, and for the away team at the start of the
second half. Implemented alongside a structural decomposition: the
monolithic `MatchEngine` is now a thin orchestrator over Protocol-typed
components (`interfaces.py`) — `movement.RoleMovementModel`,
`ball_actions.DefaultActionResolver`, `restarts.SimpleRestartPolicy`,
`conditioning.FatigueModel`/`MomentumModel` — each testable in isolation,
sharing one injected `random.Random` so seeded matches reproduce exactly.
Covered by `tests/test_restarts.py`.

### Phase 1.5 — Validation harness (implemented)

`validate.py` runs N seeded matches (identical mirrored teams) and reports
goals/match, shots, pass completion, possession split, symmetry and restart
counts against two bands per metric: a **gate band** (regression guard; CI
fails on breach via `validate.py --gate`) and a **target band** (real-football
realism goal; warns only). GitHub Actions (`.github/workflows/ci.yml`) runs
all five test suites plus the gate. The harness immediately paid off: it
exposed a 29% end-to-end pass completion. That drove the introduction of
`execution.py` — a shared execution-quality model where every contested
action (pass, first touch, interception, dribble duel, shot, save) is
resolved from the full factor stack (skill, physical fatigue, pressure vs.
composure, confidence, team momentum) instead of flat per-action constants;
`tests/test_execution.py` pins each factor's direction of effect. Current
state: all gates green; shots/corners/throw-ins still below realism targets —
that gap is attack-construction quality, i.e. the Phase 2 ball model, not
constant tuning.

### Phase 2a — Ball model (implemented)

Fixes F7: passes now aim at a frame-aware lead position ahead of the
receiver, bounded by what the receiver can reach during the ball's flight
(pace × flight time); the intended receiver breaks from role movement and
runs to meet the ball; possession never teleports (an arriving pass the
receiver didn't reach runs loose, loose balls must be run down within a
claim radius and are chased by the nearest player of each team). The
proximity gate exposed a latent flight bug — arrival was declared off an
int-floored tick counter, landing balls up to one flight-speed short of
the aim point — now positional. Covered by `tests/test_ball_model.py`.
The ASCII visualizer also gained color (light-gray pitch, blue home / red
away, yellow ball) and one unified absolute coordinate mapping for both
teams.

### Phase 2b — Psychology Phase 2: decision depth (implemented)

The hardcoded shoot/dribble/pass probability tree is gone. Decisions now
flow through a `DecisionModel` Protocol injected into
`DefaultActionResolver` (`decisions.py`), default implementation the
dual-process model from the architecture doc:

- `situation.py`: `SituationEmbedding` — pressure, frame-aware
  progression, time criticality, spatial density, passing support — a
  pure function over signals the engine already computes (the doc's
  body_orientation dimension is deferred; the engine doesn't model
  facing).
- `instincts.py`: role-seeded `InstinctBank` (System 1 comfort actions,
  e.g. finisher's box instinct, organizer's keep-it-simple under
  pressure), queried by situation similarity, shaped by aggression at
  seeding and tilted bold/safe by live confidence. Learning from
  experience is Phase 3.
- `decisions.py`: System 2 utilities per action (personality-warped by
  aggression and vision, porting the old tree's tuning) blended with the
  System 1 instinct query by the existing `psychology.system1_weight` —
  pressure against effective composure decides how much instinct
  overrides analysis. Action vocabulary grew to
  shoot/dribble/pass_forward/pass_safe; the pass intents carry into
  target scoring as a forward bias.

Acceptance criteria met and pinned in `tests/test_decisions.py`
(behavioral separation): the low-composure/high-aggression archetype
takes bold actions >1.3x the metronome's rate in identical situations;
pressure pushes nervy players to the safe ball while composed players
keep playing forward; role banks and confidence tilt verified. All
statistical gate bands stay green.

### Phase 3 — Learning & memory (implemented, first slice)

Outcomes now write into the instinct banks. `minds.py` gives decisions
and learning one shared substrate per player (`PlayerMind` = bank +
pending decision); `learning.py` pairs each on-ball decision with the
outcome event that resolves it (table-driven: one `OutcomeRule` per event
type) and calls `InstinctBank.learn`: successes form **success anchors**
that reinforce the action in similar situations, failures form **trauma**
entries that suppress it. Similar memories merge instead of duplicating
(prototype blending), banks cap at 12 learned memories (weakest pruned),
and memories decay slowly with simulated time. Confidence now selects
between memory classes at query time per doc §5.3: confident players
sample their success anchors, rattled players feel their traumas.
Memories persist across matches when the same Player objects are reused.
The behavioral-separation check is now a CI gate in `validate.py`
(archetype bold-action ratio ≥ 1.15). Deferred to later slices:
validation profiles, mentor transmission, serialization.

A code-quality pass landed alongside (dispatch tables over if/elif
chains for role movement, `DefaultActionResolver` split into
`PassResolver`/`ShotResolver` with a coordinator + stable facades,
threshold tables for utility bonuses, shared test runner/fixtures in
`tests/support.py`).

### In-game mechanics round (implemented)

Prioritized ahead of temporal continuity (see below): attack
construction and discipline, with an explicit architectural rule applied
throughout — **the vector principle**: behavioral tendencies flow
through embedding dimensions, factor stacks and similarity, never
through boolean gates or bespoke if/else formulas. Hard branches are
reserved for the laws of the game (bounds, restarts, cards).

- **Crossing**: `width` became a dimension of `SituationEmbedding`;
  cross utility is continuous in width × progression (no positional
  boolean), winger instinct prototypes match wide/advanced situations by
  similarity, strikers crash the box when the ball is wide and high, and
  `PassResolver.resolve_cross` delivers a lofted lead ball to the
  best-placed box target. Result: ~26 crosses/match and pass completion
  inside the target band (0.82) for the first time — wide players stopped
  forcing risky central passes.
- **Clearances**: defenders intercepting lofted balls near their own
  goal, and holders pressed deep, clear their lines — upfield (a real
  flight via `Ball.launch_clear`), over the touchline (throw-in), or
  behind (corner).
- **Discipline**: a challenge mechanic drives duels off *measured
  defender proximity* (the abstract press score proved dead code — it
  never exceeded 0.31). Fouls are intent (aggression) × mistimed
  execution (1 − execution quality, the shared factor stack), so tired,
  pressured, rattled challengers foul more with no bespoke formula.
  Yellow/red cards with sending off (team plays short-handed), free-kick
  restarts, and card confidence deltas. Penalties and direct free kicks
  are still TODO.
- **Stable identity** (temporal insurance): `Player.player_id`,
  `MindRegistry` keyed by identity rather than object, and distinct
  identities for mirror matchups — the substrate for match-to-match and
  career continuity, implemented now because retrofitting identity later
  touches everything.

### Perception round (implemented)

Players are no longer omniscient. `perception.py` builds the holder's
PERCEIVED world at decision time: true positions inside a focus whose
range the vision attribute widens and pressure narrows (tunnel vision via
the same System 1 weight that governs decisions); outside it, experience
fills in the formation prior — perceived = certainty·truth +
(1−certainty)·expected, with players behind the holder (frame-aware)
seen less. A player standing where expected is perceived correctly even
at low certainty; only the unexpected can be misjudged. The perceived
world is just a different input vector to the same pipeline (lanes,
support dimension, pass targets) — no decision logic branches on it —
and beliefs are consequential through the existing no-teleport ball
model: passes are aimed at the believed position, so when reality goes
against the grain the ball runs loose or is intercepted, and the
learning layer records the lesson. `OmniscientPerception` remains as the
injectable null model. Measured cost of imperfect information: pass
completion 0.82 → 0.79 (still in target); vision now has perceptual
value (high-vision squads complete measurably more). Body orientation
(true facing) is still unmodeled — the behind-penalty uses attack
direction as a proxy; teammate-familiarity priors (learned expectations
about specific teammates' runs) are a natural Phase 4 extension.

### Cognitive load round (implemented)

Load is now a core mechanism, not a bolt-on, with attribute discipline
decided explicitly (new attribute only for a stable capacity with
independent variance that can't be derived):

- **Environment** (`models.Environment`): one per-match vector —
  visibility, stakes, crowd intensity — neutral by default. Converted to
  personal mental load via the new **`sensitivity`** attribute
  (the doc's crowd/limelight sensitivity arriving early), feeding the
  previously-always-zero `PressureContext.psychological`. Because
  execution quality, tunnel vision and decision blending already key off
  pressure, the limelight propagates everywhere with zero new branches.
- **Familiarity = pre-exposure, queryable**: `InstinctBank.familiarity`
  is the best similarity×strength over everything in the bank — role
  schooling plus lived experience. Novel situations add load
  (`NOVELTY_LOAD`); familiar ones are processed cheaply. Learned
  memories ARE the veteran's fifty big nights: the same store serves
  recall and recognition.
- **Overload degrades System 1 itself**
  (`psychology.system1_integrity`): load beyond what composure absorbs
  blurs even trained automatisms toward indiscriminate noise — distinct
  from the novelty fallback (instinct with nothing to say vs. instinct
  unable to say it clearly). Composure is the absorption capacity for
  both System 2 access and System 1 integrity. Below overload, pressure
  drives the nervy to the safe ball; past it, even the safe habit
  scrambles — both regimes are pinned in tests.
- **Untrained instinct is still personality**: the novelty fallback is
  aggression-tilted, so a reckless player's panic is rasher.
- **Orientation is state, not an attribute**: facing derives from
  smoothed motion (engine-tracked velocity); perception's directional
  sight is continuous in the angle to facing, with the attack-direction
  proxy as the stationary fallback. Turn-rate limits (agility) deferred
  until a mechanic needs independent variance. **Visibility** is a
  separate perceptual channel (murk narrows focus, more for the
  sensitive). Height/occlusion and an intelligence trait (System 2
  capacity under load, currently routed through composure) are noted as
  future work, not attributes added on spec.

### Temporal continuity (deliberately deferred)

Everything measured so far is per-match by construction; there is no
continuation mechanism yet (no match/season boundary semantics, no
serialization, no season loop). Decision recorded 2026-09: finish
in-game mechanics first — continuation amplifies whatever the match loop
produces. Order when we return: identity (done) → match/season boundary
protocol (doc's Layer 1–4 consolidation) → mind serialization (doc §3.8)
→ Season runner → careers/Phase 4 social systems.

### Phase 4 — Social systems (later)

Mentor relationships, team culture buffers on feedback, validation
profiles, per the architecture doc — after temporal continuity, plus
continued statistical calibration toward the realism targets (shots,
corners, throw-ins, foul volume).

### Housekeeping (any time)

Move BMAD tooling out of the repo (or `.gitignore` it), add `pyproject.toml`,
delete the unused `PassingGraph` or mark it experimental.
