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

### Phase 0.5 — Rules-of-the-game minimum (next)

Ball out of play → possession restart (throw-in/goal-kick/corner as simple
possession events, no set-piece simulation yet); kickoff formations reset on goals
(exists) and period starts.

### Phase 1.5 — Validation harness (next, before any new features)

A `validate.py` that runs N seeded matches and reports goals/match, shots/match,
possession split, pass-completion %, against target bands from real football
(≈2.5–3.0 goals, 20–30 shots, 70–85% pass completion). Wire into CI as a
regression gate. Add pytest + GitHub Actions; keep the no-dependency test style
runnable both ways.

### Phase 2 — Ball model, then psychology Phase 2

Lead passes into space / receiver attraction to the ball (fixes F7); only then
start psychology Phase 2 (cognitive profile, instinct bank) per the architecture
doc, with the same thin-slice discipline as Phase 1.

### Housekeeping (any time)

Move BMAD tooling out of the repo (or `.gitignore` it), add `pyproject.toml`,
delete the unused `PassingGraph` or mark it experimental.
