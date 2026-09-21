---
title: 'Trace Geometry - one memory model with per-dimension receptive fields'
slug: 'trace-geometry'
created: '2026-09-21'
status: 'in-progress'
stepsCompleted: [1, 2]
tech_stack: ['Python 3.12', 'stdlib only', 'no test framework (plain assert scripts)']
files_to_modify: ['instincts.py (primary: Instinct, InstinctBank.recall/learn/decay/_prune/_merge_into_existing, ROLE_SEEDS, default_bank_for)', 'minds.py (serialization + SCHEMA_VERSION 1->2 migration, boundary decay)', 'learning.py (significance -> initial mass; delete per-tick decay path)', 'series.py (bands, probe, recognition_split, anchor/trauma aggregation)', 'decisions.py (System 2 transient hook; NOVELTY_LOAD re-derivation if familiarity shifts)', 'situation.py (SIMILARITY_SCALE deletion only; similarity() itself survives)', 'models.py (new Player.elasticity trait)', 'sweeps.py (KNOBS registry follows the constant changes)', 'tests/test_cognition.py (decay-spaced variants)', 'tests/test_learning.py', 'tests/test_continuity.py (v1 blob migration test)', 'tests/test_sweeps.py', 'tests/test_discipline.py (string marker in bank.instincts)']
code_patterns: ['table-driven rules', 'module-level tuning constants read at call time (never as default args - frozen at import)', 'dataclasses for value objects', 'gate band / target band validation', 'seeded determinism through one injected Random', 'vector principle: factors move embedding dimensions, never if/else cascades']
test_patterns: ['plain python scripts with asserts + run_tests(globals())', 'support.make_player helper', 'behavioural assertions (action_shares) alongside bank-level unit asserts', 'CRITICAL GAP: existing veteran tests learn without decay between learns, reaching strength 1.000 vs the simulation measured 0.305 - new tests must use series-realistic decayed banks']
---

# Tech-Spec: Trace Geometry - one memory model with per-dimension receptive fields

**Created:** 2026-09-21

## Overview

### Problem Statement

The engine claims a dual-process model, but it has the plasticity inverted.
**System 1 is supposed to be fast to fire and slow to change; System 2 slow to
fire and fast to change.** What is implemented is the opposite: the instinct
bank rewrites itself from a single outcome and forgets with a 4.3-match
half-life, while System 2 (`decisions.utilities`) is recomputed from tactics
every tick and never updates at all.

Three structural consequences, all measured:

1. **Schooling is a constant, experience is a decaying quantity.** Role seeds
   sit at strength 1.0, are skipped by `InstinctBank.decay()` (gated on
   `source in LEARNED_SOURCES`), and can never be reinforced because
   `_merge_into_existing()` filters on `instinct.source == source`, where
   `source` is only ever `"experience"` or `"trauma"`. Lived memories start at
   <=0.6 and lose ~15%/match on two compounding schedules
   (`0.999^90 * 0.99^7 = 0.852`). Measured: schooling recognition 0.353 vs mean
   learned strength 0.305, so lived memory out-recognises schooling for only
   **13.6% of players**.
2. **Similarity is isotropic, so `load` cannot be a retrieval key.** One global
   `SIMILARITY_SCALE = 6.0` spans all 7 dimensions. The contrast between a quiet
   Tuesday and a cup final is worth `0.8 / 6.0 = 0.13` of similarity (0.87 vs
   1.0). The design claims state-dependent memory "emerges from similarity
   geometry"; the geometry caps that emergence at one-seventh of the signal.
3. **Specificity and confidence are the same number.** A trace carries one
   scalar `strength`, so it cannot be *certain but narrow* - which is exactly
   what distinguishes a lived memory from a drilled one.

Measured headline: big-night familiarity gain **0.006** against a 0.01 target,
while every CI gate passes because the gate band (-0.02..1) cannot fail. A
47-variant sweep showed the 0.01 target sits *inside* the noise band of
unrelated knobs (`in_match_decay 1.0` alone moves it +0.021), so the success
criterion is invalid as well.

### Solution

Replace the scalar `strength` with a **per-trace receptive field**, and collapse
schooling and experience into one kind of thing governed by one set of physics.

```
prototype[d]   where in situation space the trace sits     (unchanged)
width[d]       per-axis tolerance: how far along d it still recognises   (NEW)
mass           accumulated, decaying evidence              (replaces strength)
```

Distance becomes **anisotropic per trace** (`sum_d |a_d - b_d| / width_d`), so
each trace carries its own metric. Reinforcement adds mass, *narrows* width on
axes that agreed and *widens* it on axes that differed - learning which
dimensions are diagnostic, which a scalar cannot represent. Prototype movement
scales with `1/mass`, so "slow to change" is emergent rather than a constant.

Retention is **power-law**, not exponential: `(1 + t)^-d` in matches since the
trace last fired. One curve covers both timescales - steep early loss, long
tail - so no memory class needs its own tuned half-life. Narrowing follows the
kernel-density form `width ∝ mass^(-1/11)` rather than a tuned rate, and the
decay exponent itself falls with spaced reinforcement, which is where "System 1
is slow to change" shows up in persistence rather than only in movement.

Learning is **not a purely additive, growing function**. Two bounds:

- **Mass accumulates logarithmically** (`ln(1 + reps)`, as ACT-R's base-level
  activation does). Additive accumulation would give a 600-match veteran 180x a
  debutant's evidence; logarithmic gives 6.4x.
- **Elasticity sets the BIRTH WIDTH of a new trace.** A plastic player's trace
  is born wide - one experience teaches a broad region, so it generalises. A
  rigid player's trace is born narrow: they still record the experience, but
  learn only literally what happened. At 7 dimensions a birth width of 0.9
  covers ~47.8% of situation space per experience; 0.3 covers ~0.02%. That is
  "harder to add knowledge" expressed as geometry rather than as a learning rate.

### Scope

**In Scope:**

- Per-dimension receptive fields (`width[]`, `mass`) replacing scalar `strength`
- Anisotropic, per-trace similarity replacing the global `SIMILARITY_SCALE`
- One unified trace type: schooling becomes a high-mass, wide-width trace that
  decays slowly and *can* be reinforced - not a separate `source` with separate
  physics
- Mass-dependent plasticity (System 1 slow to change, emergently)
- **Logarithmic mass accumulation** and **elasticity-driven birth width**: an
  `elasticity` trait on `Player` (mirroring the existing `sensitivity`
  precedent), consumed by total accumulated mass
- Power-law retention with a repetition-flattened exponent (the spacing effect);
  significance sets a trace's *initial mass* only
- Recognition as summed evidence over traces
- **Replacing count-based pruning** (`MAX_LEARNED_MEMORIES` + "drop the weakest")
  with redundancy-based consolidation - power-law retention and a 12-slot count
  cap are mutually incompatible (see Risk F1)
- **Decay-spaced variants of the existing veteran tests.** The behavioural tests
  already exist and pass; they pass because they learn without decay between
  learns (trace strength 1.000 vs the simulation's measured 0.305). The new
  variants must fail before this change and pass after - see Risk F4
- A transient, fast-updating System 2 adjustment that resets at the match
  boundary (the mirror of System 1; flagged as separable - see Notes)
- Serialization schema bump (`minds.SCHEMA_VERSION` 1 -> 2) with a documented
  migration for v1 records
- Re-deriving the gate and target bands for the veteran effect so the gate can
  actually fail, and so the criterion is outside the noise band
- Deleting the constants this subsumes: `ROLE_SCHOOLING_FAMILIARITY`,
  `FAMILIARITY_TOP_K`, `FAMILIARITY_BLEND_DECAY`, `HIGH_LOAD_DECAY_SHELTER`,
  `PROTOTYPE_BLEND`, and the flat decay rates

**Out of Scope:**

- Leagues, seasons, standings (still deferred per `docs/specs/temporal-continuity.md`)
- Attribute development over a career (an expected *consequence* of reinforceable
  schooling, not a deliverable here)
- Match-statistics calibration (goals/shots/fouls target bands in `validate.py`)
- Perception and the belief/fill-in model
- Redesigning System 2's utility functions beyond adding the transient hook

## Context for Development

### Codebase Patterns

- **The vector principle is the house rule**: factors influence *dimensions of
  the embedding*, never if/else cascades. A scalar rate multiplier is explicitly
  the wrong shape for a mechanism claim.
- **Tuning constants are module-level and read at call time** so
  `sweeps.KNOBS` can override them. Critical: a constant used as a *default
  argument*, or baked into a module-level structure built at import, is frozen
  and silently un-sweepable (this bit `SCHOOLING_LOAD` via `_proto` +
  `ROLE_SEEDS`). `tests/test_sweeps.py` now walks each knob's AST to enforce this.
- **Table-driven rules**: `OUTCOME_RULES`, `BANDS`, `SERIES_BANDS`,
  `MEASUREMENTS`, `ROLE_SEEDS`. New behaviour should be a row, not a branch.
- **Two-band validation**: gate bands are regression guards that fail CI; target
  bands are goals that warn. Any new metric needs both, and the gate must be
  *able* to fail.
- **Determinism**: all randomness flows through one seeded `random.Random`.
- **Tests are plain scripts** (`python3 tests/test_x.py`), asserts plus
  `run_tests(globals())`, listed explicitly in `.github/workflows/ci.yml`.

Established by the Step 2 investigation:

- **The change surface is far more contained than the scope suggests.**
  `Instinct(...)` is constructed in exactly **3 places** - `instincts.py:258`
  (`learn`), `minds.py:171` (deserialization), and 11 inline literals in
  `ROLE_SEEDS`. `situation.similarity()` is called from only **2 production
  sites**, both inside `InstinctBank` (`instincts.py:176`, `:234`). Anisotropic
  distance can therefore become a trace-local method while `situation.similarity`
  survives untouched as the isotropic baseline other code and tests still use.
- **`.strength` has 9 production readers** (`instincts.py` 178/183/235/273/282/
  289/293/376, `minds.py:141`, `series.py:171,173`) and 17 in tests. Every one
  is a mass/width decision point; none is incidental.
- **The test suite systematically over-states the mechanism.** Veteran tests
  learn repeatedly with no decay between learns, so traces merge to strength
  1.000 (`MEMORY_MERGE_SIMILARITY = 0.8` folds identical situations into one
  trace, +`gained*0.5` each time). The simulation produces a measured mean of
  0.305. This is why the suite is green while the series metric says the
  mechanism does not work - the tests and the match loop are not describing the
  same object.
- **`tests/test_discipline.py:127` appends the string `"marker"` to
  `bank.instincts`** to tag a bank. Any code that iterates instincts expecting
  trace attributes will break there.

### Files to Reference

| File | Purpose |
| ---- | ------- |
| `instincts.py` | The trace model itself: `Instinct`, `InstinctBank.recall/learn/decay/_prune/_merge_into_existing`, `ROLE_SEEDS`, `default_bank_for`. Primary surface of this change. |
| `situation.py` | `SituationEmbedding` (7 dims) and `similarity()` / `SIMILARITY_SCALE` - the isotropic kernel being replaced. |
| `decisions.py` | The two live consumers of familiarity: load relief (`NOVELTY_LOAD`) and the System-1 content gate. Where System 2's transient hook would attach. |
| `learning.py` | `OUTCOME_RULES` (valence + significance per event) and in-match decay. Significance must start driving persistence. |
| `minds.py` | Boundary protocol (`close_match`), serialization (`to_dict`/`from_dict`, `SCHEMA_VERSION`), between-match decay rates. |
| `series.py` | `BIG_NIGHT_PROBE`, `big_night_familiarity_gain`, `recognition_split`, `SERIES_BANDS` - the measurement under revision. |
| `sweeps.py` | 19-knob sweep harness; how any calibration in this spec must be justified. |
| `docs/specs/temporal-continuity.md` | House spec style ("Little Spec": Status, Principles/prior art, mechanism, acceptance gates, build order). Final spec lands in `docs/specs/`. |

### Technical Decisions

Settled during elicitation (First Principles Analysis, 2026-09-21):

1. **One trace type, not two.** Schooling and experience differ in *shape*
   (broad/shallow vs narrow/deep), not in kind. Both reinforce, both decay -
   schooling just carries far more mass, so it erodes slowly. This deletes the
   `source`-gated physics rather than retuning it.
2. **Plasticity is vector-valued.** The System 1 / System 2 asymmetry must act
   on the embedding's dimensions, not as a scalar rate. Narrowing and widening
   per axis *is* the mechanism; a decay constant is not.
3. **Familiarity is a behavioural dial, not a diagnostic.** It feeds cognitive
   load relief and System-1 gating in `decisions.py`, so any change to its
   absolute level changes every player's behaviour in every situation - not just
   veterans on big nights. Acceptance must be measured behaviourally.
4. **Top-K folding is rejected.** It cleared the target on every seed
   (0.057-0.248) but inflated *both* memory classes (role recognition
   0.376 -> 0.694), leaving the learned-vs-schooling gap roughly unchanged. It
   moved the metric, not the mechanism, and introduced two free parameters.
   Superseded by receptive fields.
5. **Raising strength ceilings is refuted by measurement.** Approach D scored
   0.003-0.008 (baseline 0.006) because the drain binds, not the cap - positive
   evidence that persistence, not encoding, is the defect.
6. **The success criterion is itself in scope.** A fix cannot be validated
   against a target that unrelated knobs move by more than the fix does.

Settled during elicitation (Occam's Razor Application, 2026-09-21):

7. **Per-dimension widths are the minimum sufficient model**, selected against
   three simpler candidates by one falsifiable test - *how hard does a
   cup-final trauma formed at load 0.9 fire on a quiet Tuesday?*

   | Model | own big night | quiet Tuesday | big night, different shape |
   | ----- | ------------- | ------------- | -------------------------- |
   | Fix the asymmetries only, keep the isotropic kernel | 0.600 | **0.451** | 0.486 |
   | One global per-dimension weight vector | same shape - a fixed reweighting, not learned | | |
   | One scalar width per trace (narrowed to 0.15) | 0.600 | 0.000 | **0.000** |
   | **Per-dimension widths (load 0.15, else 1.5)** | 0.600 | 0.000 | 0.216 |

   Fixing only the asymmetries would likely move the *metric* past target, but
   leaves the cup-final trauma firing at **75% strength on a Tuesday** - a
   memory that fires nearly everywhere with a slight preference, which is not
   state-dependent memory. That is the same "moved the metric, not the
   mechanism" failure as top-K. A single scalar width discriminates load but is
   then equally narrow on pressure and density, which genuinely vary between
   big nights, so it fires almost never. Full per-trace covariance (7x7) buys
   nothing identified, at 49 floats per trace.
8. **Complexity is free parameters, not state.** The memory path carries 14
   tunable constants today; after decisions 11-13 below this proposal nets
   **14 -> ~9**. Per-trace state grows (1 scalar -> 7 widths + mass), but state
   is learned, not tuned.
9. **Collapse the two decay schedules into one.** `DECAY_PER_MINUTE` (per tick)
   and `MEMORY_DECAY_PER_REST_DAY` (boundary) compound to 0.852/match and
   neither was derived from anything. One consolidation step at the match
   boundary. Deletes a constant and the `ExperienceLearning.decay` tick path.
10. **Widths are learned, never tuned.** Only their birth values are constants -
    one for lived traces, one for schooling. Everything after is evidence.

Settled during elicitation (Self-Consistency Validation, 2026-09-21). Four
independent derivations - IBLT/ACT-R, forgetting-curve research, this repo's
vector principle, and kernel density estimation - agree unanimously that there
is ONE trace type, and 3-of-4 agree on summed recognition and evidence-driven
narrowing. They disagreed with the draft on three points:

11. **Retention is power-law, not exponential.** ACT-R base-level activation
    (`ln(sum_j t_j^-d)`) and the Ebbinghaus/Wixted curve say so independently.
    Exponential forces a chosen timescale per memory class: tuned for the long
    tail nothing is forgotten quickly (0.994 retained after one match), tuned
    for fast forgetting nothing survives a season. Power law `(1+t)^-0.35`
    gives 0.785 after one match, 0.273 after a season, 0.107 after 600 matches.
    **This deletes the significance -> half-life table** - significance sets
    initial mass, one curve does persistence for every class.
12. **The narrowing rate is derived, not tuned.** Silverman's bandwidth rule
    `h ∝ n^(-1/(d+4))` gives `width ∝ mass^(-1/11)` at 7 dimensions. Deletes
    the posited `WIDTH_NARROW_RATE`.
13. **The decay exponent falls with spaced reinforcement** (the spacing effect).
    Absent from the draft, which decayed mass at a uniform rate. This is how
    drilled schooling erodes slowly *without* being a special case, and it is
    where "System 1 is slow to change" belongs in persistence rather than only
    in prototype movement.
14. **Per-dimension widths are a deliberate departure from ACT-R.** ACT-R's
    partial matching uses a *global* mismatch scale, not per-trace per-slot
    bandwidths - the strongest external prior art backs the global scheme. The
    vector principle and the anisotropic-KDE derivation break the tie, and
    decision 7's test is the reason: under any global scheme a cup-final trauma
    still fires at 75% strength on a Tuesday. Record the departure and its
    reason; do not claim prior art is unanimous.

Settled during elicitation (Pre-mortem Analysis, 2026-09-21):

15. **Power-law retention and count-based pruning cannot both ship.** They are
    one change, not two (Risk F1). Pruning moves from "drop the weakest" to
    consolidating redundant traces - under a long tail the weakest trace is
    always the newest one.
16. **Widths shrink toward their birth prior in proportion to mass**:
    `width = prior * (mass + k)^(-1/11)`, with `k` set so that low-mass traces
    stay wide. Estimating 7 widths from 3 observations is noise, and banks hold
    ~11 traces (Risk F2). Silverman's rule assumes an `n` this domain does not
    have.
18. **Learning is bounded, not purely additive.** Mass accumulates as
    `ln(1 + reps)`, matching ACT-R base-level activation and the power law of
    practice. Additive accumulation makes recognition run away (180x vs 6.4x
    over a career) and would make an old player's traces immovable under the
    `1/mass` plasticity rule.
19. **Elasticity acts on birth width, not on a learning rate.** Per the vector
    principle, declining plasticity must be geometric: a rigid player's new
    traces are born narrow, so an experience teaches only what literally
    happened and does not generalise. A scalar learning-rate multiplier would be
    exactly the "linear equation" this project rejects.
20. **Elasticity = trait x accumulated mass; no `age` field.** Add
    `Player.elasticity: int = 50` following the `sensitivity` precedent ("a
    trait, not a skill"). Age is deliberately NOT modelled: the engine has no
    season clock yet (leagues are deferred), so an `age` int would be a static
    number pretending to be time. Consuming elasticity by accumulated mass also
    gets the better answer for free - a 30-year-old journeyman who has played
    little is still plastic. Revisit when leagues land and age can actually
    advance.
21. **Acceptance is behavioural, and the calibration invariant is explicit.**
    The probe-point gain is demoted to a diagnostic; the gate becomes a
    veteran-vs-debutant difference in action distributions under high load. Mean
    familiarity across all decisions is pinned as an invariant, or `NOVELTY_LOAD`
    is re-derived in the same change (Risk F5).

## Implementation Plan

### Tasks

_Pending - produced in Step 3 after the Step 2 investigation._

### Acceptance Criteria

_Pending - produced in Step 3. Must be behavioural (veteran vs debutant decision
distributions under high load), not probe-point differences, per Technical
Decision 3._

## Additional Context

### Dependencies

- No new runtime dependencies (stdlib only, per repo convention).
- Depends on the sweep harness landed on `feat/familiarity-sweep-benchmark`
  (`sweeps.py`, `tests/test_sweeps.py`) for calibration evidence.
- PR #13 (`benchmark suite + spatial dynamics`, open, mergeable, CI green)
  touches `passing.py` / `movement.py` / `ball_actions.py` and shifts shot and
  possession baselines. It does not touch `instincts.py`, but every band
  re-derived here should be re-checked after it merges.

### Risks (pre-mortem, 2026-09-21)

Failure scenario: it is December 2026, trace geometry shipped, and the veteran
effect still does not work. Causes, most likely first:

| # | Failure | Evidence | Prevention |
| - | ------- | -------- | ---------- |
| **F1** | **The bank fills in month one and never turns over.** Power law extends trace life from 15.5 to **1211 matches** (78x). With a 12-slot cap and "drop the weakest", the weakest is always the newest - so nothing is learned after ~12 memorable events, and veterans become indistinguishable from juniors. The original symptom, reached from the opposite direction. | Sweep: `max_memories 6` pins at exactly 6.00; `12` measures 10.83 - the cap already binds | Decision 15: redundancy-based consolidation, not count-based pruning |
| **F2** | Widths estimated from 3 observations become noise; junk widths mean arbitrary recognition | Banks hold ~11 traces; 7 widths per trace | Decision 16: mass-proportional shrinkage toward the birth prior |
| **F3** | We measure recognition again instead of behaviour, the number moves, nobody checks whether play changed - **exactly how top-K passed** | This session | Decision 21; probe-point gain demoted to diagnostic |
| **F4** | **The tests for the headline claim exist and pass - by constructing conditions the simulation never reaches.** Corrected in Step 2: `test_veterans_keep_their_game_under_the_lights` IS behavioural (it measures bold-action share), and `test_big_night_experience_closes_the_gap_schooling_cannot` does include schooling. Both pass. They pass because they learn 4-6 times *with no decay in between*, so the veteran's trace merges up to strength **1.000** against a schooling floor of 0.353. Space those same four big nights realistically (4 matches apart) and the trace lands at **0.331 - below the floor** - giving a veteran familiarity of 0.353, identical to the debutant's. The suite cannot see the defect | Reproduced in Step 2; see Codebase Patterns | Tests must run against **series-realistic banks** - decayed, varied situations, schooling present - not hand-built ideal ones. Add a decay-spaced variant of each existing veteran test; expect them to fail before the change and pass after |
| **F5** | Mean familiarity shifts -> cognitive load shifts -> match statistics drift out of band | `load = pressure + 0.3*(1-familiarity)`; sweep showed pass%/shots moving with fold changes | Decision 21's calibration invariant, in the gates not the notes |
| **F6** | The v1 -> v2 migration silently wipes careers | `from_dict` does not check `SCHEMA_VERSION` **today** | Add the missing version check now; explicit migration (`strength -> mass`, widths -> prior) with a test loading a captured v1 blob |

### Testing Strategy

_Detailed plan pending Step 3._ Fixed points already established:

- **Write F4's behavioural test first, and let it fail.** It is the acceptance
  criterion, not a nice-to-have.
- Baseline that must stay green: all 15 test scripts, `validate.py --gate`, and
  `series.py --gate` on seeds 42 and 99.
- Every calibration claim must be justified by `sweeps.py` output across at
  least 3 seeds, never a single run - the metric swings 12x by seed.
- Benchmark `recall` at a saturated bank before and after; it runs on every
  on-ball decision, and F1 means banks would sit permanently at capacity.

### Notes

- The System 2 transient adjustment (fast to change, expensive to fire) is the
  mirror of the System 1 property and belongs to the same model, but it is
  separable into a second slice if the trace-geometry change alone proves large
  enough. Decide in Step 3.
- Open question raised, not yet answered: if unused schooling decays, a fringe
  player who rarely plays drifts toward a blank slate. Does schooling erode to a
  floor, or is atrophy intended?
- Expected emergent consequence: reinforceable schooling makes role seeds
  diverge per player over a career - development arising from traces rather than
  from attribute tables.
- **Peak age should emerge rather than be tuned.** Precision rises with
  accumulated mass while coverage-of-novelty falls with elasticity; the crossover
  is the peak. A first sketch puts it around 22-26 from two curves introduced for
  unrelated reasons. If it lands somewhere absurd, that falsifies the shape of
  one of them - which makes it a useful check, not decoration.
- **Falsifiable prediction worth testing:** an accumulated player in a NOVEL
  situation should look worse than a young one, not better - narrow traces fail
  to cover it, familiarity drops, load rises, and overload degrades System 1.
  "Veteran signing struggles to adapt to a new system" should fall out of the
  same machinery that makes them excellent in familiar ones. Nothing in the
  engine asserts this today.
