# Trace Geometry — career prototype

**Status: UNVALIDATED PROTOTYPE.** Smoke-tested only. Nothing here is wired
into the engine, and none of its constants have been calibrated. It exists to
answer one question the spec cannot answer by reasoning: *does the proposed
memory model produce plausible careers, or only a plausible-sounding
mechanism?*

Spec: `_bmad-output/implementation-artifacts/tech-spec-wip.md` (steps 1–2 of 4
complete).

## Why replay instead of simulating

100 players × 15 seasons × 38 matches is ~57,000 matches — days of compute, and
the current engine cannot show development anyway (banks cap at 12 traces with a
4.3-match half-life, so every career flatlines by construction).

So the method is: **capture the real distribution, replay it at career scale.**

1. `capture.py` instruments `InstinctBank.learn` during a real 10-match series
   and records every `(situation, action, valence, significance)` the match loop
   actually produces → `stream.json` (1,244 events).
2. `careers.py` replays that empirical stream across a synthetic population,
   under two interchangeable trace models.

This keeps the career model honest about the situation statistics the engine
really generates, rather than inventing a distribution that flatters the model.

## What the capture showed (already useful on its own)

- **5.7 learning events per player per match.** A 15-season career is ~3,250
  learning events.
- **68% of outcomes are positive** (anchors), 32% negative (traumas).
- **Significance is low and narrow**: mean 0.34, max 0.90.
- **One embedding dimension is dead.** `support` has mean 0.97, sd 0.03 — it
  never varies, so it contributes no discriminative information. `pressure`
  (sd 0.11) and `density` (sd 0.13) are compressed, while `load` has sd 0.34.
  The isotropic kernel weights all of them equally; per-dimension widths would
  discover this automatically. This is empirical support for the anisotropy
  argument in the spec.

| dimension | mean | sd | range |
| --------- | ---- | -- | ----- |
| pressure | 0.29 | 0.11 | 0.09–0.62 |
| progression | 0.53 | 0.24 | 0.05–1.00 |
| time_criticality | 0.41 | 0.26 | 0.00–0.99 |
| spatial_density | 0.22 | 0.13 | 0.00–0.60 |
| **support** | **0.97** | **0.03** | 0.78–1.00 |
| width | 0.34 | 0.26 | 0.00–1.00 |
| load | 0.39 | 0.34 | 0.09–0.90 |

## Smoke result (25 players, 6 seasons — NOT the real run)

| model | capability, season 1 → 6 | median traces |
| ----- | ----------------------- | ------------- |
| current (as shipped) | 0.568 → **0.569** | 8 |
| proposed (trace geometry) | 0.233 → **0.441** | 160 (at cap) |

The current model produces **zero career development** across six seasons —
the predicted consequence of a 12-trace cap plus a 4.3-match half-life. The
proposed model grows. That is the only claim this prototype currently supports.

## Known problems — fix these before trusting any output

1. **The proposed model saturates `MAX_TRACES = 160` and sits there.** Whether
   consolidation is doing anything sensible at the cap is untested. This is
   Risk F1 from the spec reappearing at a different scale.
2. **Absolute capability levels are not comparable between models.** Current
   recognition is a `max` (bounded by construction), proposed is a sum. The
   *shape* of the trajectory is the signal; the levels are not.
3. **Consolidation is sampled, not exhaustive.** The exhaustive O(n²) scan runs
   on every insertion once saturated and made a full run intractable. Sampling
   24 candidates is a stopgap, not a design.
4. **No peak-or-decline is observable at 6 seasons.** The elasticity curve is
   calibrated for ~900 accumulated reps; a career is ~3,250. The full 15-season
   run is what would show peak age, and it has not been run.
5. **Every constant in the proposed block is a guess.** `BIRTH_WIDTH_*`,
   `NARROW_RATE`, `WIDEN_RATE`, `RETENTION_D0`, `SPACING_GAIN`,
   `ELASTICITY_HALFLIFE_REPS` were chosen to be plausible, not fitted.
6. **Role is not modelled.** Every player draws from the same event stream, so
   strikers and centre-backs live identical lives. Real "skill vectors" need
   role-conditioned sampling — that is the next thing to build.

## Running

```bash
python3 docs/prototypes/trace-geometry/capture.py    # regenerate stream.json (~30s)
python3 docs/prototypes/trace-geometry/careers.py    # full run: 100 players, 15 seasons
```

The full run has not been completed successfully yet — see problem 3.
