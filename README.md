# Anstoss Engine

A tactical football match simulation engine inspired by the classic Anstoss game series.

## Key Features

- **Spatial control model**: Not just if-then stats, but actual pitch control calculation
- **Tactical principles system**: Behaviors emerge from composable rules, not hardcoded formations
- **Discoverable tactics**: You can recreate Guardiola, Klopp, or invent your own style
- **ASCII visualization**: Watch matches unfold in your terminal - light-gray pitch, home team in blue, away in red, ball in yellow (auto-disables without a TTY or with NO_COLOR)

## Philosophy

This engine models *why* tactics work, not just *what* they are:

- **Positional Play (Guardiola)**: Third-man combinations, half-space occupation, inverted fullbacks
- **Gegenpressing (Klopp/Rangnick)**: Immediate counter-press, high line, vertical transitions
- **Low Block Counter**: Deep defense, quick transitions, exploit space behind high lines

Tactics interact with each other, creating emergent matchups.

## Quick Start

```bash
# Visual match (default)
python demo.py

# Specific tactical matchup
python demo.py --home positional_play --away low_block_counter

# Famous manager styles
python demo.py --home pep --away klopp

# Quick statistical simulation (100 matches)
python demo.py --mode quick --home gegenpressing --away low_block_counter --matches 100

# Tactical effectiveness matrix
python demo.py --mode matrix

# Show space control visualization
python demo.py --space-control
```

## Tests

No framework needed - plain asserts:

```bash
python3 tests/test_psychology.py   # Phase 1 psychological engine
python3 tests/test_direction.py    # Attack-direction / symmetry correctness
python3 tests/test_restarts.py     # Out-of-play restarts + kickoffs + determinism
python3 tests/test_execution.py    # Factor-driven execution quality model
python3 tests/test_metrics.py      # Metrics collector + validation machinery
python3 tests/test_ball_model.py   # Lead passes, receiver runs, no-teleport possession
python3 tests/test_visualizer.py   # ASCII rendering + colors
python3 validate.py --matches 20 --seed 42 --gate   # Statistical regression gate
```

CI (`.github/workflows/ci.yml`) runs all suites plus the validation gate.
`validate.py` reports each metric against a **gate band** (regression guard,
fails CI) and a **target band** (real-football realism goal, warns only) -
tighten gates toward targets as calibration and the ball model improve.

See `docs/reviews/` for engine reviews and the current roadmap.

## Architecture

The engine is composed of small, independently testable components typed
as Protocols (see `interfaces.py`) and injected into the orchestrator:

```
anstoss-engine/
├── models.py       # Core data structures (Player, Team, Position, etc.)
├── interfaces.py   # Component Protocols (MovementModel, ActionResolver, ...)
├── spatial.py      # Space control and passing lane calculations
├── tactics.py      # Tactical principles system
├── movement.py     # RoleMovementModel: off-ball player movement
├── ball_actions.py # DefaultActionResolver: pass/shot/dribble/duel resolution
├── execution.py    # Execution quality: skill x fatigue x pressure x confidence x momentum
├── restarts.py     # SimpleRestartPolicy: kickoffs, throw-ins, goal kicks, corners
├── conditioning.py # Fatigue and momentum models
├── psychology.py   # Phase 1 psychological engine (pressure, confidence)
├── engine.py       # MatchEngine: thin orchestrator + tick/minute/match loops
├── metrics.py      # MatchMetrics: passive per-match statistics collector
├── validate.py     # Statistical validation harness (CI gate + realism targets)
├── teams.py        # Team/player creation utilities
├── visualizer.py   # ASCII rendering
└── demo.py         # Demo script
```

All simulation randomness flows through one injected `random.Random`, so
`SimulationConfig(seed=...)` reproduces a match exactly.

## Tactical Principles System

Tactics are defined as **behavioral rules**, not just stat modifiers:

```python
TacticalPrinciple(
    name="gegenpress_immediate",
    trigger=TriggerType.BALL_LOST,
    applies_to_roles=["st", "lw", "rw", "am", "cm"],
    movement=MovementInstruction(towards_ball=0.9),
    priority=10,
    ball_zone_y_min=50,  # Only in opponent's half
    min_stamina=40,
    required_attributes={"workrate": 60, "stamina": 60}
)
```

This means: "When we lose the ball in the opponent's half, forwards and midfielders immediately swarm toward it - but only if they have the stamina and workrate to do so."

## Creating Your Own Tactics

```python
from tactics import TacticalSetup, TacticalPrinciple, TriggerType, MovementInstruction

my_tactics = TacticalSetup(
    name="My Custom Style",
    principles=[
        # Your principles here
    ],
    defensive_line_height=45,
    pressing_intensity=60,
    tempo=70,
    directness=50
)
```

## TODO / Future Ideas

- [x] Rules-of-the-game minimum: throw-ins, corners, goal kicks as possession restarts
- [x] Statistical validation harness (goals/shots/possession vs. real-football bands)
- [x] Lead passes / receiver movement (passes aim into space; receivers run to meet the ball)
- [ ] Set pieces (corners, free kicks)
- [ ] Substitutions and fatigue management
- [ ] Individual player instructions
- [ ] Opposition analysis / tactical adaptation
- [ ] Multi-season simulation
- [ ] Save/load functionality
- [ ] Web-based UI

## License

MIT - Do what you want, this is a learning project.
