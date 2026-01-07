# Anstoss Engine

A tactical football match simulation engine inspired by the classic Anstoss game series.

## Key Features

- **Spatial control model**: Not just if-then stats, but actual pitch control calculation
- **Tactical principles system**: Behaviors emerge from composable rules, not hardcoded formations
- **Discoverable tactics**: You can recreate Guardiola, Klopp, or invent your own style
- **ASCII visualization**: Watch matches unfold in your terminal

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

## Architecture

```
anstoss-engine/
├── models.py       # Core data structures (Player, Team, Position, etc.)
├── spatial.py      # Space control and passing lane calculations
├── tactics.py      # Tactical principles system
├── engine.py       # Match simulation loop
├── teams.py        # Team/player creation utilities
├── visualizer.py   # ASCII rendering
└── demo.py         # Demo script
```

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

- [ ] Set pieces (corners, free kicks)
- [ ] Substitutions and fatigue management
- [ ] Individual player instructions
- [ ] Opposition analysis / tactical adaptation
- [ ] Multi-season simulation
- [ ] Save/load functionality
- [ ] Web-based UI

## License

MIT - Do what you want, this is a learning project.
