# Anstoss Engine Documentation

## What is This?

The Anstoss Engine is a tactical football (soccer) match simulation engine. Unlike traditional football games that focus on player control and graphics, this engine simulates **how players think and decide** - creating emergent, believable match outcomes driven by psychology, tactics, and individual player personalities.

Think Football Manager's match engine, but with a deeper focus on the *why* behind every decision.

## The Big Idea

**"Every position is a decision. Every decision reveals psychology."**

In most football simulations, a player with "75 Passing" simply completes 75% of passes. That's statistics, not simulation.

In this engine:
- A player under pressure from a world-class midfielder makes *different* decisions than when unmarked
- A striker who just missed a clear chance plays differently for the next 20 minutes
- A young player mentored by a composed veteran *becomes* more composed over seasons
- A player's "panic response" when pressed is shaped by their academy training from age 12

Players aren't stat sheets. They're cognitive agents with personalities, memories, and learned behaviors.

## Core Concepts

### Pressure-Driven Decisions

Every decision a player makes is filtered through **pressure** - a combination of:
- **Spatial pressure**: How close is the nearest opponent?
- **Temporal pressure**: How much time to decide?
- **Tactical pressure**: Is the team chasing the game?
- **Psychological pressure**: Cup final? Hostile crowd? Bad recent form?

Under low pressure, players analyze options carefully. Under high pressure, they fall back on **instinct** - trained patterns that bypass conscious thought.

### Dual-Process Decision Making

Based on cognitive psychology (Kahneman's System 1/System 2):

- **System 2 (Analysis)**: Full evaluation of options, tactical awareness, conscious decision-making. Available when pressure is manageable.
- **System 1 (Instinct)**: Fast, automatic responses based on training and experience. Takes over when pressure exceeds the player's composure.

A composed midfielder can access System 2 under pressure that would make a nervous winger panic.

### Psychology That Persists

Player psychology isn't reset each match:
- **Confidence** builds and erodes based on performance
- **Traumas** from significant failures resurface in similar contexts (that cup final miss haunts the player in future knockout games)
- **Success anchors** give confidence boosts in familiar winning situations
- **Mentor relationships** shape how young players develop

A player's career is a psychological journey, not just a stat progression.

## Documentation Structure

```
docs/
└── architecture/
    └── psychological-engine.md   # Full technical specification
```

### [Psychological Engine Architecture](architecture/psychological-engine.md)

The complete technical design document covering:

1. **Conceptual Model** - How decisions work, what pressure means
2. **Data Architecture** - Player psychology data structures
3. **Computational Model** - How it runs efficiently at 60fps
4. **System Interactions** - Decision flow, feedback loops, memory formation
5. **Social Systems** - Mentorship, team culture, crowd effects
6. **Implementation Phases** - MVP through full system
7. **Appendices** - Test scenarios, configuration schema, formulas

## Quick Example

Two players receive the ball in the same position, same pressure:

**Player A** (High Composure, Low Aggression):
- System 2 active: evaluates all 5 passing options
- Selects highest-retention pass (safe lateral ball)
- Confidence stable regardless of outcome

**Player B** (Low Composure, High Aggression):
- System 1 dominant: sees 2 options max
- Instinct bank suggests: "play forward or dribble"
- Attempts risky through ball
- Success: confidence spikes, reinforces behavior
- Failure: confidence drops, but aggression may increase ("I'll prove myself")

Same situation. Different minds. Different outcomes. Different stories.

## Getting Started

The engine is being built in phases:

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | MVP - Composure + Confidence + Basic Pressure | Designed |
| **Phase 2** | Full cognitive traits, instinct bank | Designed |
| **Phase 3** | Memory, trauma/anchors, cross-match persistence | Designed |
| **Phase 4** | Mentorship, team culture, career-spanning psychology | Designed |

See the [implementation phases](architecture/psychological-engine.md#7-implementation-phases) for details.

## Philosophy

This isn't about creating "realistic" football in the graphical sense. It's about creating football that **feels true** - where a player's decision makes sense given who they are, what they've experienced, and what pressure they're under.

When you watch a simulated match and think "of course he panicked there" or "that's exactly what a confident Pirlo-type would do" - that's the goal.

---

*Built with obsessive attention to how minds work under pressure.*
