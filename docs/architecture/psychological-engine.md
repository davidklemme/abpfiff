# Psychological Engine - Technical Design Document

## Executive Summary

This document defines the architecture for a psychological simulation layer within the Anstoss match engine. The system models individual player cognition, decision-making under pressure, emotional state, learned behaviors, and social dynamics to produce emergent, believable football behavior.

**Core Philosophy**: *Every position is a decision. Every decision reveals psychology.*

---

## 1. Overview & Philosophy

### 1.1 Core Premise

Traditional football simulations model players as stat-driven automatons executing tactical instructions. This engine takes a fundamentally different approach: **players are cognitive agents** whose decisions emerge from the interaction of personality, experience, psychological state, and environmental pressure.

A player doesn't simply "pass with 75% accuracy." They:
- Perceive available options (filtered by vision, fatigue, pressure)
- Evaluate options through their tactical understanding and personality
- Select based on a blend of learned instinct and conscious analysis
- Execute with quality degraded by pressure and enhanced by confidence
- Learn from the outcome, filtered through social feedback

### 1.2 Design Goals

| Goal | Description |
|------|-------------|
| **Emergent Behavior** | Player personalities create visibly different decision patterns without explicit scripting |
| **Psychological Realism** | Pressure, confidence, fatigue, and memory affect performance in believable ways |
| **Career Continuity** | Players develop psychologically over seasons; formative experiences leave permanent marks |
| **Computational Efficiency** | Full psychology for 22 players without impacting 60fps simulation |
| **Tunability** | Designers can adjust psychological parameters without code changes |
| **Debuggability** | Every decision can be traced to its psychological components |

### 1.3 Key Principles

1. **Pressure is the unifying force** - All psychological effects flow through the concept of pressure (spatial, temporal, situational, psychological)

2. **Dual-process decision making** - Players use fast instinct (System 1) under pressure, slow analysis (System 2) when comfortable

3. **Intent-driven movement** - Psychology produces continuous movement intent, not just discrete actions

4. **Experience shapes instinct** - The instinct bank is learned, not hardcoded; it evolves through career experiences

5. **Social context matters** - Outcomes are interpreted through team culture, relationships, and crowd feedback

---

## 2. Conceptual Model

### 2.1 The Dual-Process Decision System

Based on Kahneman's dual-process theory, player decisions emerge from two systems:

```
                    PRESSURE
                       │
            ┌──────────┴──────────┐
            │                     │
       LOW/MEDIUM               HIGH
            │                     │
            ▼                     ▼
    ┌───────────────┐    ┌───────────────┐
    │   SYSTEM 2    │    │   SYSTEM 1    │
    │   Analysis    │    │   Instinct    │
    ├───────────────┤    ├───────────────┤
    │ • Full option │    │ • Comfort     │
    │   generation  │    │   actions     │
    │ • Tactical    │    │ • Trauma      │
    │   evaluation  │    │   avoidance   │
    │ • Utility     │    │ • Success     │
    │   scoring     │    │   anchors     │
    │ • Personality │    │ • Muscle      │
    │   weighting   │    │   memory      │
    └───────┬───────┘    └───────┬───────┘
            │                     │
            └──────────┬──────────┘
                       │
                       ▼
                BLENDED OUTPUT
         (ratio based on pressure/composure)
```

**Blend Formula**: `system1_weight = min(pressure / composure, 1.0)`

- High composure players access System 2 under pressure others can't handle
- Under extreme pressure, even composed players fall back to instinct
- The blend is continuous, not binary

### 2.2 Pressure as a Unifying Concept

Pressure is computed from four sources:

```python
@dataclass
class PressureContext:
    spatial_pressure: float    # Inverse of nearest opponent distance
    temporal_pressure: float   # Time since receiving / time available to act
    tactical_pressure: float   # Match state urgency (score, time remaining)
    psychological_pressure: float  # Stakes, crowd, personal history

    @property
    def total(self) -> float:
        return (
            self.spatial_pressure * 0.35 +
            self.temporal_pressure * 0.30 +
            self.tactical_pressure * 0.20 +
            self.psychological_pressure * 0.15
        )
```

**Pressure affects different layers differently:**

| Layer | Pressure Effect |
|-------|-----------------|
| Strategic | Team pressure - are we being dominated? |
| Tactical | Zone pressure - is our area overloaded? |
| Intent | Player pressure - how many seconds to decide? |
| Execution | Action pressure - is someone closing me down? |

### 2.3 Intent-Driven Movement

**Critical insight**: Psychology affects ALL movement, not just ball actions.

Every player maintains a continuous `MovementIntent` that shapes their positioning:

```python
@dataclass
class MovementIntent:
    primary_target: Vector2D        # Where they want to be
    secondary_targets: List[Vector2D]  # Alternative positions
    urgency: float                  # How quickly to get there
    commitment: float               # How locked-in vs adaptable

    # Psychology-driven modifiers
    ball_seeking: float             # Show for ball vs hide
    risk_tolerance: float           # Aggressive vs conservative positioning
    energy_conservation: float      # Fatigue-driven economy of movement
```

**Observable behaviors emerge from intent:**
- Confident player: high `ball_seeking`, finds pockets, demands involvement
- Rattled player: low `ball_seeking`, hides behind opponents, avoids responsibility
- Fatigued player: high `energy_conservation`, cheats positioning, narrower movement

---

## 3. Data Architecture

### 3.1 Player Psychology Components

```
Player
├── CognitiveProfile        (relatively stable traits)
├── ValidationProfile       (social sensitivity configuration)
├── PsychologicalState      (dynamic match-to-match state)
├── InstinctBank            (learned automatic responses)
├── MemoryStore             (significant career experiences)
└── MentorImprints          (inherited patterns from mentors)
```

### 3.2 Cognitive Profile

Core personality traits that change slowly over a career:

```python
@dataclass
class CognitiveProfile:
    # Decision-making traits
    composure: float      # 0-1: Pressure resistance, maintains System 2 access
    intelligence: float   # 0-1: Tactical comprehension, option evaluation quality
    aggression: float     # 0-1: Risk appetite, warps utility toward bold actions
    resilience: float     # 0-1: Fatigue resistance, slows all degradation rates
    vision: float         # 0-1: Option awareness, size of perceived action space
    decisions: float      # 0-1: Processing speed, performance under time pressure
```

**Trait Interactions:**

| Trait | Primary Effect | Secondary Effects |
|-------|---------------|-------------------|
| Composure | System 1/2 blend threshold | Confidence stability |
| Intelligence | Utility evaluation accuracy | Tactical instruction comprehension |
| Aggression | Risk weighting in utility | Response to falling behind |
| Resilience | Fatigue curve slope | Recovery from setbacks |
| Vision | Option generation radius | Awareness of teammate runs |
| Decisions | Time pressure tolerance | First-touch quality |

### 3.3 Validation Profile

How the player interprets feedback from different sources:

```python
@dataclass
class ValidationProfile:
    internal_locus: float       # 0-1: Self-driven evaluation weight
    team_sensitivity: float     # 0-1: Response to teammates/captain
    crowd_sensitivity: float    # 0-1: Response to supporters
    authority_sensitivity: float # 0-1: Response to coach/management
```

**Key insight**: The same objective outcome produces different psychological impacts based on validation profile.

Example: Failed risky pass
- High internal locus: "I know that was the right ball, unlucky"
- High team sensitivity + critical captain: "I'm letting everyone down"
- High crowd sensitivity + hostile away fans: Amplified shame response

**Validation profile shifts over career:**
- Young players: Malleable, shaped by environment
- Veterans: More fixed, higher internal locus typically

### 3.4 Psychological State

Dynamic state that changes within and across matches:

```python
@dataclass
class PsychologicalState:
    confidence: float           # -1 to 1: Current momentum/form
    fatigue_mental: float       # 0-1: Cognitive drain (separate from physical)
    anxiety_level: float        # 0-1: Baseline nervousness elevation

    active_modifiers: List[PsychModifier]  # Temporary effects with decay
```

**Confidence Effects:**

| Confidence | Composure Modifier | Ball Seeking | Risk Tolerance |
|------------|-------------------|--------------|----------------|
| +0.8 to +1.0 | +0.2 | Very high | Elevated |
| +0.3 to +0.7 | +0.1 | High | Normal+ |
| -0.2 to +0.2 | 0 | Normal | Normal |
| -0.7 to -0.3 | -0.1 | Low | Conservative |
| -1.0 to -0.8 | -0.2 | Hiding | Very conservative |

**Mental Fatigue Effects:**

| Fatigue Level | Effect |
|---------------|--------|
| 0.0 - 0.3 | No degradation |
| 0.3 - 0.5 | Vision slightly reduced |
| 0.5 - 0.7 | Intelligence degraded, decisions slower |
| 0.7 - 0.9 | Composure drops, aggression rises (rash challenges) |
| 0.9 - 1.0 | Severe degradation, instinct dominates |

### 3.5 Instinct Bank

Learned automatic responses stored as situation→action mappings:

```python
@dataclass
class SituationEmbedding:
    """Continuous representation of game situations"""
    pressure_level: float      # 0-1
    progression_value: float   # -1 (own goal) to 1 (opponent goal)
    time_criticality: float    # 0-1
    spatial_density: float     # 0-1 (crowded vs open)
    body_orientation: float    # 0-1 (facing goal vs away)
    support_available: float   # 0-1 (teammates in passing range)

@dataclass
class InstinctMemory:
    situation_prototype: SituationEmbedding
    action_weights: Dict[ActionType, float]
    strength: float  # How ingrained (0-1)
    source: str      # "experience" | "mentor" | "academy"

class InstinctBank:
    memories: List[InstinctMemory]
    trauma_triggers: List[TraumaMemory]
    success_anchors: List[SuccessAnchor]

    def query(self, current: SituationEmbedding) -> Dict[ActionType, float]:
        """Find relevant instincts by similarity matching"""
        # Returns weighted action preferences for current situation
```

**Instinct Bank Sources:**
1. **Academy Imprint**: Base instincts from formative years (e.g., "Ajax DNA" = keep ball under pressure)
2. **Mentor Inheritance**: Patterns absorbed from senior players
3. **Career Experience**: Actions reinforced by success, blocked by trauma

### 3.6 Memory Store

Significant events that persist across matches/seasons:

```python
@dataclass
class SignificantMemory:
    context_embedding: SituationEmbedding
    event_type: str
    outcome: Outcome
    emotional_impact: float     # Magnitude of psychological effect
    timestamp: Date
    significance: float         # How important was this moment

    # Computed
    times_triggered: int        # How often has similar context occurred
    times_overcome: int         # How often succeeded in similar context since

class MemoryStore:
    trauma_memories: List[TraumaMemory]     # Negative anchors
    success_anchors: List[SuccessAnchor]    # Positive anchors
    mentor_imprints: Dict[PlayerId, MentorImprint]
    academy_imprint: AcademyImprint
```

**Memory Persistence Layers:**

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: CORE IDENTITY (permanent after ~age 23)           │
│  └── Base validation profile, fundamental personality       │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: CAREER SEDIMENT (very slow decay)                 │
│  └── Major traumas, defining successes, mentor imprints     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: SEASONAL CONTEXT (moderate decay)                 │
│  └── Current form, team relationships, manager rapport      │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4: MATCH STATE (rapid decay, not persisted)          │
│  └── Immediate confidence, recent actions, current fatigue  │
└─────────────────────────────────────────────────────────────┘
```

### 3.7 Trauma and Success Memories

**Trauma Memory:**
```python
@dataclass
class TraumaMemory:
    trigger_context: SituationEmbedding  # When does this resurface?
    action_blocked: ActionType           # What action to avoid
    intensity: float                     # How powerful when triggered
    decay_rate: float                    # Very slow for major events
    times_overcome: int                  # Successfully facing it weakens it
```

**Key insight**: Trauma doesn't live in time, it lives in **similar contexts**. A keeper's cup final fumble is dormant in league games but resurfaces in knockout matches.

**Success Anchor:**
```python
@dataclass
class SuccessAnchor:
    trigger_context: SituationEmbedding
    action_reinforced: ActionType
    confidence_boost: float
    last_reinforced: Date               # Fades if never repeated
```

### 3.8 Serialization & Persistence

Player psychology must persist across save/load cycles and span multiple seasons. This section defines the serialization contract.

**Persistence Scope by Layer:**

| Layer | Persisted? | When Saved | Format |
|-------|-----------|------------|--------|
| Layer 1 (Core Identity) | Yes | Season end + save | Full snapshot |
| Layer 2 (Career Sediment) | Yes | After significant events + save | Append-only log + snapshot |
| Layer 3 (Seasonal Context) | Yes | Match end + save | Full snapshot |
| Layer 4 (Match State) | No | Never | Reconstructed each match |

**Serialization Schema:**

```python
@dataclass
class PlayerPsychologySaveData:
    """Complete serializable psychology state"""
    version: str  # Schema version for migration

    # Layer 1: Core Identity
    core_identity: CoreIdentitySaveData

    # Layer 2: Career Sediment
    career_memories: List[MemorySaveData]
    instinct_bank: InstinctBankSaveData
    mentor_imprints: List[MentorImprintSaveData]
    academy_imprint: Optional[AcademyImprintSaveData]

    # Layer 3: Seasonal Context
    seasonal_state: SeasonalStateSaveData


@dataclass
class CoreIdentitySaveData:
    cognitive_profile: Dict[str, float]  # trait_name → value
    validation_profile: Dict[str, float]
    formed_at_age: int
    is_locked: bool  # True after age 23


@dataclass
class MemorySaveData:
    id: str
    memory_type: str  # "trauma" | "success" | "neutral"
    context_embedding: List[float]  # 6-dimensional vector
    action_type: str
    emotional_impact: float
    significance: float
    timestamp: str  # ISO date
    times_triggered: int
    times_overcome: int
    decay_rate: float
    current_strength: float


@dataclass
class InstinctBankSaveData:
    memories: List[InstinctMemorySaveData]
    trauma_triggers: List[str]  # Memory IDs
    success_anchors: List[str]  # Memory IDs


@dataclass
class SeasonalStateSaveData:
    confidence: float
    form_trajectory: List[float]  # Last N match ratings
    team_relationships: Dict[str, float]  # player_id → relationship_strength
    manager_relationship: float
    fan_standing: float
    matches_played_this_season: int
```

**Save/Load Operations:**

```python
class PlayerPsychology:
    def serialize(self) -> PlayerPsychologySaveData:
        """Export for save game"""
        return PlayerPsychologySaveData(
            version=PSYCHOLOGY_SCHEMA_VERSION,
            core_identity=self._serialize_core(),
            career_memories=self._serialize_memories(),
            instinct_bank=self._serialize_instincts(),
            mentor_imprints=self._serialize_imprints(),
            academy_imprint=self._serialize_academy(),
            seasonal_state=self._serialize_seasonal()
        )

    @classmethod
    def deserialize(cls, data: PlayerPsychologySaveData) -> 'PlayerPsychology':
        """Restore from save game"""
        # Handle schema migration if version differs
        if data.version != PSYCHOLOGY_SCHEMA_VERSION:
            data = migrate_save_data(data)

        psychology = cls()
        psychology._restore_core(data.core_identity)
        psychology._restore_memories(data.career_memories)
        psychology._restore_instincts(data.instinct_bank)
        psychology._restore_imprints(data.mentor_imprints)
        psychology._restore_academy(data.academy_imprint)
        psychology._restore_seasonal(data.seasonal_state)

        return psychology

    def on_season_end(self):
        """Seasonal transition - consolidate and decay"""
        # Merge similar memories
        self.career_memories = self._consolidate_memories()

        # Decay weak memories
        self.career_memories = [m for m in self.career_memories
                                if m.current_strength > MEMORY_PRUNE_THRESHOLD]

        # Reset seasonal state
        self.seasonal_state.reset_for_new_season()

        # Age-based identity locking
        if self.player.age >= 23 and not self.core_identity.is_locked:
            self.core_identity.is_locked = True
```

**Memory Consolidation Algorithm:**

```python
def _consolidate_memories(self) -> List[SignificantMemory]:
    """Merge similar memories, keeping the strongest"""
    clusters = []

    for memory in self.career_memories:
        matched = False
        for cluster in clusters:
            if cluster.can_absorb(memory):
                cluster.absorb(memory)
                matched = True
                break
        if not matched:
            clusters.append(MemoryCluster(memory))

    return [c.to_consolidated_memory() for c in clusters]


class MemoryCluster:
    SIMILARITY_THRESHOLD = 0.8

    def can_absorb(self, memory: SignificantMemory) -> bool:
        return (
            memory.memory_type == self.memory_type and
            memory.action_type == self.action_type and
            self.context_similarity(memory) > self.SIMILARITY_THRESHOLD
        )

    def absorb(self, memory: SignificantMemory):
        # Strengthen the cluster, update prototype
        self.strength = min(self.strength + memory.strength * 0.5, 1.0)
        self.count += 1
        self.prototype = self._blend_contexts(self.prototype, memory.context)
```

---

## 4. Computational Model

### 4.1 Update Frequencies

Psychology computation is **event-driven**, not continuous:

```
┌─────────────────────────────────────────────────────────────┐
│                    UPDATE FREQUENCY TIERS                   │
├─────────────────────────────────────────────────────────────┤
│  PHYSICS TICK (60 fps)                                      │
│  └── Intent → Movement execution only                       │
│  └── Cost: ~5μs per player                                  │
├─────────────────────────────────────────────────────────────┤
│  INTENT ADAPTATION (2-5 Hz per player)                      │
│  └── Lightweight adjustment to existing intent              │
│  └── Cost: ~50-100μs per player                             │
├─────────────────────────────────────────────────────────────┤
│  FULL PSYCHOLOGY EVALUATION (0.2-3 Hz per player)           │
│  └── Complete decision engine execution                     │
│  └── Cost: ~500-2000μs per player                           │
├─────────────────────────────────────────────────────────────┤
│  FEEDBACK PROCESSING (on significant events)                │
│  └── Outcome interpretation, memory formation               │
│  └── Cost: ~100-500μs per event                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Intent Confidence System

Players don't recalculate psychology every frame. They maintain **intent confidence** that decays:

```python
class PlayerBrain:
    current_intent: MovementIntent
    intent_confidence: float  # 1.0 = fresh, decays over time

    def update(self, dt: float, match_state: MatchState):
        # Decay confidence
        self.intent_confidence -= dt * self.decay_rate(match_state)

        # Check triggers for reevaluation
        if self.should_reevaluate(match_state):
            self.full_reevaluation(match_state)  # Expensive
        elif self.intent_confidence < ADAPTATION_THRESHOLD:
            self.lightweight_adaptation(match_state)  # Cheap

        return self.current_intent
```

**Reevaluation Triggers:**
- Ball zone changed
- Possession changed
- Intent confidence below minimum threshold
- Threat landscape significantly changed
- Time since last evaluation exceeds maximum

**Decay Rate Factors:**
- Distance to ball (closer = faster decay)
- Match phase (transition = faster decay)
- Mental fatigue (higher = slower adaptation)

### 4.3 Performance Budget

```
COMPUTE BUDGET PER SECOND
─────────────────────────────────────────────────────────────
Full Reevaluation
  Near ball (3-4 players × 2-3/sec × 1000μs)     =  9ms
  Mid distance (6-8 players × 0.5/sec × 1000μs)  =  4ms
  Far (10-12 players × 0.2/sec × 1000μs)         =  2ms
                                          Subtotal: 15ms

Lightweight Adaptation
  All players (22 × 3/sec × 75μs)                =  5ms

Intent → Physics (every tick)
  All players (22 × 60/sec × 5μs)                =  7ms

Feedback Processing
  Events (~2/sec × 300μs)                        =  0.6ms
─────────────────────────────────────────────────────────────
TOTAL: ~28ms per second = 2.8% of frame budget
```

### 4.4 Caching Strategy

```python
@dataclass
class CachedDecisionContext:
    timestamp: float
    pressure_context: PressureContext
    situation_embedding: SituationEmbedding
    available_options: List[ActionType]
    instinct_weights: Dict[ActionType, float]

    def is_valid(self, current_time: float, invalidation_events: List) -> bool:
        if (current_time - self.timestamp) > MAX_CACHE_AGE:
            return False
        if any(e.invalidates_cache for e in invalidation_events):
            return False
        return True
```

**Cache Invalidation Events:**
- Player received/lost ball
- Significant confidence change (goal, error)
- Zone transition
- Major pressure threshold crossed

---

## 5. System Interactions

### 5.1 Decision Engine Flow

```
INPUT
─────────────────────────────────────────────────────────────
│ MatchState        │ Ball position, player positions, score, time
│ PlayerPhysical    │ Position, stamina, has ball
│ PlayerPsychology  │ CognitiveProfile + PsychologicalState
│ TeamTactics       │ Formation, instructions, mentality
└───────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     1. PRESSURE CALCULATION                                 │
│     spatial + temporal + tactical + psychological           │
│     → PressureContext                                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     2. SYSTEM SELECTION                                     │
│     system1_weight = pressure / effective_composure         │
│     (clamped 0-1)                                           │
└─────────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────┐
         ▼                                 ▼
┌─────────────────────┐     ┌─────────────────────────────────┐
│ SYSTEM 1 (Instinct) │     │ SYSTEM 2 (Analysis)             │
├─────────────────────┤     ├─────────────────────────────────┤
│ Query instinct bank │     │ Generate all options            │
│ with current        │     │ Filter by vision/pressure       │
│ situation embedding │     │ Evaluate tactical utility       │
│                     │     │ Apply personality weights       │
│ Apply trauma blocks │     │ Score each option               │
│ Apply success boosts│     │                                 │
└──────────┬──────────┘     └──────────────┬──────────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│     3. BLEND & SELECT                                       │
│     Weighted combination of System 1 & 2 outputs            │
│     → Selected ActionType with parameters                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     4. INTENT FORMATION                                     │
│     Action → Movement intent + action readiness             │
│     → MovementIntent (continuous) + ActionIntent (discrete) │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 System 2 Utility Evaluation

When System 2 (Analysis) is active, options are scored using a multi-factor utility function:

```python
def evaluate_option_utility(option: ActionType, context: Context,
                            player: Player) -> float:
    """
    Score an action option based on tactical value and personality.
    Higher scores = more attractive options.
    """
    # Base tactical value (from team instructions + game state)
    tactical_value = calculate_tactical_value(option, context)

    # Strategic alignment (does this serve current team goal?)
    strategic_weight = context.match_state.get_strategic_weights()
    strategic_score = strategic_weight.score(option)

    # Risk/reward evaluation
    risk = estimate_risk(option, context)
    reward = estimate_reward(option, context)

    # Personality warping
    personality_modifier = apply_personality(player, risk, reward)

    # Combine
    utility = (
        tactical_value * 0.35 +
        strategic_score * 0.25 +
        (reward - risk * player.cognitive.risk_aversion) * 0.25 +
        personality_modifier * 0.15
    )

    return utility
```

**Tactical Value Calculation:**

| Factor | Weight | Description |
|--------|--------|-------------|
| Progression | 0.3 | Does action move ball toward goal? |
| Retention | 0.25 | Likelihood of maintaining possession |
| Space Creation | 0.2 | Does action create/exploit space? |
| Teammate Benefit | 0.15 | Does action put teammate in good position? |
| Instruction Alignment | 0.1 | Does action follow tactical instructions? |

```python
def calculate_tactical_value(option: ActionType, context: Context) -> float:
    progression = score_progression(option, context) * 0.30
    retention = score_retention_probability(option, context) * 0.25
    space = score_space_creation(option, context) * 0.20
    teammate = score_teammate_benefit(option, context) * 0.15
    instruction = score_instruction_alignment(option, context) * 0.10

    return progression + retention + space + teammate + instruction
```

**Risk Estimation:**

```python
def estimate_risk(option: ActionType, context: Context) -> float:
    """0 = no risk, 1 = maximum risk"""

    base_risk = OPTION_BASE_RISK[option]  # e.g., through_ball = 0.6, pass_back = 0.1

    # Modifiers
    pressure_risk = context.pressure.spatial * 0.3
    interception_risk = calculate_interception_probability(option, context) * 0.4
    turnover_cost = context.field_position.turnover_danger * 0.3

    return clamp(base_risk + pressure_risk + interception_risk + turnover_cost, 0, 1)
```

**Personality Warping:**

```python
def apply_personality(player: Player, risk: float, reward: float) -> float:
    """Personality distorts the risk/reward landscape"""

    cog = player.cognitive

    # Aggression increases appetite for high-risk/high-reward
    aggression_bonus = cog.aggression * reward * 0.5 if risk > 0.5 else 0

    # Low composure under pressure penalizes complex options
    pressure_penalty = 0
    if player.current_pressure > 0.6 and cog.composure < 0.5:
        pressure_penalty = -0.3  # Bias toward simple options

    # Intelligence affects accuracy of utility estimation
    # (low intelligence = more noise in evaluation)
    noise = (1 - cog.intelligence) * random.uniform(-0.2, 0.2)

    return aggression_bonus + pressure_penalty + noise
```

**Strategic Weights by Match State:**

| Match State | Progression Weight | Retention Weight | Risk Tolerance |
|-------------|-------------------|------------------|----------------|
| Losing, late | 1.5 | 0.5 | High |
| Losing, early | 1.2 | 0.8 | Medium-High |
| Drawing | 1.0 | 1.0 | Medium |
| Winning, early | 0.8 | 1.2 | Medium-Low |
| Winning, late | 0.5 | 1.5 | Low |

### 5.3 Confidence-Instinct Sampling Relationship

**Critical insight**: Confidence doesn't just affect composure—it changes WHICH instincts are sampled.

```python
class InstinctBank:
    def query(self, situation: SituationEmbedding,
              confidence: float) -> Dict[ActionType, float]:
        """
        Query instinct bank with confidence-weighted sampling.
        High confidence → sample from success anchors
        Low confidence → sample from safe defaults
        """

        # Get base instinct matches
        base_matches = self._similarity_query(situation)

        # Separate by source type
        success_anchors = [m for m in base_matches if m.source == "success"]
        safe_defaults = [m for m in base_matches if m.source == "comfort"]
        experience_based = [m for m in base_matches if m.source == "experience"]

        # Weight by confidence
        if confidence > 0.3:
            # Confident: lean into success anchors
            anchor_weight = 0.4 + (confidence * 0.4)  # 0.4 to 0.8
            safe_weight = 0.1
            experience_weight = 1.0 - anchor_weight - safe_weight
        elif confidence < -0.3:
            # Rattled: retreat to safe defaults
            safe_weight = 0.4 + (abs(confidence) * 0.4)  # 0.4 to 0.8
            anchor_weight = 0.1
            experience_weight = 1.0 - anchor_weight - safe_weight
        else:
            # Neutral: balanced sampling
            anchor_weight = 0.25
            safe_weight = 0.25
            experience_weight = 0.5

        # Blend and return
        return self._blend_instincts(
            success_anchors, anchor_weight,
            safe_defaults, safe_weight,
            experience_based, experience_weight
        )
```

**Confidence Sampling Visualization:**

```
CONFIDENCE LEVEL → INSTINCT SOURCE WEIGHTS
─────────────────────────────────────────────────────────────
+1.0  ████████████████░░░░░░░░ Success Anchors (80%)
      ██░░░░░░░░░░░░░░░░░░░░░░ Safe Defaults (10%)
      ██░░░░░░░░░░░░░░░░░░░░░░ Experience (10%)

+0.5  ██████████░░░░░░░░░░░░░░ Success Anchors (50%)
      ████░░░░░░░░░░░░░░░░░░░░ Safe Defaults (20%)
      ██████░░░░░░░░░░░░░░░░░░ Experience (30%)

 0.0  █████░░░░░░░░░░░░░░░░░░░ Success Anchors (25%)
      █████░░░░░░░░░░░░░░░░░░░ Safe Defaults (25%)
      ██████████░░░░░░░░░░░░░░ Experience (50%)

-0.5  ████░░░░░░░░░░░░░░░░░░░░ Success Anchors (20%)
      ██████████░░░░░░░░░░░░░░ Safe Defaults (50%)
      ██████░░░░░░░░░░░░░░░░░░ Experience (30%)

-1.0  ██░░░░░░░░░░░░░░░░░░░░░░ Success Anchors (10%)
      ████████████████░░░░░░░░ Safe Defaults (80%)
      ██░░░░░░░░░░░░░░░░░░░░░░ Experience (10%)
```

**Behavioral Implications:**

| Confidence | Instinct Behavior | Observable Result |
|------------|------------------|-------------------|
| High (+0.5 to +1.0) | Samples from "this worked brilliantly" memories | Attempts ambitious plays, recreates past glory |
| Neutral (-0.2 to +0.2) | Balanced experience-based | Plays to training, standard patterns |
| Low (-1.0 to -0.5) | Samples from "this is always safe" | Conservative, predictable, avoids responsibility |

### 5.4 Feedback Processing

```
EVENT OCCURS (e.g., pass attempted → intercepted)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     1. OBJECTIVE OUTCOME                                    │
│     success: false, risk_level: high, visibility: high      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     2. GATHER REACTIONS                                     │
│     ┌─────────┬─────────┬─────────┬─────────┐              │
│     │  SELF   │  TEAM   │ CROWD   │ COACH   │              │
│     │ eval    │ reaction│ reaction│ reaction│              │
│     └─────────┴─────────┴─────────┴─────────┘              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     3. WEIGHT BY VALIDATION PROFILE                         │
│     weighted_impact = Σ(reaction × sensitivity)             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     4. APPLY TEAM CULTURE BUFFER                            │
│     Supportive culture softens negative feedback            │
│     Blame culture amplifies it                              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     5. UPDATE PSYCHOLOGICAL STATE                           │
│     • Confidence delta                                      │
│     • Active modifier creation                              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     6. MEMORY FORMATION CHECK                               │
│     If |impact| × significance > threshold:                 │
│       → Create trauma or success anchor                     │
│       → Update instinct bank weights                        │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Confidence Update Rules

| Event | Base Confidence Δ | Duration | Notes |
|-------|-------------------|----------|-------|
| Goal scored | +0.30 | 15 min | Major positive anchor potential |
| Assist | +0.20 | 10 min | |
| Key pass completed | +0.10 | 5 min | |
| Successful tackle | +0.10 | 5 min | |
| Simple pass completed | +0.02 | 1 min | Cumulative stability |
| Pass misplaced | -0.05 | 3 min | |
| Dispossessed | -0.10 | 5 min | |
| Missed clear chance | -0.25 | 20 min | Trauma potential |
| Caused goal against | -0.30 | Rest of match | High trauma potential |
| Got nutmegged | -0.15 | 10 min | Aggression spike |
| Yellow card | -0.10 | Rest of match | Aggression dampened |

**Modifiers:**
- Event significance multiplies base delta (cup final = 2x)
- Validation profile weights modify based on feedback source
- Team culture can buffer up to 30% of negative impact
- Confidence is clamped to [-1, 1]

### 5.4 Memory Formation

**Significance Calculation:**

```python
def calculate_significance(event, context, player) -> float:
    base = event.base_significance

    # Context multipliers
    base *= context.match_importance      # 1.0 (friendly) to 3.0 (cup final)
    base *= context.time_criticality      # 1.0 (early) to 2.0 (final minutes)
    base *= event.outcome_extremity       # 1.0 (normal) to 2.0 (exceptional)

    # Personal multipliers
    if player.is_first_time(event.type):
        base *= 1.5                       # First goal, debut, etc.

    base *= context.social_amplification  # Crowd size, media presence

    return min(base, MAX_SIGNIFICANCE)
```

**Memory Persistence Thresholds:**

| Significance Score | Result |
|-------------------|--------|
| < 0.3 | No memory formed |
| 0.3 - 0.6 | Seasonal memory (Layer 3) |
| 0.6 - 0.8 | Career memory (Layer 2) |
| > 0.8 | Core memory (Layer 1-2, near permanent) |

---

## 6. Social Systems

### 6.1 Mentor Relationships

Young players (plasticity > 0.5) form mentor relationships with senior players:

```python
@dataclass
class MentorRelationship:
    mentor_id: PlayerId
    mentee_id: PlayerId
    bond_strength: float          # 0-1, grows with shared time
    influence_domains: List[str]  # What aspects transfer
```

**Mentor Selection Criteria:**
- Age gap: 3-12 years (optimal: 5-8)
- Same position group: Strong bonus
- Captain/senior status: Bonus
- Personality compatibility: Moderate match preferred

**Trait Transmission:**

| Trait | Transfer Rate | Mechanism |
|-------|--------------|-----------|
| Composure | Medium | Modeling under pressure |
| Instinct Bank | High | Observation during play |
| Reaction Templates | High | Social learning |
| Aggression | Low | Bounded suggestion |
| Risk Evaluation | Medium | Calibration of "acceptable" |

**Transmission Formula:**
```python
transfer_rate = (
    base_rate[trait] *
    mentee.plasticity *
    mentor.leadership_quality *
    relationship.bond_strength *
    shared_pitch_time_factor
)
```

### 6.2 Team Culture

Team-wide psychological environment:

```python
@dataclass
class TeamCulture:
    blame_tolerance: float     # 0 (punitive) to 1 (supportive)
    risk_encouragement: float  # 0 (safe) to 1 (creative)
    hierarchy_strength: float  # 0 (flat) to 1 (captain dominant)
    cohesion: float           # 0 (fragmented) to 1 (unified)
```

**Culture Effects:**
- `blame_tolerance` buffers negative feedback impact
- `risk_encouragement` affects aggression trait evolution
- `hierarchy_strength` weights captain reaction in team feedback
- `cohesion` affects team-wide confidence spillover

### 6.3 Crowd Model

```python
@dataclass
class CrowdState:
    hostility: float          # -1 (home support) to 1 (hostile away)
    intensity: float          # Current noise/involvement level
    patience: float           # Tolerance for mistakes (degrades with match state)

    # Memory
    recent_positive: List[PlayerId]  # Players with recent goodwill
    recent_negative: List[PlayerId]  # Players crowd is targeting
```

**Crowd Reaction Calculation:**
```python
def crowd_reaction(event, crowd, player) -> float:
    base = event.visibility * event.outcome_valence  # +/- based on good/bad

    # Hostility inverts reaction
    if crowd.hostility > 0:  # Away
        base *= -1 * crowd.hostility
    else:  # Home
        base *= (1 - crowd.hostility)

    # Patience modifier
    if base < 0:
        base *= (1 + (1 - crowd.patience))  # Less patience = harsher reaction

    # Personal history
    if player.id in crowd.recent_negative:
        base *= 1.3  # Crowd is already on them

    return base * crowd.intensity
```

---

## 7. Implementation Phases

### 7.0 Phase Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     IMPLEMENTATION DEPENDENCY GRAPH                      │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 1: MVP                         PHASE 2: Decision Depth
┌─────────────────────┐              ┌─────────────────────┐
│ • Composure trait   │              │ • Full 6 traits     │
│ • Confidence state  │─────────────▶│ • System 1/2 blend  │
│ • Basic pressure    │   REQUIRES   │ • Situation embed   │
│ • Intent modifiers  │              │ • Instinct bank     │
│ • Simple feedback   │              │ • Option generation │
└─────────────────────┘              └──────────┬──────────┘
                                                │
                                                │ REQUIRES
                                                ▼
PHASE 4: Social Systems              PHASE 3: Learning & Memory
┌─────────────────────┐              ┌─────────────────────┐
│ • Mentor system     │◀─────────────│ • Trauma/anchors    │
│ • Team culture      │   REQUIRES   │ • Validation profile│
│ • Career persist    │              │ • Memory formation  │
│ • Academy imprints  │              │ • Cross-match state │
└─────────────────────┘              └─────────────────────┘

─────────────────────────────────────────────────────────────────────────

DETAILED DEPENDENCIES:

Phase 1 provides:
  ├── PressureContext calculation
  ├── Basic PsychologicalState
  ├── MovementIntent structure
  └── Feedback → Confidence loop

Phase 2 requires Phase 1, adds:
  ├── CognitiveProfile (extends composure-only)
  ├── SituationEmbedding (extends basic pressure)
  ├── InstinctBank (requires situation embedding)
  ├── System blend logic (requires full cognitive profile)
  └── Utility evaluation (requires cognitive profile)

Phase 3 requires Phase 2, adds:
  ├── ValidationProfile (extends feedback processing)
  ├── MemoryStore (requires situation embedding)
  ├── Trauma/Anchor formation (requires instinct bank)
  └── Persistence layer (requires all data structures)

Phase 4 requires Phase 3, adds:
  ├── MentorRelationship (requires memory + instinct systems)
  ├── TeamCulture (requires validation profile)
  ├── Academy imprints (requires instinct bank seeding)
  └── Career-spanning persistence (requires all layers)
```

**Critical Path:**
```
Pressure Calculation → Situation Embedding → Instinct Bank → Memory System → Social Systems
       (P1)                  (P2)               (P2)           (P3)            (P4)
```

**Parallel Work Opportunities:**

| During Phase | Can Parallel Develop |
|--------------|---------------------|
| Phase 1 | Unit tests, visualization tools, debug logging |
| Phase 2 | Memory store schema, mentor relationship data model |
| Phase 3 | Team culture model, academy imprint templates |
| Phase 4 | N/A - integration phase |

### Phase 1: Minimum Viable Psychology (MVP)

**Scope:**
- 1 trait: Composure
- 1 state: Confidence
- Binary system selection: Instinct vs Analysis (simplified)
- 3 actions: Pass safe / Pass risky / Clear
- Basic feedback: Outcome → Confidence

**Goal:** See a player's movement and decisions change based on psychological state.

**Data Structures:**
```python
@dataclass
class CognitiveProfile_MVP:
    composure: float

@dataclass
class PsychologicalState_MVP:
    confidence: float

@dataclass
class MovementIntent_MVP:
    target_position: Vector2D
    urgency: float
    ball_seeking: float
```

**Core Loop:**
```python
def tick(self, dt):
    for player in self.players:
        base_intent = self.tactics.get_base_intent(player, self.state)
        pressure = self.calculate_pressure(player, self.state)
        modified_intent = self.modify_intent(base_intent, player, pressure)
        self.physics.move_player(player, modified_intent, dt)

    for event in self.events_this_tick:
        self.process_feedback(event.player, event)
```

### Phase 2: Decision Depth

**Adds:**
- Full cognitive profile (6 traits)
- System 1/2 blending with proper weighting
- Basic instinct bank with comfort actions
- Situation embedding for instinct queries

**Goal:** Different players make visibly different decisions under identical circumstances.

### Phase 3: Learning & Memory

**Adds:**
- Trauma and success anchor formation
- Confidence momentum across matches
- Validation profile and social feedback weighting
- Memory persistence (Layer 2-3)

**Goal:** Players develop psychological patterns over a season.

### Phase 4: Social Systems

**Adds:**
- Mentor relationships with trait transmission
- Team culture effects
- Career-spanning persistence (Layer 1-2)
- Academy imprints

**Goal:** Full psychological life simulation with emergent career narratives.

---

## 8. Appendices

### A. Action Types

```python
class ActionType(Enum):
    # Ball actions
    PASS_SHORT = auto()
    PASS_LONG = auto()
    PASS_THROUGH = auto()
    PASS_BACK = auto()
    CROSS = auto()
    SHOOT = auto()
    DRIBBLE = auto()
    CLEAR = auto()
    HOLD = auto()

    # Defensive
    PRESS = auto()
    COVER = auto()
    TACKLE = auto()
    INTERCEPT = auto()
    MARK = auto()

    # Movement
    RUN_FORWARD = auto()
    RUN_WIDE = auto()
    DROP_DEEP = auto()
    HOLD_POSITION = auto()
```

### B. Situation Embedding Dimensions

| Dimension | Range | Description |
|-----------|-------|-------------|
| pressure_level | 0-1 | Current pressure from all sources |
| progression_value | -1 to 1 | Field position relative to goals |
| time_criticality | 0-1 | Match time urgency |
| spatial_density | 0-1 | Player density in zone |
| body_orientation | 0-1 | Facing goal vs away |
| support_available | 0-1 | Teammate passing options |

### C. Player Archetypes

Emergent from trait combinations:

**The Metronome** (High Composure, High Intelligence, Low Aggression)
- Pressure barely affects option space
- Always finds the "correct" pass
- Sometimes frustratingly safe
- Fatigue: Options narrow but quality stays

**The Chaos Engine** (Low Composure, High Aggression, High Vision)
- Sees everything, wants to try everything
- Pressure makes them MORE dangerous (or disastrous)
- High variance outcomes
- Fatigue: Aggression stays, intelligence drops

**The Soldier** (High Resilience, Medium everything)
- Consistent regardless of match state
- Minute 90 same as minute 10
- Won't win you the game, won't lose it
- Fatigue: What fatigue?

**The Playmaker** (High Intelligence, High Vision, Low Resilience)
- Sees passes others can't imagine
- First 60 minutes only
- Fatigue: Catastrophic decline

### D. Formula Reference

**Effective Composure:**
```
effective_composure = composure + (confidence × 0.3) - (mental_fatigue × 0.2)
```

**System Blend:**
```
system1_weight = min(pressure.total / effective_composure, 1.0)
system2_weight = 1.0 - system1_weight
```

**Plasticity by Age:**
```
if age < 20: plasticity = 1.0
elif age < 23: plasticity = 0.8
elif age < 27: plasticity = 0.5
elif age < 30: plasticity = 0.3
else: plasticity = 0.15
```

**Confidence Update:**
```
impact = base_delta × significance × validation_weights × culture_buffer
confidence = clamp(confidence + impact, -1, 1)
```

**Memory Decay:**
```
strength(t) = initial_strength × e^(-decay_rate × t)
# But trauma resurfaces: if context_similarity > 0.7, strength temporarily restored
```

### E. Test Scenarios

Comprehensive test scenarios for validating psychological engine behavior.

#### E.1 Unit Tests - Core Mechanics

```python
# TEST: Pressure affects system selection
def test_pressure_triggers_system1():
    player = create_player(composure=0.5)

    low_pressure = PressureContext(spatial=0.2, temporal=0.2, tactical=0.1, psychological=0.1)
    high_pressure = PressureContext(spatial=0.8, temporal=0.9, tactical=0.5, psychological=0.6)

    blend_low = calculate_system_blend(player, low_pressure)
    blend_high = calculate_system_blend(player, high_pressure)

    assert blend_low.system1_weight < 0.5  # Analysis dominates
    assert blend_high.system1_weight > 0.8  # Instinct dominates


# TEST: Confidence modifies effective composure
def test_confidence_buffs_composure():
    player = create_player(composure=0.5, confidence=0.0)
    pressure = PressureContext(spatial=0.6, temporal=0.5, tactical=0.3, psychological=0.3)

    base_blend = calculate_system_blend(player, pressure)

    player.state.confidence = 0.8  # High confidence
    confident_blend = calculate_system_blend(player, pressure)

    # High confidence should give more System 2 access
    assert confident_blend.system2_weight > base_blend.system2_weight


# TEST: Feedback updates confidence correctly
def test_positive_event_increases_confidence():
    player = create_player(confidence=0.0)
    event = MatchEvent(type="goal_scored", player=player, success=True, significance=0.8)

    process_feedback(player, event)

    assert player.state.confidence > 0.2


# TEST: Confidence affects instinct sampling
def test_low_confidence_samples_safe_defaults():
    player = create_player(confidence=-0.7)
    situation = create_situation(pressure=0.6)

    player.instinct_bank.add_success_anchor(situation, ActionType.PASS_THROUGH, strength=0.8)
    player.instinct_bank.add_safe_default(situation, ActionType.PASS_BACK, strength=0.6)

    samples = player.instinct_bank.query(situation, player.state.confidence)

    # Low confidence should weight safe defaults higher
    assert samples[ActionType.PASS_BACK] > samples[ActionType.PASS_THROUGH]
```

#### E.2 Integration Tests - Match Behavior

```python
# TEST: Player movement changes with confidence
def test_rattled_player_hides():
    match = create_match()
    player = match.get_player("midfielder_1")

    # Baseline position
    player.state.confidence = 0.0
    base_intent = calculate_intent(player, match.state)
    base_ball_seeking = base_intent.ball_seeking

    # Simulate negative events
    player.state.confidence = -0.6
    rattled_intent = calculate_intent(player, match.state)

    assert rattled_intent.ball_seeking < base_ball_seeking
    # Rattled player should position further from passing lanes


# TEST: Different players make different decisions
def test_personality_creates_variance():
    match = create_match()
    situation = create_identical_situation()

    composed_player = create_player(composure=0.9, aggression=0.3)
    aggressive_player = create_player(composure=0.4, aggression=0.9)

    # Same situation, same pressure
    decision_1 = decide(composed_player, situation)
    decision_2 = decide(aggressive_player, situation)

    # Decisions should differ based on personality
    # (run multiple times for statistical validation)
    decisions_composed = [decide(composed_player, situation) for _ in range(100)]
    decisions_aggressive = [decide(aggressive_player, situation) for _ in range(100)]

    # Aggressive player should choose high-risk options more often
    risky_composed = sum(1 for d in decisions_composed if d.action in RISKY_ACTIONS)
    risky_aggressive = sum(1 for d in decisions_aggressive if d.action in RISKY_ACTIONS)

    assert risky_aggressive > risky_composed * 1.5
```

#### E.3 Career Arc Tests

```python
# TEST: Trauma persists and resurfaces
def test_trauma_resurfaces_in_similar_context():
    player = create_player(age=25)

    # Create trauma in cup final context
    cup_final_context = create_situation(
        match_importance=0.95,
        time_criticality=0.9,
        pressure=0.8
    )
    trauma_event = MatchEvent(
        type="missed_penalty",
        context=cup_final_context,
        success=False,
        significance=0.95
    )
    process_feedback(player, trauma_event)

    # Verify trauma stored
    assert len(player.psychology.memory_store.trauma_memories) == 1

    # Simulate 50 league matches (low importance)
    for _ in range(50):
        league_context = create_situation(match_importance=0.3, time_criticality=0.5)
        performance = simulate_match_performance(player, league_context)
        # Trauma should be dormant
        assert not player.psychology.has_active_trauma()

    # Simulate similar high-stakes context
    knockout_context = create_situation(
        match_importance=0.9,
        time_criticality=0.85,
        pressure=0.75
    )

    # Trauma should resurface
    activated = player.psychology.check_trauma_activation(knockout_context)
    assert activated
    assert player.psychology.get_active_trauma().action_blocked == ActionType.SHOOT


# TEST: Mentor influence shapes young player
def test_mentorship_transmission():
    veteran = create_player(age=32, composure=0.9)
    veteran.instinct_bank.add_comfort_action(
        situation=create_high_pressure_situation(),
        action=ActionType.PASS_BACK,
        strength=0.8
    )

    youth = create_player(age=18, composure=0.5, plasticity=1.0)

    # Create mentor relationship
    relationship = MentorRelationship(
        mentor=veteran,
        mentee=youth,
        bond_strength=0.8
    )

    # Simulate 2 seasons together
    for season in range(2):
        for match in range(38):
            process_mentor_influence(relationship, shared_pitch_time=90)

    # Youth should have absorbed some composure
    assert youth.cognitive.composure > 0.6

    # Youth should have inherited instinct pattern
    youth_instincts = youth.instinct_bank.query(create_high_pressure_situation())
    assert ActionType.PASS_BACK in youth_instincts
    assert youth_instincts[ActionType.PASS_BACK] > 0.3


# TEST: Youth in toxic culture develops differently
def test_toxic_culture_affects_development():
    youth_supportive = create_player(age=18, internal_locus=0.5)
    youth_toxic = create_player(age=18, internal_locus=0.5)

    supportive_culture = TeamCulture(blame_tolerance=0.8, risk_encouragement=0.7)
    toxic_culture = TeamCulture(blame_tolerance=0.2, risk_encouragement=0.3)

    # Simulate season with equal negative events
    for match in range(38):
        negative_event = create_negative_event(significance=0.4)

        process_feedback_with_culture(youth_supportive, negative_event, supportive_culture)
        process_feedback_with_culture(youth_toxic, negative_event, toxic_culture)

    # Youth in toxic culture should have:
    # - Lower confidence
    assert youth_toxic.state.confidence < youth_supportive.state.confidence

    # - Higher external validation need
    assert youth_toxic.validation.team_sensitivity > youth_supportive.validation.team_sensitivity

    # - More conservative instinct bank
    toxic_instincts = youth_toxic.instinct_bank.get_dominant_patterns()
    supportive_instincts = youth_supportive.instinct_bank.get_dominant_patterns()
    assert count_safe_actions(toxic_instincts) > count_safe_actions(supportive_instincts)
```

#### E.4 Regression Tests

```python
# TEST: Decision logging is complete and traceable
def test_decision_audit_trail():
    player = create_player()
    match = create_match()

    with decision_logging_enabled():
        intent = decide(player, match.state)

    log = get_last_decision_log()

    # All components must be logged
    assert log.pressure_context is not None
    assert log.system_blend is not None
    assert log.options_considered is not None
    assert log.utility_scores is not None
    assert log.instinct_weights is not None
    assert log.selected_action is not None
    assert log.selection_reason is not None


# TEST: Serialization round-trip preserves state
def test_save_load_preserves_psychology():
    player = create_player_with_rich_history()

    # Add various psychological elements
    player.psychology.state.confidence = 0.7
    player.psychology.add_trauma(create_trauma())
    player.psychology.add_success_anchor(create_anchor())
    player.psychology.add_mentor_imprint(create_imprint())

    # Serialize
    save_data = player.psychology.serialize()

    # Deserialize to new player
    new_player = create_empty_player()
    new_player.psychology = PlayerPsychology.deserialize(save_data)

    # Verify all state preserved
    assert new_player.psychology.state.confidence == 0.7
    assert len(new_player.psychology.memory_store.trauma_memories) == 1
    assert len(new_player.psychology.memory_store.success_anchors) == 1
    assert len(new_player.psychology.mentor_imprints) == 1


# TEST: Performance stays within budget
def test_psychology_performance_budget():
    match = create_match_with_22_players()

    # Warm up
    for _ in range(60):
        tick(match, dt=1/60)

    # Measure
    times = []
    for _ in range(600):  # 10 seconds of simulation
        start = time.perf_counter_ns()
        tick(match, dt=1/60)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)

    avg_psychology_time = sum(times) / len(times)

    # Psychology should use < 3% of 16ms frame budget
    assert avg_psychology_time < 480_000  # 0.48ms in nanoseconds
```

### F. Configuration Schema

Designer-tunable parameters separated from hardcoded engine values.

#### F.1 Configuration File Structure

```yaml
# psychology_config.yaml
# Designer-tunable parameters for the psychological engine

version: "1.0"

# ─────────────────────────────────────────────────────────────
# PRESSURE WEIGHTS
# How different pressure sources contribute to total pressure
# ─────────────────────────────────────────────────────────────
pressure:
  weights:
    spatial: 0.35      # Opponent proximity
    temporal: 0.30     # Time pressure
    tactical: 0.20     # Match state urgency
    psychological: 0.15 # Stakes, crowd, history

  # Distance thresholds for spatial pressure
  spatial:
    max_pressure_distance: 2.0   # meters - full pressure
    zero_pressure_distance: 15.0 # meters - no pressure

  # Time thresholds for temporal pressure
  temporal:
    max_pressure_time: 0.5       # seconds - full pressure
    zero_pressure_time: 3.0      # seconds - no pressure

# ─────────────────────────────────────────────────────────────
# CONFIDENCE SYSTEM
# How events affect confidence and its effects on behavior
# ─────────────────────────────────────────────────────────────
confidence:
  # Base confidence deltas by event type
  events:
    goal_scored: 0.30
    assist: 0.20
    key_pass: 0.10
    successful_tackle: 0.10
    pass_completed: 0.02
    pass_misplaced: -0.05
    dispossessed: -0.10
    missed_chance: -0.25
    caused_goal_against: -0.30
    nutmegged: -0.15
    yellow_card: -0.10

  # How confidence affects other systems
  effects:
    composure_modifier_scale: 0.3  # confidence × this = composure bonus
    ball_seeking_scale: 0.4        # confidence × this = ball seeking modifier

  # Confidence decay between matches
  decay:
    match_to_match: 0.1    # Decay toward 0 per match
    season_reset: 0.3      # Reset toward 0 at season start

# ─────────────────────────────────────────────────────────────
# SYSTEM BLEND (Instinct vs Analysis)
# Controls when players use System 1 vs System 2
# ─────────────────────────────────────────────────────────────
system_blend:
  # System 1 weight = pressure / (composure × this scale)
  composure_scale: 1.0

  # Minimum System 2 weight even under extreme pressure
  min_system2_weight: 0.05

  # Skip System 2 entirely below this weight (performance optimization)
  system2_skip_threshold: 0.1

# ─────────────────────────────────────────────────────────────
# UTILITY EVALUATION (System 2)
# Weights for option scoring
# ─────────────────────────────────────────────────────────────
utility:
  tactical_value:
    progression: 0.30
    retention: 0.25
    space_creation: 0.20
    teammate_benefit: 0.15
    instruction_alignment: 0.10

  # Overall utility component weights
  components:
    tactical: 0.35
    strategic: 0.25
    risk_reward: 0.25
    personality: 0.15

  # Base risk values by action type
  action_base_risk:
    pass_short: 0.15
    pass_back: 0.10
    pass_long: 0.35
    pass_through: 0.55
    cross: 0.40
    shoot: 0.50
    dribble: 0.45
    clear: 0.20
    hold: 0.25

# ─────────────────────────────────────────────────────────────
# INSTINCT BANK
# Parameters for instinct queries and learning
# ─────────────────────────────────────────────────────────────
instinct_bank:
  # Similarity threshold for instinct activation
  similarity_threshold: 0.6

  # Maximum instincts to blend per query
  max_blend_count: 5

  # Confidence-based source weighting
  confidence_sampling:
    high_confidence_anchor_weight: 0.8
    high_confidence_safe_weight: 0.1
    neutral_anchor_weight: 0.25
    neutral_safe_weight: 0.25
    low_confidence_anchor_weight: 0.1
    low_confidence_safe_weight: 0.8

  # Learning rates
  learning:
    success_reinforcement: 0.15
    failure_weakening: 0.10
    mentor_transfer_base: 0.05

# ─────────────────────────────────────────────────────────────
# MEMORY SYSTEM
# Thresholds for memory formation and persistence
# ─────────────────────────────────────────────────────────────
memory:
  # Significance thresholds for memory persistence
  thresholds:
    no_memory: 0.3
    seasonal: 0.6       # Layer 3
    career: 0.8         # Layer 2
    core: 0.95          # Layer 1

  # Significance multipliers
  multipliers:
    match_importance_max: 3.0      # Cup final multiplier
    time_criticality_max: 2.0      # Last minute multiplier
    first_time_bonus: 1.5

  # Decay rates (per season)
  decay:
    seasonal_memory: 0.5
    career_memory: 0.1
    trauma_base: 0.05

  # Trauma system
  trauma:
    formation_threshold: 0.7       # Min impact for trauma
    resurfacing_similarity: 0.7    # Context similarity to trigger
    overcome_weakening: 0.3        # Strength reduction when overcome

# ─────────────────────────────────────────────────────────────
# VALIDATION & FEEDBACK
# How social feedback affects players
# ─────────────────────────────────────────────────────────────
feedback:
  # Team culture buffer (max negative feedback reduction)
  culture_buffer_max: 0.30

  # Crowd intensity scaling
  crowd:
    max_intensity: 1.5
    patience_decay_per_mistake: 0.1

  # Validation profile development
  plasticity:
    age_thresholds: [20, 23, 27, 30]
    plasticity_values: [1.0, 0.8, 0.5, 0.3, 0.15]

# ─────────────────────────────────────────────────────────────
# MENTORSHIP
# Parameters for mentor-mentee relationships
# ─────────────────────────────────────────────────────────────
mentorship:
  # Ideal age gap range
  age_gap:
    min: 3
    max: 12
    optimal: 7

  # Bond strength growth
  bond:
    growth_per_match: 0.02
    max_bond: 1.0
    decay_when_separated: 0.1

  # Trait transmission rates (base, before modifiers)
  transmission:
    composure: 0.08
    instinct_bank: 0.12
    reaction_templates: 0.10
    aggression: 0.04
    risk_evaluation: 0.06

# ─────────────────────────────────────────────────────────────
# PERFORMANCE TUNING
# Computational budget controls
# ─────────────────────────────────────────────────────────────
performance:
  # Intent confidence system
  intent:
    initial_confidence: 1.0
    adaptation_threshold: 0.4
    reevaluation_threshold: 0.2
    max_age_seconds: 2.0

  # Decay rate modifiers
  decay:
    near_ball_multiplier: 3.0
    transition_multiplier: 2.0
    fatigue_reduction: 0.5

  # Cache settings
  cache:
    max_age_ms: 500
    invalidation_confidence_delta: 0.3
```

#### F.2 Loading Configuration

```python
class PsychologyConfig:
    _instance: Optional['PsychologyConfig'] = None

    @classmethod
    def load(cls, config_path: str = "psychology_config.yaml") -> 'PsychologyConfig':
        if cls._instance is None:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            cls._instance = cls(data)
        return cls._instance

    @classmethod
    def reload(cls, config_path: str = "psychology_config.yaml"):
        """Hot-reload configuration during development"""
        cls._instance = None
        return cls.load(config_path)

    def __init__(self, data: dict):
        self.pressure = PressureConfig(data['pressure'])
        self.confidence = ConfidenceConfig(data['confidence'])
        self.system_blend = SystemBlendConfig(data['system_blend'])
        self.utility = UtilityConfig(data['utility'])
        self.instinct_bank = InstinctBankConfig(data['instinct_bank'])
        self.memory = MemoryConfig(data['memory'])
        self.feedback = FeedbackConfig(data['feedback'])
        self.mentorship = MentorshipConfig(data['mentorship'])
        self.performance = PerformanceConfig(data['performance'])


# Usage in engine code:
config = PsychologyConfig.load()

spatial_weight = config.pressure.weights['spatial']
goal_confidence_delta = config.confidence.events['goal_scored']
```

#### F.3 Parameter Tuning Guidelines

| Parameter Category | Tuning Approach | Observable Effect |
|-------------------|-----------------|-------------------|
| Pressure weights | Adjust if wrong actions under pressure | Player seems too calm or too panicked |
| Confidence deltas | Adjust if form swings too much/little | Confidence too volatile or too stable |
| Utility weights | Adjust if tactical choices feel wrong | Players ignore tactics or are too rigid |
| Memory thresholds | Adjust if memories form too easily/rarely | Too many traumas or none at all |
| Plasticity values | Adjust if youth develop too fast/slow | 18-year-olds too mature or never develop |
| Mentor transmission | Adjust if trait transfer too strong/weak | Mentorship meaningless or overpowering |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-08 | Party Mode Session | Initial architecture from brainstorming |
| 0.2 | 2026-01-08 | Party Mode Session | Added: System 2 utility formulas, confidence-instinct sampling, serialization, test scenarios, configuration schema, phase dependencies |

---

*"Every position is a decision. Every decision reveals psychology."*
