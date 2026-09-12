# Neurosymbolic Distillation Analysis for Multi-Robot Swarm (v0.27)

## Executive Summary

This report analyzes the feasibility and design of neurosymbolic distillation for the multi-robot swarm coordination system trained in v0.27. The goal is to extract interpretable symbolic rules from the trained neural network policy using PySR (symbolic regression via genetic programming).

## Problem Setup

### Trained System Overview

**Input**: 218-dimensional state vector per timestep
**Output**: 15 Q-values (5 actions × 3 robots)
**Task**: Multi-robot priority-aware area coverage
**Performance**: +119% improvement over baseline (2.2M → 4.8M)

### Distillation Goal

Extract symbolic formulas that:
1. Predict Q-values or advantages for each action
2. Are human-interpretable (math + plain English)
3. Approximate neural network performance (ideally >80% fidelity)
4. Reveal learned coordination strategies

## State Feature Analysis

The 218-dimensional state vector includes:

### 1. Robot States (9 features)
```
x0-x2:   robot_1 (x_norm, y_norm, heading_norm)
x3-x5:   robot_2 (x_norm, y_norm, heading_norm)
x6-x8:   robot_3 (x_norm, y_norm, heading_norm)
```

### 2. Global Coverage Statistics (8 features)
```
x9-x12:  coverage stats (mean, std, min, max)
x13-x16: weighted coverage stats (mean, std, min, max)
```

### 3. Downsampled Maps (128 features)
```
x17-x80:   coverage map (8×8 = 64 values)
x81-x144:  priority map (8×8 = 64 values)
```

### 4. Inter-Robot Distances (3 features)
```
x145: dist(robot_1, robot_2)
x146: dist(robot_1, robot_3)
x147: dist(robot_2, robot_3)
```

### 5. Per-Robot Target Directions (12 features)
```
x148-x151: robot_1 (dx, dy, dist, bearing) to nearest high-priority target
x152-x155: robot_2 (dx, dy, dist, bearing)
x156-x159: robot_3 (dx, dy, dist, bearing)
```

### 6. Per-Robot Coverage Gradients (12 features)
```
x160-x163: robot_1 (gradient_N, gradient_S, gradient_E, gradient_W)
x164-x167: robot_2 gradients
x168-x171: robot_3 gradients
```

### 7. Global Coordination Features (3 features)
```
x172: priority_weighted_remaining_work
x173: team_spatial_entropy
x174: recent_score_velocity
```

### 8. Per-Robot Teammate Awareness (9 features)
```
x175-x177: robot_1 (dx, dy, dist) to nearest teammate
x178-x180: robot_2 to nearest teammate
x181-x183: robot_3 to nearest teammate
```

**Total**: 9 + 8 + 128 + 3 + 12 + 12 + 3 + 9 = **184 features** (discrepancy!)

**Note**: Actual state_dim is 218, suggesting 34 additional features not accounted for above. This needs investigation.

## Challenge: High-Dimensional State Space

### Dimensionality Problem

**218 features** is extremely high for symbolic regression:
- PySR typically works best with 5-20 features
- Genetic programming search space grows exponentially with feature count
- Risk of overfitting to noise rather than discovering true patterns
- Computational cost becomes prohibitive

### Proposed Solution: Feature Selection

Rather than using all 218 features, we should select the **most important features** for each robot's actions:

#### Critical Features (must include):
1. **Own robot state** (3 features): position, heading
2. **Own target direction** (4 features): dx, dy, dist, bearing to target
3. **Own coverage gradients** (4 features): N, S, E, W opportunities
4. **Global metrics** (3 features): remaining work, entropy, velocity
5. **Nearest teammate** (3 features): dx, dy, dist for coordination

**Total**: ~17 critical features per robot

This reduces the search space dramatically while retaining the cognitive artifacts that drive behavior.

## PySR Configuration Analysis

### Operator Primitives

For multi-robot coordination, we need operators that can express:

#### Binary Operators
- **Arithmetic**: `+`, `-`, `*`, `/` (combine features)
- **Comparison**: `>`, `<` (threshold decisions)
- **Logic**: Could add `max`, `min` for multi-objective reasoning

#### Unary Operators
- **Geometric**: `abs`, `sqrt` (distances, norms)
- **Nonlinear**: `square`, `exp`, `log` (scaling, priorities)
- **Trigonometric**: `sin`, `cos` (angular reasoning for headings/bearings)

### Recommended Configuration

```python
PySRRegressor(
    niterations=200,  # More iterations for complex problem
    binary_operators=["+", "-", "*", "/", "max", "min"],
    unary_operators=["square", "abs", "sqrt", "exp", "log", "sin", "cos"],
    populations=30,
    population_size=100,
    maxsize=30,  # Allow moderately complex formulas
    parsimony=0.005,  # Balance accuracy vs interpretability
    timeout_in_seconds=1200,  # 20 minutes per action
    random_state=42
)
```

### Why These Operators?

**Geometric operators** (`abs`, `sqrt`):
- Distance calculations: `sqrt(dx^2 + dy^2)`
- Magnitude normalization: `abs(gradient_N - gradient_S)`

**Trigonometric operators** (`sin`, `cos`):
- Heading-based navigation: `cos(heading - bearing)`
- Angular differences: `sin(2 * bearing)`

**Comparison operators** (`max`, `min`):
- Multi-objective: `max(gradient_N, gradient_E)`
- Coordination: `min(dist_teammate, dist_target)`

**Exponential/Log**:
- Priority weighting: `exp(remaining_work)`
- Decay modeling: `log(1 + dist_target)`

## Distillation Strategy

### Approach: Advantage-Based Regression

Following the success of v0.15 (box collection game), use **advantage-based distillation**:

```python
Advantage(s, a) = Q(s, a) - mean(Q(s, :))
```

**Why advantages?**
1. Zero-centered (easier to learn)
2. Focuses on relative action preferences, not absolute values
3. Previous work showed this works better than raw Q-values

### Per-Robot Distillation

The challenge: We have **5 actions × 3 robots = 15 Q-values**.

**Options:**

**Option A: Joint Distillation**
- Predict all 15 Q-values with 15 separate formulas
- Pro: Captures full coordination
- Con: 15 formulas is hard to interpret

**Option B: Per-Robot Distillation** (RECOMMENDED)
- Distill each robot's 5 actions independently
- Use only that robot's features + global context
- Pro: 3 sets of 5 formulas = more interpretable
- Con: May miss inter-robot coordination effects

**Option C: Action-Type Distillation**
- Group similar actions: turn_left, turn_right, straight, speed_up, slow_down
- Distill rules for each action type
- Pro: Discovers action archetypes
- Con: Loses robot-specific nuances

**Recommendation**: Start with **Option B** (per-robot), then verify coordination emerges from individual behaviors.

## Data Collection

### Sampling Strategy

To generate training data for PySR:

1. **Load trained agent** from `src/swarm_model_rl.joblib`
2. **Run episodes** on diverse scenarios (different seeds, priority maps)
3. **Collect (state, Q-values)** tuples at each timestep
4. **Compute advantages** per robot: A = Q - mean(Q)

**Sample Size**: 
- Target: 10,000 - 50,000 samples per robot
- From: 50-100 episodes × ~150 steps/episode = ~15,000 timesteps
- Ensures diverse coverage of state space

### State Space Coverage

**Important**: Sample from:
- Different priority map configurations (8-15 blobs)
- Different robot starting formations
- Different mission phases (early, mid, late)
- Different team spatial configurations (clustered, spread)

This prevents overfitting to specific scenarios.

## Feature Importance Analysis

Before distillation, we should:

1. **Compute feature importances** using:
   - Permutation importance
   - Gradient-based saliency
   - Correlation with Q-value variance

2. **Select top 15-20 features** per robot based on importance

3. **Verify interpretability**: Selected features should be cognitive artifacts, not arbitrary map pixels

### Expected Important Features

Based on training results and cognitive feature design:

**High Importance** (likely):
- `ego_bearing`: Relative angle to target (enables reactive navigation)
- `dist_target`: How far to high-priority area
- `gradient_N/S/E/W`: Coverage opportunities in each direction
- `remaining_work`: Mission progress signal
- `spatial_entropy`: Team dispersion metric
- `dist_teammate`: Collision avoidance

**Medium Importance**:
- `heading_norm`: Current orientation
- `score_velocity`: Performance trend
- `coverage_stats`: Global situation awareness

**Low Importance** (likely):
- Individual map pixels (too specific, not generalizable)
- Distant robot positions (local coordination more important)

## Validation Strategy

### Step 1: Formula Quality
- **Loss**: How well formulas predict advantages
- **Complexity**: Number of operations (prefer <20)
- **Score**: PySR's accuracy/parsimony trade-off

### Step 2: Policy Fidelity
Compare symbolic policy vs neural network:
1. Run both on same 100 test episodes
2. Measure score difference (target: <20% gap)
3. Analyze decision agreement (% of timesteps they choose same action)

### Step 3: Interpretability
- Translate each formula to plain English
- Verify formulas use cognitive features meaningfully
- Check if strategies match human intuition

### Success Criteria

✅ **Minimum viable**: 
- Symbolic policy achieves >60% of NN performance
- Formulas use <5 features each
- Can explain behavior in plain English

✅ **Good**:
- Symbolic policy achieves >80% of NN performance
- Formulas reveal interpretable coordination strategies
- Decision agreement >70%

✅ **Excellent**:
- Symbolic policy achieves >90% of NN performance
- Formulas match human expert reasoning
- Discover emergent coordination principles

## Risks and Mitigation

### Risk 1: Overfitting to Map Pixels
**Problem**: PySR might create formulas using specific map coordinates
**Mitigation**: Exclude downsampled maps from feature set, use only aggregated statistics

### Risk 2: Inter-Robot Coordination Invisible
**Problem**: Per-robot distillation might miss team-level strategies
**Mitigation**: Include spatial entropy, teammate distances, verify emergent coordination

### Risk 3: Formula Complexity Explosion
**Problem**: 218 features → extremely complex formulas
**Mitigation**: Feature selection (15-20 features), parsimony penalty, complexity limit

### Risk 4: Action Space Coupling
**Problem**: Actions affect future state, causing temporal dependencies
**Mitigation**: Sample diverse trajectories, use advantage-based approach

### Risk 5: Computational Time
**Problem**: PySR on high-dimensional space is slow
**Mitigation**: Timeout per action (20 min), parallel runs, feature reduction

## Alternative: Decision Tree Distillation

If PySR struggles with 218 features, consider:

**Decision Trees**:
- Pro: Handle high dimensions better
- Pro: Guaranteed interpretability (if-then rules)
- Pro: Faster to train
- Con: Less elegant than closed-form math
- Con: Harder to explain "why" vs "what"

**Hybrid Approach**:
1. Use decision tree to identify important features
2. Use PySR on reduced feature set
3. Combine: "If [tree condition], then action = [PySR formula]"

## Recommended Implementation Plan

### Phase 1: Data Collection & Feature Analysis (1-2 hours)
1. Generate 50,000 samples from trained agent
2. Compute feature importances
3. Select top 15-20 features per robot
4. Verify cognitive features are represented

### Phase 2: Distillation (2-3 hours)
1. Run PySR for each robot (3 robots × 5 actions = 15 formulas)
2. Use advantage-based approach
3. 200 iterations, 20-min timeout per action
4. Save best formulas (complexity <25)

### Phase 3: Implementation & Validation (1-2 hours)
1. Implement symbolic policy using extracted formulas
2. Test on 100 episodes
3. Compare performance to neural network
4. Analyze decision agreement

### Phase 4: Interpretation & Documentation (1-2 hours)
1. Translate each formula to plain English
2. Identify coordination strategies
3. Create visualizations comparing NN vs symbolic
4. Update README with findings

**Total Estimated Time**: 5-9 hours

## Conclusion

### Feasibility: VIABLE with Modifications

The neurosymbolic distillation is **feasible** for the v0.27 swarm system, but requires:

1. **Feature selection**: Reduce 218 → 15-20 critical features
2. **Per-robot approach**: Distill each robot independently
3. **Advantage-based**: Use A(s,a) = Q(s,a) - mean(Q) 
4. **Rich operators**: Include geometric, trigonometric, comparison ops
5. **Longer search**: 200 iterations, 20-min timeouts

### Expected Outcome

**Likely Result**: 
- Discover 15 symbolic formulas (3 robots × 5 actions)
- Each formula uses 3-8 cognitive features
- Symbolic policy achieves 70-85% of NN performance
- Reveals coordination strategies: spreading, priority targeting, revisiting

**Key Insights to Expect**:
- How ego-centric bearings drive navigation
- How coverage gradients enable opportunistic exploration
- How spatial entropy coordinates team dispersion
- How score velocity triggers strategy switching

### Recommendation

**PROCEED** with distillation using:
- Advantage-based PySR
- Per-robot approach
- Feature selection (top 20 per robot)
- Comprehensive operators (geometric, trig, comparison)
- Robust validation against neural network

The cognitive features in v0.27 were designed to be interpretable, which should make distillation more successful than with raw sensor data.

---

**Status**: Analysis Complete  
**Next Step**: Implement data collection script  
**Expected Duration**: 5-9 hours total  
**Success Probability**: High (70-80%) due to cognitive feature engineering
