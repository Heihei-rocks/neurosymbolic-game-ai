# Cognitive Features Analysis - v0.27 Alpha

## Overview

Version 0.27 introduces enhanced cognitive artifacts to help the neural network learn better policies for multi-robot area coverage. These features go beyond raw sensor data to provide derived insights that enable more sophisticated decision-making.

## Why Cognitive Artifacts Matter

Raw state (robot positions, coverage map, priority map) tells the network **what** the situation is. Cognitive artifacts tell the network **how to think about it**:

- They reduce the learning burden by pre-computing useful relationships
- They provide the network with "hunches" that good strategies might follow
- They encode domain knowledge about what matters in coverage missions
- They enable the network to learn faster by focusing on the right abstractions

## New Features Added in v0.27

### 1. Ego-Centric Target Bearing (4 values per robot)

**What it is**: The relative angle to the nearest high-priority target, measured from the robot's current heading.

**Why it helps**:
- **Before**: Robot knew "target is at absolute bearing 045°" 
- **After**: Robot knows "turn left 30° to face target"
- Enables reactive navigation without complex geometric reasoning
- Analogous to how pilots use relative bearings for quick navigation

**Features per robot**:
- `dx`: Horizontal distance to target (normalized)
- `dy`: Vertical distance to target (normalized)  
- `dist`: Euclidean distance to target (normalized)
- `bearing_norm`: Relative angle to target in robot's frame [-1, 1]

**Example**: If robot heading is 90° (north) and target is at 120° (northeast), bearing_norm would be +0.17 (turn right ~30°).

### 2. Local Coverage Gradients (4 values per robot)

**What it is**: Priority-weighted coverage samples in 4 cardinal directions (N, S, E, W) from each robot's position.

**Why it helps**:
- **Before**: Robot had to infer good directions from global coverage map
- **After**: Robot can directly sense "north has good opportunities, south is saturated"
- Enables gradient-following behavior without explicit path planning
- Analogous to chemotaxis (bacteria following chemical gradients)

**Features per robot**:
- `gradient_north`: Weighted coverage 20px north
- `gradient_south`: Weighted coverage 20px south  
- `gradient_east`: Weighted coverage 20px east
- `gradient_west`: Weighted coverage 20px west

**Example**: If north has low coverage (0.1) and high priority (0.9), gradient_north = 0.09 (good opportunity).

### 3. Priority-Weighted Remaining Work (1 global value)

**What it is**: Fraction of total priority mass that remains uncovered.

**Why it helps**:
- **Before**: Network had to compute mission progress from raw coverage
- **After**: Direct signal "we're 30% done" or "we're 95% done"
- Enables adaptive behavior (aggressive early, thorough late)
- Helps with credit assignment in long missions

**Formula**: `uncovered_priority / total_priority`

**Example**: Early mission = 0.9 (lots of work left), late mission = 0.1 (mopping up).

### 4. Team Spatial Entropy (1 global value)

**What it is**: Standard deviation of inter-robot distances, normalized by world size.

**Why it helps**:
- **Before**: Network saw individual distances but not spread pattern
- **After**: Direct signal "we're clustered" vs "we're spread out"
- Encourages dispersion for better coverage
- Prevents robots from bunching up

**Formula**: `std(pairwise_distances) / world_size`

**Example**: 
- Clustered formation: entropy = 0.05 (all robots close together)
- Spread formation: entropy = 0.30 (robots well distributed)

### 5. Recent Score Velocity (1 global value)

**What it is**: Change in priority-weighted coverage from last timestep to current.

**Why it helps**:
- **Before**: Network only saw current score, not trend
- **After**: Knows "we're improving quickly" vs "we're plateauing"
- Enables adaptive exploration (intensify if improving, diversify if stuck)
- Provides immediate performance feedback

**Formula**: `(coverage_sum[t] - coverage_sum[t-1]) / 1000`

**Example**:
- Velocity > 0: Finding new valuable areas (keep going!)
- Velocity ≈ 0: Diminishing returns (try something else)
- Velocity < 0: Losing coverage to decay (revisit faster)

### 6. Nearest Teammate Vectors (3 values per robot)

**What it is**: Direction and distance to each robot's closest teammate.

**Why it helps**:
- **Before**: Robot saw all robot positions but had to compute nearest
- **After**: Direct signal "my nearest ally is 10 nmi northeast"
- Enables collision avoidance and coordination
- Helps maintain formation without explicit communication

**Features per robot**:
- `dx_teammate`: Horizontal distance to nearest teammate (normalized)
- `dy_teammate`: Vertical distance to nearest teammate (normalized)
- `dist_teammate`: Euclidean distance to nearest teammate (normalized)

**Example**: If robots are too close, network can learn to spread out preemptively.

## State Vector Size

**v0.26**: ~191 features
- 3 × num_robots (robot states)
- 8 (coverage/priority stats)
- 128 (8×8 downsampled maps)
- 3 (inter-robot distances for 3 robots)
- 4 × num_robots (target directions)

**v0.27**: ~230 features (39 new features for 3-robot system)
- All v0.26 features
- +4 × num_robots (ego-centric bearings)
- +4 × num_robots (local gradients)
- +1 (remaining work)
- +1 (spatial entropy)
- +1 (score velocity)
- +3 × num_robots (teammate vectors)

For 3 robots: 191 + (4 + 4 + 3) × 3 + 3 = 191 + 33 + 3 = **227 features**

## Expected Performance Improvements

### Better Long-Term Planning
- 20x longer episodes (3000 steps) allow learning sustained coordination
- Score velocity helps agent recognize when strategies pay off over time
- Remaining work enables endgame optimization

### Improved Coordination
- Teammate vectors reduce collisions
- Spatial entropy encourages spreading out
- Local gradients help robots find complementary search areas

### More Adaptive Behavior  
- Ego-centric bearings enable reactive navigation
- Coverage gradients allow opportunistic exploitation
- Score velocity triggers strategy switching when stuck

### Faster Learning
- Cognitive artifacts reduce effective state space
- Network can focus on high-level strategy, not geometry
- Fewer parameters needed to represent useful policies

## Future Enhancements

Potential additional cognitive features for v0.28+:
- **Historical coverage memory**: "I was here 50 timesteps ago"
- **Team velocity consensus**: Average heading of all robots
- **Priority mass distribution**: Are high-priority areas clustered or scattered?
- **Coverage persistence**: How long do areas stay covered before decay?
- **Frontier detection**: Direction to edge of explored region

## References

Similar cognitive feature engineering in RL:
- [RIDE: Rewarding Impact-Driven Exploration](https://arxiv.org/abs/2002.12292) - Novelty-based features
- [Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363) - Prediction error as feature  
- [World Models](https://arxiv.org/abs/1803.10122) - Learned latent features

## Training Details

- Episodes: 500
- Steps per episode: 3000 (20x longer than v0.26)
- Total experiences: 1,500,000
- Network: (256, 128, 64) hidden layers
- Batch size: 64
- Learning rate: 0.001 → 0.0002 (decay 0.999)
- Epsilon: 1.0 → 0.05 (decay 0.9975)
- Priority map: Randomized per episode (8-15 Gaussian blobs)

---

**Version**: v0.27 alpha  
**Date**: 2024  
**Authors**: Heihei team
