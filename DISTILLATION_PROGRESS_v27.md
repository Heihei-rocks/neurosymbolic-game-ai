# Neurosymbolic Distillation Process - v0.27 Alpha

## What's Happening

The distillation script is currently running in the background, extracting symbolic rules from the trained neural network using PySR (Python Symbolic Regression).

## Process Steps

### 1. Data Collection (In Progress)
- Running 50 episodes with diverse priority maps
- Each episode: 150 steps maximum
- Collecting (state, Q-values) pairs at each timestep
- **Expected samples**: ~7,500 state-action observations
- **Purpose**: Ensure diverse coverage of state space

### 2. Feature Importance Analysis
- Computing correlation between features and Q-value variance
- Selecting top 20 most important features out of 218
- **Purpose**: Reduce dimensionality for PySR (218 → 20 features)
- **Expected result**: Cognitive features (bearings, gradients, entropy) will rank highest

### 3. Advantage Computation
- Converting Q-values to advantages: A(s,a) = Q(s,a) - mean(Q(s,:))
- **Purpose**: Zero-center values, focus on relative action preferences
- **Why this works**: Past research (v0.15) showed advantages work better than raw Q-values

### 4. Symbolic Regression (Will Take Longest)
- Running PySR for each robot (3 robots) × each action (5 actions) = **15 formulas**
- **Per formula**: 200 iterations of genetic programming, max 20 minutes
- **Total time**: 15 formulas × up to 20 min = **up to 5 hours**
- **Search space**: Combinations of 20 features using operators: +, -, *, /, max, min, sqrt, square, abs, exp, log, sin, cos

### 5. Formula Selection
- For each formula, PySR generates many candidates
- Selection criteria: **Pareto frontier** of accuracy vs complexity
- Best formula: Highest "score" (accuracy/complexity trade-off)
- **Goal**: Simple, interpretable formulas that predict advantages well

## Expected Output Files

1. **src/distillation_data_v27.npz**
   - Raw collected states and Q-values
   - For future analysis and retraining

2. **src/distilled_rules_v27.npz**
   - Extracted symbolic formulas for all 15 actions
   - Feature mappings
   - Performance metrics

3. **src/SYMBOLIC_RULES_v27.md**
   - Human-readable summary
   - Formulas for each robot and action
   - Feature explanations

## What We're Looking For

### Good Symbolic Rules Should:

1. **Use Cognitive Features**
   - Ego-centric bearings (turn toward target)
   - Coverage gradients (move toward high opportunity)
   - Spatial entropy (coordinate spreading)
   - Score velocity (adapt when stuck)

2. **Be Interpretable**
   - Few features per formula (3-8 ideal)
   - Clear logical structure
   - Match human intuition about coordination

3. **Capture Strategies**
   - **Navigation**: "Turn toward high-priority target"
   - **Exploration**: "Move where gradient is highest"
   - **Coordination**: "Spread out when entropy is low"
   - **Adaptation**: "Change strategy when velocity drops"

### Example Formula (Hypothetical)

```
A_turn_left(state) = -1.2 * ego_bearing + 0.8 * gradient_west - 0.3 * spatial_entropy
```

**Plain English**: "Turn left when the target is to the left (negative bearing), when there's good coverage opportunity to the west (positive gradient), and when the team is too clustered (low entropy signals need to spread)."

## Success Criteria

### Minimum Success (60% fidelity):
- ✅ Formulas use cognitive features meaningfully
- ✅ Can explain each formula in plain English
- ✅ Symbolic policy achieves >60% of NN performance

### Good Success (80% fidelity):
- ✅ Formulas reveal coordination strategies
- ✅ Behavior matches observations from GIFs
- ✅ Symbolic policy achieves >80% of NN performance

### Excellent Success (90% fidelity):
- ✅ Discover emergent coordination principles
- ✅ Formulas generalize to new scenarios
- ✅ Symbolic policy achieves >90% of NN performance

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Data Collection | 15-30 min | In Progress |
| Feature Analysis | 5 min | Pending |
| PySR Distillation | 2-5 hours | Pending |
| Implementation | 30 min | Pending |
| Validation | 30 min | Pending |
| Documentation | 30 min | Pending |
| **Total** | **4-7 hours** | **Running** |

## Monitoring

The distillation is running in background process ID: b628frtoq

Check progress with:
```bash
tail -f /private/tmp/claude-*/tasks/b628frtoq.output
```

Or wait for completion notification when the script finishes.

## Next Steps (After Completion)

1. **Analyze Extracted Formulas**
   - Check which features were selected
   - Verify formulas are interpretable
   - Look for coordination patterns

2. **Implement Symbolic Policy**
   - Create heuristic-based agent using extracted formulas
   - Compute advantages for all actions
   - Select action with highest advantage

3. **Validate Performance**
   - Run symbolic policy on 100 test episodes
   - Compare scores to neural network
   - Measure decision agreement (% same action)

4. **Translate to Plain English**
   - Explain each formula in human terms
   - Identify learned strategies
   - Connect to cognitive features

5. **Update Documentation**
   - Add formulas to README
   - Create visualizations comparing NN vs symbolic
   - Document discovered coordination principles

---

**Started**: Current time  
**Expected Completion**: 4-7 hours from start  
**Status**: Running PySR distillation in background
