# PySR Symbolic Distillation Results

## Summary

PySR successfully discovered mathematical formulas through multiple approaches:
1. **Q-value regression (v1):** Failed (0 points)
2. **Contextual formulas (v2):** Limited improvement
3. **Advantage-based regression (v3):** **SUCCESS (1306 points)** ✅

## The Breakthrough: Advantage-Based Formulas

### What Changed

**Original approach (FAILED):**
- Predicted: Q(s,a) - absolute values (0-500)
- Result: Formulas using only global state (`remaining`, `reward_counter`)
- Score: 0 points

**Fixed approach (SUCCESS):**
- Predicted: A(s,a) = Q(s,a) - mean(Q(s,:)) - relative advantages
- Result: Formulas using directional features (`nearest_dx`, `nearest_dy`)
- Score: 1306 points

### Discovered Formulas (Advantage-Based)

```python
A_UP    = (nearest_angle × -32.68) × reward_counter
A_DOWN  = (reward_counter × (nearest_dy / dist_nearest)) × 23.17
A_LEFT  = nearest_dx × -105.05
A_RIGHT = nearest_dx × 118.34
```

**How they work:**
- **LEFT/RIGHT:** Linear function of `nearest_dx` (move toward target horizontally)
- **DOWN:** Considers vertical direction normalized by distance, scaled by time pressure
- **UP:** Uses angle and time pressure

These formulas capture **reactive navigation** - they respond to where the target is!

### Why This Fails

**Problem:** The formulas are static with respect to actions.
- When `remaining = 48`, all Q-values are similar (~550-570)
- Agent picks RIGHT (slightly highest)
- Agent never collects boxes → `remaining` stays 48
- Agent moves RIGHT forever → score = 0

**Root Cause:** PySR found formulas that minimize loss across all states, but the resulting policy lacks **reactivity** to local conditions.

### Discovered Formulas

**Complexity-4 (simplest):**
```
Q_UP    = reward_counter × 347.90
Q_DOWN  = reward_counter × 351.17
Q_LEFT  = reward_counter × 356.53
Q_RIGHT = reward_counter × 358.74
```
- All scale with time pressure
- Directional bias: RIGHT > LEFT > DOWN > UP

**Complexity-6 (more accurate):**
```
Q_UP    = (remaining - 24.39)²
Q_DOWN  = (remaining - 24.56)²
Q_LEFT  = (remaining - 24.20)²
Q_RIGHT = (remaining - 24.13)²
```
- Q-values minimized when ~24 boxes remain
- Quadratic growth away from this critical point
- Still only global state

## What PySR Learned

1. **Time pressure matters:** Complexity-4 formulas capture reward scaling
2. **Critical point at ~24 boxes:** Complexity-6 formulas reveal Q-value minimum
3. **Directional bias:** RIGHT is slightly preferred in both formula sets

## What PySR Missed

PySR could not discover formulas incorporating:
- `pos_x`, `pos_y` - Agent position
- `green_N`, `green_S`, `green_E`, `green_W` - Adjacent boxes
- `nearest_dx`, `nearest_dy` - Direction to target
- `target2_dx`, `target2_dy`, `target3_dx`, `target3_dy` - Multi-target planning

**Why:** These features create complex, non-linear interactions that simple mathematical expressions cannot capture efficiently.

## Performance

- **Random:** 201 points
- **Symbolic (Q-based, v1):** 0 points (stuck moving right)
- **Symbolic (Advantage-based, v3):** **1306 points** ✅
- **Greedy:** 1306 points
- **Decision Tree:** 1346 points (100% NN fidelity)
- **Neural Network:** 1346 points

## Why Advantage-Based Works

### Problem with Q-Values

When predicting Q(s,a), all Q-values are large positive numbers (~0-500) that scale with:
- Time pressure (reward_counter)
- Remaining boxes

PySR found these **global scaling patterns** because they explain the most variance:
```
Q_UP = remaining² × 347  (explains variance but loses action preference)
```

### Solution: Predict Advantages

Advantages are **zero-centered** and show which action is better than average:
```
A(s,a) = Q(s,a) - mean(Q(s,:))
```

For each state:
- Sum of advantages = 0
- Positive advantage = better than average
- Negative advantage = worse than average

This forces PySR to learn **relative preferences**, not absolute scaling.

### Result

PySR discovered formulas using **directional features**:
```
A_LEFT = nearest_dx × -105
```
- When target is left (dx < 0): A_LEFT > 0, agent moves left
- When target is right (dx > 0): A_LEFT < 0, agent avoids left

This is exactly what the decision tree learned (47% importance on `nearest_dx`)!

## Conclusions

1. **PySR works** - it discovered interpretable formulas that approximate Q-values
2. **Simple formulas insufficient** - the game requires reactive behavior based on local state
3. **Decision trees win** - for this task, tree-based distillation preserves the reactive logic better than regression
4. **Need more complex formulas** - future work could try:
   - Higher complexity budget (20-30 operations)
   - Piecewise functions or conditional formulas
   - Separate formulas for different game phases

## Implications for Future Iterations

**Current game (simple):**
- Symbolic (advantage) matches greedy (1306)
- Decision tree matches NN (1346)
- Gap is small because greedy is nearly optimal

**Future game (complex: obstacles, red boxes, larger grid):**
- Symbolic formulas will degrade first (can't capture complex spatial logic)
- Decision tree will degrade next (too many branches)
- Neural network will maintain performance (learns arbitrary functions)

**Expected hierarchy with complexity:**
```
Simple game:  Random < Symbolic ≈ Greedy < Tree ≈ NN
Complex game: Random < Symbolic < Greedy < Tree < NN
```

This validates neurosymbolic distillation while showing the trade-off:
- **Simple formulas:** Interpretable but limited expressiveness
- **Decision trees:** More expressive, still interpretable
- **Neural networks:** Maximum expressiveness, requires distillation for interpretation
