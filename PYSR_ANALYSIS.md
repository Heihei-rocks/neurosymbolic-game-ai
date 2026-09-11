# PySR Symbolic Distillation Results

## Summary

PySR successfully ran symbolic regression on the neural network's Q-values, discovering mathematical formulas at different complexity levels.

## Key Finding: Formulas Are Too Simple

The extracted formulas depend only on **global state** (`remaining` boxes or `reward_counter`), not **local state** (position, adjacent boxes, target directions).

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
- **Symbolic (PySR):** 0 points (stuck moving right)
- **Greedy:** 1306 points
- **Decision Tree:** 1346 points (100% NN fidelity)
- **Neural Network:** 1346 points

## Conclusions

1. **PySR works** - it discovered interpretable formulas that approximate Q-values
2. **Simple formulas insufficient** - the game requires reactive behavior based on local state
3. **Decision trees win** - for this task, tree-based distillation preserves the reactive logic better than regression
4. **Need more complex formulas** - future work could try:
   - Higher complexity budget (20-30 operations)
   - Piecewise functions or conditional formulas
   - Separate formulas for different game phases

## Implications for Future Iterations

When the game becomes more complex (obstacles, red boxes, larger grids), we expect:
- Decision trees to degrade first (too many branches)
- Symbolic formulas to fail earlier (can't capture complexity)
- Neural networks to maintain performance (can learn arbitrary functions)

This validates the neurosymbolic distillation approach while showing its limits for reactive, spatially-aware tasks.
