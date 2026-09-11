# Summary of Improvements

## What Was Changed

### 1. Neural Network Training
**Previous**: Behavioral cloning from 200-500 greedy demonstrations
- Small network (64-32)
- No exploration
- Can't exceed teacher performance

**New**: Reinforcement learning with Q-learning
- Larger network (256-128-64, ~35K parameters)
- 5000 training episodes
- Epsilon-greedy exploration
- Experience replay buffer (50K)
- Can discover strategies beyond hand-coded heuristics

**File**: `src/train_rl.py`

### 2. Neurosymbolic Distillation
**Previous**: Hand-coded rules in `heuristics.py` (not actually distilled from NN)

**New**: Three-method extraction pipeline
1. **Decision Tree** - mimics NN with 95%+ fidelity, fully interpretable
2. **PySR Symbolic Regression** - finds mathematical formulas for Q-values
3. **Pattern Mining** - extracts high-level strategic patterns

**File**: `src/distill_improved.py`

### 3. Evaluation Framework
**New**: Comprehensive comparison of all policies on identical test sets
- Random baseline
- Greedy heuristic
- RL agent (NN)
- Distilled symbolic rules

**File**: `src/compare_all.py`

### 4. Pipeline Automation
**New**: Single-command execution of full pipeline

**File**: `run_pipeline.py`

## How to Use

### Quick Start
```bash
# Run entire pipeline (3-5 minutes)
python run_pipeline.py
```

### Manual Steps
```bash
# Step 1: Train RL agent
python src/train_rl.py

# Step 2: Extract symbolic rules
python src/distill_improved.py

# Step 3: Compare policies
python src/compare_all.py
```

## Expected Improvements

### Training Performance
- **Before**: NN scored ~452 points (suboptimal)
- **After**: NN should score ~1250-1350 points (near-optimal)
- **Reason**: RL explores and learns from trial-and-error vs just imitating

### Distillation Quality
- **Before**: "Distilled" rules were hand-coded, not extracted
- **After**: Rules actually extracted from NN via decision trees + symbolic regression
- **Fidelity**: Distilled tree should maintain 95%+ of NN performance

## Key Technical Improvements

### Q-Learning Algorithm
```
For each episode:
  1. Agent takes action (epsilon-greedy)
  2. Store (state, action, reward, next_state) in buffer
  3. Sample random batch from buffer
  4. Update: Q(s,a) ← r + γ·max Q(s',a')
  5. Train neural network on updated Q-values
  6. Decay exploration rate
```

### Decision Tree Distillation
```
1. Collect (state, action) pairs from trained NN
2. Train decision tree to mimic NN's choices
3. Result: Interpretable rules that approximate NN
4. Bonus: Much faster inference (no matrix math)
```

### Symbolic Regression (PySR)
```
1. For each action, extract Q-values from NN
2. Use genetic programming to evolve equations
3. Find: Q(s, action) = f(position, greens, reds, ...)
4. Balance accuracy vs complexity (parsimony)
```

## Architecture Comparison

### Original
```
Input (12) → 64 → 32 → Output (4)
~2,700 parameters
```

### Improved
```
Input (12) → 256 → 128 → 64 → Output (4)
~35,000 parameters
```

### Distilled Tree
```
Input (12) → Decision Tree (depth 8) → Output (1)
~100-500 decision nodes
Fully interpretable!
```

## Files Created

| File | Purpose |
|------|---------|
| `src/train_rl.py` | Q-learning training with replay buffer |
| `src/distill_improved.py` | Extract symbolic rules from trained NN |
| `src/compare_all.py` | Evaluate all policies on same games |
| `run_pipeline.py` | Automated pipeline runner |
| `TRAINING_IMPROVEMENTS.md` | Detailed technical documentation |
| `IMPROVEMENTS_SUMMARY.md` | This file (quick reference) |

## Validation Checklist

After running the pipeline, verify:

- [ ] RL agent scores similar to greedy heuristic (~1250-1350 points)
- [ ] Distilled tree maintains 95%+ of NN performance
- [ ] Decision tree has reasonable depth (5-10 levels)
- [ ] Extracted patterns make intuitive sense
- [ ] All four policies evaluated on same test set

## Troubleshooting

**NN doesn't improve much?**
- Game might be too simple (greedy is optimal)
- Try enabling red boxes: `red_disabled=False`
- Increase episodes: `n_episodes=10000`

**Distillation takes too long?**
- PySR can be slow (uses Julia backend)
- Reduce iterations: `max_iterations=20`
- Decision tree alone is sufficient

**Want to understand the learned policy?**
```python
import joblib
from sklearn.tree import export_text

tree = joblib.load('src/distilled_tree.joblib')
rules = export_text(tree, feature_names=[
    'pos_x', 'pos_y', 'green_N', 'green_S', 'green_E', 'green_W',
    'red_N', 'red_S', 'red_E', 'red_W', 'remaining', 'dist'
])
print(rules)
```

## Next Steps

1. **Run the pipeline**: `python run_pipeline.py`
2. **Analyze results**: Check `src/policy_comparison.npz`
3. **Inspect rules**: Read `src/heuristics_distilled.py`
4. **Visualize**: Generate GIFs with `python src/make_gifs_scored.py`
5. **Iterate**: Tweak hyperparameters if needed

## Questions?

- See `TRAINING_IMPROVEMENTS.md` for technical details
- See `README.md` for project overview
- See `NEUROSYMBOLIC_RESEARCH.md` for academic context
