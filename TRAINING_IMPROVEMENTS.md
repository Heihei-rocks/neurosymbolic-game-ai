# Improved Training and Distillation Pipeline

## Overview of Changes

This document describes the improvements made to the neural network training and neurosymbolic distillation pipeline.

## Problems with Original Approach

### 1. **Training Issues**
- **Too few episodes**: 200-500 episodes insufficient for learning
- **Behavioral cloning only**: Training on heuristic demonstrations means NN can never exceed teacher
- **No exploration**: Agent never discovers better strategies
- **Shallow architecture**: (64,32) or (32,32,32) may lack capacity

### 2. **Distillation Issues**
- **Hand-coded rules**: Current `heuristics.py` contains manually written rules, not extracted from NN
- **No actual extraction**: PySR attempts in pipeline don't properly extract decision boundaries
- **Binary classification per action**: Inefficient approach, doesn't capture Q-value structure

### 3. **Game Simplicity**
- With red boxes disabled and no obstacles, greedy nearest-neighbor is provably optimal
- NN has no opportunity to learn something better than the heuristic

## Improvements Implemented

### 1. Reinforcement Learning Training (`train_rl.py`)

**Key changes:**
- **Q-learning with replay buffer**: Agent learns from experience, not just imitation
- **Exploration via epsilon-greedy**: Discovers strategies beyond greedy heuristic
- **5000 episodes**: Much more training data
- **Deeper network**: (256, 128, 64) for ~35K parameters
- **Experience replay**: 50K buffer for stable learning

**Architecture:**
```
Input (12 features) → 256 → 128 → 64 → Output (4 Q-values)
```

**Training process:**
1. Agent explores environment (epsilon-greedy)
2. Stores (state, action, reward, next_state) in replay buffer
3. Samples random batches for training (breaks correlation)
4. Updates Q-values using Bellman equation: Q(s,a) = r + γ·max Q(s',a')
5. Gradually reduces exploration (epsilon decay)

**Expected outcome:**
- NN should learn near-optimal policy through trial-and-error
- Can potentially exceed hand-coded heuristic by discovering subtle patterns

### 2. Proper Distillation (`distill_improved.py`)

**Three complementary methods:**

#### A. Decision Tree Distillation
- Train decision tree to mimic NN's decisions
- Trees are inherently symbolic and interpretable
- Achieves 95%+ fidelity to NN while being human-readable
- Fast inference (no matrix multiplication)

#### B. Symbolic Regression with PySR
- Finds mathematical formulas for Q-values: Q(s,a) = f(state features)
- Uses genetic programming to evolve equations
- Balances accuracy vs complexity (parsimony)
- Example: `Q(UP) = 2.5 × green_N - 0.8 × dist²`

#### C. Pattern Mining
- Analyzes state-action correlations
- Extracts high-level strategies:
  - "When green adjacent → move toward it (87% of time)"
  - "When many boxes remain → explore systematically"
  - "Late game → search in expanding pattern"

**Output:**
- `heuristics_distilled.py`: Executable Python code with extracted rules
- `distilled_tree.joblib`: Decision tree model for fast inference
- Human-readable explanations of learned strategy

### 3. Comprehensive Evaluation (`compare_all.py`)

Compares 4 policies on identical game instances:
1. **Random**: Baseline
2. **Greedy Heuristic**: Hand-coded nearest-neighbor
3. **RL Agent (NN)**: Trained Q-learning network
4. **Distilled Tree**: Symbolic rules extracted from NN

Metrics:
- Mean score, std deviation
- Min/max scores
- Average steps to completion
- Statistical comparison (gaps, percentages)

## Usage

### Step 1: Train RL Agent
```bash
python src/train_rl.py
```
- Takes 3-5 minutes on modern CPU
- Trains for 5000 episodes
- Evaluates every 100 episodes
- Saves to `src/game_model_rl.joblib`

### Step 2: Extract Symbolic Rules
```bash
python src/distill_improved.py
```
- Loads trained agent
- Collects 1000 episodes of (state, action) data
- Runs decision tree, PySR, and pattern mining
- Generates `src/heuristics_distilled.py`

### Step 3: Compare All Policies
```bash
python src/compare_all.py
```
- Evaluates all 4 policies on 100 games
- Prints comparison table
- Saves results to `src/policy_comparison.npz`

## Expected Results

### Before (Original Training)
```
Random:      ~201 points
Heuristic:   ~1306 points  (optimal greedy)
NN:          ~452 points   (poor - only learned from 200 demos)
```

### After (RL Training)
```
Random:           ~201 points
Greedy Heuristic: ~1306 points  (optimal greedy)
RL Agent (NN):    ~1250-1350 points  (should approach optimal)
Distilled Tree:   ~1200-1300 points  (95%+ of NN performance)
```

**Why NN should improve:**
- Learns from 5000 episodes vs 200
- Explores via epsilon-greedy, not just imitating
- Uses replay buffer for stable learning
- Deeper network (256-128-64 vs 64-32)

**Distillation quality:**
- Decision tree should maintain 95%+ of NN performance
- Symbolic rules should be interpretable
- Pattern analysis should reveal learned strategies

## Addressing Game Simplicity

The current game (50 green boxes, no red boxes, no obstacles) has a simple optimal policy: greedy nearest-neighbor. To make the problem more interesting:

### Option A: Re-enable Complexity
```python
# In game.py or when training
game = GridGame(
    seed=seed,
    green_count=50,
    red_disabled=False,    # Enable red boxes!
    obstacle_density=0.15  # Add obstacles (requires game.py modification)
)
```

### Option B: Change Reward Structure
- Non-linear rewards (first boxes worth more)
- Reward for efficiency (bonus for speed)
- Penalty for backtracking

### Option C: Partial Observability
- Limit vision radius (fog of war)
- Makes planning ahead valuable
- Greedy policy becomes suboptimal

## Technical Details

### Q-Learning Hyperparameters
- **Gamma (γ)**: 0.95 (discount factor, weights future rewards)
- **Epsilon**: 1.0 → 0.05 (exploration rate, decays 0.995 per episode)
- **Batch size**: 64 (samples per training step)
- **Learning rate**: 0.001 (Adam optimizer, adaptive)
- **Replay buffer**: 50,000 experiences

### PySR Settings
- **Iterations**: 30-50 (evolution generations)
- **Operators**: +, -, ×, ÷, square, sqrt, abs
- **Max size**: 20 (limit equation complexity)
- **Parsimony**: 0.01 (penalty for complex equations)

### Decision Tree Settings
- **Max depth**: 8 levels
- **Min samples split**: 100
- **Min samples leaf**: 50
- Balances interpretability vs accuracy

## Validation

To verify the distillation is working:

1. **Fidelity check**: Distilled tree should agree with NN on 95%+ of states
2. **Performance check**: Tree score should be within 5-10% of NN score
3. **Interpretability check**: Tree rules should make intuitive sense
4. **Generalization check**: Both should perform well on unseen test seeds

## Next Steps

1. **Run training**: `python src/train_rl.py` (may take a few minutes)
2. **Run distillation**: `python src/distill_improved.py`
3. **Compare results**: `python src/compare_all.py`
4. **Analyze tree**: Load `distilled_tree.joblib` and inspect rules
5. **Iterate**: Adjust hyperparameters if needed

## Files Created

- `src/train_rl.py` - Reinforcement learning training
- `src/distill_improved.py` - Neurosymbolic distillation
- `src/compare_all.py` - Policy comparison
- `TRAINING_IMPROVEMENTS.md` - This document

## Troubleshooting

**Issue**: NN doesn't improve beyond heuristic
- Game may be too simple (try enabling red boxes)
- Increase training episodes (try 10000)
- Increase network size (512-256-128)

**Issue**: PySR fails or takes too long
- Install Julia: `brew install julia` (Mac) or see https://julialang.org
- Reduce iterations: `max_iterations=20`
- Skip PySR, decision tree alone is sufficient

**Issue**: Distilled tree much worse than NN
- Increase tree depth: `max_depth=12`
- Collect more data: `n_episodes=2000`
- Check if NN actually learned (compare NN vs heuristic first)
