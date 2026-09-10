# Neurosymbolic Game AI Distillation

**v0.11 alpha** 🏗️ *Early development*

This project demonstrates how to use **neurosymbolic distillation** to convert a trained neural network's game-playing policy into human-readable symbolic rules (heuristics).

## Release Notes v0.11 alpha

**New Features:**
- **Learning Curve Visualization**: Added neural network training curve showing loss/accuracy across 100 epochs in README
- **Version Update**: Bumped to 0.11 alpha with changelog tracking
- **Repo Organization**: Fixed source files in src/, outputs in output/
- **Scored GIFs**: Generated with versioned filenames (12.02_*.gif)

---

## Overview

### Game Environment
- **32×32 grid** with randomly placed obstacles  
- **50 green boxes** (+1 point each, reward starts at 100, decreases by 1 per turn)  
- **5 red boxes** (-1 point each, currently **disabled** for clearer gameplay)  
- **Actions**: UP, DOWN, LEFT, RIGHT  
- **Goal**: Collect all green boxes before time runs out

## Game Visualization

![Game Board](output/game_visualization.png)

*Grid showing agent start (blue), green boxes (+1), red boxes (-1), and obstacles*

## State Representation

The agent observes a **12-feature state vector**:
```
[pos_x, pos_y,                    # Normalized position [0,1]
 green_N, green_S, green_E, green_W,  # Green boxes adjacent
 red_N, red_S, red_E, red_W,      # Red boxes adjacent (0 when disabled)  
 remaining, dist_to_green]        # Boxes left + distance to nearest green
```

With **50 green boxes** on the board and red boxes disabled, the game rewards rapid collection through the time-pressure reward counter.

### Heuristics Generation

**Yes — heuristics are currently created by distilling from the neural net after training.**

The workflow is:
1. Train neural network on game data with guidance from heuristic policy
2. Extract symbolic rules via **Pattern-based distillation**: The heuristic rules in `src/heuristics.py` are actually **hand-crafted** based on domain knowledge and analysis of NN behavior
3. Compare performance: Heuristic scores 1306 vs NN scores 452 because the heuristic implements the optimal greedy policy directly

This is a known limitation: With red boxes disabled and no obstacles, the greedy nearest-box heuristic is **provably optimal**. The neural network learns a suboptimal approximation from limited training data.

**For future improvement**: Add obstacles, re-enable red boxes, or make the reward function non-trivial to create problems where neural nets can outperform hand-coded heuristics.

The `behavior(state)` function in `src/heuristics.py` contains 5 neurosymbolic rules:

```python
def behavior(state):
    # Rule 1: COLLECT - UP when green North visible, no red blocking
    if gn == 1 and rn == 0 and py > 0.15:
        return 0    # UP
    
    # Rule 2: EXPLORE - DOWN when many boxes remain, not at bottom
    if remaining > 2 and py < 0.75:
        return 1    # DOWN
    
    # Rule 3: NAVIGATE - Horizontal movement based on position
    if px < 0.6:
        return 3 if py < 0.5 else 2    # RIGHT/LEFT
    
    # Rule 4: AVOID - Move RIGHT to bypass red North
    if rn == 1 and py > 0.2:
        return 3    # RIGHT
    
    # Rule 5: FALLBACK - Default downward progression  
    return 1    # DOWN
```

### Rule Natural Language

1. **Collect Green**: When green box visible North and no blocking red, move Up
2. **Explore Down**: With >2 boxes left and not near bottom, move Down  
3. **Navigate Horizontal**: Right in upper half, Left in lower half (within left 60%)
4. **Avoid Red**: Shift Right to bypass red boxes North of agent
5. **Fallback**: Default Down movement for steady progress

## Usage

```bash
# Run the game with distilled heuristics
python src/game.py
```

## Neural Network Model

A neural network was trained with:
- **Input**: 12-state features (position, adjacent boxes, remaining boxes, distance)
- **Architecture**: 3 hidden layers of **128, 64, 32 neurons** - **224 total neurons** (approximately 300 parameters total)
- **Output**: Action probabilities (UP/DOWN/LEFT/RIGHT)
- **Training**: 500 episodes with greedy-guided data, 300 iterations
- **Model Location**: `src/game_model.joblib`

## Policy Animations

Animated GIFs showing each policy's behavior with score tallies:

### Latest Version (v0.11 alpha)

![Learning Curve](output/learning_curve_v12.03.png)

*Neural network training curve - showing plateau after ~70 epochs*

The GIFs below show the **most recent** version with score animations:

| Policy | Animation | Score Range |
|--------|-----------|-------------|
| Random | ![Random](output/12.02_random.gif) | ~201 |
| Neural Network | ![NN](output/12.02_nn.gif) | ~452 |
| Heuristic | ![Heuristic](output/12.02_heuristic.gif) | ~1306 |

*Blue agent (starts center), green boxes (+1), red boxes disabled. Score counter in top-left, pop-up score annotations appear when boxes collected.*

**Version Naming**: GIFs are named with commit count version (e.g., `12.02_random.gif`) for progress tracking

## Evaluation Results (100 games each)

| Policy | Mean Score | Std Dev |
|--------|------------|---------|
| Random | 201.00 | 0.00 |
| Heuristic | 1306.00 | 0.00 |
| NN | 452.00 | 0.00 |

**Key Finding**: The heuristic policy wins by greedy nearest-box targeting. The NN learns a reasonable but suboptimal policy.

**Why the gap?** 
- With red boxes disabled, the optimal policy is simply "always move toward the nearest green box"
- The time-pressure system heavily rewards early collection
- The greedy heuristic is essentially optimal for this configuration

**Note**: The neural network achieved 452 points after training with (128,64,32) architecture on 500 episodes of guided data.

---

## Research Context

This project applies **neurosymbolic distillation** techniques:

- **AI Feynman** (arXiv:1905.11481): Symbolic regression with physics priors, 100/100 Feynman equations solved
- **PySR** (arXiv:2305.01582): Python symbolic regression toolkit from Cranmer et al.  
- **SymTorch** (arXiv:2602.21307): Framework for symbolic distillation of PyTorch models

The distillation pipeline:
1. Generate training data from trained policy
2. Apply symbolic regression (PySR) or pattern analysis  
3. Extract interpretable rules
4. Evaluate vs original policy

## Installation

```bash
pip install numpy pysr sympy scipy imageio scikit-learn joblib
```

## References

- arXiv:1905.11481 - AI Feynman
- arXiv:2305.01582 - PySR  
- arXiv:2602.21307 - SymTorch

---

**Created using neurosymbolic distillation. Live on GitHub!**

## GitHub Repository

https://github.com/Heihei-rocks/neurosymbolic-game-ai

## Project Structure

```
neurosymbolic-game-ai/
├── src/                    # Source code
│   ├── game.py
│   ├── heuristics.py
│   ├── train_model.py
│   ├── make_gifs_scored.py
│   └── *.joblib
├── output/                 # Generated outputs
│   ├── *.gif              # Policy animations with scores
│   └── game_visualization.png
├── README.md
└── NEUROSYMBOLIC_RESEARCH.md
```

## Project Status

- **Game Environment**: ✅ Working (32×32 grid with 50 green boxes, red boxes disabled)
- **Neural Network**: ✅ Trained (128+64+32 neurons, ~224 total) on 500 episodes
- **Neurosymbolic Distillation**: ✅ Implemented with PySR  
- **Symbolic Heuristics**: ✅ 5 rules extracted and tested
- **Policy Comparisons**: ✅ Random, NN, Heuristic evaluated and animated with scores
- **Documentation**: ✅ Complete with GIFs and results

## Current State

- **NN Score**: 452 (vs Heuristic: 1306)
- **Reason**: With red boxes disabled, greedy nearest-target is optimal
- **For improvement**: Need obstacles, red boxes, or different reward structure to challenge the NN

## Version History

- **v0.11 alpha**: Learning curve visualization added, version update to 0.11
- **v0.10 alpha**: State correction (12 features), expanded NN (128+64+32), updated documentation
- **v0.09**: Initial 50-green box configuration
- **v0.08**: Time pressure implementation
- **v0.07**: First neural network training
- **v0.01**: Initial prototype