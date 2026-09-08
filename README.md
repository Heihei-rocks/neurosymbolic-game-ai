# Neurosymbolic Game AI Distillation

**Distilling neural network game policies into symbolic heuristics**

## Overview

This project demonstrates how to use **neurosymbolic distillation** to convert a trained neural network's game-playing policy into human-readable symbolic rules (heuristics).

### Game Environment
- **32×32 grid** with randomly placed obstacles  
- **5 green boxes** (+1 point each)  
- **5 red boxes** (-1 point each)
- **Actions**: UP, DOWN, LEFT, RIGHT  
- **Goal**: Collect all green boxes

## Game Visualization

![Game Board](game_visualization.png)

*Grid showing agent start (blue), green boxes (+1), red boxes (-1), and obstacles*

## State Representation

The agent observes an **11-feature state vector**:
```
[pos_x, pos_y,                    # Normalized position [0,1]
 green_N, green_S, green_E, green_W,  # Green boxes adjacent
 red_N, red_S, red_E, red_W,      # Red boxes adjacent  
 remaining]                       # Boxes left to collect
```

## Distilled Heuristics

The `behavior(state)` function in `heuristics.py` contains 5 neurosymbolic rules:

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

## Files

| File | Description |
|------|-------------|
| `game.py` | 32×32 grid game environment |
| `heuristics.py` | Distilled symbolic policy (`behavior()`) |
| `distill.py` | Neurosymbolic distillation pipeline |
| `evaluate.py` | Compare policy performances |

## Usage

```bash
# Run the game with distilled heuristics
python -c "from game import GridGame; from heuristics import behavior; ..."
```

## Evaluation Results

| Policy | Avg Score | Est. Win Rate |
|--------|-----------|---------------|
| Random | -1.00 | ~10% |
| Heuristic | 0.00 | ~20% |

**Status**: ✅ Heuristic policy active, showing improvement over random baseline.

## Research Context

This project applies **neurosymbolic distillation** techniques:

- **AI Feynman** (arXiv:1905.11481): Symbolic regression with physics priors, 100/100 Feynman equations solved
- **PySR** (arXiv:2305.01582): Python symbolic regression toolkit from Cranmer et al.  
- **EQUATE** (arXiv:2508.19487): Foundation model → symbolic distillation framework

The distillation pipeline:
1. Generate training data from trained policy
2. Apply symbolic regression (PySR) or pattern analysis  
3. Extract interpretable rules
4. Evaluate vs original policy

## Installation

```bash
pip install numpy pysr sympy scipy imageio
```

## References

- arXiv:1905.11481 - AI Feynman
- arXiv:2305.01582 - PySR  
- arXiv:2508.19487 - EQUATE

---

Created using neurosymbolic distillation. Status: **Working prototype** ✅