#!/usr/bin/env python3
"""
Symbolic Policy from PySR Distillation
=======================================

This policy uses the mathematical formulas discovered by PySR
to approximate the neural network's Q-value function.
"""

import numpy as np


def behavior_symbolic(state):
    """
    Policy based on symbolic formulas extracted by PySR (complexity-6 formulas).

    The formulas discovered are:
        Q_UP    = (remaining - 24.39)²
        Q_DOWN  = (remaining - 24.56)²
        Q_LEFT  = (remaining - 24.20)²
        Q_RIGHT = (remaining - 24.13)²

    These formulas have much lower loss than the simpler complexity-4 versions.
    They reveal that the NN learned Q-values based on distance from a critical
    point (~24 remaining boxes), with slight directional bias in the offsets.
    """
    # Extract remaining boxes (x10 in PySR notation)
    remaining = state[10]

    # Compute Q-values using symbolic formulas (complexity-6, lower loss)
    q_up    = (remaining - 24.392365) ** 2
    q_down  = (remaining - 24.560000) ** 2  # Approximate from output
    q_left  = (remaining - 24.198797) ** 2
    q_right = (remaining - 24.133709) ** 2

    # Select action with highest Q-value
    q_values = [q_up, q_down, q_left, q_right]
    return int(np.argmax(q_values))


def explain_symbolic_policy():
    """
    Human-readable explanation of the symbolic policy.
    """
    return """
SYMBOLIC POLICY EXPLANATION
============================

PySR discovered multiple formulas approximating the NN's Q-function.

COMPLEXITY-4 FORMULAS (simpler, higher loss):
    Q(state, action) ≈ reward_counter * coefficient

    Coefficients: UP=347.90, DOWN=351.17, LEFT=356.53, RIGHT=358.74
    These capture time pressure optimization with directional bias.

COMPLEXITY-6 FORMULAS (more accurate, lower loss):
    Q(state, action) ≈ (remaining - offset)²

    Offsets: UP=24.39, DOWN=24.56, LEFT=24.20, RIGHT=24.13

    These formulas reveal a critical insight: Q-values are minimized
    when ~24 boxes remain, and grow quadratically as remaining count
    deviates from this point.

INTERPRETATION:
---------------

1. **Critical Point at 24 Boxes**
   The NN learned that when ~24 boxes remain, the game is at a
   transitional state. Q-values are lowest there.

2. **Quadratic Growth**
   Q-values grow quadratically with distance from this point, meaning:
   - Early game (many boxes): High Q-values encourage exploration
   - Late game (few boxes): High Q-values drive completion

3. **Directional Bias in Offsets**
   RIGHT has lowest offset (24.13), meaning it's preferred when
   remaining count is higher than ~24. This creates systematic
   rightward movement bias.

WHAT'S MISSING:
---------------

These formulas still don't incorporate:
- Position (pos_x, pos_y)
- Adjacent boxes (green_N, green_S, etc.)
- Target directions (nearest_dx, nearest_dy, target2/3)
- Distance metrics

This reveals that while the NN uses these features internally,
the Q-value function can be reasonably approximated using only
the global state (remaining boxes).

EXPECTED PERFORMANCE:
---------------------

Complexity-6 formulas should perform better than complexity-4
(lower loss), but still worse than decision tree and NN since
they lack spatial awareness.

Expected hierarchy: Random < Symbolic < Greedy < Tree ≈ NN
"""
