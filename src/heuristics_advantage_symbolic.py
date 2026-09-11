#!/usr/bin/env python3
"""
Advantage-Based Symbolic Policy
================================

Uses PySR-discovered advantage formulas (v3 results).
These formulas use directional features and should actually work!
"""

import numpy as np


def behavior_advantage_symbolic(state):
    """
    Policy using advantage-based symbolic formulas from PySR.

    Formulas discovered:
        A_UP    = (nearest_angle × -32.68) × reward_counter
        A_DOWN  = (reward_counter × (nearest_dy / dist_nearest)) × 23.17
        A_LEFT  = nearest_dx × -105.05
        A_RIGHT = nearest_dx × 118.34

    These use directional features (dx, dy, angle) - should be much better!
    """
    # Extract features
    nearest_angle = state[14]  # x14
    nearest_dx = state[12]     # x12
    nearest_dy = state[13]     # x13
    dist_nearest = state[11]   # x11
    reward_counter = state[16] # x16

    # Compute advantages using symbolic formulas
    a_up    = (nearest_angle * -32.683876) * reward_counter
    a_down  = (reward_counter * (nearest_dy / (dist_nearest + 1e-8))) * 23.171993
    a_left  = nearest_dx * -105.0549
    a_right = nearest_dx * 118.336235

    # Select action with highest advantage
    advantages = [a_up, a_down, a_left, a_right]
    return int(np.argmax(advantages))


def explain_advantage_policy():
    """Explain the advantage-based symbolic policy."""
    return """
ADVANTAGE-BASED SYMBOLIC POLICY
================================

PySR discovered formulas predicting A(s,a) = Q(s,a) - mean(Q(s,:))

KEY INSIGHT: By predicting advantages instead of raw Q-values, PySR
learned to focus on RELATIVE action preferences, not absolute values.

DISCOVERED FORMULAS:
--------------------

LEFT/RIGHT (horizontal movement):
    A_LEFT  = nearest_dx × -105
    A_RIGHT = nearest_dx × +118

    Interpretation:
    - When nearest_dx < 0 (target is left): A_LEFT positive, A_RIGHT negative
    - When nearest_dx > 0 (target is right): A_RIGHT positive, A_LEFT negative
    - Magnitudes slightly favor RIGHT (118 > 105) for exploration bias

UP (vertical movement):
    A_UP = (nearest_angle × -32.7) × reward_counter

    Interpretation:
    - Uses angle to nearest target
    - Scaled by time pressure (reward_counter)
    - Negative coefficient means prefer UP when angle indicates upward direction

DOWN (vertical movement):
    A_DOWN = (reward_counter × (nearest_dy / dist_nearest)) × 23.2

    Interpretation:
    - Proportional to vertical direction (nearest_dy)
    - Normalized by distance
    - Scaled by time pressure

COMPARISON TO FIRST ATTEMPT:
-----------------------------

Original PySR (Q-values):
    Q_UP = remaining × 347.90  (failed - 0 points)

Advantage PySR:
    A_LEFT = nearest_dx × -105  (uses directional features!)

The advantage-based formulas actually capture the reactive navigation
logic needed to play the game.

EXPECTED PERFORMANCE:
---------------------

This should significantly outperform the original symbolic policy
because it uses the same directional features the decision tree uses.

Hierarchy: Random < Original Symbolic < Advantage Symbolic < Greedy < Tree ≈ NN
"""
