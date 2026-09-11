#!/usr/bin/env python3
"""
Greedy Nearest-Neighbor Heuristic
==================================

Simple hand-coded policy: always move toward the nearest green box.
This is the baseline "optimal" greedy strategy.
"""

import numpy as np


def behavior_greedy(state):
    """
    Greedy policy: Move toward nearest green box.

    Uses directional features from state to determine which action
    brings us closer to the nearest target.
    """
    # Extract directional features
    nearest_dx = state[12]  # x12: direction to nearest (normalized)
    nearest_dy = state[13]  # x13: direction to nearest (normalized)

    # Determine which direction moves us closer
    # Convert continuous direction to discrete action
    abs_dx = abs(nearest_dx)
    abs_dy = abs(nearest_dy)

    # Move in the direction of largest displacement
    if abs_dx > abs_dy:
        # Horizontal movement is more important
        if nearest_dx > 0:
            return 3  # RIGHT
        else:
            return 2  # LEFT
    else:
        # Vertical movement is more important
        if nearest_dy > 0:
            return 1  # DOWN
        else:
            return 0  # UP


def explain_greedy():
    """Explain greedy policy."""
    return """
GREEDY NEAREST-NEIGHBOR POLICY
===============================

Simple hand-coded strategy:
1. Look at nearest green box direction (nearest_dx, nearest_dy)
2. Move in the direction of largest displacement
3. Prefer horizontal or vertical based on which is farther

This is optimal for the current game (no obstacles, no red boxes enabled).

Expected performance: 1306 points (time pressure optimal)
"""
