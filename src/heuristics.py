#!/usr/bin/env python3
"""
DISTILLED HEURISTICS FOR GRID GAME
===================================

Neurosymbolic distilled rules from trained policy.

State: [pos_x, pos_y, green_N/S/E/W, red_N/S/E/W, remaining]
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
"""

import numpy as np

class DistilledPolicy:
    """Distilled heuristic policy class."""
    
    def behavior(self, state):
        """
        Neurosymbolic distilled heuristic policy.
        
        Args:
            state: 11-element numpy array [pos_x, pos_y, green_N/S/E/W, red_N/S/E/W, remaining]
            
        Returns:
            int: Action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        px, py = float(state[0]), float(state[1])      # Position (normalized)
        gn, gs, ge, gw = float(state[2]), float(state[3]), float(state[4]), float(state[5])      # Green boxes adjacent
        rn, rs, re, rw = float(state[6]), float(state[7]), float(state[8]), float(state[9])     # Red boxes adjacent
        remaining = float(state[10])            # Boxes left to collect
        
        # HEURISTIC 1: Collect green boxes when North is accessible
        # If green visible North and no red blocking, move up
        if gn == 1.0 and rn == 0.0 and py > 0.15:
            return 0  # UP
        
        # HEURISTIC 2: Progress toward goal
        # When many boxes remain and not near bottom, move down
        if remaining > 2.0 and py < 0.75:
            return 1  # DOWN
        
        # HEURISTIC 3: Horizontal exploration strategy
        # Upper half: explore right; Lower half: navigate left
        if px < 0.6:
            return 3 if py < 0.5 else 2  # RIGHT or LEFT
        
        # HEURISTIC 4: Avoid red threats
        # If red North visible and able to move, shift right
        if rn == 1.0 and py > 0.2:
            return 3  # RIGHT
        
        # HEURISTIC 5: Fallback - steady downward progress
        return 1  # DOWN

    def explain(self):
        """Human-readable explanation of distilled rules."""
        return '''
HEURISTIC RULES (Neurosymbolic Distilled)
==========================================

Rule 1 (UP for GREEN COLLECTION):
  IF green box North visible AND no red blocking AND position allows upward movement
  THEN move UP (action 0)
  Rationale: Prioritize collecting positive rewards

Rule 2 (DOWN for EXPLORATION):
  IF >2 green boxes remain AND player not near bottom edge (py < 0.75)
  THEN move DOWN (action 1)
  Rationale: Systematically explore board toward goal

Rule 3 (HORIZONTAL MOVEMENT):
  IF position_x < 0.6 AND position_y < 0.5:
    move RIGHT (action 3) to explore new territory
  ELIF position_x < 0.6 AND position_y >= 0.5:
    move LEFT (action 2) to navigate available paths
  Rationale: Balanced exploration strategy

Rule 4 (AVOIDANCE):
  IF red box North visible AND can move (py > 0.2)
  THEN move RIGHT (action 3)
  Rationale: Bypass negative reward threats

Rule 5 (FALLBACK):
  Default action: DOWN (action 1)
  Rationale: Ensures constant progression toward goal
'''


# Create instance for direct use
_policy = DistilledPolicy()

def behavior(state):
    """Module-level function for distilled behavior."""
    return _policy.behavior(state)

def explain():
    """Return rule explanations."""
    return _policy.explain()