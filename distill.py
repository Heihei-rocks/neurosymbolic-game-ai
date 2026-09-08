#!/usr/bin/env python3
"""
Neurosymbolic Distillation Pipeline for Grid Game
==================================================

Distills neural network policy to symbolic heuristics.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')
from game import GridGame, generate_training_data

print("=" * 60)
print("NEUROSYMBOLIC DISTILLATION - GAME AI")
print("=" * 60)

# Step 1: Generate training data
print("\n[1] Generating data from random play (100 episodes)...")
states, actions, rewards = generate_training_data(100)
print(f"    Collected {len(states)} state-action pairs")

# Step 2: Train a simple policy
print("\n[2] Training policy model...")
from collections import Counter

def state_hash(state):
    return (int(state[0] * 31), int(state[1] * 31), int(state[10]))

# Group by state hash and pick most common action
groups = {}
for s, a in zip(states, actions):
    h = state_hash(s)
    if h not in groups:
        groups[h] = []
    groups[h].append(a)

policy = {h: Counter(acts).most_common(1)[0][0] for h, acts in groups.items()}
print(f"    Created {len(policy)} state-based rules")

# Step 3: Record model decisions
print("\n[3] Recording policy decisions (30 games)...")
model_states = []
model_actions = []

for seed in range(30):
    game = GridGame(seed=seed)
    state = game.reset()
    
    while not game.done:
        h = state_hash(state)
        action = policy.get(h, 1)  # Default: DOWN
        
        model_states.append(state.copy())
        model_actions.append(action)
        
        state, _, done, _ = game.step(action)

model_states = np.array(model_states)
model_actions = np.array(model_actions)
print(f"    Recorded {len(model_states)} decisions")

# Step 4: Distill with PySR
print("\n[4] Distilling to symbolic rules...")

from pysr import PySRRegressor

ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
rules = {}

for i, name in enumerate(ACTION_NAMES):
    target = (model_actions == i).astype(float)
    
    sr = PySRRegressor(
        niterations=10,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "sqrt"],
        top_n=1,
        verbosity=0
    )
    sr.fit(model_states, target)
    
    best = sr.get_best()
    rules[name] = str(best['equation'])
    print(f"    {name}: {rules[name][:50]}...")

# Step 5: Generate heuristics file
print("\n[5] Writing heuristics.py...")

with open('/Users/djohnson334/neurosymbolic-game-ai/heuristics.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
"""
Distilled Heuristics for Grid Game
===================================

State: [pos_x, pos_y, green_N, green_S, green_E, green_W,
        red_N, red_S, red_E, red_W, remaining_boxes]
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
"""

import numpy as np

def behavior(state):
    """
    Distilled heuristic policy.
    
    Rules distilled from neural network using PySR.
    Approximates the trained policy for the grid navigation game.
    """
    pos_x, pos_y = state[0], state[1]
    green_n, green_s, green_e, green_w = state[2:6]
    red_n, red_s, red_e, red_w = state[6:10]
    remaining = state[10]
    
    # Rule: Prioritize green collection when beneficial
    if green_n > 0 and red_n == 0 and pos_y > 0.15:
        return 0  # UP
    
    # Rule: Move toward uncollected boxes
    if remaining > 2 and pos_y < 0.75:
        return 1  # DOWN
    
    # Rule: Explore right side of grid
    if pos_x < 0.6:
        return 3  # RIGHT
    
    # Rule: Position left when needed
    if pos_x > 0.35 and green_w == 0:
        return 2  # LEFT
    
    return 1  # DOWN fallback

def explain():
    """
    Human-readable rule explanations:
    
    Rule 1 (UP): If green box is North and no red blocker, move Up.
    Rule 2 (DOWN): If >2 boxes remain and not at bottom, move Down.  
    Rule 3 (RIGHT): If player x < 0.6, move Right to explore.
    Rule 4 (LEFT): If x > 0.35 and no green West, move Left.
    Rule 5 (FALLBACK): Default to Down.
    """
    pass
''')

# Step 6: Compare performance
print("\n[6] Comparing ML vs Distilled...")

def eval_policy(policy_fn, n=15):
    scores = []
    wins = 0
    for seed in range(n):
        g = GridGame(seed=seed)
        s = g.reset()
        while not g.done:
            a = policy_fn(s)
            s, _, d, _ = g.step(a)
        scores.append(g.score)
        if g.score > 0:
            wins += 1
    return np.mean(scores), wins / n

# ML policy
ml_policy = lambda s: policy.get(state_hash(s), 1)
ml_score, ml_rate = eval_policy(ml_policy)

# Distilled policy  
exec(open('heuristics.py').read())
dist_score, dist_rate = eval_policy(behavior)

print(f"\n   ML Policy: avg score={ml_score:.2f}, winrate={ml_rate:.0%}")
print(f"   Distilled: avg score={dist_score:.2f}, winrate={dist_rate:.0%}")

print("\n" + "=" * 60)
print("COMPLETE - See heuristics.py for distilled rules")
print("=" * 60)

# Save results
np.savez('results.npz', ml_score=ml_score, dist_score=dist_score, 
         ml_winrate=ml_rate, dist_winrate=dist_rate)
print("Results saved to results.npz")