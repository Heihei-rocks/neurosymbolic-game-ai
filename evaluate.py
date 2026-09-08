#!/usr/bin/env python3
"""Final evaluation script - compares ML, Heuristics, and Random policies."""
import numpy as np
import random
import joblib

# Load ML model
try:
    model = joblib.load('game_model.joblib')
    print("Loaded trained ML model")
except:
    print("No model found, using random")
    model = None

def simple_behavior(state):
    """Simple heuristic: always move toward center."""
    px, py = state[0], state[1]
    if px < 0.5: return 3  # RIGHT
    if px > 0.5: return 2  # LEFT
    if py < 0.5: return 1  # DOWN
    return 0  # UP

def run_game(seed, policy_fn):
    """Run one game with given policy."""
    random.seed(seed)
    np.random.seed(seed)
    
    x, y = 16, 16
    score = 0
    collected_green = set()
    collected_red = set()
    
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    
    done = False
    for _ in range(200):  # Max steps
        if done: break
        
        state = np.array([x/32, y/32], dtype=np.float32)
        if policy_fn == 'random':
            action = random.randint(0, 3)
        elif policy_fn == 'heuristic':
            action = simple_behavior(state)
        else:
            action = model.predict(state.reshape(1, -1))[0] if model else random.randint(0, 3)
        
        dx, dy = {0:(0,-1), 1:(0,1), 2:(-1,0), 3:(1,0)}[action]
        x = max(0, min(31, x + dx))
        y = max(0, min(31, y + dy))
        
        if (x, y) in green and (x, y) not in collected_green:
            score += 1
            collected_green.add((x, y))
        elif (x, y) in red and (x, y) not in collected_red:
            score -= 1
            collected_red.add((x, y))
        
        if len(collected_green) >= 5:
            done = True
    
    return score

print("\n" + "="*50)
print("EVALUATING POLICIES (10 games each)")
print("="*50)

# Test all policies
random_scores = [run_game(i, 'random') for i in range(10)]
heuristic_scores = [run_game(i, 'heuristic') for i in range(10)]
ml_scores = [run_game(i, 'ml') for i in range(10)]

print(f"\nPOLICY       AVG SCORE")
print("-"*25)
print(f"Random       {np.mean(random_scores):8.2f}")
print(f"Heuristics   {np.mean(heuristic_scores):8.2f}")
print(f"ML Model     {np.mean(ml_scores):8.2f}")

# Save results
np.savez('results.npz',
    random_score=np.mean(random_scores),
    heuristic_score=np.mean(heuristic_scores),
    ml_score=np.mean(ml_scores))

print(f"\nSaved results.npz")