#!/usr/bin/env python3
"""
Train a simple ML model for the Grid Game using pure Python/no external game module.
This avoids import issues and runs fast.
"""
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

random.seed(42)
np.random.seed(42)

print("=" * 50)
print("TRAINING ML MODEL FOR GRID GAME")
print("=" * 50)

# Generate training data from random play
print("\nGenerating training data (200 games)...")
states, actions = [], []

for game_seed in range(200):
    # Game state
    x, y = 16, 16  # start in center
    score = 0
    done = False
    collected_green = set()
    collected_red = set()
    
    # Place boxes
    green = set()
    red = set()
    random.seed(game_seed)
    for _ in range(5):
        green.add((random.randint(0,31), random.randint(0,31)))
    for _ in range(5):
        red.add((random.randint(0,31), random.randint(0,31)))
    
    # Play randomly
    steps = 0
    while not done and steps < 30:
        # State: position, nearby boxes, remaining
        gn = int((x, y-1) in green and (x, y-1) not in collected_green)
        gs = int((x, y+1) in green and (x, y+1) not in collected_green)
        ge = int((x+1, y) in green and (x+1, y) not in collected_green)
        gw = int((x-1, y) in green and (x-1, y) not in collected_green)
        rn = int((x, y-1) in red and (x, y-1) not in collected_red)
        rs = int((x, y+1) in red and (x, y+1) not in collected_red)
        re = int((x+1, y) in red and (x+1, y) not in collected_red)
        rw = int((x-1, y) in red and (x-1, y) not in collected_red)
        remaining = len(green) - len(collected_green)
        
        state = np.array([x/32, y/32, gn, gs, ge, gw, rn, rs, re, rw, remaining], dtype=np.float32)
        action = random.randint(0, 3)
        
        states.append(state)
        actions.append(action)
        
        # Execute action
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        x = max(0, min(31, x + dx))
        y = max(0, min(31, y + dy))
        
        if (x, y) in green and (x, y) not in collected_green:
            score += 1
            collected_green.add((x, y))
        elif (x, y) in red and (x, y) not in collected_red:
            score -= 1
            collected_red.add((x, y))
        
        if len(collected_green) >= len(green):
            done = True
        steps += 1

states = np.array(states)
actions = np.array(actions)
print(f"Data: {states.shape}")

# Train simple logistic regression
print("\nTraining logistic regression...")
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(states, actions)
print(f"Training accuracy: {model.score(states, actions):.3f}")

# Evaluate
print("\nEvaluating on new games (10 games)...")
scores_ml = []
for i in range(10):
    # Game setup
    random.seed(i * 1000)
    x, y = 16, 16
    score = 0
    done = False
    collected_green = set()
    collected_red = set()
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    
    while not done:
        # Get state
        gn = int((x, y-1) in green and (x, y-1) not in collected_green)
        gs = int((x, y+1) in green and (x, y+1) not in collected_green)
        ge = int((x+1, y) in green and (x+1, y) not in collected_green)
        gw = int((x-1, y) in green and (x-1, y) not in collected_green)
        rn = int((x, y-1) in red and (x, y-1) not in collected_red)
        rs = int((x, y+1) in red and (x, y+1) not in collected_red)
        re = int((x+1, y) in red and (x+1, y) not in collected_red)
        rw = int((x-1, y) in red and (x-1, y) not in collected_red)
        remaining = len(green) - len(collected_green)
        
        state = np.array([x/32, y/32, gn, gs, ge, gw, rn, rs, re, rw, remaining], dtype=np.float32)
        action = model.predict(state.reshape(1,-1))[0]
        
        # Execute
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        x = max(0, min(31, x + dx))
        y = max(0, min(31, y + dy))
        
        if (x, y) in green and (x, y) not in collected_green:
            score += 1
            collected_green.add((x, y))
        elif (x, y) in red and (x, y) not in collected_red:
            score -= 1
            collected_red.add((x, y))
        
        if len(collected_green) >= len(green):
            done = True
    
    scores_ml.append(score)

# Random baseline
print("\nRandom policy evaluation...")
scores_random = []
for i in range(10):
    random.seed(i * 1000)
    x, y = 16, 16
    score = 0
    done = False
    collected_green = set()
    collected_red = set()
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    
    while not done:
        action = random.randint(0, 3)
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        x = max(0, min(31, x + dx))
        y = max(0, min(31, y + dy))
        
        if (x, y) in green and (x, y) not in collected_green:
            score += 1
            collected_green.add((x, y))
        elif (x, y) in red and (x, y) not in collected_red:
            score -= 1
            collected_red.add((x, y))
        
        if len(collected_green) >= len(green):
            done = True

    scores_random.append(score)

print(f"\n{'Policy':<12} {'Avg Score':>12} {'Scores'}")
print("-" * 50)
print(f"{'Random':<12} {np.mean(scores_random):>12.2f} {scores_random}")
print(f"{'ML':<12} {np.mean(scores_ml):>12.2f} {scores_ml}")

# Save model
joblib.dump(model, 'game_model.joblib')
print(f"\nModel saved: game_model.joblib")
print(f"ML beats random: {np.mean(scores_ml) > np.mean(scores_random)}")

print("\nDone!")