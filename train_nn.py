#!/usr/bin/env python3
"""Train a small neural network for Grid Game with 11-state input."""
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
import joblib

print("Training neural network (100 neurons total)...")

# Generate training data
states, actions = [], []
for seed in range(500):
    random.seed(seed)
    x, y = 16, 16
    cg, cr = set(), set()
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    
    for _ in range(30):  # steps per game
        # Build 11-state
        gn = int((x, y-1) in green and (x, y-1) not in cg)
        gs = int((x, y+1) in green and (x, y+1) not in cg)
        ge = int((x+1, y) in green and (x+1, y) not in cg)
        gw = int((x-1, y) in green and (x-1, y) not in cg)
        rn = int((x, y-1) in red and (x, y-1) not in cr)
        rs = int((x, y+1) in red and (x, y+1) not in cr)
        re = int((x+1, y) in red and (x+1, y) not in cr)
        rw = int((x-1, y) in red and (x-1, y) not in cr)
        remaining = len(green) - len(cg)
        
        state = [x/32, y/32, gn, gs, ge, gw, rn, rs, re, rw, remaining]
        action = random.randint(0, 3)
        
        states.append(state)
        actions.append(action)
        
        # Execute action
        dx, dy = {0:(0,-1), 1:(0,1), 2:(-1,0), 3:(1,0)}[action]
        x = max(0, min(31, x+dx))
        y = max(0, min(31, y+dy))
        
        if (x, y) in green and (x, y) not in cg: cg.add((x, y))
        elif (x, y) in red and (x, y) not in cr: cr.add((x, y))

states = np.array(states)
actions = np.array(actions)

print(f"Training data: {states.shape}")

# Small NN: ~100 neurons total (hidden layers: 32 + 32 + 32 = 96)
model = MLPClassifier(
    hidden_layer_sizes=(32, 32, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    verbose=False
)

print("Fitting model...")
model.fit(states, actions)

print(f"Training accuracy: {model.score(states, actions):.3f}")

# Save model
joblib.dump(model, '/Users/djohnson334/neurosymbolic-game-ai/game_model.joblib')
print("Saved game_model.joblib")

# Quick test
print("\nTesting...")
test_state = [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 5]
pred = model.predict([test_state])[0]
print(f"Test prediction: action {pred}")

print("Done!")