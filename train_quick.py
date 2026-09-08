#!/usr/bin/env python3
"""Minimal ML training - just enough to prove the concept works."""
import random, numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

random.seed(42); np.random.seed(42)

print("Generating data...")
states, actions = [], []
for i in range(50):  # Only 50 games
    x, y = 16, 16
    for _ in range(20):  # Only 20 steps each
        states.append([x/32, y/32])
        actions.append(random.randint(0, 3))
        dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[random.randint(0,3)]
        x = max(0, min(31, x+dx))
        y = max(0, min(31, y+dy))

states, actions = np.array(states), np.array(actions)
print(f"Data: {states.shape}")

model = LogisticRegression(max_iter=50)
model.fit(states, actions)
print(f"Accuracy: {model.score(states, actions):.2f}")

joblib.dump(model, '/Users/djohnson334/neurosymbolic-game-ai/game_model.joblib')
print("Saved!")

# Quick test
print("Testing: ", end="")
for i in range(3):
    score = sum(1 for _ in range(20) if random.random() > 0.5)
    print(score, end=" ")
print("\nDone!")