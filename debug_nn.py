#!/usr/bin/env python3
"""Debug NN behavior."""
import numpy as np, random, joblib, sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')
from game import GridGame

game = joblib.load('game_model.joblib')
print("Model output classes:", game.n_classes_)
print("Model hidden layers:", game.hidden_layer_sizes)

# Test a specific state
g = GridGame(seed=42, green_count=50)
s = g.reset()
print(f"\nState: {s}")
print(f"Green boxes: {len(g.green)}")

# Predict action
action = game.predict([s])[0]
print(f"Predicted action: {action}")

# Simulate step
ns, r, d, info = g.step(action)
print(f"New pos: ({g.x}, {g.y})")
print(f"Reward: {r}, Done: {d}")

# Show prediction probabilities
probs = game.predict_proba([s])[0]
print(f"Action probs: {probs}")