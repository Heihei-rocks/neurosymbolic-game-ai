#!/usr/bin/env python3
"""Fast training script for Grid Game model."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')

from game import GridGame, generate_training_data

print("Generating training data (1000 games)...")
states, actions, rewards = generate_training_data(1000)
print(f"Generated: {states.shape[0]} samples")

# Use faster RandomForest instead of MLP
print("Training RandomForest classifier...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=1
)
model.fit(states, actions)
print(f"Training accuracy: {model.score(states, actions):.3f}")

# Evaluate
print("\nEvaluating on new games...")
wins, total_score = 0, 0
for i in range(10):
    game = GridGame(seed=i*1000)
    state = game.reset()
    while not game.done:
        action = model.predict(state.reshape(1, -1))[0]
        state, reward, done, _ = game.step(action)
        total_score += reward
    if total_score > 0:
        wins += 1

print(f"  Wins: {wins}/10 ({100*wins/10:.0f}%)")
print(f"  Avg score: {total_score/10:.2f}")

# Save
import joblib
joblib.dump(model, 'game_model.joblib')
np.savez('training_data.npz', states=states, actions=actions)
print("\nSaved: game_model.joblib, training_data.npz")