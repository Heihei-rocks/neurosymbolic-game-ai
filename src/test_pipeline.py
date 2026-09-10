#!/usr/bin/env python3
"""Minimal training - just test the pipeline works."""
import numpy as np
import os
os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

# Import game module directly
import importlib.util
spec = importlib.util.spec_from_file_location("game", "/Users/djohnson334/neurosymbolic-game-ai/game.py")
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)

GridGame = game.GridGame
generate_training_data = game.generate_training_data

print("Generating 100 training samples...")
states, actions, rewards = generate_training_data(100)
print(f"Generated: {states.shape}")

# Train simple model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(states, actions)
print(f"Accuracy: {model.score(states, actions):.3f}")

# Save
import joblib
joblib.dump(model, 'game_model.joblib')
np.savez('training_data.npz', states=states, actions=actions)
print("Model saved!")

# Quick test
print("\nTesting model on 5 games...")
scores = []
for i in range(5):
    g = GridGame(seed=i*100)
    s = g.reset()
    while not g.done:
        a = model.predict(s.reshape(1,-1))[0]
        s, r, d, _ = g.step(a)
    scores.append(g.score)
    
print(f"ML model scores: {scores}")
print(f"Average: {np.mean(scores):.2f}")

# Compare with random
print("\nComparing with random policy (5 games)...")
random_scores = []
for i in range(5):
    g = GridGame(seed=i*100)
    s = g.reset()
    while not g.done:
        a = np.random.randint(0,4)
        s, r, d, _ = g.step(a)
    random_scores.append(g.score)
    
print(f"Random scores: {random_scores}")
print(f"Average: {np.mean(random_scores):.2f}")

print(f"\n✅ ML better than random: {np.mean(scores) > np.mean(random_scores)}")