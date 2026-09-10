#!/usr/bin/env python3
"""Quick ML training with minimal data."""
import sys
import os
os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

# Add path
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')

# Minimal game implementation inline
import random
import numpy as np

class GridGame:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.w, self.h = 32, 32
        self.x, self.y = 16, 16
        self.score = 0
        self.collected_green = set()
        self.collected_red = set()
        self.done = False
        self.green = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(5)}
        self.red = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(5)}

    def get_state(self):
        return np.array([self.x/self.w, self.y/self.h], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        self.x = max(0, min(self.w-1, self.x + dx))
        self.y = max(0, min(self.h-1, self.y + dy))
        if (self.x, self.y) in self.green and (self.x, self.y) not in self.collected_green:
            self.score += 1
            self.collected_green.add((self.x, self.y))
        elif (self.x, self.y) in self.red and (self.x, self.y) not in self.collected_red:
            self.score -= 1
            self.collected_red.add((self.x, self.y))
        if len(self.collected_green) >= len(self.green):
            self.done = True
        return self.get_state(), self.score, self.done, {}

    def reset(self):
        self.__init__(seed=42)
        return self.get_state()

print("Generating 500 training samples...")
states, actions = [], []
for seed in range(500):
    g = GridGame(seed=seed)
    s = g.reset()
    while not g.done:
        a = random.randint(0, 3)
        ns, r, d, _ = g.step(a)
        states.append(s)
        actions.append(a)
        s = ns

states = np.array(states)
actions = np.array(actions)
print(f"Data: {states.shape}")

print("Training tiny model...")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=100, random_state=42)
model.fit(states, actions)
print(f"Accuracy: {model.score(states, actions):.3f}")

# Save model
import joblib
joblib.dump(model, 'game_model.joblib')
print("Saved game_model.joblib")

# Test
print("\nTesting model...")
scores = []
for i in range(5):
    g = GridGame(seed=i*1000)
    s = g.reset()
    while not g.done:
        a = model.predict(s.reshape(1,-1))[0]
        s, r, d, _ = g.step(a)
    scores.append(g.score)

print(f"ML scores: {scores}")
print(f"Average: {np.mean(scores):.2f}")

print("\nDone!")