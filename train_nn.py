#!/usr/bin/env python3
"""Train NN with improved state."""
import numpy as np, random, joblib
from sklearn.neural_network import MLPClassifier
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')
from game import GridGame, generate_training_data

print("Generating training data...")
states, actions = generate_training_data(n_eps=200, red_disabled=True, green_count=50)
print(f"Samples: {len(states)}, State dim: {states.shape[1]}")

print("Training NN (32+32+32)...")
model = MLPClassifier(hidden_layer_sizes=(32,32,32), activation='relu', solver='adam', max_iter=150, random_state=42)
model.fit(states, actions)
print(f"Loss: {model.loss_:.4f}")

print("\nEvaluating 100 games per policy...")
import time
def eval_policy(pol, seed_offset):
    scores = []
    for s in range(100):
        g = GridGame(seed=s+seed_offset, green_count=50, red_disabled=True)
        g.reset()
        while not g.done:
            if pol == 'random': a = random.randint(0,3)
            elif pol == 'heuristic':
                greens = [(gx, gy) for gx, gy in g.green if (gx, gy) not in g.collected_green]
                a = random.randint(0,3)
                if greens:
                    gx, gy = min(greens, key=lambda b: (b[0]-g.x)**2 + (b[1]-g.y)**2)
                    a = 2 if gx < g.x else (3 if gx > g.x else (0 if gy < g.y else 1))
            else: a = model.predict([g.get_state()])[0]
            g.step(a)
        scores.append(g.score)
    return np.array(scores)

for pol in ['random', 'heuristic', 'nn']:
    scores = eval_policy(pol, 1000)
    print(f"{pol:10} | Mean: {np.mean(scores):7.2f} | Std: {np.std(scores):6.2f}")

joblib.dump(model, 'game_model.joblib')
print("\nModel saved.")