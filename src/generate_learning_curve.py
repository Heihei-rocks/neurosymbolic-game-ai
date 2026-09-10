#!/usr/bin/env python3
"""Generate learning curve with actual game scores."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai/src')
from game import GridGame, generate_training_data
from sklearn.neural_network import MLPClassifier

print("Generating training data for learning curve...")
states, actions = generate_training_data(n_eps=200, red_disabled=True, green_count=50)

# Simulate training with partial fits
print("Training with incremental evaluation...")
n_iters = 300
scores_per_iter = []
batch_size = max(1, len(states) // 10)

model = MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam', max_iter=1, random_state=42)
model.fit(states[:batch_size], actions[:batch_size])

# Evaluate every 10 iterations
for i in range(1, n_iters+1):
    if i % 10 == 0:
        # Evaluate on 20 games
        scores = []
        for seed in range(20):
            g = GridGame(seed=seed+1000, green_count=50, red_disabled=True)
            g.reset()
            while not g.done and g.turn < 60:
                state = g.get_state()
                try:
                    a = int(model.predict([state])[0])
                except:
                    a = 0
                g.step(a)
            scores.append(g.score)
        mean_score = np.mean(scores)
        scores_per_iter.append((i, mean_score))
        print(f"Iter {i}: Mean score {mean_score:.1f}")
        # Continue training
        if i < n_iters:
            model.fit(states[:batch_size*i], actions[:batch_size*i])

# Plot
fig, ax = plt.subplots(figsize=(10,6))
iters, scores = zip(*scores_per_iter)
ax.plot(iters, scores, 'b-', linewidth=2, label='Mean Game Score')
ax.scatter(iters, scores, c='red', alpha=0.5, s=30, label='Score per iteration')
ax.set_xlabel('Training Iterations', fontsize=12)
ax.set_ylabel('Mean Game Score (20 games)', fontsize=12)
ax.set_title('Neural Network Training: Game Score vs Training Iterations', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('/Users/djohnson334/neurosymbolic-game-ai/output/learning_curve_v13.04.png', dpi=150)
print("Saved learning curve")
