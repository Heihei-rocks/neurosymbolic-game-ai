#!/usr/bin/env python3
"""Fast evaluation script."""
import numpy as np
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')
from game import GridGame

# Load heuristics
exec(open('heuristics.py').read())

def quick_eval(fn, n=5):
    scores = []
    for seed in range(n):
        g = GridGame(seed=seed)
        s = g.reset()
        steps = 0
        while not g.done and steps < 30:
            a = fn(s)
            s, _, d, _ = g.step(a)
            steps += 1
        scores.append(g.score)
    return np.mean(scores)

rand = lambda s: np.random.randint(0,4)
rand_score = quick_eval(rand, 5)
heur_score = quick_eval(behavior, 5)

print(f"Random avg: {rand_score:.2f}")
print(f"Heur avg: {heur_score:.2f}")
np.savez('results.npz', rand_score=rand_score, heur_score=heur_score)
print("Saved results.npz")