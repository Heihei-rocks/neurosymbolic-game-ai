#!/usr/bin/env python3
"""Quick evaluation of heuristic vs random policy."""
import numpy as np
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')
from game import GridGame

exec(open('heuristics.py').read())

def evaluate(fn, n=20):
    scores, wins = [], 0
    for seed in range(n):
        g = GridGame(seed=seed)
        s = g.reset()
        while not g.done:
            a = fn(s)
            s, _, d, _ = g.step(a)
        scores.append(g.score)
        if g.score > 0:
            wins += 1
    return np.mean(scores), wins/n

# Test
rand = lambda s: np.random.randint(0,4)
rand_s, rand_w = evaluate(rand)

exec(open('heuristics.py').read())
h_s, h_w = evaluate(behavior)

print(f"\n{'Policy':<12} {'Avg Score':>12} {'Win Rate':>10}")
print("-" * 40)
print(f"{'Random':<12} {rand_s:>12.2f} {rand_w:>10.0%}")
print(f"{'Heuristics':<12} {h_s:>12.2f} {h_w:>10.0%}")

# Save
np.savez('results.npz', rand_score=rand_s, heur_score=h_s, 
         rand_winrate=rand_w, heur_winrate=h_w)
print("\nResults saved to results.npz")