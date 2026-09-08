#!/usr/bin/env python3
"""Final evaluation with 100 games, save results."""
import numpy as np, random, joblib, sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')

model = joblib.load('game_model.joblib')
from game import GridGame

def eval_policy(pol, n_games=100):
    scores = []
    for seed in range(n_games):
        g = GridGame(seed=seed, green_count=50, red_disabled=True)
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

print("Evaluating 100 games per policy...")
print("Settings: 50 green boxes, red disabled, time pressure")

r_scores, h_scores, nn_scores = [], [], []
for pol, name in [('random','Random'),('heuristic','Heuristic'),('nn','NN')]:
    scores = eval_policy(pol)
    if pol == 'random': r_scores = scores
    elif pol == 'heuristic': h_scores = scores
    else: nn_scores = scores
    print(f"{name:12} | μ={np.mean(scores):.1f} | σ={np.std(scores):.2f} | range=[{np.min(scores)}, {np.max(scores)}]")

np.savez('results.npz', random=r_scores, heuristic=h_scores, nn=nn_scores)
print("\nSaved results.npz")