#!/usr/bin/env python3
"""Create animated GIFs for all three policies."""
import numpy as np, random, os, imageio
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')

import joblib
model = joblib.load('game_model.joblib')
from game import GridGame

def make_gif(filename, policy, seed):
    random.seed(seed); np.random.seed(seed)
    g = GridGame(seed=seed, green_count=50, red_disabled=True)
    g.reset()
    
    frames = []
    
    for step in range(40):
        if g.done: break
        
        img = Image.new('RGB', (400, 400), 'black')
        draw = ImageDraw.Draw(img)
        
        # Draw grid with boxes
        for gx, gy in g.green:
            if (gx, gy) not in g.collected_green:
                cx, cy = gx*10 + 20, gy*10 + 20
                draw.ellipse([cx, cy, cx+8, cy+8], fill='green')
        
        # Agent
        ax, ay = g.x*10+20, g.y*10+20
        draw.ellipse([ax, ay, ax+10, ay+10], fill='blue')
        
        frames.append(np.array(img))
        
        # Get action
        if policy == 'random':
            a = random.randint(0,3)
        elif policy == 'heuristic':
            greens = [(gx, gy) for gx, gy in g.green if (gx, gy) not in g.collected_green]
            a = random.randint(0,3)
            if greens:
                gx, gy = min(greens, key=lambda b: (b[0]-g.x)**2 + (b[1]-g.y)**2)
                a = 2 if gx < g.x else (3 if gx > g.x else (0 if gy < g.y else 1))
        else:
            a = model.predict([g.get_state()])[0]
        
        g.step(a)
    
    imageio.mimsave(filename, frames, fps=8)
    return g.score

os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

print("Creating GIFs...")
print("Random:", make_gif('random.gif', 'random', 42))
print("NN:", make_gif('nn_policy.gif', 'nn', 43))
print("Heuristic:", make_gif('heuristic.gif', 'heuristic', 44))
print("Done!")