#!/usr/bin/env python3
"""Generate animated GIFs for Random, NN, and Heuristic policies."""
import numpy as np
import random
import imageio
import os

os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

# Load NN model
import joblib
model = joblib.load('game_model.joblib')

# Load heuristics
import importlib.util
spec = importlib.util.spec_from_file_location("heuristics", "/Users/djohnson334/neurosymbolic-game-ai/heuristics.py")
heur_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heur_mod)

def run_game_visual(seed, policy_type, max_frames=100):
    """Run game and return frames for visualization."""
    random.seed(seed)
    np.random.seed(seed)
    
    x, y = 16, 16
    frames = []
    cg, cr = set(), set()
    collected_order_green = []
    collected_order_red = []
    
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    
    # Ensure no overlap
    green -= red
    red -= green
    while len(green) < 5: green.add((random.randint(0,31), random.randint(0,31)))
    while len(red) < 5: red.add((random.randint(0,31), random.randint(0,31)))
    
    def make_frame():
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Green boxes (collected ones fade)
        for bx, by in green - cg:
            frame[by*2+10:(by+1)*2+10, bx*2+10:(bx+1)*2+10] = [0, 200, 0]
        for bx, by in cg:
            frame[by*2+10:(by+1)*2+10, bx*2+10:(bx+1)*2+10] = [100, 150, 100]
        # Red boxes
        for bx, by in red - cr:
            frame[by*2+10:(by+1)*2+10, bx*2+10:(bx+1)*2+10] = [200, 0, 0]
        # Collected red
        for bx, by in cr:
            frame[by*2+10:(by+1)*2+10, bx*2+10:(bx+1)*2+10] = [150, 100, 100]
        # Agent
        frame[y*2+12:(y+1)*2+12, x*2+12:(x+1)*2+12] = [0, 0, 255]
        return frame
    
    frames.append(make_frame())
    done = False
    
    while not done and len(frames) < max_frames:
        # Build state
        gn = int((x, y-1) in green and (x, y-1) not in cg)
        gs = int((x, y+1) in green and (x, y+1) not in cg)
        ge = int((x+1, y) in green and (x+1, y) not in cg)
        gw = int((x-1, y) in green and (x-1, y) not in cg)
        rn = int((x, y-1) in red and (x, y-1) not in cr)
        rs = int((x, y+1) in red and (x, y+1) not in cr)
        re = int((x+1, y) in red and (x+1, y) not in cr)
        rw = int((x-1, y) in red and (x-1, y) not in cr)
        remaining = len(green) - len(cg)
        
        state = np.array([x/32, y/32, gn, gs, ge, gw, rn, rs, re, rw, remaining], dtype=np.float32)
        
        if policy_type == 'random':
            action = random.randint(0, 3)
        elif policy_type == 'heuristic':
            action = heur_mod.behavior(state)
        else:  # nn
            action = model.predict(state.reshape(1, -1))[0]
        
        dx, dy = {0:(0,-1), 1:(0,1), 2:(-1,0), 3:(1,0)}[action]
        x = max(0, min(31, x + dx))
        y = max(0, min(31, y + dy))
        
        if (x, y) in green and (x, y) not in cg:
            cg.add((x, y))
            collected_order_green.append((x, y))
        elif (x, y) in red and (x, y) not in cr:
            cr.add((x, y))
            collected_order_red.append((x, y))
        
        if len(cg) >= len(green):
            done = True
        
        frames.append(make_frame())
    
    return frames

print("Generating animations...")

# Random
print("Random policy...")
rf = run_game_visual(42, 'random')
imageio.mimsave('random.gif', rf, fps=4)
print(f"  {len(rf)} frames")

# Neural Network
print("NN policy...")
nf = run_game_visual(43, 'nn')
imageio.mimsave('nn_policy.gif', nf, fps=4)
print(f"  {len(nf)} frames")

# Heuristic
print("Heuristic policy...")
hf = run_game_visual(44, 'heuristic')
imageio.mimsave('heuristic.gif', hf, fps=4)
print(f"  {len(hf)} frames")

print("\nDone! Created: random.gif, nn_policy.gif, heuristic.gif")