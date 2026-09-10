#!/usr/bin/env python3
"""Generate animated GIFs for all three policies with time pressure."""
import numpy as np
import random
import imageio
import os

os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

# Load NN model
import joblib
model = joblib.load('game_model.joblib')

# Simple heuristic for comparison
def heuristic_policy(state):
    board = state[:1024]
    px, py = int(state[1024] * 32), int(state[1025] * 32)
    
    # Find nearest green and red boxes
    greens = [(i % 32, i // 32) for i, v in enumerate(board) if v > 0]
    reds = [(i % 32, i // 32) for i, v in enumerate(board) if v < 0]
    
    if greens and (not reds or random.random() > 0.3):
        gx, gy = min(greens, key=lambda b: (b[0]-px)**2 + (b[1]-py)**2)
        if gx < px: return 2  # LEFT
        if gx > px: return 3  # RIGHT
        if gy < py: return 0  # UP
        return 1  # DOWN
    
    if reds:
        return random.choice([2, 3])  # Move horizontally to avoid
    
    return random.choice([0, 1, 2, 3])

def create_gif(filename, policy, seed=42, frames=64):
    """Create animation of agent gameplay."""
    random.seed(seed)
    np.random.seed(seed)
    
    x, y = 16, 16
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    cg, cr = set(), set()
    reward = 100
    score = 0
    
    frames_list = []
    
    for step in range(frames):
        # Create frame (32x32 grid)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        # Empty = black
        # Green = green
        for gx, gy in green:
            if (gx, gy) not in cg:
                frame[gy, gx] = [0, 255, 0]
        # Red = red
        for rx, ry in red:
            if (rx, ry) not in cr:
                frame[ry, rx] = [255, 0, 0]
        # Agent = blue
        frame[y, x] = [0, 0, 255]
        
        frames_list.append(frame)
        
        if reward <= 0:
            break
            
        # Build state
        board = np.zeros(1024)
        for gx, gy in green:
            if (gx, gy) not in cg: board[gy*32+gx] = 1
        for rx, ry in red:
            if (rx, ry) not in cr: board[ry*32+rx] = -1
        state = np.concatenate([board, [x/32, y/32]])
        
        # Select action by policy
        if policy == 'random':
            a = random.randint(0, 3)
        elif policy == 'nn':
            a = int(model.predict([state])[0])
        else:  # heuristic
            a = heuristic_policy(state)
        
        dx, dy = {0:(0,-1), 1:(0,1), 2:(-1,0), 3:(1,0)}[a]
        old_x, old_y = x, y
        x = max(0, min(31, x+dx))
        y = max(0, min(31, y+dy))
        
        if (x, y) in green and (x, y) not in cg:
            score += reward
            cg.add((x, y))
        elif (x, y) in red and (x, y) not in cr:
            score -= reward
            cr.add((x, y))
        
        reward -= 1
    
    # Save GIF
    imageio.mimsave(filename, frames_list, fps=10)
    return score

print("Creating GIFs...")
print("Random policy...")
s1 = create_gif('random.gif', 'random')
print("NN policy...")
s2 = create_gif('nn_policy.gif', 'nn')
print("Heuristic policy...")
s3 = create_gif('heuristic.gif', 'heuristic')

print(f"\nScores: Random={s1:.1f}, NN={s2:.1f}, Heuristic={s3:.1f}")
print("GIFs created!")