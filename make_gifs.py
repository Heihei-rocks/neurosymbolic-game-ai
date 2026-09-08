#!/usr/bin/env python3
"""Generate simple GIFs using PIL."""
import numpy as np, random
from PIL import Image, ImageDraw
import imageio
import os

os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

# Load model
import joblib
try:
    model = joblib.load('game_model.joblib')
except:
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(16,16))

def hex_to_rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

def make_simple_gif(filename, policy, seed):
    random.seed(seed); np.random.seed(seed)
    x, y, cg, cr = 16, 16, set(), set()
    green = [(rx, ry) for rx, ry in [(random.randint(0,31), random.randint(0,31)) for _ in range(5)]]
    red = [(rx, ry) for rx, ry in [(random.randint(0,31), random.randint(0,31)) for _ in range(5)]]
    
    # Ensure no overlap
    overlap = set(green) & set(red)
    for ox, oy in overlap:
        green.remove((ox, oy)) if (ox, oy) in green else red.remove((ox, oy))
    while len(green) < 5: green.append((random.randint(0,31), random.randint(0,31)))
    while len(red) < 5: red.append((random.randint(0,31), random.randint(0,31)))
    
    frames = []
    reward = 100
    
    for step in range(48):
        if reward <= 0: break
        
        # Create frame: 64x64 pixels per cell, 32x32 cells = 2048x2048
        img = Image.new('RGB', (640, 640), 'black')
        draw = ImageDraw.Draw(img)
        
        # Draw boxes
        for gx, gy in green:
            if (gx, gy) not in cg:
                draw.rectangle([gx*20+2, gy*20+2, gx*20+18, gy*20+18], fill='green')
        for rx, ry in red:
            if (rx, ry) not in cr:
                draw.rectangle([rx*20+2, ry*20+2, rx*20+18, ry*20+18], fill='red')
        
        # Draw agent
        draw.rectangle([x*20+5, y*20+5, x*20+15, y*20+15], fill='blue')
        
        frames.append(np.array(img))
        
        # Get action
        gn = int((x, y-1) in green and (x, y-1) not in cg)
        gs = int((x, y+1) in green and (x, y+1) not in cg)
        ge = int((x+1, y) in green and (x+1, y) not in cg)
        gw = int((x-1, y) in green and (x-1, y) not in cg)
        rn = int((x, y-1) in red and (x, y-1) not in cr)
        rs = int((x, y+1) in red and (x, y+1) not in cr)
        re = int((x+1, y) in red and (x+1, y) not in cr)
        rw = int((x-1, y) in red and (x-1, y) not in cr)
        state = [x/32, y/32, gn, gs, ge, gw, rn, rs, re, rw, 5-len(cg)]
        
        if policy == 'random': a = random.randint(0,3)
        elif policy == 'heuristic':
            # Simple heuristic: move toward nearest green
            greens = [(gx, gy) for gx, gy in green if (gx, gy) not in cg]
            reds = [(rx, ry) for rx, ry in red if (rx, ry) not in cr]
            if greens:
                gx, gy = min(greens, key=lambda b: (b[0]-x)**2 + (b[1]-y)**2)
                if gx < x: a = 2  # LEFT
                elif gx > x: a = 3  # RIGHT
                elif gy < y: a = 0  # UP
                else: a = 1  # DOWN
            else:
                a = random.randint(0,3)
        else:
            a = model.predict([state])[0]
        
        dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[a]
        old_x, old_y = x, y
        x = max(0, min(31, x+dx))
        y = max(0, min(31, y+dy))
        
        if (x, y) in green and (x, y) not in cg:
            cg.add((x, y))
        elif (x, y) in red and (x, y) not in cr:
            cr.add((x, y))
        
        reward -= 1
    
    imageio.mimsave(filename, frames, fps=10)
    return len(cg) - len(cr)

print("Creating GIFs with PIL...")
print("Random:", make_simple_gif('random.gif', 'random', 42))
print("NN:", make_simple_gif('nn_policy.gif', 'nn', 43))
print("Heuristic:", make_simple_gif('heuristic.gif', 'heuristic', 44))
print("Done!")