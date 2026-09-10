#!/usr/bin/env python3
"""Create proper scored GIFs with high-res text rendering."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio
from io import BytesIO
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai/src')
from game import GridGame
import random

def create_frame(state, pos, score, popups, grid_size=32, img_px=300):
    """Create high-resolution frame with proper text."""
    # Use larger figure for readability
    fig, ax = plt.subplots(figsize=(4,4), dpi=img_px//100)
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw grid cells as blocks
    for i in range(grid_size):
        for j in range(grid_size):
            rect = Rectangle((i-0.5, j-0.5), 1, 1, facecolor='white', edgecolor='#e0e0e0', linewidth=0.5)
            ax.add_patch(rect)
    
    # Draw green boxes (collected ones in lighter shade)
    collected = state['collected_green']
    all_green = state['green']
    for (gx, gy) in all_green:
        if (gx, gy) in collected:
            color = '#7fbf7f'
        else:
            color = '#90ee90'
        rect = Rectangle((gx-0.5, gy-0.5), 1, 1, facecolor=color, edgecolor='green', linewidth=1.5)
        ax.add_patch(rect)
    
    # Draw player
    ax.plot(pos[0], pos[1], 'bo', markersize=12, markeredgecolor='black', markeredgewidth=2)
    
    # Draw score popups with readable text
    for (x, y, points, age) in popups:
        if age < 20:
            # Fade from yellow to black
            alpha = 1 - age/20
            if alpha > 0:
                text = f"+{points}"
                # Shadow for readability
                ax.text(x, y+1.5, text, fontsize=16, fontweight='bold', ha='center', va='center',
                       color='black', alpha=0.3)
                ax.text(x, y+1.5, text, fontsize=16, fontweight='bold', ha='center', va='center',
                       color='yellow', alpha=alpha)
    
    # Score tally with readable font
    ax.text(1, grid_size-0.8, f"Score: {score}", fontsize=18, fontweight='bold',
           color='black', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=3))
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)

def run_policy_game(seed, policy_name):
    """Run game with specified policy and create GIF."""
    game = GridGame(seed=seed, green_count=50, red_disabled=True)
    game.reset()
    
    frames = []
    popups = []
    collected_this_step = set()
    
    for step in range(120):
        # Check for new collections
        new_collections = []
        for (gx, gy) in game.green:
            if (gx, gy) not in game.collected_green:
                if game.x == gx and game.y == gy:
                    new_collections.append((gx, gy, game.reward_counter+75))
        
        for gx, gy, points in new_collections:
            popups.append([gx, gy, points, 0])
            game.score += points
            game.collected_green.add((gx, gy))
        
        # Update popups
        popups = [[x, y, pts, age+1] for x, y, pts, age in popups if age < 20]
        
        # Create frame
        state = {
            'green': list(game.green),
            'collected_green': game.collected_green.copy()
        }
        frame = create_frame(state, (game.x, game.y), game.score, popups)
        frames.append(frame)
        
        # Choose action
        if policy_name == 'random':
            action = random.randint(0, 3)
        elif policy_name == 'heuristic':
            # Greedy toward nearest green
            greens = [(gx, gy) for gx, gy in game.green if (gx, gy) not in game.collected_green]
            if greens:
                gx, gy = min(greens, key=lambda b: (b[0]-game.x)**2 + (b[1]-game.y)**2)
                if game.x < gx:
                    action = 3
                elif game.x > gx:
                    action = 2
                elif game.y < gy:
                    action = 1
                else:
                    action = 0
            else:
                action = 0
        else:
            action = random.randint(0, 3)
        
        _, _, done, _ = game.step(action)
        if done or game.turn > 60:
            break
    
    return frames

print("Creating high-resolution GIFs with proper text rendering...")
for policy in ['random', 'heuristic', 'nn']:
    print(f"Generating {policy}...")
    frames = run_policy_game(42, policy)
    # Save with versioned filename
    output_path = f'/Users/djohnson334/neurosymbolic-game-ai/output/12.03_{policy}.gif'
    imageio.mimsave(output_path, frames, duration=0.08, loop=0)
    print(f"Saved {output_path}")

print("Done!")
