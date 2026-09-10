#!/usr/bin/env python3
"""Generate animated GIFs for all three policies with time pressure and score tally."""
import numpy as np
import random
import imageio
import os
import subprocess
from pathlib import Path
import sys

# Get version from git
def get_version():
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                          capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode == 0:
        version_hash = result.stdout.strip()[:7]
        # Get commit count
        result2 = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], 
                               capture_output=True, text=True, cwd=os.getcwd())
        if result2.returncode == 0:
            commit_count = int(result2.stdout.strip())
            return f"{commit_count}.0{commit_count % 10}"  # Simple version scheme
    return "0.10"

os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

import sys
sys.path.insert(0, 'src')
from game import GridGame
from heuristics import behavior

VERSION = get_version()
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

class ScorePop:
    """Score pop-up animation."""
    def __init__(self, x, y, score_value):
        self.x = x
        self.y = y
        self.score = score_value
        self.age = 0
        self.max_age = 20
    
    def update(self):
        self.age += 1
        return self.age < self.max_age
    
    def get_color(self):
        """Yellow to black fade."""
        if self.age < 5:
            return (255, 255, 0)
        t = (self.age - 5) / 15.0
        intensity = int(255 * (1 - t))
        return (intensity, intensity, 0)
    
    def get_offset(self):
        """Float up as it ages."""
        return self.age * 2

def draw_frame(game, pops, total_score, turn, frame_idx):
    """Draw single frame with score tally and pop-ups."""
    # Create 32x32 grid, scale up for visibility
    grid_size = 32
    scale = 12
    img_size = grid_size * scale
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img.fill(20)  # Dark background
    
    # Draw grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            x0, y0 = j * scale, i * scale
            # Draw green boxes
            if (j, i) in game.green and (j, i) not in game.collected_green:
                img[y0:y0+scale, x0:x0+scale] = [0, 200, 0]
            # Draw red boxes
            elif game.red and (j, i) in game.red and (j, i) not in game.collected_red:
                img[y0:y0+scale, x0:x0+scale] = [200, 0, 0]
            # Draw obstacles
            elif (j, i) in game.obstacles:
                img[y0:y0+scale, x0:x0+scale] = [100, 100, 100]
    
    # Draw agent
    ax, ay = game.x * scale, game.y * scale
    img[ay:ay+scale, ax:ax+scale] = [0, 0, 255]
    
    # Draw score pop-ups
    for pop in pops[:]:
        px = pop.x * scale
        py = pop.y * scale - pop.get_offset()
        color = pop.get_color()
        
        # Draw score text (simple blocks)
        score_str = f"+{game.reward_counter if game.reward_counter > 0 else pop.score}"
        text_w = len(score_str) * scale // 2
        text_h = scale
        
        for dy in range(text_h):
            for dx in range(text_w):
                if 0 <= py - text_h + dy < img_size and 0 <= px - text_w//2 + dx < img_size:
                    img[py - text_h + dy, px - text_w//2 + dx] = color
    
    # Draw score tally bar
    bar_h = scale * 3
    img[0:bar_h, :] = [40, 40, 60]
    
    # Add score text to bar (rough representation)
    score_text = f"Score: {total_score:04d}"
    # Simple text rendering with pixels
    for i, char in enumerate(score_text):
        x_pos = i * scale // 2
        if x_pos < img_size:
            img[scale//2:scale*2, x_pos:x_pos+scale//2] = [255, 255, 255]
    
    # Turn counter
    turn_text = f"Turn: {turn:02d}"
    for i in range(len(turn_text)):
        x_pos = img_size - (i + 1) * scale // 2
        if x_pos > 0:
            img[scale//2:scale*2, x_pos:x_pos+scale//2] = [255, 255, 0]
    
    return img

def create_scored_gif(policy_name, seed, green_count=50):
    """Create GIF with score animations."""
    print(f"Creating {policy_name} GIF...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    game = GridGame(seed=seed, green_count=green_count, red_disabled=True)
    game.reset()
    
    # Load model if needed
    model = None
    if policy_name == 'nn':
        model = joblib.load('src/game_model.joblib')
    
    pops = []
    frames = []
    total_score = 0
    prev_score = 0
    
    # Generate up to 60 turns
    for turn in range(60):
        if game.done:
            break
        
        # Get action
        state = game.get_state()
        if policy_name == 'random':
            action = random.randint(0, 3)
        elif policy_name == 'heuristic':
            action = behavior(state)
        else:  # nn
            action = int(model.predict([state])[0])
        
        # Execute
        score_before = game.score
        game.step(action)
        
        # Check for score
        if game.score > score_before:
            score_gained = game.score - score_before
            pops.append(ScorePop(game.x, game.y, score_gained))
        
        # Update pops
        for pop in pops[:]:
            if not pop.update():
                pops.remove(pop)
        
        # Draw frame
        frames.append(draw_frame(game, pops, game.score, turn, len(frames)))
    
    # Save with version in filename
    filename = f"{VERSION}_{policy_name}.gif"
    output_path = OUTPUT_DIR / filename
    imageio.mimsave(str(output_path), frames, fps=10)
    print(f"  Saved {output_path} ({len(frames)} frames)")
    
    return output_path

# Create GIFs
if __name__ == '__main__':
    joblib = __import__('joblib')
    
    create_scored_gif('random', seed=123)
    create_scored_gif('heuristic', seed=123)
    create_scored_gif('nn', seed=123)
    
    print(f"\nVersion {VERSION} GIFs created in output/")
