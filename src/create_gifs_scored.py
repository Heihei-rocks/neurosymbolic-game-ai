#!/usr/bin/env python3
"""Generate animated GIFs with score tally and score pop-up animations."""
import numpy as np
import random
import imageio
import os
from pathlib import Path

# Get version from git or use default
def get_version():
    import subprocess
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return "dev"

os.chdir('/Users/djohnson334/neurosymbolic-game-ai')

# Load NN model
import joblib
import sys
sys.path.insert(0, 'src')
from game import GridGame
from src.heuristics import behavior

VERSION = get_version()
OUTPUT_DIR = Path('output')

class ScoreAnimator:
    """Manages score pop-up animations."""
    def __init__(self):
        self.active_pops = []  # list of (x, y, score_str, frame_index)
    
    def add_pop(self, x, y, score_points):
        """Add new score pop at position."""
        score_str = f"+{score_points}"
        self.active_pops.append({'x': x, 'y': y, 'score': score_str, 'age': 0})
    
    def update(self):
        """Age pops and remove old ones."""
        for pop in self.active_pops:
            pop['age'] += 1
        # Remove pops older than 20 frames
        self.active_pops = [p for p in self.active_pops if p['age'] < 20]
    
    def get_pop_color(self, age):
        """Yellow to black fade."""
        if age < 5:
            # Stay bright yellow
            return (255, 255, 0)
        elif age < 15:
            # Fade to black
            t = (age - 5) / 10.0
            intensity = int(255 * (1 - t))
            return (intensity, intensity, 0)
        else:
            return (0, 0, 0)

def draw_grid(frame, game, score_animator, total_score, turn):
    """Draw game grid with score pop-ups and tally."""
    # Create 32x32 grid (scaled up for visibility)
    grid_size = 32
    scale = 8
    img_size = grid_size * scale
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Draw grid background
    img.fill(10)
    
    # Draw game elements
    for gx, gy in game.green:
        if (gx, gy) not in game.collected_green:
            x, y = gx * scale, gy * scale
            img[y:y+scale, x:x+scale] = [0, 255, 0]  # Green
    
    for rx, ry in game.red:
        if (rx, ry) not in game.collected_red:
            x, y = rx * scale, ry * scale
            img[y:y+scale, x:x+scale] = [255, 0, 0]  # Red
    
    # Draw obstacles
    for ox, oy in game.obstacles:
        x, y = ox * scale, oy * scale
        img[y:y+scale, x:x+scale] = [128, 128, 128]  # Gray
    
    # Draw agent
    x, y = game.x * scale, game.y * scale
    img[y:y+scale, x:x+scale] = [0, 0, 255]  # Blue
    
    # Draw score pop-ups
    for pop in score_animator.active_pops:
        px, py = pop['x'] * scale, pop['y'] * scale
        pop_color = score_animator.get_pop_color(pop['age'])
        
        # Draw score text (simple block representation)
        # In real implementation, use PIL for text rendering
        text_h = 2 * scale
        text_w = len(pop['score']) * scale // 2
        text_y = py - text_h - 5
        text_x = px - text_w // 2
        
        # Simple text rendering using colored pixels
        for ty in range(text_h):
            for tx in range(text_w):
                if 0 <= text_y + ty < img_size and 0 <= text_x + tx < img_size:
                    # Create simple text pattern
                    img[text_y + ty, text_x + tx] = pop_color
    
    # Add score tally overlay
    # Score in top-left corner
    score_text = f"Score: {total_score}"
    # Simple representation - draw colored bar
    bar_height = scale * 2
    bar_y = 0
    for ty in range(bar_height):
        img[ty, :img_size] = [50, 50, 50]  # Dark bar
    
    # Add turn counter
    turn_text = f"Turn: {turn}"
    
    return img

def create_gif_with_scores(filename, policy, seed=42, max_turns=60, green_count=50):
    """Create animation with score tally and pop-ups."""
    print(f"Creating GIF: {filename}")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize game
    game = GridGame(seed=seed, green_count=green_count, red_disabled=True)
    game.reset()
    
    score_animator = ScoreAnimator()
    frames = []
    total_score = 0
    prev_score = 0
    
    # Track score pops for each frame
    score_history = []
    
    # Generate frames
    for turn in range(max_turns):
        if game.done:
            break
        
        # Determine action
        state = game.get_state()
        if policy == 'random':
            action = random.randint(0, 3)
        elif policy == 'heuristic':
            action = behavior(state)
        else:  # nn
            model = joblib.load('src/game_model.joblib')
            action = int(model.predict([state])[0])
        
        # Execute action
        prev_state = game.get_state()
        prev_score = game.score
        
        _, _, done, _ = game.step(action)
        
        # Check if scored
        if game.score > prev_score:
            score_gained = game.score - prev_score
            score_animator.add_pop(game.x, game.y, score_gained)
            total_score = game.score
        
        # Update animator
        score_animator.update()
        
        # Draw frame
        frame = draw_grid(None, game, score_animator, game.score, turn)
        frames.append(frame)
    
    # Save GIF
    output_path = OUTPUT_DIR / filename
    imageio.mimsave(str(output_path), frames, fps=10, duration=0.1)
    print(f"Saved to {output_path}")
    return len(frames)

# Create GIFs with score animations
print(f"Using version: {VERSION}")

# Only create a few frames to test
create_gif_with_scores(f"v{VERSION}_random.gif", 'random', seed=123, max_turns=30)
create_gif_with_scores(f"v{VERSION}_heuristic.gif", 'heuristic', seed=123, max_turns=30)
create_gif_with_scores(f"v{VERSION}_nn.gif", 'nn', seed=123, max_turns=30)

print("GIFs created with score pop-ups!")
