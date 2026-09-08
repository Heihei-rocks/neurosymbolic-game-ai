#!/usr/bin/env python3
"""Generate project image for README."""
import numpy as np

# Create a visual representation of the grid
grid = [[' ' for _ in range(32)] for _ in range(32)]

# Place obstacles (simple pattern)
for x in range(0, 32, 4):
    for y in range(32):
        if np.random.random() < 0.3:
            grid[y][x] = '#'

np.random.seed(42)

# Place green boxes (+1)
green_positions = []
while len(green_positions) < 5:
    x, y = np.random.randint(0, 32), np.random.randint(0, 32)
    if grid[y][x] == ' ' and (x, y) != (16, 16):
        grid[y][x] = 'G'
        green_positions.append((x, y))

# Place red boxes (-1)
red_positions = []
while len(red_positions) < 5:
    x, y = np.random.randint(0, 32), np.random.randint(0, 32)
    if grid[y][x] == ' ' and (x, y) not in green_positions:
        grid[y][x] = 'R'
        red_positions.append((x, y))

# Create image data (32x32 grid with larger pixels for visibility)
img_size = 320  # 10x per pixel
img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # white background

# Colors
GRAY = [200, 200, 200]    # obstacles
GREEN = [0, 180, 0]       # green boxes  
RED = [180, 0, 0]         # red boxes
BLUE = [0, 0, 180]        # agent

# Draw elements
for y in range(32):
    for x in range(32):
        pixel_y0, pixel_y1 = y * 10, (y + 1) * 10
        pixel_x0, pixel_x1 = x * 10, (x + 1) * 10
        
        if grid[y][x] == '#':
            img[pixel_y0:pixel_y1, pixel_x0:pixel_x1] = GRAY
        elif grid[y][x] == 'G':
            img[pixel_y0:pixel_y1, pixel_x0:pixel_x1] = GREEN
        elif grid[y][x] == 'R':
            img[pixel_y0:pixel_y1, pixel_x0:pixel_x1] = RED

# Agent at center (16, 16)
img[160:170, 160:170] = BLUE

# Save
try:
    import imageio
    imageio.imwrite('/Users/djohnson334/neurosymbolic-game-ai/game_visualization.png', img)
    print("Saved game_visualization.png")
except ImportError:
    # Create ASCII for fallback
    with open('/Users/djohnson334/neurosymbolic-game-ai/game_visualization.txt', 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')
    print("Saved game_visualization.txt")
    
print(f"\nGreen boxes: {len(green_positions)}")
print(f"Red boxes: {len(red_positions)}")