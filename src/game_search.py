#!/usr/bin/env python3
"""
Multi-Robot Area Coverage Game
===============================

Optimal search problem with multiple robots covering a continuous space.
- Board: 50x50 nmi continuous space (500x500 pixel grid)
- Robots: 1-10 blue agents with sensors, starting in loose lattice at bottom
- Sensor: Field of view (FOV) and range (20 nmi)
- Coverage: Pixels in FOV decay over time (age-out)
- Goal: Maximize cumulative area coverage
- Score: Sum of coverage matrix at each timestep
"""

import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class MultiRobotSearchGame:
    """
    Multi-robot area coverage game.

    State: Coverage map showing how recently each area was observed
    Action: Move direction for each robot
    Reward: Sum of coverage matrix (situational awareness value)
    """

    def __init__(self,
                 world_size_nmi=50.0,
                 grid_size=500,
                 num_robots=None,
                 sensor_range=20.0,  # nmi (increased from 5.0)
                 sensor_fov_degrees=90,
                 decay_rate=0.97,  # Changed from 0.99 to 0.97 (faster decay)
                 ingress_formation='clump',  # Changed from 'lattice' to 'clump'
                 colormap='inferno',  # inferno, bone, jet, plasma, viridis
                 num_priority_blobs=5,  # Number of Gaussian blobs for priority map
                 seed=42):
        """
        Initialize multi-robot search game.

        Args:
            world_size_nmi: Size of square world in nautical miles
            grid_size: Resolution of coverage grid (pixels)
            num_robots: Number of robots (random 1-10 if None)
            sensor_range: Sensor range in nmi
            sensor_fov_degrees: Field of view in degrees
            decay_rate: Coverage decay per timestep (0.97 = 3% decay)
            ingress_formation: 'clump' for tight start, 'lattice' for spread
            colormap: Matplotlib colormap name
            num_priority_blobs: Number of Gaussian blobs for priority map
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)

        # World parameters
        self.world_size_nmi = world_size_nmi
        self.grid_size = grid_size
        self.nmi_per_pixel = world_size_nmi / grid_size

        # Robot parameters
        self.num_robots = num_robots if num_robots else random.randint(1, 10)
        self.sensor_range = sensor_range
        self.sensor_fov_degrees = sensor_fov_degrees
        self.sensor_fov_radians = np.radians(sensor_fov_degrees)
        self.decay_rate = decay_rate
        self.ingress_formation = ingress_formation

        # Priority map parameters
        self.num_priority_blobs = num_priority_blobs

        # Visualization
        self.colormap = colormap
        try:
            self.cmap = cm.get_cmap(colormap)
        except AttributeError:
            # Newer matplotlib versions
            self.cmap = plt.get_cmap(colormap)

        # Initialize priority map (static for game lifetime)
        self.priority_map = self._generate_priority_map()

        # Initialize robots in formation
        self.robots = self._init_robots_formation()

        # Coverage map: 0 (never seen) to 1 (just seen)
        self.coverage = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Game state
        self.timestep = 0
        self.done = False
        self.max_timesteps = 100

        # Scoring: cumulative sum of all coverage values over time
        self.score = 0.0
        self.score_history = []
        self.coverage_sum_history = []

    def _generate_priority_map(self):
        """
        Generate priority map as mixture of random Gaussian blobs.

        Returns:
            Priority map (grid_size x grid_size) with values [0, 1]
        """
        priority_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Generate random Gaussian blobs
        for _ in range(self.num_priority_blobs):
            # Random center position (in pixels)
            center_x = random.randint(0, self.grid_size - 1)
            center_y = random.randint(0, self.grid_size - 1)

            # Random sigma (spread) - range from narrow to wide blobs
            sigma = random.uniform(30, 100)  # pixels

            # Random intensity
            intensity = random.uniform(0.3, 1.0)

            # Generate Gaussian blob
            y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
            dist_squared = (x_coords - center_x)**2 + (y_coords - center_y)**2
            blob = intensity * np.exp(-dist_squared / (2 * sigma**2))

            # Add to priority map
            priority_map = np.maximum(priority_map, blob)

        # Normalize to [0, 1]
        if priority_map.max() > 0:
            priority_map = priority_map / priority_map.max()

        return priority_map

    def _init_robots_formation(self):
        """Initialize robots in formation at bottom."""
        robots = []

        if self.ingress_formation == 'clump':
            # Tight clump at center bottom
            center_x = self.world_size_nmi / 2

            for i in range(self.num_robots):
                # X position: clustered near center with small random variation
                x = center_x + random.uniform(-3, 3)
                x = np.clip(x, 0, self.world_size_nmi)

                # Y position: near bottom with small random variation
                y = random.uniform(0, 3)

                # Heading: generally north (90°) with small random variation
                heading = np.radians(90) + random.uniform(-0.2, 0.2)

                robots.append([x, y, heading])

        elif self.ingress_formation == 'lattice':
            # Loose lattice spacing at bottom (y ~ 0-5 nmi)
            spacing_x = self.world_size_nmi / (self.num_robots + 1)

            for i in range(self.num_robots):
                # X position: evenly spaced with random variation
                x = spacing_x * (i + 1) + random.uniform(-2, 2)
                x = np.clip(x, 0, self.world_size_nmi)

                # Y position: near bottom with small random variation
                y = random.uniform(0, 5)

                # Heading: generally north (90°) with random variation
                heading = np.radians(90) + random.uniform(-0.3, 0.3)

                robots.append([x, y, heading])

        return np.array(robots)

    def reset(self):
        """Reset the game to initial state."""
        # Reinitialize robot formation
        self.robots = self._init_robots_formation()

        self.coverage = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.timestep = 0
        self.done = False
        self.score = 0.0
        self.score_history = []
        self.coverage_sum_history = []

        # Apply initial sensor coverage
        self._update_coverage()

        # Record initial score (weighted by priority map)
        weighted_coverage = self.coverage * self.priority_map
        coverage_sum = np.sum(weighted_coverage)
        self.score += coverage_sum
        self.score_history.append(self.score)
        self.coverage_sum_history.append(coverage_sum)

        return self.get_state()

    def _update_coverage(self):
        """Update coverage map with current robot sensor observations."""
        # Decay existing coverage
        self.coverage *= self.decay_rate

        # Add new observations from each robot
        for robot_idx in range(self.num_robots):
            x_nmi, y_nmi, heading = self.robots[robot_idx]

            # Convert to pixel coordinates
            x_px = int(x_nmi / self.nmi_per_pixel)
            y_px = int(y_nmi / self.nmi_per_pixel)

            # Sensor range in pixels
            range_px = int(self.sensor_range / self.nmi_per_pixel)

            # Calculate FOV coverage
            self._add_fov_coverage(x_px, y_px, heading, range_px)

    def _add_fov_coverage(self, x_px, y_px, heading, range_px):
        """Add sensor FOV coverage to the map."""
        # Create a mask for the FOV cone
        half_fov = self.sensor_fov_radians / 2

        # Check each pixel in the sensor range
        for dy in range(-range_px, range_px + 1):
            for dx in range(-range_px, range_px + 1):
                # Target pixel
                tx = x_px + dx
                ty = y_px + dy

                # Check bounds
                if tx < 0 or tx >= self.grid_size or ty < 0 or ty >= self.grid_size:
                    continue

                # Distance from robot
                dist = np.sqrt(dx**2 + dy**2)
                if dist > range_px:
                    continue

                # Angle to target
                if dist == 0:
                    # Robot's own position
                    self.coverage[ty, tx] = 1.0
                    continue

                # FIXED: angle_to_target in image coordinates (y-axis points down)
                angle_to_target = np.arctan2(-dy, dx)  # Negative dy because y increases downward in image

                # Angle difference from heading
                angle_diff = angle_to_target - heading
                # Normalize to [-pi, pi]
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

                # Check if within FOV
                if abs(angle_diff) <= half_fov:
                    self.coverage[ty, tx] = 1.0

    def step(self, actions):
        """
        Execute actions for all robots.

        Args:
            actions: List of actions for each robot
                    Each action: [delta_heading, speed]
                    delta_heading: change in heading (-pi to pi)
                    speed: forward speed in nmi/timestep

        Returns:
            state, reward, done, info
        """
        # Move each robot
        for robot_idx, action in enumerate(actions):
            delta_heading, speed = action

            # Update heading
            self.robots[robot_idx, 2] += delta_heading
            # Normalize heading to [0, 2pi]
            self.robots[robot_idx, 2] = self.robots[robot_idx, 2] % (2 * np.pi)

            # Update position
            heading = self.robots[robot_idx, 2]
            dx = speed * np.cos(heading)
            dy = speed * np.sin(heading)

            self.robots[robot_idx, 0] += dx
            self.robots[robot_idx, 1] += dy

            # Clamp to boundaries (no wrap-around)
            self.robots[robot_idx, 0] = np.clip(self.robots[robot_idx, 0], 0, self.world_size_nmi)
            self.robots[robot_idx, 1] = np.clip(self.robots[robot_idx, 1], 0, self.world_size_nmi)

        # Update coverage
        self._update_coverage()

        # Calculate reward: sum of (coverage * priority_map) - weighted SA value
        weighted_coverage = self.coverage * self.priority_map
        coverage_sum = np.sum(weighted_coverage)
        reward = coverage_sum

        # Update cumulative score
        self.score += coverage_sum
        self.score_history.append(self.score)
        self.coverage_sum_history.append(coverage_sum)

        # Update timestep
        self.timestep += 1
        if self.timestep >= self.max_timesteps:
            self.done = True

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        """
        Get current state representation for RL agent.

        Returns:
            numpy array with flattened state information
        """
        # State includes:
        # - Robot positions and headings (3 * num_robots)
        # - Coverage statistics (4 values)
        # - Downsampled coverage map (32x32)

        robot_state = self.robots.flatten()

        coverage_stats = np.array([
            np.mean(self.coverage),
            np.std(self.coverage),
            np.min(self.coverage),
            np.max(self.coverage)
        ])

        # Downsample coverage for state representation
        downsample_factor = self.grid_size // 32
        coverage_downsampled = self.coverage[::downsample_factor, ::downsample_factor]

        state = np.concatenate([
            robot_state,
            coverage_stats,
            coverage_downsampled.flatten()
        ])

        return state

    def render(self, save_path=None):
        """
        Render the current game state as an image with colormap.

        Args:
            save_path: If provided, save image to this path

        Returns:
            PIL Image
        """
        # Apply colormap to coverage (flip vertically so y=0 is at bottom)
        coverage_flipped = np.flipud(self.coverage)
        coverage_colored = self.cmap(coverage_flipped)[:, :, :3]  # RGB only
        img_array = (coverage_colored * 255).astype(np.uint8)

        # Convert to PIL for drawing robots
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        # Draw each robot
        for robot_idx in range(self.num_robots):
            x_nmi, y_nmi, heading = self.robots[robot_idx]
            x_px = int(x_nmi / self.nmi_per_pixel)
            # Flip y for rendering: y_nmi=0 should be at bottom of image
            y_px = self.grid_size - 1 - int(y_nmi / self.nmi_per_pixel)

            # Robot body (cyan circle for visibility)
            robot_radius = 5
            draw.ellipse(
                [x_px - robot_radius, y_px - robot_radius,
                 x_px + robot_radius, y_px + robot_radius],
                fill=(0, 255, 255),
                outline=(255, 255, 255),
                width=2
            )

            # Heading indicator (white line)
            # heading=0 is East, heading=π/2 is North (up in game)
            # Since y is flipped in rendering, negate sin component
            line_length = 10
            end_x = x_px + line_length * np.cos(heading)
            end_y = y_px - line_length * np.sin(heading)  # Negative because y rendering is flipped
            draw.line([x_px, y_px, end_x, end_y], fill=(255, 255, 255), width=2)

        if save_path:
            img.save(save_path)

        return img

    def render_priority_map(self, save_path=None):
        """
        Render the priority map.

        Args:
            save_path: If provided, save image to this path

        Returns:
            PIL Image
        """
        # Apply colormap to priority map
        priority_colored = self.cmap(self.priority_map)[:, :, :3]  # RGB only
        img_array = (priority_colored * 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        if save_path:
            img.save(save_path)

        return img

    def render_weighted_coverage(self, save_path=None, figsize_multiplier=2.0):
        """
        Render the weighted coverage (SA * priority) with priority map overlay.

        Args:
            save_path: If provided, save image to this path
            figsize_multiplier: Scale factor for output image size

        Returns:
            PIL Image
        """
        # Weighted coverage
        weighted = self.coverage * self.priority_map

        # Flip vertically so y=0 is at bottom
        weighted_flipped = np.flipud(weighted)
        priority_flipped = np.flipud(self.priority_map)

        # Apply colormap
        weighted_colored = self.cmap(weighted_flipped)[:, :, :3]  # RGB only
        img_array = (weighted_colored * 255).astype(np.uint8)

        # Convert to PIL
        img = Image.fromarray(img_array)

        # Resize for larger output
        if figsize_multiplier != 1.0:
            new_size = (int(img.width * figsize_multiplier), int(img.height * figsize_multiplier))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            priority_flipped_resized = Image.fromarray((priority_flipped * 255).astype(np.uint8), mode='L')
            priority_flipped_resized = priority_flipped_resized.resize(new_size, Image.Resampling.LANCZOS)
            priority_gray = np.array(priority_flipped_resized)
        else:
            priority_gray = (priority_flipped * 255).astype(np.uint8)

        # Create priority map overlay (grayscale, mostly transparent)
        priority_overlay = Image.fromarray(priority_gray, mode='L')

        # Convert to RGBA for transparency control
        priority_overlay = priority_overlay.convert('RGBA')

        # Adjust alpha channel: priority map with 30% opacity
        priority_data = priority_overlay.getdata()
        priority_with_alpha = []
        for item in priority_data:
            # White where priority is high, with 30% opacity
            gray_value = item[0]
            alpha = int(gray_value * 0.3)  # 30% max opacity
            priority_with_alpha.append((255, 255, 255, alpha))

        priority_overlay.putdata(priority_with_alpha)

        # Composite: weighted coverage + priority overlay
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, priority_overlay)
        img = img.convert('RGB')

        # Draw robots
        draw = ImageDraw.Draw(img)

        for robot_idx in range(self.num_robots):
            x_nmi, y_nmi, heading = self.robots[robot_idx]
            x_px = int(x_nmi / self.nmi_per_pixel * figsize_multiplier)
            # Flip y for rendering: y_nmi=0 should be at bottom of image
            y_px = int((self.grid_size - 1 - y_nmi / self.nmi_per_pixel) * figsize_multiplier)

            # Robot body (cyan circle for visibility) - larger for bigger image
            robot_radius = int(10 * figsize_multiplier)
            draw.ellipse(
                [x_px - robot_radius, y_px - robot_radius,
                 x_px + robot_radius, y_px + robot_radius],
                fill=(0, 255, 255),
                outline=(255, 255, 255),
                width=int(4 * figsize_multiplier)
            )

            # Heading indicator (white line) - longer for bigger image
            line_length = int(20 * figsize_multiplier)
            end_x = x_px + line_length * np.cos(heading)
            end_y = y_px - line_length * np.sin(heading)  # Negative because y rendering is flipped
            draw.line([x_px, y_px, end_x, end_y], fill=(255, 255, 255), width=int(4 * figsize_multiplier))

        if save_path:
            img.save(save_path)

        return img


def simple_behavior(game_state, robot_idx):
    """
    Simple coordinated behavior for multi-robot search.

    Strategy: Move north with slight variations to maintain formation.

    Args:
        game_state: Current game state dict
        robot_idx: Index of robot to control

    Returns:
        [delta_heading, speed]: Action for this robot
    """
    robots = game_state['robots']
    robot_x, robot_y, robot_heading = robots[robot_idx]

    # Target heading: north (90 degrees = pi/2 radians)
    target_heading = np.pi / 2

    # Current heading
    current_heading = robot_heading

    # Heading error
    heading_error = target_heading - current_heading
    # Normalize to [-pi, pi]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    # Proportional control: turn toward north
    delta_heading = np.clip(heading_error * 0.3, -0.2, 0.2)

    # Speed: constant northward progress
    speed = 2.0  # nmi per timestep

    return [delta_heading, speed]


def demo_game():
    """Demonstrate the multi-robot search game with behavior control."""
    print("="*70)
    print("MULTI-ROBOT AREA COVERAGE GAME DEMO - v2")
    print("="*70)

    # Create game
    game = MultiRobotSearchGame(
        world_size_nmi=50.0,
        grid_size=500,
        num_robots=6,
        sensor_range=20.0,
        sensor_fov_degrees=90,
        decay_rate=0.97,  # Faster decay
        ingress_formation='clump',  # Tight clump
        colormap='inferno',
        num_priority_blobs=5,
        seed=42
    )

    print(f"\nGame Configuration:")
    print(f"  World size: {game.world_size_nmi} nmi x {game.world_size_nmi} nmi")
    print(f"  Grid resolution: {game.grid_size} x {game.grid_size} pixels")
    print(f"  Number of robots: {game.num_robots}")
    print(f"  Sensor range: {game.sensor_range} nmi")
    print(f"  Sensor FOV: {game.sensor_fov_degrees}°")
    print(f"  Coverage decay rate: {game.decay_rate}")
    print(f"  Formation: {game.ingress_formation}")
    print(f"  Colormap: {game.colormap}")
    print(f"  Priority blobs: {game.num_priority_blobs}")

    # Render priority map
    print(f"\n{'='*70}")
    print("PRIORITY MAP (Gaussian Mixture)")
    print('='*70)
    game.render_priority_map(save_path='output/search_v3_priority_map.png')
    print(f"✓ Priority map saved to output/search_v3_priority_map.png")
    print(f"  Priority statistics:")
    print(f"    Mean: {np.mean(game.priority_map):.3f}")
    print(f"    Max: {np.max(game.priority_map):.3f}")
    print(f"    Min: {np.min(game.priority_map):.3f}")

    # Initialize
    state = game.reset()

    print(f"\n{'='*70}")
    print("INITIAL STATE (Clump Formation at Bottom Center)")
    print('='*70)

    print("\nRobot positions:")
    for i, robot in enumerate(game.robots):
        print(f"  Robot {i}: pos=({robot[0]:.1f}, {robot[1]:.1f}) nmi, "
              f"heading={np.degrees(robot[2]):.1f}°")

    # Render initial state (both raw and weighted)
    img = game.render(save_path='output/search_v3_initial_coverage.png')
    print(f"\n✓ Initial coverage saved to output/search_v3_initial_coverage.png")

    img_weighted = game.render_weighted_coverage(save_path='output/search_v3_initial_weighted.png')
    print(f"✓ Initial weighted coverage saved to output/search_v3_initial_weighted.png")

    # Run simulation with coordinated behavior
    print(f"\n{'='*70}")
    print("RUNNING SIMULATION (50 timesteps with coordinated behavior)")
    print('='*70)

    for t in range(50):
        # Get actions from behavior routine for all robots
        actions = []
        for robot_idx in range(game.num_robots):
            action = simple_behavior({'robots': game.robots}, robot_idx)
            actions.append(action)

        state, reward, done, _ = game.step(actions)

        if t % 10 == 0:
            weighted_sum = game.coverage_sum_history[-1]
            print(f"  t={t:2d}: weighted_coverage={weighted_sum:7.1f}, "
                  f"cumulative_score={game.score:.1f}, "
                  f"mean_cov={np.mean(game.coverage):.3f}")

        # Save intermediate frames
        if t in [10, 25, 49]:
            game.render(save_path=f'output/search_v3_t{t:02d}_coverage.png')
            game.render_weighted_coverage(save_path=f'output/search_v3_t{t:02d}_weighted.png')

    # Final renders
    game.render(save_path='output/search_v3_final_coverage.png')
    print(f"\n✓ Final coverage saved to output/search_v3_final_coverage.png")

    game.render_weighted_coverage(save_path='output/search_v3_final_weighted.png')
    print(f"✓ Final weighted coverage saved to output/search_v3_final_weighted.png")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cumulative score over time
    ax1.plot(game.score_history, linewidth=2)
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Cumulative Score (Weighted)', fontsize=12)
    ax1.set_title('Total Weighted Score Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Weighted coverage sum per timestep
    ax2.plot(game.coverage_sum_history, linewidth=2, color='orange')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Weighted Coverage Sum', fontsize=12)
    ax2.set_title('Instantaneous Weighted Coverage Per Timestep', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/search_v3_scores.png', dpi=150)
    print(f"✓ Score plots saved to output/search_v3_scores.png")

    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print('='*70)
    print(f"\nFinal statistics:")
    print(f"  Cumulative score (weighted): {game.score:.1f}")
    print(f"  Mean raw coverage: {np.mean(game.coverage):.3f}")
    print(f"  Mean weighted coverage: {np.mean(game.coverage * game.priority_map):.3f}")
    print(f"  Max coverage: {np.max(game.coverage):.3f}")
    print(f"  Timesteps: {game.timestep}")

    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print('='*70)
    print("Robots must learn to:")
    print("  1. Identify high-priority regions (hotspots in priority map)")
    print("  2. Balance coverage vs revisiting (0.97 decay is aggressive)")
    print("  3. Coordinate to avoid redundant coverage")
    print("  4. Maximize weighted coverage sum = SA * Priority")


if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    demo_game()
