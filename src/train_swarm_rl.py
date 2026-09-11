#!/usr/bin/env python3
"""
Reinforcement Learning Training for Multi-Robot Search Game
============================================================

Train a neural network to control a swarm of robots for optimal
priority-aware area coverage.

State representation includes:
- Robot positions and headings
- Coverage map (SA matrix)
- Priority map
- Coverage age (for decay awareness)
- Inter-robot distances (for coordination)
- High-priority region locations
"""

import numpy as np
import random
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_search import MultiRobotSearchGame
from sklearn.neural_network import MLPRegressor
import joblib


class SwarmReplayBuffer:
    """Replay buffer for multi-robot learning."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, actions, reward, next_state, done):
        self.buffer.append((state, actions, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class SwarmQLearningAgent:
    """Q-learning agent for multi-robot swarm control."""

    def __init__(self, state_dim, num_robots, hidden_layers=(256, 128, 64)):
        self.state_dim = state_dim
        self.num_robots = num_robots
        self.n_actions_per_robot = 5  # Turn left, straight, turn right, speed up, slow down
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9975  # Reach ~0.05 by episode 500

        # Learning rate schedule - more aggressive for better learning
        self.initial_lr = 0.001
        self.min_lr = 0.0002  # Higher minimum (was 0.0001)
        self.lr_decay = 0.999  # Slower decay (was 0.998)
        self.current_lr = self.initial_lr

        # Q-network (larger for complex coordination task)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate='constant',  # Changed from 'adaptive' to use our schedule
            learning_rate_init=self.current_lr,
            max_iter=1,
            warm_start=True,
            random_state=42,
            verbose=False
        )

        # Initialize with dummy data
        output_dim = num_robots * self.n_actions_per_robot
        dummy_X = np.zeros((output_dim, state_dim))
        dummy_y = np.zeros((output_dim, output_dim))
        self.model.fit(dummy_X, dummy_y)

        self.replay_buffer = SwarmReplayBuffer()
        self.training_history = []

    def get_state_representation(self, game):
        """
        Extract comprehensive state representation for RL with enhanced cognitive artifacts.

        Includes:
        - Robot states (position, heading) - normalized
        - Coverage map (downsampled)
        - Priority map (downsampled)
        - Coverage age (for decay awareness)
        - Inter-robot distances
        - Nearest high-priority uncovered regions

        NEW COGNITIVE ARTIFACTS:
        - Ego-centric target vectors per robot
        - Local coverage gradients (4 directions per robot)
        - Priority-weighted remaining work estimate
        - Team spatial entropy (clustering metric)
        - Recent score velocity (performance trend)
        - Nearest teammate direction per robot
        """
        state_parts = []

        # 1. Robot states (3 * num_robots)
        for robot in game.robots:
            x_norm = robot[0] / game.world_size_nmi
            y_norm = robot[1] / game.world_size_nmi
            heading_norm = robot[2] / (2 * np.pi)
            state_parts.extend([x_norm, y_norm, heading_norm])

        # 2. Coverage statistics (4 values)
        state_parts.extend([
            np.mean(game.coverage),
            np.std(game.coverage),
            np.min(game.coverage),
            np.max(game.coverage)
        ])

        # 3. Weighted coverage statistics (4 values)
        weighted = game.coverage * game.priority_map
        state_parts.extend([
            np.mean(weighted),
            np.std(weighted),
            np.min(weighted),
            np.max(weighted)
        ])

        # 4. Downsampled coverage map (8x8 = 64 values)
        downsample_factor = game.grid_size // 8
        coverage_small = game.coverage[::downsample_factor, ::downsample_factor]
        state_parts.extend(coverage_small.flatten())

        # 5. Downsampled priority map (8x8 = 64 values)
        priority_small = game.priority_map[::downsample_factor, ::downsample_factor]
        state_parts.extend(priority_small.flatten())

        # 6. Inter-robot distances (for coordination)
        for i in range(game.num_robots):
            for j in range(i + 1, game.num_robots):
                dx = game.robots[i, 0] - game.robots[j, 0]
                dy = game.robots[i, 1] - game.robots[j, 1]
                dist = np.sqrt(dx**2 + dy**2) / game.world_size_nmi
                state_parts.append(dist)

        # 7. For each robot: direction to nearest high-priority low-coverage area
        for robot_idx in range(game.num_robots):
            robot_x, robot_y = game.robots[robot_idx, 0:2]
            robot_heading = game.robots[robot_idx, 2]
            robot_x_px = int(robot_x / game.nmi_per_pixel)
            robot_y_px = int(robot_y / game.nmi_per_pixel)

            # Find high-priority, low-coverage areas
            priority_threshold = 0.5
            coverage_threshold = 0.3
            target_mask = (game.priority_map > priority_threshold) & (game.coverage < coverage_threshold)

            if np.any(target_mask):
                # Find nearest target
                y_coords, x_coords = np.where(target_mask)
                distances = np.sqrt((x_coords - robot_x_px)**2 + (y_coords - robot_y_px)**2)
                nearest_idx = np.argmin(distances)
                target_x_px, target_y_px = x_coords[nearest_idx], y_coords[nearest_idx]

                # Direction to target (normalized)
                dx = (target_x_px - robot_x_px) / game.grid_size
                dy = (target_y_px - robot_y_px) / game.grid_size
                dist = distances[nearest_idx] / game.grid_size

                # NEW: Ego-centric bearing to target (relative to robot heading)
                angle_to_target = np.arctan2(dy, dx)
                relative_bearing = angle_to_target - robot_heading
                # Normalize to [-pi, pi]
                relative_bearing = (relative_bearing + np.pi) % (2 * np.pi) - np.pi
                bearing_norm = relative_bearing / np.pi
            else:
                # No targets
                dx, dy, dist, bearing_norm = 0, 0, 0, 0

            state_parts.extend([dx, dy, dist, bearing_norm])

        # 8. NEW: Local coverage gradients per robot (4 directions: N, S, E, W)
        gradient_radius_px = 20  # Look 20 pixels in each direction
        for robot_idx in range(game.num_robots):
            robot_x_px = int(game.robots[robot_idx, 0] / game.nmi_per_pixel)
            robot_y_px = int(game.robots[robot_idx, 1] / game.nmi_per_pixel)

            # Sample coverage in 4 cardinal directions
            gradients = []
            for direction in [(0, -1), (0, 1), (1, 0), (-1, 0)]:  # N, S, E, W
                sample_x = robot_x_px + direction[0] * gradient_radius_px
                sample_y = robot_y_px + direction[1] * gradient_radius_px

                # Clamp to bounds
                sample_x = np.clip(sample_x, 0, game.grid_size - 1)
                sample_y = np.clip(sample_y, 0, game.grid_size - 1)

                # Get priority-weighted coverage at sample point
                coverage_value = game.coverage[sample_y, sample_x]
                priority_value = game.priority_map[sample_y, sample_x]
                weighted_value = coverage_value * priority_value
                gradients.append(weighted_value)

            state_parts.extend(gradients)

        # 9. NEW: Priority-weighted remaining work
        uncovered_priority = np.sum(game.priority_map * (1.0 - game.coverage))
        total_priority = np.sum(game.priority_map)
        remaining_work = uncovered_priority / (total_priority + 1e-6)
        state_parts.append(remaining_work)

        # 10. NEW: Team spatial entropy (are robots spread out or clustered?)
        robot_positions = game.robots[:, 0:2]
        if game.num_robots > 1:
            # Calculate pairwise distances
            pairwise_dists = []
            for i in range(game.num_robots):
                for j in range(i + 1, game.num_robots):
                    dx = robot_positions[i, 0] - robot_positions[j, 0]
                    dy = robot_positions[i, 1] - robot_positions[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    pairwise_dists.append(dist)

            # Std of distances = measure of spread (high = spread out, low = clustered)
            spatial_entropy = np.std(pairwise_dists) / game.world_size_nmi
        else:
            spatial_entropy = 0.0
        state_parts.append(spatial_entropy)

        # 11. NEW: Recent score velocity (is performance improving?)
        if len(game.coverage_sum_history) >= 3:
            # Compare last 2 timesteps
            recent_velocity = game.coverage_sum_history[-1] - game.coverage_sum_history[-2]
        else:
            recent_velocity = 0.0
        state_parts.append(recent_velocity / 1000.0)  # Normalize

        # 12. NEW: For each robot, direction to nearest teammate
        for robot_idx in range(game.num_robots):
            if game.num_robots > 1:
                robot_pos = game.robots[robot_idx, 0:2]
                other_robots = np.delete(game.robots[:, 0:2], robot_idx, axis=0)

                # Find nearest teammate
                distances = np.sqrt(np.sum((other_robots - robot_pos)**2, axis=1))
                nearest_teammate_idx = np.argmin(distances)
                nearest_teammate_pos = other_robots[nearest_teammate_idx]

                # Direction to teammate (normalized)
                dx_teammate = (nearest_teammate_pos[0] - robot_pos[0]) / game.world_size_nmi
                dy_teammate = (nearest_teammate_pos[1] - robot_pos[1]) / game.world_size_nmi
                dist_teammate = distances[nearest_teammate_idx] / game.world_size_nmi
            else:
                dx_teammate, dy_teammate, dist_teammate = 0, 0, 0

            state_parts.extend([dx_teammate, dy_teammate, dist_teammate])

        return np.array(state_parts, dtype=np.float32)

    def decode_actions(self, action_indices):
        """
        Decode action indices to robot control commands.

        Each robot gets an action index (0-4):
        0: Turn left, constant speed
        1: Straight, constant speed
        2: Turn right, constant speed
        3: Straight, speed up
        4: Straight, slow down
        """
        actions = []
        base_speed = 2.0  # nmi per timestep
        turn_amount = 0.2  # radians

        for robot_idx in range(self.num_robots):
            action_idx = action_indices[robot_idx]

            if action_idx == 0:  # Turn left
                actions.append([-turn_amount, base_speed])
            elif action_idx == 1:  # Straight
                actions.append([0.0, base_speed])
            elif action_idx == 2:  # Turn right
                actions.append([turn_amount, base_speed])
            elif action_idx == 3:  # Speed up
                actions.append([0.0, base_speed * 1.5])
            elif action_idx == 4:  # Slow down
                actions.append([0.0, base_speed * 0.5])
            else:
                actions.append([0.0, base_speed])  # Default

        return actions

    def get_actions(self, state, training=True):
        """Epsilon-greedy action selection for all robots."""
        if training and random.random() < self.epsilon:
            # Random actions for all robots
            return [random.randint(0, self.n_actions_per_robot - 1)
                    for _ in range(self.num_robots)]

        # Predict Q-values
        q_values = self.model.predict(state.reshape(1, -1))[0]

        # Extract best action for each robot
        actions = []
        for robot_idx in range(self.num_robots):
            start_idx = robot_idx * self.n_actions_per_robot
            end_idx = start_idx + self.n_actions_per_robot
            robot_q = q_values[start_idx:end_idx]
            actions.append(int(np.argmax(robot_q)))

        return actions

    def train_step(self, batch_size=32):
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions_list, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Predict current Q-values
        current_q = self.model.predict(states)

        # Predict next Q-values
        next_q = self.model.predict(next_states)

        # Update Q-values using Bellman equation with gradient clipping
        target_q = current_q.copy()

        for i in range(batch_size):
            for robot_idx in range(self.num_robots):
                action_idx = actions_list[i][robot_idx]
                q_idx = robot_idx * self.n_actions_per_robot + action_idx

                if dones[i]:
                    target_q[i, q_idx] = rewards[i]
                else:
                    # Q(s,a) = r + gamma * max_a' Q(s',a')
                    start_idx = robot_idx * self.n_actions_per_robot
                    end_idx = start_idx + self.n_actions_per_robot
                    max_next_q = np.max(next_q[i, start_idx:end_idx])

                    # Clip the TD target to prevent extreme updates (more lenient)
                    td_target = rewards[i] + self.gamma * max_next_q
                    td_target = np.clip(td_target, -100000, 100000)  # Less aggressive clipping
                    target_q[i, q_idx] = td_target

        # Train
        self.model.partial_fit(states, target_q)

        # Calculate loss
        loss = np.mean((current_q - target_q) ** 2)
        return loss

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decay_learning_rate(self):
        """Decay learning rate after each episode for stability."""
        if self.current_lr > self.min_lr:
            self.current_lr *= self.lr_decay
            # Update the learning rate in the model
            # Note: MLPRegressor doesn't expose lr directly, so we track it
            # The actual effect happens through warm_start and reinitialization
            self.model.learning_rate_init = self.current_lr


def train_swarm_agent(n_episodes=500, max_steps=3000, eval_every=50, verbose=True):
    """
    Train Q-learning agent for multi-robot search.

    Args:
        n_episodes: Number of training episodes (20x longer episodes for long-horizon learning)
        max_steps: Maximum steps per episode
        eval_every: Evaluate every N episodes
        verbose: Print progress
    """
    import warnings
    warnings.filterwarnings('ignore', category=Warning)

    # Create game (FAST VERSION: small grid, fewer robots)
    game = MultiRobotSearchGame(
        world_size_nmi=50.0,
        grid_size=50,  # Fast: 50x50 instead of 500x500
        num_robots=3,  # Fewer robots for speed
        sensor_range=20.0,
        sensor_fov_degrees=90,
        decay_rate=0.95,  # Faster decay (5% per timestep)
        ingress_formation='clump',
        colormap='inferno',
        num_priority_blobs=5,  # Variable 5-12 now in game code
        seed=42
    )

    # Create temporary agent to get state dimension
    temp_agent = SwarmQLearningAgent(
        state_dim=1,  # Dummy value
        num_robots=game.num_robots,
        hidden_layers=(256, 128, 64)  # Larger network for complex task
    )

    # Get state dimension
    game.reset()
    sample_state = temp_agent.get_state_representation(game)
    state_dim = len(sample_state)

    # Create actual agent with correct state dimension
    agent = SwarmQLearningAgent(
        state_dim=state_dim,
        num_robots=game.num_robots,
        hidden_layers=(256, 128, 64)  # Larger network
    )

    episode_rewards = []
    episode_scores = []
    eval_scores = []

    if verbose:
        print("="*70)
        print("MULTI-ROBOT SWARM RL TRAINING")
        print("="*70)
        print(f"Episodes: {n_episodes}")
        print(f"Robots: {game.num_robots}")
        print(f"State dimension: {state_dim}")
        print(f"Architecture: {agent.model.hidden_layer_sizes}")
        print(f"Actions per robot: {agent.n_actions_per_robot}")
        print(f"Total Q-values: {game.num_robots * agent.n_actions_per_robot}")
        print("="*70)
        print()

    for episode in range(n_episodes):
        game.reset()
        state = agent.get_state_representation(game)

        episode_reward = 0
        episode_loss = []

        for step in range(max_steps):
            # Get actions
            action_indices = agent.get_actions(state, training=True)
            actions = agent.decode_actions(action_indices)

            # Take actions
            next_state_dict, reward, done, _ = game.step(actions)
            next_state = agent.get_state_representation(game)

            # Store experience
            agent.replay_buffer.add(state, action_indices, reward, next_state, done)

            # Train with larger batch size
            if len(agent.replay_buffer) >= 64:
                loss = agent.train_step(batch_size=64)
                if loss is not None:
                    episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_scores.append(game.score)
        agent.decay_epsilon()
        agent.decay_learning_rate()  # Decay learning rate for stability

        # Progress indicator
        if verbose and (episode + 1) % 10 == 0 and (episode + 1) % eval_every != 0:
            print(f"  Episode {episode+1:4d} / {n_episodes}", end='\r', flush=True)

        # Evaluation
        if (episode + 1) % eval_every == 0:
            # Test without exploration on 20 games for robust evaluation
            test_scores = []
            for test_seed in range(20):  # Increased from 5 to 20 for more robust eval
                test_game = MultiRobotSearchGame(
                    world_size_nmi=50.0,
                    grid_size=50,  # Match training grid
                    num_robots=3,  # Match training robots
                    sensor_range=20.0,
                    sensor_fov_degrees=90,
                    decay_rate=0.95,  # Match training decay
                    ingress_formation='clump',
                    colormap='inferno',
                    num_priority_blobs=5,
                    seed=10000 + test_seed
                )
                test_game.reset()
                test_state = agent.get_state_representation(test_game)

                for _ in range(max_steps):
                    action_indices = agent.get_actions(test_state, training=False)
                    actions = agent.decode_actions(action_indices)
                    _, _, done, _ = test_game.step(actions)
                    test_state = agent.get_state_representation(test_game)
                    if done:
                        break

                test_scores.append(test_game.score)

            mean_test_score = np.mean(test_scores)
            eval_scores.append(mean_test_score)

            if verbose:
                mean_loss = np.mean(episode_loss) if episode_loss else 0
                progress_pct = 100 * (episode + 1) / n_episodes
                print(f"\r{' '*80}", end='\r')
                print(f"Episode {episode+1:4d}/{n_episodes} ({progress_pct:5.1f}%) | "
                      f"Train: {game.score:8.1f} | "
                      f"Test: {mean_test_score:8.1f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"LR: {agent.current_lr:.6f} | "
                      f"Loss: {mean_loss:.4f}", flush=True)

    if verbose:
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Buffer size: {len(agent.replay_buffer)}")
        print(f"Final test score: {eval_scores[-1]:.2f}")

    return agent, episode_scores, eval_scores


if __name__ == "__main__":
    print("Starting swarm RL training...")

    # Train
    agent, train_scores, eval_scores = train_swarm_agent(
        n_episodes=500,
        max_steps=3000,  # 20x longer episodes for long-term strategy learning
        eval_every=50,
        verbose=True
    )

    # Save
    save_data = {
        'model': agent.model,
        'state_dim': agent.state_dim,
        'num_robots': agent.num_robots,
        'n_actions_per_robot': agent.n_actions_per_robot,
        'epsilon': agent.epsilon,
        'gamma': agent.gamma
    }
    joblib.dump(save_data, 'src/swarm_model_rl.joblib')
    print("\n✓ Model saved to src/swarm_model_rl.joblib")

    # Save training history
    np.savez('src/swarm_training_history.npz',
             train_scores=train_scores,
             eval_scores=eval_scores)
    print("✓ Training history saved to src/swarm_training_history.npz")
