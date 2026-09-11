#!/usr/bin/env python3
"""
Demo: Visualize Trained Swarm Agent
====================================

Run trained swarm agent and create animated GIFs comparing:
1. Random policy
2. Trained neural network
"""

import numpy as np
import sys
import os
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_search import MultiRobotSearchGame
from src.train_swarm_rl import SwarmQLearningAgent
import joblib


def load_trained_agent():
    """Load trained swarm agent."""
    save_data = joblib.load('src/swarm_model_rl.joblib')

    agent = SwarmQLearningAgent(
        state_dim=save_data['state_dim'],
        num_robots=save_data['num_robots'],
        hidden_layers=(128, 64)
    )
    agent.model = save_data['model']
    agent.epsilon = 0.0  # No exploration for demo

    return agent


def run_episode(agent, game, policy_type='random', max_steps=30, save_gif=True):
    """
    Run one episode and optionally save as GIF.

    Args:
        agent: SwarmQLearningAgent or None for random
        game: MultiRobotSearchGame instance
        policy_type: 'random' or 'trained'
        max_steps: Maximum steps
        save_gif: Whether to save frames as GIF
    """
    game.reset()
    frames = []

    # Initial frame
    if save_gif:
        img = game.render_weighted_coverage()
        frames.append(np.array(img))

    for step in range(max_steps):
        if policy_type == 'random':
            # Random actions for all robots
            actions = []
            base_speed = 2.0
            turn_amount = 0.2
            for _ in range(game.num_robots):
                action_idx = np.random.randint(0, 5)
                if action_idx == 0:
                    actions.append([-turn_amount, base_speed])
                elif action_idx == 1:
                    actions.append([0.0, base_speed])
                elif action_idx == 2:
                    actions.append([turn_amount, base_speed])
                elif action_idx == 3:
                    actions.append([0.0, base_speed * 1.5])
                else:
                    actions.append([0.0, base_speed * 0.5])
        else:
            # Trained policy
            state = agent.get_state_representation(game)
            action_indices = agent.get_actions(state, training=False)
            actions = agent.decode_actions(action_indices)

        # Step
        _, _, done, _ = game.step(actions)

        # Render
        if save_gif:
            img = game.render_weighted_coverage()
            frames.append(np.array(img))

        if done:
            break

    score = game.score

    # Save GIF
    if save_gif and len(frames) > 0:
        filename = f'output/swarm_{policy_type}_v18.gif'
        imageio.mimsave(filename, frames, duration=0.2, loop=0)
        print(f"✓ Saved {filename} (score: {score:.0f})")

    return score


def main():
    """Run comparison demo."""
    print("="*70)
    print("MULTI-ROBOT SWARM DEMO")
    print("="*70)
    print()

    # Load trained agent
    print("Loading trained agent...")
    agent = load_trained_agent()
    print(f"✓ Agent loaded (state_dim={agent.state_dim}, robots={agent.num_robots})")
    print()

    # Create game
    game = MultiRobotSearchGame(
        world_size_nmi=50.0,
        grid_size=50,
        num_robots=3,
        sensor_range=20.0,
        sensor_fov_degrees=90,
        decay_rate=0.97,
        ingress_formation='clump',
        colormap='inferno',
        num_priority_blobs=5,
        seed=42
    )

    # Run random policy
    print("Running random policy...")
    random_score = run_episode(None, game, policy_type='random', max_steps=30, save_gif=True)

    # Run trained policy
    print("Running trained policy...")
    game.seed = 42  # Same seed for fair comparison
    trained_score = run_episode(agent, game, policy_type='trained', max_steps=30, save_gif=True)

    # Summary
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Random policy:  {random_score:>10.0f}")
    print(f"Trained policy: {trained_score:>10.0f}")
    print(f"Improvement:    {trained_score - random_score:>10.0f} (+{100*(trained_score/random_score - 1):.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
