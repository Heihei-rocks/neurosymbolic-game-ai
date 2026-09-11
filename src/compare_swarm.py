#!/usr/bin/env python3
"""
Compare Swarm Policies: Random vs Trained
==========================================

Evaluate both policies over multiple test games.
"""

import numpy as np
import sys
import os

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
    agent.epsilon = 0.0  # No exploration

    return agent


def evaluate_policy(agent, policy_type='random', n_games=20, max_steps=30):
    """Evaluate policy over multiple games."""
    scores = []

    for seed in range(10000, 10000 + n_games):
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
            seed=seed
        )
        game.reset()

        for step in range(max_steps):
            if policy_type == 'random':
                # Random actions
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

            _, _, done, _ = game.step(actions)
            if done:
                break

        scores.append(game.score)

        if (len(scores) % 5 == 0):
            print(f"  {len(scores)}/{n_games} games complete...", end='\r', flush=True)

    print(f"  {n_games}/{n_games} games complete.   ")
    return np.array(scores)


def main():
    print("="*70)
    print("SWARM POLICY COMPARISON (20 games each)")
    print("="*70)
    print()

    # Load agent
    print("Loading trained agent...")
    agent = load_trained_agent()
    print("✓ Agent loaded")
    print()

    # Evaluate random
    print("Evaluating random policy...")
    random_scores = evaluate_policy(None, policy_type='random', n_games=20, max_steps=30)

    # Evaluate trained
    print("Evaluating trained policy...")
    trained_scores = evaluate_policy(agent, policy_type='trained', n_games=20, max_steps=30)

    # Results
    print()
    print("="*70)
    print("RESULTS (20 games each)")
    print("="*70)
    print(f"{'Policy':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*70)
    print(f"{'Random':<20} {np.mean(random_scores):>10.0f} {np.std(random_scores):>10.0f} "
          f"{np.min(random_scores):>10.0f} {np.max(random_scores):>10.0f}")
    print(f"{'Trained':<20} {np.mean(trained_scores):>10.0f} {np.std(trained_scores):>10.0f} "
          f"{np.min(trained_scores):>10.0f} {np.max(trained_scores):>10.0f}")
    print("-"*70)

    improvement = np.mean(trained_scores) - np.mean(random_scores)
    pct_improvement = 100 * (np.mean(trained_scores) / np.mean(random_scores) - 1)
    print(f"\nImprovement: +{improvement:.0f} points ({pct_improvement:.1f}% better)")
    print("="*70)


if __name__ == "__main__":
    main()
