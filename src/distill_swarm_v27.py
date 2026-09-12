#!/usr/bin/env python3
"""
Neurosymbolic Distillation for Multi-Robot Swarm (v0.27)
========================================================

Extracts symbolic rules from trained neural network using PySR.
Uses advantage-based regression and feature selection for interpretability.
"""

import numpy as np
import sys
import os
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_search import MultiRobotSearchGame
from src.train_swarm_rl import SwarmQLearningAgent

# Check for PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
    print("✓ PySR is available")
except ImportError:
    print("✗ PySR not available - install with: pip install pysr")
    print("  Also requires Julia: https://julialang.org/downloads/")
    PYSR_AVAILABLE = False


def load_trained_agent():
    """Load the trained swarm agent."""
    print("\nLoading trained agent...")
    save_data = joblib.load('src/swarm_model_rl.joblib')

    agent = SwarmQLearningAgent(
        state_dim=save_data['state_dim'],
        num_robots=save_data['num_robots'],
        hidden_layers=(256, 128, 64)
    )
    agent.model = save_data['model']
    agent.epsilon = 0.0  # No exploration

    print(f"✓ Agent loaded:")
    print(f"  State dim: {agent.state_dim}")
    print(f"  Robots: {agent.num_robots}")
    print(f"  Actions/robot: {agent.n_actions_per_robot}")

    return agent


def collect_data(agent, n_episodes=50, max_steps=150):
    """
    Collect (state, Q-values) pairs from diverse scenarios.

    Args:
        agent: Trained agent
        n_episodes: Number of episodes to sample
        max_steps: Steps per episode

    Returns:
        states: (N, 218) array
        q_values: (N, 15) array
    """
    print(f"\nCollecting data from {n_episodes} episodes...")

    states_list = []
    q_values_list = []

    for episode in range(n_episodes):
        # Create game with random seed and priority map
        game = MultiRobotSearchGame(
            world_size_nmi=50.0,
            grid_size=50,
            num_robots=3,
            sensor_range=20.0,
            sensor_fov_degrees=90,
            decay_rate=0.95,
            ingress_formation='clump',
            colormap='inferno',
            num_priority_blobs=np.random.randint(8, 16),  # Random complexity
            seed=10000 + episode
        )

        game.reset()

        for step in range(max_steps):
            # Get state and Q-values
            state = agent.get_state_representation(game)
            q_values = agent.model.predict(state.reshape(1, -1))[0]

            states_list.append(state)
            q_values_list.append(q_values)

            # Take action
            action_indices = agent.get_actions(state, training=False)
            actions = agent.decode_actions(action_indices)
            _, _, done, _ = game.step(actions)

            if done:
                break

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{n_episodes} complete ({len(states_list)} samples)")

    states = np.array(states_list)
    q_values = np.array(q_values_list)

    print(f"\n✓ Collected {len(states)} samples")
    print(f"  State shape: {states.shape}")
    print(f"  Q-values shape: {q_values.shape}")

    return states, q_values


def compute_feature_importance(states, q_values, top_k=20):
    """
    Compute feature importance using variance in Q-values.

    Simple heuristic: Features that have high correlation with Q-value
    variance are likely important for decision-making.
    """
    print(f"\nComputing feature importance (selecting top {top_k})...")

    # Compute Q-value variance across actions
    q_variance = np.var(q_values, axis=1)

    # Compute correlation between each feature and Q-variance
    importances = []
    for i in range(states.shape[1]):
        feature_values = states[:, i]
        # Correlation between feature and Q-variance
        corr = np.abs(np.corrcoef(feature_values, q_variance)[0, 1])
        importances.append((i, corr))

    # Sort by importance
    importances.sort(key=lambda x: x[1], reverse=True)

    top_features = [idx for idx, _ in importances[:top_k]]

    print(f"  Top {top_k} features by Q-variance correlation:")
    for rank, (idx, score) in enumerate(importances[:top_k], 1):
        print(f"    {rank:2d}. Feature {idx:3d}: {score:.4f}")

    return top_features


def get_feature_names():
    """Get human-readable names for features."""
    names = []

    # Robot states (9)
    for i in range(3):
        names.extend([f"robot{i+1}_x", f"robot{i+1}_y", f"robot{i+1}_heading"])

    # Coverage stats (8)
    names.extend(["cov_mean", "cov_std", "cov_min", "cov_max"])
    names.extend(["wcov_mean", "wcov_std", "wcov_min", "wcov_max"])

    # Downsampled maps (128)
    for i in range(64):
        names.append(f"coverage_map_{i}")
    for i in range(64):
        names.append(f"priority_map_{i}")

    # Inter-robot distances (3)
    names.extend(["dist_r1_r2", "dist_r1_r3", "dist_r2_r3"])

    # Target directions (12)
    for i in range(3):
        names.extend([f"robot{i+1}_target_dx", f"robot{i+1}_target_dy",
                      f"robot{i+1}_target_dist", f"robot{i+1}_target_bearing"])

    # Coverage gradients (12)
    for i in range(3):
        names.extend([f"robot{i+1}_grad_N", f"robot{i+1}_grad_S",
                      f"robot{i+1}_grad_E", f"robot{i+1}_grad_W"])

    # Global features (3)
    names.extend(["remaining_work", "spatial_entropy", "score_velocity"])

    # Teammate vectors (9)
    for i in range(3):
        names.extend([f"robot{i+1}_teammate_dx", f"robot{i+1}_teammate_dy",
                      f"robot{i+1}_teammate_dist"])

    # Pad to 218 if needed
    while len(names) < 218:
        names.append(f"unknown_{len(names)}")

    return names[:218]


def compute_advantages(q_values):
    """Compute advantage values: A(s,a) = Q(s,a) - mean(Q(s,:))"""
    print("\nComputing advantages...")
    q_mean = q_values.mean(axis=1, keepdims=True)
    advantages = q_values - q_mean
    print(f"  Advantage range: [{advantages.min():.2f}, {advantages.max():.2f}]")
    print(f"  Mean: {advantages.mean():.6f} (should be ~0)")
    return advantages


def run_pysr_distillation(states, advantages, robot_idx, feature_indices, feature_names):
    """
    Run PySR distillation for one robot's actions.

    Args:
        states: Full state array (N, 218)
        advantages: Advantage values (N, 15)
        robot_idx: Robot index (0, 1, 2)
        feature_indices: Indices of selected features
        feature_names: Names of all features
    """
    if not PYSR_AVAILABLE:
        print("  ✗ PySR not available, skipping")
        return None

    # Extract features
    X = states[:, feature_indices]
    selected_names = [feature_names[i] for i in feature_indices]

    print(f"\n{'='*70}")
    print(f"DISTILLING ROBOT {robot_idx + 1}")
    print('='*70)
    print(f"  State features: {len(feature_indices)}")
    print(f"  Selected features:")
    for i, (idx, name) in enumerate(zip(feature_indices, selected_names)):
        print(f"    x{i}: {name} (orig: x{idx})")

    action_names = ['turn_left', 'straight', 'turn_right', 'speed_up', 'slow_down']
    results = {}

    for action_idx, action_name in enumerate(action_names):
        # Get advantages for this robot's action
        q_idx = robot_idx * 5 + action_idx
        y = advantages[:, q_idx]

        print(f"\n  Action: {action_name}")
        print(f"  Advantage range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  Training PySR (200 iterations, max 20 min)...")

        model = PySRRegressor(
            niterations=200,
            binary_operators=["+", "-", "*", "/", "max", "min"],
            unary_operators=["square", "abs", "sqrt", "exp", "log", "sin", "cos"],
            populations=30,
            population_size=100,
            maxsize=25,
            parsimony=0.005,
            timeout_in_seconds=1200,  # 20 minutes
            verbosity=0,
            random_state=42
        )

        try:
            model.fit(X, y)

            # Get best equation
            best = model.equations_.iloc[model.equations_['score'].idxmax()]

            print(f"  ✓ Found equation:")
            print(f"    {best['equation']}")
            print(f"    Loss: {best['loss']:.4f}, Complexity: {best['complexity']}")

            results[action_name] = {
                'equation': str(best['equation']),
                'loss': float(best['loss']),
                'complexity': int(best['complexity']),
                'score': float(best['score']),
                'feature_names': selected_names
            }

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[action_name] = None

    return results


def main():
    print("="*70)
    print("NEUROSYMBOLIC DISTILLATION FOR MULTI-ROBOT SWARM")
    print("="*70)

    # Load agent
    agent = load_trained_agent()

    # Collect data
    states, q_values = collect_data(agent, n_episodes=50, max_steps=150)

    # Save raw data
    np.savez('src/distillation_data_v27.npz', states=states, q_values=q_values)
    print("\n✓ Raw data saved to src/distillation_data_v27.npz")

    # Compute advantages
    advantages = compute_advantages(q_values)

    # Get feature names
    feature_names = get_feature_names()

    # Compute feature importance
    top_features = compute_feature_importance(states, q_values, top_k=20)

    # Run PySR for each robot
    all_results = {}
    for robot_idx in range(3):
        robot_results = run_pysr_distillation(
            states, advantages, robot_idx, top_features, feature_names
        )
        all_results[f'robot_{robot_idx + 1}'] = robot_results

    # Save results
    np.savez('src/distilled_rules_v27.npz', **all_results, top_features=top_features)
    print("\n" + "="*70)
    print("DISTILLATION COMPLETE")
    print("="*70)
    print("  Results saved to: src/distilled_rules_v27.npz")

    # Generate summary report
    with open('src/SYMBOLIC_RULES_v27.md', 'w') as f:
        f.write("# Symbolic Rules from Neural Network Distillation (v0.27)\n\n")
        f.write("Extracted using PySR advantage-based regression.\n\n")
        f.write("## Selected Features\n\n")
        for i, idx in enumerate(top_features):
            f.write(f"- `x{i}`: {feature_names[idx]}\n")
        f.write("\n## Rules by Robot\n\n")

        for robot_name, robot_results in all_results.items():
            if robot_results:
                f.write(f"### {robot_name}\n\n")
                for action_name, rule in robot_results.items():
                    if rule:
                        f.write(f"#### {action_name}\n\n")
                        f.write(f"```\nA_{action_name}(state) = {rule['equation']}\n```\n\n")
                        f.write(f"- Loss: {rule['loss']:.4f}\n")
                        f.write(f"- Complexity: {rule['complexity']}\n\n")

    print("  Summary saved to: src/SYMBOLIC_RULES_v27.md")
    print("\nDone!")


if __name__ == "__main__":
    if not PYSR_AVAILABLE:
        print("\nERROR: PySR is required but not installed.")
        print("Install with: pip install pysr")
        print("Note: Also requires Julia runtime from https://julialang.org")
        sys.exit(1)

    main()
