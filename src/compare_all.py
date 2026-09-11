#!/usr/bin/env python3
"""
Complete comparison: Random vs Heuristic vs RL Agent vs Distilled
==================================================================
"""

import numpy as np
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.game import GridGame
except ImportError:
    from game import GridGame

def greedy_heuristic(state, game):
    """Hand-coded greedy nearest-green heuristic."""
    greens = [(gx, gy) for gx, gy in game.green if (gx, gy) not in game.collected_green]
    if not greens:
        return random.randint(0, 3)

    gx, gy = min(greens, key=lambda b: (b[0]-game.x)**2 + (b[1]-game.y)**2)

    # Move toward nearest green
    if abs(gx - game.x) > abs(gy - game.y):
        return 2 if gx < game.x else 3  # LEFT or RIGHT
    else:
        return 0 if gy < game.y else 1  # UP or DOWN


def evaluate_all_policies(n_games=100, seed_offset=40000):
    """Evaluate all policies on same set of games."""

    print("=" * 70)
    print("POLICY COMPARISON")
    print("=" * 70)

    policies = {
        'Random': None,
        'Greedy Heuristic': None,
        'RL Agent (NN)': None,
        'Distilled Tree': None,
    }

    # Load models
    try:
        from train_rl import load_agent
        rl_agent = load_agent('src/game_model_rl.joblib')
        policies['RL Agent (NN)'] = rl_agent
        print("✓ Loaded RL agent")
    except Exception as e:
        print(f"✗ Could not load RL agent: {e}")

    try:
        import joblib
        tree_model = joblib.load('src/distilled_tree.joblib')
        policies['Distilled Tree'] = tree_model
        print("✓ Loaded distilled tree")
    except Exception as e:
        print(f"✗ Could not load distilled tree: {e}")

    print(f"\nEvaluating on {n_games} games...\n")

    results = {}

    for policy_name in policies.keys():
        scores = []
        steps = []

        for seed in range(seed_offset, seed_offset + n_games):
            game = GridGame(seed=seed, green_count=50, red_disabled=True)
            state = game.reset()

            while not game.done and game.turn < 100:
                # Select action based on policy
                if policy_name == 'Random':
                    action = random.randint(0, 3)

                elif policy_name == 'Greedy Heuristic':
                    action = greedy_heuristic(state, game)

                elif policy_name == 'RL Agent (NN)':
                    if policies[policy_name] is not None:
                        q_values = policies[policy_name].model.predict(state.reshape(1, -1))[0]
                        action = np.argmax(q_values[:4])
                    else:
                        action = random.randint(0, 3)

                elif policy_name == 'Distilled Tree':
                    if policies[policy_name] is not None:
                        action = policies[policy_name].predict(state.reshape(1, -1))[0]
                    else:
                        action = random.randint(0, 3)

                state, _, done, _ = game.step(action)

            scores.append(game.score)
            steps.append(game.turn)

        results[policy_name] = {
            'scores': np.array(scores),
            'steps': np.array(steps),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'mean_steps': np.mean(steps)
        }

    # Print results
    print("-" * 70)
    print(f"{'Policy':<20} {'Mean Score':<12} {'Std Dev':<10} {'Min':<8} {'Max':<8} {'Steps':<8}")
    print("-" * 70)

    for policy_name, res in results.items():
        print(f"{policy_name:<20} {res['mean']:>10.1f}   {res['std']:>8.1f}   "
              f"{res['min']:>6.0f}   {res['max']:>6.0f}   {res['mean_steps']:>6.1f}")

    print("-" * 70)

    # Statistical comparison
    print("\nPerformance gaps:")
    baseline = results['Greedy Heuristic']['mean']
    for policy_name, res in results.items():
        if policy_name != 'Greedy Heuristic':
            gap = res['mean'] - baseline
            pct = 100 * gap / baseline
            print(f"  {policy_name:<20} vs Heuristic: {gap:+7.1f} ({pct:+.1f}%)")

    # Save results
    np.savez('src/policy_comparison.npz', **results)
    print("\nResults saved to src/policy_comparison.npz")

    return results


if __name__ == "__main__":
    results = evaluate_all_policies(n_games=100)
