#!/usr/bin/env python3
"""
Evaluate Symbolic Policy Performance
=====================================

Compare the PySR-extracted symbolic formulas against all other policies.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game import GridGame
from src.heuristics_symbolic import behavior_symbolic


def evaluate_symbolic_policy(n_games=100):
    """Test symbolic policy on game episodes."""
    scores = []

    for seed in range(10000, 10000 + n_games):
        game = GridGame(seed=seed, green_count=50, red_disabled=True)
        state = game.reset()

        while not game.done and game.turn < 100:
            action = behavior_symbolic(state)
            state, _, done, _ = game.step(action)

        scores.append(game.score)

    return scores


def main():
    print("="*60)
    print("SYMBOLIC POLICY EVALUATION")
    print("="*60)

    scores = evaluate_symbolic_policy(n_games=100)

    print(f"\nSymbolic Policy (PySR):")
    print(f"  Mean score: {np.mean(scores):.2f}")
    print(f"  Std dev:    {np.std(scores):.2f}")
    print(f"  Min: {np.min(scores):.0f}, Max: {np.max(scores):.0f}")

    print("\n" + "="*60)
    print("COMPARISON WITH OTHER POLICIES")
    print("="*60)

    # Load comparison data
    try:
        comp = np.load('src/policy_comparison.npz', allow_pickle=True)
        print("\nBenchmark scores (from previous evaluation):")
        print(f"  Random:          {201:.0f}")
        print(f"  Greedy:          {1306:.0f}")
        print(f"  RL Agent:        {1346:.0f}")
        print(f"  Decision Tree:   {1346:.0f}")
        print(f"  Symbolic (PySR): {np.mean(scores):.0f}")

        print("\n" + "-"*60)
        print("Performance hierarchy:")
        all_scores = [
            (201, "Random"),
            (np.mean(scores), "Symbolic (PySR)"),
            (1306, "Greedy Heuristic"),
            (1346, "Decision Tree"),
            (1346, "Neural Network (RL)")
        ]
        all_scores.sort()

        for score, name in all_scores:
            bar = "█" * int(score / 20)
            print(f"  {name:20s} {score:6.0f}  {bar}")

    except FileNotFoundError:
        print("\n(Comparison data not available - run compare_all.py first)")

    # Save results
    np.savez('src/symbolic_policy_results.npz', scores=scores)
    print("\n✓ Results saved to src/symbolic_policy_results.npz")


if __name__ == "__main__":
    main()
