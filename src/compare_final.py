#!/usr/bin/env python3
"""
Final Policy Comparison: 4 Core Policies
=========================================

Compare the 4 main policies:
1. Random baseline
2. Greedy heuristic (hand-coded)
3. PySR Symbolic (advantage-based formulas)
4. Neural Network (RL trained)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game import GridGame
from src.heuristics_greedy import behavior_greedy
from src.heuristics_advantage_symbolic import behavior_advantage_symbolic
import random


def evaluate_policy(policy_fn, n_games=100, verbose=False):
    """Evaluate a policy on n_games."""
    scores = []

    for seed in range(10000, 10000 + n_games):
        game = GridGame(seed=seed, green_count=50, red_disabled=True)
        state = game.reset()

        while not game.done and game.turn < 100:
            action = policy_fn(state, game)
            state, _, done, _ = game.step(action)

        scores.append(game.score)

        if verbose and (len(scores) % 20 == 0):
            print(f"  {len(scores)}/{n_games} games complete...", end='\r', flush=True)

    if verbose:
        print(f"  {n_games}/{n_games} games complete.   ")

    return np.array(scores)


# Policy wrappers
def random_policy(state, game):
    return random.randint(0, 3)

def greedy_policy(state, game):
    return behavior_greedy(state)

def symbolic_policy(state, game):
    return behavior_advantage_symbolic(state)

def nn_policy(state, game):
    if not hasattr(nn_policy, 'agent'):
        from train_rl import load_agent
        nn_policy.agent = load_agent('src/game_model_rl.joblib')
    q_values = nn_policy.agent.model.predict(state.reshape(1, -1))[0]
    return int(np.argmax(q_values[:4]))


def main():
    print("="*70)
    print("FINAL POLICY COMPARISON")
    print("="*70)
    print("\nEvaluating 4 core policies on 100 games each...\n")

    policies = [
        ("Random", random_policy),
        ("Greedy Heuristic", greedy_policy),
        ("PySR Symbolic", symbolic_policy),
        ("Neural Network", nn_policy),
    ]

    results = {}

    for name, policy_fn in policies:
        print(f"Evaluating {name}...")
        scores = evaluate_policy(policy_fn, n_games=100, verbose=True)
        results[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'scores': scores
        }
        print(f"  Mean: {results[name]['mean']:.2f} ± {results[name]['std']:.2f}\n")

    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Policy':<20} {'Mean Score':>12} {'Std Dev':>10} {'Min':>8} {'Max':>8}")
    print("-"*70)

    for name in ["Random", "Greedy Heuristic", "PySR Symbolic", "Neural Network"]:
        r = results[name]
        print(f"{name:<20} {r['mean']:>12.2f} {r['std']:>10.2f} {r['min']:>8.0f} {r['max']:>8.0f}")

    # Visual comparison
    print("\n" + "="*70)
    print("PERFORMANCE HIERARCHY")
    print("="*70)

    sorted_policies = sorted(results.items(), key=lambda x: x[1]['mean'])

    for name, r in sorted_policies:
        bar = "█" * int(r['mean'] / 20)
        print(f"  {name:<20} {r['mean']:>6.0f}  {bar}")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    nn_score = results["Neural Network"]["mean"]
    greedy_score = results["Greedy Heuristic"]["mean"]
    symbolic_score = results["PySR Symbolic"]["mean"]
    random_score = results["Random"]["mean"]

    print(f"\n1. Neural Network vs Greedy:")
    print(f"   NN: {nn_score:.0f}, Greedy: {greedy_score:.0f}")
    print(f"   Gap: +{nn_score - greedy_score:.0f} points ({100*(nn_score/greedy_score-1):.1f}% better)")

    print(f"\n2. PySR Symbolic vs Greedy:")
    print(f"   Symbolic: {symbolic_score:.0f}, Greedy: {greedy_score:.0f}")
    if abs(symbolic_score - greedy_score) < 1:
        print(f"   Result: MATCHES greedy (neurosymbolic distillation success!)")
    else:
        print(f"   Gap: {symbolic_score - greedy_score:.0f} points")

    print(f"\n3. Improvement over Random:")
    print(f"   Greedy:   {100*(greedy_score/random_score):.1f}x better")
    print(f"   Symbolic: {100*(symbolic_score/random_score):.1f}x better")
    print(f"   NN:       {100*(nn_score/random_score):.1f}x better")

    # Save results
    np.savez('src/final_comparison.npz', **{name: r['scores'] for name, r in results.items()})
    print("\n✓ Results saved to src/final_comparison.npz")
    print("="*70)


if __name__ == "__main__":
    main()
