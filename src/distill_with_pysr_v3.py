#!/usr/bin/env python3
"""
PySR v3: Advantage-Based Symbolic Regression
=============================================

Instead of predicting Q(s,a), predict Advantage(s,a) = Q(s,a) - mean(Q(s,:))

This focuses PySR on learning which action is BETTER, not absolute values.
Advantage values are centered around 0, which may be easier to learn.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
    print("✓ PySR is available")
except ImportError:
    print("✗ PySR not available")
    sys.exit(1)


def load_data():
    """Load pre-computed Q-values and states."""
    print("\nLoading data...")
    data = np.load('src/qvalues_for_pysr.npz', allow_pickle=True)
    states = data['states']
    q_values = data['q_values']
    print(f"  Loaded {len(states)} samples")
    return states, q_values


def compute_advantages(q_values):
    """
    Compute advantage values: A(s,a) = Q(s,a) - mean(Q(s,:))

    Advantages are zero-centered and show which actions are better/worse
    than average for that state.
    """
    q_mean = q_values.mean(axis=1, keepdims=True)
    advantages = q_values - q_mean
    return advantages


def run_advantage_pysr(states, advantages, action_idx, action_name, max_iterations=150):
    """Run PySR on advantage values."""

    target = advantages[:, action_idx]

    print(f"\n{'='*60}")
    print(f"PySR Advantage: {action_name}")
    print('='*60)
    print(f"  Training on {len(states)} samples")
    print(f"  Advantage range: [{target.min():.2f}, {target.max():.2f}]")
    print(f"  Mean: {target.mean():.2f}, Std: {target.std():.2f}")

    model = PySRRegressor(
        niterations=max_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "abs", "sqrt", "exp"],
        populations=30,
        population_size=100,
        maxsize=25,
        verbosity=1,
        random_state=42,
        parsimony=0.003,  # Allow slightly more complex formulas
        timeout_in_seconds=600,
    )

    try:
        model.fit(states, target)

        # Show top 5
        print(f"\n  Top 5 equations:")
        for i in range(min(5, len(model.equations_))):
            eq = model.equations_.iloc[i]
            print(f"    [{i+1}] C:{eq['complexity']:2.0f} | Loss:{eq['loss']:8.2f} | {eq['equation']}")

        best = model.equations_.iloc[model.equations_['score'].idxmax()]

        print(f"\n  ⭐ SELECTED EQUATION:")
        print(f"     {best['equation']}")
        print(f"     Loss: {best['loss']:.2f}, Complexity: {best['complexity']}")

        return {
            'equation': str(best['equation']),
            'loss': float(best['loss']),
            'complexity': int(best['complexity']),
            'score': float(best['score'])
        }
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def main():
    print("="*60)
    print("ADVANTAGE-BASED SYMBOLIC REGRESSION")
    print("="*60)

    states, q_values = load_data()

    # Compute advantages
    print("\nComputing advantage values...")
    advantages = compute_advantages(q_values)
    print(f"  Advantage range: [{advantages.min():.2f}, {advantages.max():.2f}]")
    print(f"  Mean: {advantages.mean():.6f} (should be ~0)")

    # Feature names for reference
    feature_names = [
        'pos_x', 'pos_y',
        'green_N', 'green_S', 'green_E', 'green_W',
        'red_N', 'red_S', 'red_E', 'red_W',
        'remaining', 'dist_nearest',
        'nearest_dx', 'nearest_dy', 'nearest_angle', 'nearest_manhattan',
        'reward_counter', 'dist_delta',
        'target2_dx', 'target2_dy', 'target3_dx', 'target3_dy'
    ]

    print(f"\nFeatures:")
    for i, name in enumerate(feature_names):
        print(f"  x{i}: {name}")

    # Run PySR for each action
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    results = {}

    for action_idx, action_name in enumerate(action_names):
        result = run_advantage_pysr(states, advantages, action_idx, action_name, max_iterations=150)
        if result:
            results[action_name] = result

    # Save results
    np.savez('src/pysr_advantage_rules.npz', **results)
    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print("  Output: src/pysr_advantage_rules.npz")

    # Generate summary
    with open('src/ADVANTAGE_RULES.md', 'w') as f:
        f.write("# Advantage-Based Symbolic Rules\n\n")
        f.write("These formulas predict A(s,a) = Q(s,a) - mean(Q(s,:))\n\n")
        f.write("To use: Compute advantage for each action, pick action with highest advantage.\n\n")
        f.write("## State Features\n\n")
        for i, name in enumerate(feature_names):
            f.write(f"- `x{i}`: {name}\n")
        f.write("\n## Formulas\n\n")

        for action_name, rule in results.items():
            f.write(f"### {action_name}\n\n")
            f.write(f"```\nA_{action_name}(state) = {rule['equation']}\n```\n\n")
            f.write(f"- Loss: {rule['loss']:.2f}\n")
            f.write(f"- Complexity: {rule['complexity']}\n")
            f.write(f"- Score: {rule['score']:.3f}\n\n")

    print("  Summary: src/ADVANTAGE_RULES.md")
    print("\nDone! Advantage-based formulas may better capture relative action preferences.")


if __name__ == "__main__":
    main()
