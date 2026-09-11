#!/usr/bin/env python3
"""
PySR v5: Direction-Only Symbolic Regression
============================================

The decision tree uses 75% importance on directional features:
- nearest_dx (47%)
- nearest_angle (28%)

Let's train PySR on ONLY directional features to see if it can learn
the threshold logic: "move in direction of nearest box"
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


def extract_directional_features(states):
    """
    Extract only the features the decision tree found important:
    - nearest_dx (x12)
    - nearest_dy (x13)
    - nearest_angle (x14)
    - nearest_manhattan (x15)
    - target2_dx (x18)
    - target2_dy (x19)
    - target3_dx (x20)
    - target3_dy (x21)
    """
    directional_indices = [12, 13, 14, 15, 18, 19, 20, 21]
    return states[:, directional_indices]


def run_directional_pysr(states_dir, advantages, action_idx, action_name, max_iterations=150):
    """Run PySR on directional features only."""

    target = advantages[:, action_idx]

    print(f"\n{'='*60}")
    print(f"PySR Directional: {action_name}")
    print('='*60)
    print(f"  Features: nearest_dx, nearest_dy, angle, manhattan, target2/3")
    print(f"  Advantage range: [{target.min():.2f}, {target.max():.2f}]")

    model = PySRRegressor(
        niterations=max_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "abs", "sqrt", "sign"],  # sign() for thresholds!
        populations=30,
        population_size=100,
        maxsize=20,
        verbosity=1,
        random_state=42,
        parsimony=0.002,
        timeout_in_seconds=600,
    )

    try:
        model.fit(states_dir, target)

        # Show top 5
        print(f"\n  Top 5 equations:")
        for i in range(min(5, len(model.equations_))):
            eq = model.equations_.iloc[i]
            print(f"    [{i+1}] C:{eq['complexity']:2.0f} | Loss:{eq['loss']:8.2f} | {eq['equation']}")

        best = model.equations_.iloc[model.equations_['score'].idxmax()]

        print(f"\n  ⭐ SELECTED:")
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
    print("DIRECTION-ONLY SYMBOLIC REGRESSION")
    print("="*60)

    states, q_values = load_data()

    # Extract directional features only
    print("\nExtracting directional features...")
    states_dir = extract_directional_features(states)
    print(f"  Shape: {states_dir.shape} (8 directional features)")

    # Compute advantages
    advantages = q_values - q_values.mean(axis=1, keepdims=True)

    # Feature names
    feature_names = [
        'nearest_dx', 'nearest_dy', 'nearest_angle', 'nearest_manhattan',
        'target2_dx', 'target2_dy', 'target3_dx', 'target3_dy'
    ]

    print("\nDirectional features:")
    for i, name in enumerate(feature_names):
        print(f"  x{i}: {name}")

    # Run PySR for each action
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    results = {}

    for action_idx, action_name in enumerate(action_names):
        result = run_directional_pysr(states_dir, advantages, action_idx, action_name, max_iterations=150)
        if result:
            results[action_name] = result

    # Save results
    np.savez('src/pysr_directional_rules.npz', **results)
    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print("  Output: src/pysr_directional_rules.npz")

    # Generate summary
    with open('src/DIRECTIONAL_RULES.md', 'w') as f:
        f.write("# Direction-Based Symbolic Rules\n\n")
        f.write("Formulas using ONLY directional features (what decision tree uses).\n\n")
        f.write("## Features\n\n")
        for i, name in enumerate(feature_names):
            f.write(f"- `x{i}`: {name}\n")
        f.write("\n## Advantage Formulas\n\n")

        for action_name, rule in results.items():
            f.write(f"### {action_name}\n\n")
            f.write(f"```\nA_{action_name}(dirs) = {rule['equation']}\n```\n\n")
            f.write(f"- Loss: {rule['loss']:.2f}\n")
            f.write(f"- Complexity: {rule['complexity']}\n\n")

    print("  Summary: src/DIRECTIONAL_RULES.md")
    print("\nDone! This focuses on the features the decision tree found most important.")


if __name__ == "__main__":
    main()
