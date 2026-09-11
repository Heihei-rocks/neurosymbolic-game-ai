#!/usr/bin/env python3
"""
PySR v2: Context-Aware Symbolic Regression
===========================================

Instead of finding one formula for all states, find separate formulas
for different game contexts:
1. Adjacent to green box
2. Near target (< 5 tiles away)
3. Far from target (> 5 tiles away)

This creates a piecewise symbolic policy that can be more reactive.
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


def run_contextual_pysr(states, q_values, action_idx, action_name, context_name, mask, max_iterations=100):
    """Run PySR on a specific context (subset of states)."""

    context_states = states[mask]
    context_q = q_values[mask, action_idx]

    if len(context_states) < 50:
        print(f"  ⚠️  Only {len(context_states)} samples - skipping")
        return None

    print(f"\n{'='*60}")
    print(f"PySR: {action_name} / {context_name}")
    print('='*60)
    print(f"  Training on {len(context_states)} samples")
    print(f"  Q-value range: [{context_q.min():.2f}, {context_q.max():.2f}]")

    model = PySRRegressor(
        niterations=max_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "abs", "sqrt"],
        populations=30,
        population_size=100,
        maxsize=20,
        verbosity=1,
        random_state=42,
        parsimony=0.005,  # Slightly prefer simpler formulas
        timeout_in_seconds=300,  # 5 min per context
    )

    try:
        model.fit(context_states, context_q)
        best = model.equations_.iloc[model.equations_['score'].idxmax()]

        print(f"\n  ⭐ BEST EQUATION:")
        print(f"     {best['equation']}")
        print(f"     Loss: {best['loss']:.2f}, Complexity: {best['complexity']}")

        return {
            'equation': str(best['equation']),
            'loss': float(best['loss']),
            'complexity': int(best['complexity']),
            'n_samples': len(context_states),
            'context': context_name
        }
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def main():
    print("="*60)
    print("CONTEXTUAL SYMBOLIC REGRESSION WITH PYSR")
    print("="*60)

    states, q_values = load_data()

    # Define contexts
    print("\n" + "="*60)
    print("DEFINING CONTEXTS")
    print("="*60)

    # Context 1: Adjacent to green box
    green_adjacent = (states[:, 2:6].sum(axis=1) > 0)
    print(f"\n1. Adjacent to green: {green_adjacent.sum()} samples ({100*green_adjacent.mean():.1f}%)")

    # Context 2: Near target (manhattan distance < 5)
    near_target = (states[:, 15] < 0.15)  # x15 = nearest_manhattan (normalized)
    print(f"2. Near target (<5 tiles): {near_target.sum()} samples ({100*near_target.mean():.1f}%)")

    # Context 3: Far from target
    far_target = (states[:, 15] >= 0.15)
    print(f"3. Far from target (>=5 tiles): {far_target.sum()} samples ({100*far_target.mean():.1f}%)")

    # Context 4: Early game (many boxes remaining)
    early_game = (states[:, 10] > 30)  # x10 = remaining
    print(f"4. Early game (>30 boxes): {early_game.sum()} samples ({100*early_game.mean():.1f}%)")

    # Context 5: Late game (few boxes remaining)
    late_game = (states[:, 10] < 10)
    print(f"5. Late game (<10 boxes): {late_game.sum()} samples ({100*late_game.mean():.1f}%)")

    # Run PySR for each action × context combination
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    contexts = [
        ('Adjacent to green', green_adjacent),
        ('Near target', near_target),
        ('Far from target', far_target),
        ('Early game', early_game),
        ('Late game', late_game),
    ]

    results = {}

    for action_idx, action_name in enumerate(action_names):
        results[action_name] = {}
        for context_name, mask in contexts:
            result = run_contextual_pysr(
                states, q_values, action_idx, action_name,
                context_name, mask, max_iterations=50
            )
            if result:
                results[action_name][context_name] = result

    # Save results
    np.savez('src/pysr_contextual_rules.npz', **results)
    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print("  Output: src/pysr_contextual_rules.npz")

    # Generate summary
    with open('src/CONTEXTUAL_RULES.md', 'w') as f:
        f.write("# Contextual Symbolic Rules from PySR\n\n")
        f.write("These formulas are context-specific, creating a piecewise policy.\n\n")

        for action_name in action_names:
            f.write(f"## {action_name}\n\n")
            for context_name, rule in results[action_name].items():
                f.write(f"### {context_name}\n")
                f.write(f"```\n{rule['equation']}\n```\n")
                f.write(f"- Samples: {rule['n_samples']}\n")
                f.write(f"- Loss: {rule['loss']:.2f}\n")
                f.write(f"- Complexity: {rule['complexity']}\n\n")

    print("  Summary: src/CONTEXTUAL_RULES.md")


if __name__ == "__main__":
    main()
