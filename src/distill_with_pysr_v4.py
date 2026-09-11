#!/usr/bin/env python3
"""
PySR v4: Direct Action Classification
======================================

Instead of predicting Q-values, predict a binary classifier:
"Is UP better than DOWN?" for state s

Then combine multiple pairwise comparisons to select best action.

This is closer to how decision trees work - learning decision boundaries.
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
    actions = data['actions']
    q_values = data['q_values']
    print(f"  Loaded {len(states)} samples")
    return states, actions, q_values


def create_pairwise_comparisons(states, q_values):
    """
    Create binary classification targets:
    For each pair of actions (i,j), is Q(s,i) > Q(s,j)?

    Returns: targets[action_pair] = 1 if action_i better, 0 otherwise
    """
    action_pairs = [
        ('UP', 'DOWN', 0, 1),
        ('UP', 'LEFT', 0, 2),
        ('UP', 'RIGHT', 0, 3),
        ('DOWN', 'LEFT', 1, 2),
        ('DOWN', 'RIGHT', 1, 3),
        ('LEFT', 'RIGHT', 2, 3),
    ]

    comparisons = {}
    for name1, name2, idx1, idx2 in action_pairs:
        # 1.0 if action1 better, 0.0 if action2 better
        target = (q_values[:, idx1] > q_values[:, idx2]).astype(float)
        comparisons[f"{name1}_vs_{name2}"] = target
        print(f"  {name1} vs {name2}: {name1} wins {100*target.mean():.1f}% of time")

    return comparisons


def run_comparison_pysr(states, target, comparison_name, max_iterations=100):
    """Run PySR to predict binary comparison."""

    print(f"\n{'='*60}")
    print(f"PySR: {comparison_name}")
    print('='*60)

    model = PySRRegressor(
        niterations=max_iterations,
        binary_operators=["+", "-", "*", "/", ">", "<"],
        unary_operators=["square", "abs", "sqrt"],
        populations=30,
        population_size=80,
        maxsize=15,
        verbosity=0,
        random_state=42,
        parsimony=0.01,
        timeout_in_seconds=200,
    )

    try:
        model.fit(states, target)
        best = model.equations_.iloc[model.equations_['score'].idxmax()]

        print(f"  ⭐ {best['equation']}")
        print(f"     Loss: {best['loss']:.4f}, Complexity: {best['complexity']}")

        # Check accuracy
        pred = model.predict(states)
        pred_binary = (pred > 0.5).astype(float)
        accuracy = (pred_binary == target).mean()
        print(f"     Accuracy: {100*accuracy:.1f}%")

        return {
            'equation': str(best['equation']),
            'loss': float(best['loss']),
            'complexity': int(best['complexity']),
            'accuracy': float(accuracy)
        }
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def main():
    print("="*60)
    print("PAIRWISE COMPARISON SYMBOLIC REGRESSION")
    print("="*60)

    states, actions, q_values = load_data()

    # Create pairwise comparison targets
    print("\nCreating pairwise comparisons...")
    comparisons = create_pairwise_comparisons(states, q_values)

    # Run PySR for each comparison
    results = {}
    for comp_name, target in comparisons.items():
        result = run_comparison_pysr(states, target, comp_name, max_iterations=50)
        if result:
            results[comp_name] = result

    # Save results
    np.savez('src/pysr_pairwise_rules.npz', **results)
    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print("  Output: src/pysr_pairwise_rules.npz")

    # Generate summary
    with open('src/PAIRWISE_RULES.md', 'w') as f:
        f.write("# Pairwise Comparison Rules\n\n")
        f.write("Binary classifiers for each action pair.\n\n")
        f.write("To use: Evaluate all 6 comparisons, use majority vote to select action.\n\n")

        for comp_name, rule in results.items():
            f.write(f"## {comp_name.replace('_', ' ')}\n\n")
            f.write(f"```\n{rule['equation']}\n```\n\n")
            f.write(f"- Accuracy: {100*rule['accuracy']:.1f}%\n")
            f.write(f"- Loss: {rule['loss']:.4f}\n")
            f.write(f"- Complexity: {rule['complexity']}\n\n")

    print("  Summary: src/PAIRWISE_RULES.md")
    print("\nDone! Pairwise comparisons may capture decision boundaries better.")


if __name__ == "__main__":
    main()
