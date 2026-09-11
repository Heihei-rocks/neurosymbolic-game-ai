#!/usr/bin/env python3
"""
PySR Symbolic Regression for Neurosymbolic Distillation
========================================================

Extract symbolic mathematical formulas from trained RL agent using PySR.
Run this in the 'ares' conda environment where PySR is installed.
"""

import numpy as np
import joblib
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
    print("✓ PySR is available")
except ImportError:
    print("✗ PySR not available. Install with: pip install pysr")
    sys.exit(1)


def load_distillation_data():
    """Load pre-collected state-action data and Q-values."""
    print("\nLoading Q-values and states...")
    try:
        data = np.load('src/qvalues_for_pysr.npz', allow_pickle=True)
        states = data['states']
        actions = data['actions']
        q_values = data['q_values']
        print(f"  Loaded {len(states)} state-action pairs")
        print(f"  Q-values shape: {q_values.shape}")
        print(f"  Q-value range: [{q_values.min():.2f}, {q_values.max():.2f}]")
        return states, actions, q_values
    except FileNotFoundError:
        print("  ✗ qvalues_for_pysr.npz not found")
        print("  Run: python src/compute_qvalues_for_pysr.py first")
        sys.exit(1)


def symbolic_regression_per_action(states, q_values_all, action_idx, action_name, max_iterations=100):
    """
    Use PySR to find symbolic formula for Q(s, action).

    This discovers human-readable mathematical expressions that approximate
    the neural network's Q-value function for a specific action.
    """
    print(f"\n{'='*60}")
    print(f"PySR: Finding formula for {action_name} Q-value")
    print('='*60)

    # Get Q-values for this action
    target_q = q_values_all[:, action_idx]

    print(f"  Target Q-value range: [{target_q.min():.2f}, {target_q.max():.2f}]")
    print(f"  Mean: {target_q.mean():.2f}, Std: {target_q.std():.2f}")

    # Configure PySR
    print(f"\n  Starting symbolic regression ({max_iterations} iterations)...")
    print("  This may take several minutes...")

    pysr_model = PySRRegressor(
        niterations=max_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cube", "sqrt", "abs", "exp", "log"],
        populations=30,
        population_size=100,
        ncycles_per_iteration=550,
        maxsize=25,
        verbosity=1,
        random_state=42,
        temp_equation_file=True,
        parsimony=0.01,  # Prefer simpler equations
        weight_optimize=0.001,
        complexity_of_operators={
            "+": 1, "-": 1, "*": 2, "/": 2,
            "square": 3, "cube": 4, "sqrt": 3, "abs": 2,
            "exp": 5, "log": 5
        },
        constraints={
            "/": (1, 1),  # Prevent division by zero issues
        },
        timeout_in_seconds=600,  # 10 minute timeout per action
    )

    try:
        pysr_model.fit(states, target_q)

        # Get equation history (sorted by complexity and accuracy)
        equations = pysr_model.equations_

        print(f"\n  Found {len(equations)} candidate equations")
        print("\n  Top 5 equations (complexity vs accuracy):")
        print("  " + "-"*80)

        for idx in range(min(5, len(equations))):
            eq = equations.iloc[idx]
            print(f"  [{idx+1}] Complexity: {eq['complexity']:2.0f} | "
                  f"Loss: {eq['loss']:.6f} | "
                  f"Score: {eq['score']:.3f}")
            print(f"      {eq['equation']}")

        # Select best equation (balance between simplicity and accuracy)
        best_idx = pysr_model.equations_['score'].idxmax()
        best_eq = pysr_model.equations_.iloc[best_idx]

        print(f"\n  ⭐ SELECTED EQUATION:")
        print(f"     Complexity: {best_eq['complexity']}")
        print(f"     Loss: {best_eq['loss']:.6f}")
        print(f"     {best_eq['equation']}")

        return {
            'action': action_name,
            'equation': str(best_eq['equation']),
            'sympy_equation': best_eq.get('sympy_format', None),
            'loss': float(best_eq['loss']),
            'complexity': int(best_eq['complexity']),
            'score': float(best_eq['score']),
            'all_equations': equations.to_dict('records')
        }

    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        return None


def main():
    print("="*60)
    print("NEUROSYMBOLIC DISTILLATION WITH PYSR")
    print("="*60)

    # Load data
    states, actions, q_values_all = load_distillation_data()

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

    print(f"\nState features ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  [{i:2d}] {name}")

    # Run symbolic regression for each action
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    symbolic_rules = {}

    for action_idx, action_name in enumerate(action_names):
        result = symbolic_regression_per_action(
            states, q_values_all, action_idx, action_name,
            max_iterations=100
        )
        if result:
            symbolic_rules[action_name] = result

    # Save results
    output_path = 'src/pysr_symbolic_rules.npz'
    np.savez(output_path, **symbolic_rules)
    print(f"\n{'='*60}")
    print("SYMBOLIC RULES SAVED")
    print('='*60)
    print(f"  Output: {output_path}")

    # Generate human-readable summary
    summary_path = 'src/SYMBOLIC_RULES.md'
    with open(summary_path, 'w') as f:
        f.write("# Symbolic Rules Extracted from Neural Network\n\n")
        f.write("These mathematical formulas were discovered by PySR through symbolic regression.\n")
        f.write("They approximate the neural network's Q-value function for each action.\n\n")
        f.write("## State Features\n\n")
        for i, name in enumerate(feature_names):
            f.write(f"- `x{i}`: {name}\n")
        f.write("\n## Extracted Formulas\n\n")

        for action_name, rule in symbolic_rules.items():
            f.write(f"### {action_name}\n\n")
            f.write(f"**Q-value formula:**\n")
            f.write(f"```\n{rule['equation']}\n```\n\n")
            f.write(f"- **Complexity:** {rule['complexity']} operations\n")
            f.write(f"- **Loss:** {rule['loss']:.6f}\n")
            f.write(f"- **Score:** {rule['score']:.3f}\n\n")

    print(f"  Summary: {summary_path}")
    print("\nDone! Use these symbolic rules to understand what the NN learned.")


if __name__ == "__main__":
    main()
