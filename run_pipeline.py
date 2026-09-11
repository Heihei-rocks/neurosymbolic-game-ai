#!/usr/bin/env python3
"""
Full Pipeline Runner
====================

Runs the complete training and distillation pipeline:
1. Train RL agent (5000 episodes, ~3-5 minutes)
2. Distill to symbolic rules (decision tree + PySR)
3. Compare all policies
4. Generate visualizations
"""

import sys
import time
import subprocess

def run_step(step_num, description, script_path):
    """Run a pipeline step and report timing."""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {description}")
    print("="*70)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Step {step_num} completed in {elapsed:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Step {step_num} failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False

    except KeyboardInterrupt:
        print(f"\n⚠ Step {step_num} interrupted by user")
        return False


def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║  NEUROSYMBOLIC GAME AI - FULL PIPELINE                            ║
║  Train RL agent → Extract symbolic rules → Compare performance    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    pipeline = [
        (1, "Train RL Agent (5000 episodes, ~3-5 min)", "src/train_rl.py"),
        (2, "Extract Symbolic Rules (distillation)", "src/distill_improved.py"),
        (3, "Compare All Policies", "src/compare_all.py"),
    ]

    overall_start = time.time()
    results = []

    for step_num, description, script_path in pipeline:
        success = run_step(step_num, description, script_path)
        results.append((step_num, description, success))

        if not success:
            print(f"\n⚠ Pipeline stopped at step {step_num}")
            print("You can resume by running individual scripts:")
            for s, d, p in pipeline[step_num:]:
                print(f"  python {p}")
            break
    else:
        # All steps completed
        overall_time = time.time() - overall_start
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Total time: {overall_time:.1f} seconds ({overall_time/60:.1f} minutes)")

        print("\nGenerated files:")
        print("  • src/game_model_rl.joblib      - Trained Q-learning agent")
        print("  • src/training_history.npz      - Training curves")
        print("  • src/heuristics_distilled.py   - Extracted symbolic rules")
        print("  • src/distilled_tree.joblib     - Decision tree model")
        print("  • src/distillation_results.npz  - Distillation analysis")
        print("  • src/policy_comparison.npz     - Performance comparison")

        print("\nNext steps:")
        print("  1. Check results: python -c 'import numpy as np; d=np.load(\"src/policy_comparison.npz\", allow_pickle=True); print(d.files)'")
        print("  2. View distilled rules: cat src/heuristics_distilled.py")
        print("  3. Analyze tree: python -c 'import joblib; t=joblib.load(\"src/distilled_tree.joblib\"); print(f\"Depth: {t.get_depth()}, Leaves: {t.get_n_leaves()}\")'")
        print("  4. Generate GIFs: python src/make_gifs_scored.py")

    print("\n" + "="*70)
    print("Summary:")
    for step_num, description, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} Step {step_num}: {description}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
