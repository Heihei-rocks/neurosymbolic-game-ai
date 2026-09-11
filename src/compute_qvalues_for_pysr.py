#!/usr/bin/env python3
"""
Compute Q-values from RL agent and save for PySR.

This script runs in the base environment to extract Q-values,
then PySR can use them in the ares environment.
"""

import numpy as np
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("Loading distillation data...")
    data = np.load('src/distillation_results.npz', allow_pickle=True)
    states = data['states']
    actions = data['actions']
    print(f"  Loaded {len(states)} state-action pairs")

    print("\nLoading trained RL agent...")
    model_data = joblib.load('src/game_model_rl.joblib')
    model = model_data['model']
    print("  ✓ Loaded RL agent")

    print("\nComputing Q-values...")
    q_values = model.predict(states)
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-value range: [{q_values.min():.2f}, {q_values.max():.2f}]")

    # Save for PySR
    output_path = 'src/qvalues_for_pysr.npz'
    np.savez(output_path,
             states=states,
             actions=actions,
             q_values=q_values)
    print(f"\n✓ Saved to {output_path}")

if __name__ == "__main__":
    main()
