#!/usr/bin/env python3
"""
Plot Multi-Robot Swarm Training Progress
=========================================

Visualize training and evaluation scores from swarm RL training.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_progress():
    """Plot training curves from saved history."""

    # Load training history
    data = np.load('src/swarm_training_history.npz')
    train_scores = data['train_scores']
    eval_scores = data['eval_scores']

    # Episodes
    train_episodes = np.arange(1, len(train_scores) + 1)
    # Eval episodes: need to match the actual eval_scores length
    eval_interval = len(train_scores) // len(eval_scores) if len(eval_scores) > 0 else 50
    eval_episodes = np.arange(eval_interval, len(train_scores) + 1, eval_interval)[:len(eval_scores)]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot scores
    ax.plot(train_episodes, train_scores, 'b-', alpha=0.3, linewidth=0.5, label='Training Score')
    ax.plot(eval_episodes, eval_scores, 'r-o', linewidth=2, markersize=6, label='Test Score (20 games avg)', zorder=5)

    # Smoothed training curve
    window = 25
    if len(train_scores) >= window:
        smoothed = np.convolve(train_scores, np.ones(window)/window, mode='valid')
        ax.plot(train_episodes[window-1:], smoothed, 'darkblue', linewidth=2, label=f'Training (smoothed, {window}-episode avg)')

    # Add baseline reference
    ax.axhline(y=eval_scores[0], color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Baseline: {eval_scores[0]:,.0f}')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score (Priority-Weighted Coverage)', fontsize=12)
    ax.set_title('v0.27 Alpha: Multi-Robot Swarm Training Progress\n20x Longer Episodes (3000 steps) + Enhanced Cognitive Features (218 state dims)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add improvement annotation
    final_eval = eval_scores[-1]
    improvement_pct = 100 * (final_eval / eval_scores[0] - 1)
    ax.text(0.98, 0.05, f'Final: {final_eval:,.0f}\n+{improvement_pct:.1f}% improvement',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig('output/swarm_training_progress_v27.png', dpi=150, bbox_inches='tight')
    print("✓ Training plot saved to output/swarm_training_progress_v27.png")

    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total episodes: {len(train_scores)}")
    print(f"Initial eval score: {eval_scores[0]:.0f}")
    print(f"Final eval score: {eval_scores[-1]:.0f}")
    print(f"Improvement: +{eval_scores[-1] - eval_scores[0]:.0f} ({100*(eval_scores[-1]/eval_scores[0] - 1):.1f}%)")
    print(f"Best eval score: {np.max(eval_scores):.0f} (episode {eval_episodes[np.argmax(eval_scores)]})")
    print(f"Final training score: {train_scores[-1]:.0f}")
    print("="*60)


if __name__ == "__main__":
    plot_training_progress()
