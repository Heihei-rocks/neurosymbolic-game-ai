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
    eval_episodes = np.arange(20, len(train_scores) + 1, 20)  # Every 20 episodes

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot scores
    ax.plot(train_episodes, train_scores, 'b-', alpha=0.3, linewidth=0.5, label='Training Score')
    ax.plot(eval_episodes, eval_scores, 'r-o', linewidth=2, markersize=4, label='Evaluation Score (5 games avg)')

    # Smoothed training curve
    window = 10
    if len(train_scores) >= window:
        smoothed = np.convolve(train_scores, np.ones(window)/window, mode='valid')
        ax.plot(train_episodes[window-1:], smoothed, 'b-', linewidth=2, label=f'Training (smoothed, window={window})')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score (Priority-Weighted Coverage)', fontsize=12)
    ax.set_title('Multi-Robot Swarm RL Training Progress\n3 Robots, 50×50 Grid, Priority-Aware Coverage',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add final score annotation
    final_eval = eval_scores[-1]
    ax.annotate(f'Final: {final_eval:.0f}',
                xy=(eval_episodes[-1], final_eval),
                xytext=(eval_episodes[-1] - 30, final_eval + 2000),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))

    plt.tight_layout()
    plt.savefig('output/swarm_training_progress_v18.png', dpi=150, bbox_inches='tight')
    print("✓ Training plot saved to output/swarm_training_progress_v18.png")

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
