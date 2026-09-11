#!/usr/bin/env python3
"""
Reinforcement Learning Training for Grid Game
==============================================

Uses Q-learning style training with replay buffer to learn optimal policy.
This allows the NN to potentially surpass hand-coded heuristics.
"""

import numpy as np
import random
from collections import deque
import joblib
from sklearn.neural_network import MLPRegressor
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.game import GridGame
except ImportError:
    from game import GridGame

class ReplayBuffer:
    """Store and sample experience tuples."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """Q-learning agent using neural network function approximator."""

    def __init__(self, state_dim=22, n_actions=4, hidden_layers=(256, 128, 64)):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9993  # Decays to ~0.05 after ~5000 episodes

        # Q-network (predicts Q-value for each action given state)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1,  # We'll call partial_fit repeatedly
            warm_start=True,
            random_state=42,
            verbose=False  # Suppress sklearn warnings
        )

        # Initialize with dummy data
        dummy_X = np.zeros((n_actions, state_dim))
        dummy_y = np.zeros((n_actions, n_actions))
        self.model.fit(dummy_X, dummy_y)

        self.replay_buffer = ReplayBuffer()
        self.training_history = []

    def get_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Predict Q-values for all actions
        q_values = self.model.predict(state.reshape(1, -1))[0]
        return np.argmax(q_values[:self.n_actions])

    def train_step(self, batch_size=64):
        """Perform one training step on a batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Predict Q-values for current states
        current_q = self.model.predict(states)

        # Predict Q-values for next states
        next_q = self.model.predict(next_states)

        # Update Q-values using Bellman equation
        target_q = current_q.copy()
        for i in range(batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                # Q(s,a) = r + gamma * max_a' Q(s',a')
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i, :self.n_actions])

        # Train the network
        self.model.partial_fit(states, target_q)

        # Calculate loss for monitoring
        loss = np.mean((current_q - target_q) ** 2)
        return loss

    def decay_epsilon(self):
        """Decay epsilon - call once per episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(n_episodes=5000, max_steps=100, eval_every=100, verbose=True):
    """
    Train a Q-learning agent on the grid game.

    Args:
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_every: Evaluate performance every N episodes
        verbose: Print progress
    """
    import warnings
    warnings.filterwarnings('ignore', category=Warning)

    agent = QLearningAgent(state_dim=22, n_actions=4, hidden_layers=(256, 128, 64))

    episode_rewards = []
    episode_scores = []
    eval_scores = []

    if verbose:
        print("=" * 60)
        print("REINFORCEMENT LEARNING TRAINING")
        print("=" * 60)
        print(f"Episodes: {n_episodes}")
        print(f"Architecture: {agent.model.hidden_layer_sizes}")
        print(f"Replay buffer: {agent.replay_buffer.buffer.maxlen}")
        print(f"Progress updates every {eval_every} episodes")
        print("=" * 60)
        print()

    for episode in range(n_episodes):
        game = GridGame(seed=episode, green_count=50, red_disabled=True)
        state = game.reset()

        episode_reward = 0
        episode_loss = []

        for step in range(max_steps):
            # Select action
            action = agent.get_action(state, training=True)

            # Take action
            next_state, reward, done, info = game.step(action)

            # Store experience
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Train on batch
            if len(agent.replay_buffer) >= 64:
                loss = agent.train_step(batch_size=64)
                if loss is not None:
                    episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_scores.append(game.score)

        # Decay epsilon once per episode
        agent.decay_epsilon()

        # Show progress indicator
        if verbose and (episode + 1) % 10 == 0 and (episode + 1) % eval_every != 0:
            print(f"  Episode {episode+1:5d} / {n_episodes} ({100*(episode+1)/n_episodes:.1f}%)", end='\r', flush=True)

        # Evaluation
        if (episode + 1) % eval_every == 0:
            # Test on fixed seeds without exploration
            test_scores = []
            for test_seed in range(10000, 10020):  # 20 test games
                game = GridGame(seed=test_seed, green_count=50, red_disabled=True)
                state = game.reset()

                for _ in range(max_steps):
                    action = agent.get_action(state, training=False)
                    state, _, done, _ = game.step(action)
                    if done:
                        break

                test_scores.append(game.score)

            mean_test_score = np.mean(test_scores)
            eval_scores.append(mean_test_score)

            if verbose:
                mean_loss = np.mean(episode_loss) if episode_loss else 0
                progress_pct = 100 * (episode + 1) / n_episodes
                print(f"\r{'':80}", end='\r')  # Clear progress line
                print(f"Episode {episode+1:5d}/{n_episodes} ({progress_pct:5.1f}%) | "
                      f"Train: {game.score:6.1f} | "
                      f"Test: {mean_test_score:6.1f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Loss: {mean_loss:.4f} | "
                      f"Buf: {len(agent.replay_buffer)}", flush=True)

    if verbose:
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Replay buffer size: {len(agent.replay_buffer)}")
        print(f"Final test score: {eval_scores[-1]:.2f}")

    return agent, episode_scores, eval_scores


def evaluate_agent(agent, n_games=100, verbose=True):
    """Evaluate trained agent."""
    scores = []

    for seed in range(10000, 10000 + n_games):
        game = GridGame(seed=seed, green_count=50, red_disabled=True)
        state = game.reset()

        while not game.done and game.turn < 100:
            action = agent.get_action(state, training=False)
            state, _, done, _ = game.step(action)

        scores.append(game.score)

    if verbose:
        print(f"\nEvaluation on {n_games} games:")
        print(f"  Mean score: {np.mean(scores):.2f}")
        print(f"  Std dev: {np.std(scores):.2f}")
        print(f"  Min: {np.min(scores):.0f}, Max: {np.max(scores):.0f}")

    return scores


def save_agent(agent, filepath='src/game_model_rl.joblib'):
    """Save trained agent."""
    model_data = {
        'model': agent.model,
        'state_dim': agent.state_dim,
        'n_actions': agent.n_actions,
        'epsilon': agent.epsilon,
        'gamma': agent.gamma
    }
    joblib.dump(model_data, filepath)
    print(f"Agent saved to {filepath}")


def load_agent(filepath='src/game_model_rl.joblib'):
    """Load trained agent."""
    model_data = joblib.load(filepath)
    agent = QLearningAgent(
        state_dim=model_data['state_dim'],
        n_actions=model_data['n_actions']
    )
    agent.model = model_data['model']
    agent.epsilon = model_data.get('epsilon', 0.05)
    agent.gamma = model_data.get('gamma', 0.95)
    return agent


if __name__ == "__main__":
    # Train the agent
    agent, train_scores, eval_scores = train_agent(
        n_episodes=5000,
        max_steps=100,
        eval_every=100,
        verbose=True
    )

    # Final evaluation
    final_scores = evaluate_agent(agent, n_games=100)

    # Save the trained agent
    save_agent(agent, 'src/game_model_rl.joblib')

    # Save training history
    np.savez('src/training_history.npz',
             train_scores=train_scores,
             eval_scores=eval_scores,
             final_scores=final_scores)

    print("\nTraining data saved to src/training_history.npz")
