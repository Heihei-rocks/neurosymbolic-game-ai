#!/usr/bin/env python3
"""
Train Neural Network for Grid Game
==================================

Simple DQN-style network trained to play the grid game.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle

# Add path to import game
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')

from game import GridGame, generate_training_data

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class GameAgent(nn.Module):
    """Simple neural network for game policy."""
    
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# TRAINING
# ============================================================================

def train_agent(n_episodes=2000, batch_size=64, lr=0.01):
    """Train the agent on the grid game."""
    
    print("=" * 50)
    print("TRAINING NEURAL NETWORK")
    print("=" * 50)
    
    # Create model
    model = GameAgent(input_dim=15, hidden_dim=64, output_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Generate training data
    print(f"\nGenerating training data from {n_episodes} episodes...")
    states, actions, _ = generate_training_data(n_episodes)
    
    # Split into train/eval
    n_train = int(0.8 * len(states))
    X_train = torch.tensor(states[:n_train], dtype=torch.float32)
    y_train = torch.tensor(actions[:n_train], dtype=torch.long)
    X_eval = torch.tensor(states[n_train:], dtype=torch.float32)
    y_eval = torch.tensor(actions[n_train:], dtype=torch.long)
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Eval data: {len(X_eval)} samples")
    
    # Training loop
    losses = []
    for episode in range(100):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
        
        # Evaluate every 10 episodes
        if episode % 10 == 0:
            model.eval()
            with torch.no_grad():
                eval_outputs = model(X_eval)
                eval_preds = torch.argmax(eval_outputs, dim=1)
                accuracy = (eval_preds == y_eval).float().mean().item()
            
            print(f"Episode {episode:3d} | Loss: {avg_loss:.4f} | Eval Acc: {accuracy:.3f}")
    
    return model, losses


def evaluate_trained_model(model, n_games=50):
    """Evaluate trained model on games."""
    print("\n" + "=" * 50)
    print("EVALUATING TRAINED MODEL")
    print("=" * 50)
    
    model.eval()
    scores = []
    wins = 0
    
    for i in range(n_games):
        game = GridGame(seed=i*100)
        state = game.reset()
        
        total_reward = 0
        steps = 0
        
        while not game.done and steps < 200:
            with torch.no_grad():
                state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                action_logits = model(state_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            
            state, reward, done, info = game.step(action)
            total_reward += reward
            steps += 1
        
        scores.append(total_reward)
        if total_reward > 0:  # Positive score means more greens collected
            wins += 1
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{n_games} games | Avg score: {np.mean(scores):.2f} | Wins: {wins}")
    
    print(f"\nFinal Results:")
    print(f"  Average score: {np.mean(scores):.2f}")
    print(f"  Win rate: {wins}/{n_games} ({100*wins/n_games:.1f}%)")
    print(f"  Score range: [{min(scores):.1f}, {max(scores):.1f}]")
    
    return scores


# ============================================================================
# SAVE MODEL AND DATA
# ============================================================================

if __name__ == "__main__":
    # Train the model
    model, losses = train_agent(n_episodes=1000)
    
    # Evaluate
    scores = evaluate_trained_model(model, n_games=30)
    
    # Save model and data for distillation
    print("\n" + "=" * 50)
    print("SAVING MODEL AND DATA")
    print("=" * 50)
    
    # Save model weights
    torch.save(model.state_dict(), 'model_weights.pt')
    print("Saved: model_weights.pt")
    
    # Save some sample data for distillation
    states, actions, _ = generate_training_data(500)
    np.savez('training_data.npz', states=states, actions=actions)
    print("Saved: training_data.npz")
    
    # Save game state info
    info = {
        'grid_size': 32,
        'n_green_boxes': 5,
        'n_red_boxes': 5,
        'action_names': ['UP', 'DOWN', 'LEFT', 'RIGHT'],
        'state_features': ['pos_x', 'pos_y', 'green_N', 'green_S', 'green_E', 'green_W',
                          'red_N', 'red_S', 'red_E', 'red_W', 'remaining_boxes']
    }
    with open('game_info.pkl', 'wb') as f:
        pickle.dump(info, f)
    print("Saved: game_info.pkl")
    
    # Save training metrics
    np.savez('training_metrics.npz', losses=np.array(losses), scores=np.array(scores))
    print("Saved: training_metrics.npz")
    
    print("\nTraining complete! Model ready for distillation.")