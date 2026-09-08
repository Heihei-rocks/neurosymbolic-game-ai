#!/usr/bin/env python3
"""
Train simple model for Grid Game using scikit-learn
==================================

Fast neural network equivalent using MLPClassifier.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Add path to import game
import sys
sys.path.insert(0, '/Users/djohnson334/neurosymbolic-game-ai')

from game import GridGame, generate_training_data

def main():
    print("=" * 50)
    print("TRAINING MODEL FOR GRID GAME")
    print("=" * 50)
    
    # Generate training data
    print("\nGenerating training data...")
    states, actions, rewards = generate_training_data(2000)
    print(f"Generated: {states.shape[0]} samples")
    
    # Train a simple MLP (fast neural network)
    print("\nTraining MLP classifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        verbose=False
    )
    
    model.fit(states, actions)
    print(f"Training accuracy: {model.score(states, actions):.3f}")
    
    # Evaluate on new games
    print("\nEvaluating on new games...")
    wins = 0
    total_score = 0
    n_games = 20
    
    for i in range(n_games):
        game = GridGame(seed=i*1000)
        state = game.reset()
        
        while not game.done:
            action = model.predict(state.reshape(1, -1))[0]
            state, reward, done, info = game.step(action)
            total_score += reward
        
        if total_score > 0:
            wins += 1
    
    print(f"  Wins: {wins}/{n_games} ({100*wins/n_games:.0f}%)")
    print(f"  Avg score: {total_score/n_games:.2f}")
    
    # Save model
    print("\nSaving model and data...")
    
    import joblib
    joblib.dump(model, 'game_model.joblib')
    print("Saved: game_model.joblib")
    
    # Save training data for distillation
    np.savez('training_data.npz', states=states, actions=actions)
    print("Saved: training_data.npz")
    
    # Game info
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
    
    # Also save as sklearn model
    np.savez('model.npz', coefs_=model.coefs_, intercepts_=model.intercepts_)
    print("Saved: model.npz")
    
    print("\nModel training complete!")
    return model, states, actions

if __name__ == "__main__":
    main()