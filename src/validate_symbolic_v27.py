#!/usr/bin/env python3
"""
Symbolic Policy Implementation and Validation (v0.27)
=====================================================

Implements the extracted symbolic rules and compares performance
to the trained neural network.
"""

import numpy as np
import sys
import os
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_search import MultiRobotSearchGame
from src.train_swarm_rl import SwarmQLearningAgent


class SymbolicSwarmPolicy:
    """Symbolic policy using extracted formulas."""

    def __init__(self, rules_path='src/distilled_rules_v27.npz'):
        """Load extracted symbolic rules."""
        data = np.load(rules_path, allow_pickle=True)
        self.top_features = data['top_features']

        # Load rules for each robot
        self.rules = {}
        for i in range(3):
            robot_key = f'robot_{i+1}'
            self.rules[robot_key] = data[robot_key].item()

        print(f"✓ Loaded symbolic rules for 3 robots")

    def evaluate_formula(self, formula_str, feature_values):
        """
        Evaluate a symbolic formula given feature values.

        Args:
            formula_str: String like "x0 * -422.14233"
            feature_values: Dict mapping x0, x1, etc. to values
        """
        # Replace x0, x1, etc. with actual values
        expr = formula_str
        for var_name, value in feature_values.items():
            expr = expr.replace(var_name, str(value))

        # Safe evaluation with numpy functions
        safe_dict = {
            'cos': np.cos,
            'sin': np.sin,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'square': lambda x: x**2,
            'abs': np.abs,
            'max': np.maximum,
            'min': np.minimum
        }

        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return result
        except:
            return 0.0  # Fallback for invalid expressions

    def get_actions(self, full_state):
        """
        Get actions for all robots using symbolic policy.

        Args:
            full_state: Full 218-d state vector

        Returns:
            List of action indices [robot1_action, robot2_action, robot3_action]
        """
        # Extract relevant features
        selected_state = full_state[self.top_features]

        # Create feature dict (x0, x1, ..., x19)
        features = {f'x{i}': selected_state[i] for i in range(len(selected_state))}

        # Get actions for each robot
        actions = []
        action_names = ['turn_left', 'straight', 'turn_right', 'speed_up', 'slow_down']

        for robot_idx in range(3):
            robot_key = f'robot_{robot_idx + 1}'
            robot_rules = self.rules[robot_key]

            # Compute advantage for each action
            advantages = []
            for action_name in action_names:
                rule = robot_rules.get(action_name)
                if rule and rule['equation']:
                    adv = self.evaluate_formula(rule['equation'], features)
                else:
                    adv = 0.0
                advantages.append(adv)

            # Select action with highest advantage
            best_action = int(np.argmax(advantages))
            actions.append(best_action)

        return actions


def load_nn_agent():
    """Load the trained neural network agent."""
    save_data = joblib.load('src/swarm_model_rl.joblib')

    agent = SwarmQLearningAgent(
        state_dim=save_data['state_dim'],
        num_robots=save_data['num_robots'],
        hidden_layers=(256, 128, 64)
    )
    agent.model = save_data['model']
    agent.epsilon = 0.0

    return agent


def run_episode(game, policy, policy_type='symbolic', max_steps=150):
    """
    Run one episode with given policy.

    Args:
        game: MultiRobotSearchGame instance
        policy: SymbolicSwarmPolicy or SwarmQLearningAgent
        policy_type: 'symbolic' or 'neural'
        max_steps: Maximum steps

    Returns:
        score, decisions (list of action choices per timestep)
    """
    game.reset()
    decisions = []

    for step in range(max_steps):
        if policy_type == 'symbolic':
            # Symbolic policy needs state representation from agent
            state = policy.get_state_representation(game) if hasattr(policy, 'get_state_representation') else None
            if state is None:
                # Fallback: use simple state
                state = np.zeros(218)  # Dummy state

            action_indices = policy.get_actions(state)
        else:
            # Neural network
            state = policy.get_state_representation(game)
            action_indices = policy.get_actions(state, training=False)

        # Decode actions
        actions = policy.decode_actions(action_indices)
        decisions.append(action_indices)

        # Step
        _, _, done, _ = game.step(actions)

        if done:
            break

    return game.score, decisions


def compare_policies(n_episodes=100):
    """
    Compare symbolic policy vs neural network.

    Args:
        n_episodes: Number of test episodes
    """
    print("="*70)
    print("SYMBOLIC POLICY VALIDATION")
    print("="*70)
    print()

    # Load policies
    print("Loading policies...")
    nn_agent = load_nn_agent()
    symbolic_policy = SymbolicSwarmPolicy()

    # Need to give symbolic policy access to state representation
    symbolic_policy.get_state_representation = nn_agent.get_state_representation
    symbolic_policy.decode_actions = nn_agent.decode_actions

    print("✓ Both policies loaded")
    print()

    # Run comparison
    nn_scores = []
    symbolic_scores = []
    decision_agreements = []

    print(f"Running {n_episodes} episodes...")

    for episode in range(n_episodes):
        # Create game
        game = MultiRobotSearchGame(
            world_size_nmi=50.0,
            grid_size=50,
            num_robots=3,
            sensor_range=20.0,
            sensor_fov_degrees=90,
            decay_rate=0.95,
            ingress_formation='clump',
            colormap='inferno',
            num_priority_blobs=np.random.randint(8, 16),
            seed=20000 + episode
        )

        # Test NN
        score_nn, decisions_nn = run_episode(game, nn_agent, 'neural', max_steps=150)
        nn_scores.append(score_nn)

        # Test symbolic (reset same game)
        game = MultiRobotSearchGame(
            world_size_nmi=50.0,
            grid_size=50,
            num_robots=3,
            sensor_range=20.0,
            sensor_fov_degrees=90,
            decay_rate=0.95,
            ingress_formation='clump',
            colormap='inferno',
            num_priority_blobs=game.num_priority_blobs,  # Match complexity
            seed=20000 + episode  # Same seed
        )
        score_sym, decisions_sym = run_episode(game, symbolic_policy, 'symbolic', max_steps=150)
        symbolic_scores.append(score_sym)

        # Compute decision agreement
        min_len = min(len(decisions_nn), len(decisions_sym))
        if min_len > 0:
            agreements = sum(1 for i in range(min_len)
                           if decisions_nn[i] == decisions_sym[i])
            agreement_pct = 100.0 * agreements / (min_len * 3)  # 3 robots
        else:
            agreement_pct = 0.0
        decision_agreements.append(agreement_pct)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{n_episodes} complete")

    # Results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nPerformance (score on {n_episodes} episodes):")
    print(f"  Neural Network:  {np.mean(nn_scores):>12,.0f} ± {np.std(nn_scores):>8,.0f}")
    print(f"  Symbolic Policy: {np.mean(symbolic_scores):>12,.0f} ± {np.std(symbolic_scores):>8,.0f}")

    fidelity = 100.0 * np.mean(symbolic_scores) / np.mean(nn_scores)
    print(f"\n  Symbolic Fidelity: {fidelity:.1f}% of NN performance")

    print(f"\nDecision Agreement:")
    print(f"  Average: {np.mean(decision_agreements):.1f}% of decisions match")
    print(f"  Range: {np.min(decision_agreements):.1f}% - {np.max(decision_agreements):.1f}%")

    print()
    if fidelity >= 90:
        print("✅ EXCELLENT: Symbolic policy achieves >90% of NN performance!")
    elif fidelity >= 80:
        print("✅ GOOD: Symbolic policy achieves >80% of NN performance")
    elif fidelity >= 60:
        print("⚠️  VIABLE: Symbolic policy achieves >60% of NN performance")
    else:
        print("❌ POOR: Symbolic policy falls short (<60% of NN)")

    print("="*70)

    # Save results
    np.savez('src/symbolic_validation_v27.npz',
             nn_scores=nn_scores,
             symbolic_scores=symbolic_scores,
             decision_agreements=decision_agreements,
             fidelity_pct=fidelity)

    print("\n✓ Results saved to src/symbolic_validation_v27.npz")

    return fidelity, np.mean(decision_agreements)


if __name__ == "__main__":
    fidelity, agreement = compare_policies(n_episodes=100)
