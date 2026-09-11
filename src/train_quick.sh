#!/bin/bash
# Quick training script - runs training in background and shows progress

echo "Starting RL training (5000 episodes, ~3-5 minutes)..."
echo "Progress will be displayed below:"
echo ""

python src/train_rl.py
