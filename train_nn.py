#!/usr/bin/env python3
"""Train neural network - minimal version."""
import numpy as np, random, joblib
from sklearn.neural_network import MLPClassifier

# Generate training data
states, actions = [], []
for seed in range(500):
    random.seed(seed)
    x, y = 16, 16
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    cg, cr = set(), set()
    reward = 100
    
    for _ in range(50):
        board = np.zeros(1024)
        for gx, gy in green:
            if (gx, gy) not in cg: board[gy*32+gx] = 1
        for rx, ry in red:
            if (rx, ry) not in cr: board[ry*32+rx] = -1
        state = np.concatenate([board, [x/32, y/32]])
        action = random.randint(0,3)
        states.append(state)
        actions.append(action)
        
        dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[action]
        x, y = max(0,min(31,x+dx)), max(0,min(31,y+dy))
        if (x,y) in green: cg.add((x,y))
        elif (x,y) in red: cr.add((x,y))
        reward -= 1

states = np.array(states); actions = np.array(actions)
print(f"Data: {states.shape}")

model = MLPClassifier(hidden_layer_sizes=(16,16), max_iter=100, random_state=42)
model.fit(states, actions)
print(f"Train acc: {model.score(states, actions):.3f}")

joblib.dump(model, 'game_model.joblib')
print("Saved!")

# Quick eval
def eval_game(s, policy):
    random.seed(s); np.random.seed(s)
    x, y = 16, 16
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    cg, cr = set(), set()
    score = 0; reward = 100
    
    for _ in range(50):
        board = np.zeros(1024)
        for gx, gy in green:
            if (gx, gy) not in cg: board[gy*32+gx] = 1
        for rx, ry in red:
            if (rx, ry) not in cr: board[ry*32+rx] = -1
        state = np.concatenate([board, [x/32, y/32]])
        
        a = random.randint(0,3) if policy=='random' else model.predict([state])[0]
        dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[a]
        x, y = max(0,min(31,x+dx)), max(0,min(31,y+dy))
        if (x,y) in green: score += reward; cg.add((x,y))
        elif (x,y) in red: score -= reward; cr.add((x,y))
        reward -= 1
    return score

r = [eval_game(i, 'random') for i in range(20)]
n = [eval_game(i, 'nn') for i in range(20)]
print(f"Random: {np.mean(r):.2f}, NN: {np.mean(n):.2f}")

np.savez('results.npz', random_score=np.mean(r), nn_score=np.mean(n))