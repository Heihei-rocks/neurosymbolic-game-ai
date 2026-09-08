#!/usr/bin/env python3
"""Final evaluation with time pressure - simple version."""
import numpy as np, random, joblib

# Load model trained with 11 features
try:
    model = joblib.load('game_model.joblib')
    print(f"Model loaded, expecting {model.n_features_in_} features")
except:
    # Create new model
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(16,16))
    states, actions = [], []
    for seed in range(300):
        random.seed(seed)
        x, y, cg, cr = 16, 16, set(), set()
        green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
        red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
        for _ in range(30):
            gn = int((x, y-1) in green and (x, y-1) not in cg)
            gs = int((x, y+1) in green and (x, y+1) not in cg)
            ge = int((x+1, y) in green and (x+1, y) not in cg)
            gw = int((x-1, y) in green and (x-1, y) not in cg)
            rn = int((x, y-1) in red and (x, y-1) not in cr)
            rs = int((x, y+1) in red and (x, y+1) not in cr)
            re = int((x+1, y) in red and (x+1, y) not in cr)
            rw = int((x-1, y) in red and (x-1, y) not in cr)
            state = [x/32, y/32, gn, gs, ge, gw, rn, rs, re, rw, 5-len(cg)]
            actions.append(random.randint(0,3))
            states.append(state)
            dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[actions[-1]]
            x, y = max(0,min(31,x+dx)), max(0,min(31,y+dy))
            if (x,y) in green: cg.add((x,y))
            elif (x,y) in red: cr.add((x,y))
    model.fit(states, actions)
    joblib.dump(model, 'game_model.joblib')

def eval_policy(seed, policy):
    random.seed(seed); np.random.seed(seed)
    x, y, cg, cr, score, reward = 16, 16, set(), set(), 0, 100
    green = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    red = {(random.randint(0,31), random.randint(0,31)) for _ in range(5)}
    
    for _ in range(50):
        if reward <= 0: break
        gn = int((x, y-1) in green and (x, y-1) not in cg)
        gs = int((x, y+1) in green and (x, y+1) not in cg)
        ge = int((x+1, y) in green and (x+1, y) not in cg)
        gw = int((x-1, y) in green and (x-1, y) not in cg)
        rn = int((x, y-1) in red and (x, y-1) not in cr)
        rs = int((x, y+1) in red and (x, y+1) not in cr)
        re = int((x+1, y) in red and (x+1, y) not in cr)
        rw = int((x-1, y) in red and (x-1, y) not in cr)
        state = [x/32, y/32, gn, gs, ge, gw, rn, rs, re, rw, 5-len(cg)]
        
        if policy == 'random': a = random.randint(0,3)
        else: a = model.predict([state])[0]
        
        dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[a]
        x, y = max(0,min(31,x+dx)), max(0,min(31,y+dy))
        if (x,y) in green: score += reward; cg.add((x,y))
        elif (x,y) in red: score -= reward; cr.add((x,y))
        reward -= 1
    return score

print("Evaluating policies (50 games each)...")
r = [eval_policy(i, 'random') for i in range(50)]
n = [eval_policy(i, 'nn') for i in range(50)]

print(f"Random: {np.mean(r):.2f} ± {np.std(r):.2f}")
print(f"NN:     {np.mean(n):.2f} ± {np.std(n):.2f}")

np.savez('results.npz', random_score=np.mean(r), nn_score=np.mean(n), 
         random_scores=r, nn_scores=n)