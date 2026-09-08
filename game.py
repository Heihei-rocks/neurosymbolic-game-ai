#!/usr/bin/env python3
"""
Grid Game - Minimal Version for Fast Distillation
=================================================

32x32 grid, simple movement, box collection.
"""

import random
import numpy as np

class GridGame:
    def __init__(self, width=32, height=32, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.w, self.h = width, height
        self.x, self.y = width // 2, height // 2
        self.score = 0
        self.collected_green = set()
        self.collected_red = set()
        self.done = False
        
        # Place boxes randomly
        self.green = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(5)}
        self.red = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(5)}
        self.obstacles = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(30)}

    def get_state(self):
        """11 features"""
        gn = int((self.x, self.y-1) in self.green and (self.x, self.y-1) not in self.collected_green)
        gs = int((self.x, self.y+1) in self.green and (self.x, self.y+1) not in self.collected_green)
        ge = int((self.x+1, self.y) in self.green and (self.x+1, self.y) not in self.collected_green)
        gw = int((self.x-1, self.y) in self.green and (self.x-1, self.y) not in self.collected_green)
        rn = int((self.x, self.y-1) in self.red and (self.x, self.y-1) not in self.collected_red)
        rs = int((self.x, self.y+1) in self.red and (self.x, self.y+1) not in self.collected_red)
        re = int((self.x+1, self.y) in self.red and (self.x+1, self.y) not in self.collected_red)
        rw = int((self.x-1, self.y) in self.red and (self.x-1, self.y) not in self.collected_red)
        
        return np.array([
            self.x/self.w, self.y/self.h,
            gn, gs, ge, gw,
            rn, rs, re, rw,
            len(self.green) - len(self.collected_green)
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        self.x = max(0, min(self.w-1, self.x + dx))
        self.y = max(0, min(self.h-1, self.y + dy))
        
        if (self.x, self.y) in self.obstacles:
            self.x -= dx; self.y -= dy  # revert
            return self.get_state(), -0.1, False, {'blocked': True}
        
        if (self.x, self.y) in self.green and (self.x, self.y) not in self.collected_green:
            self.score += 1
            self.collected_green.add((self.x, self.y))
        elif (self.x, self.y) in self.red and (self.x, self.y) not in self.collected_red:
            self.score -= 1
            self.collected_red.add((self.x, self.y))
        
        if len(self.collected_green) >= len(self.green):
            self.done = True
        
        return self.get_state(), self.score, self.done, {}

    def reset(self):
        self.__init__(seed=42)
        return self.get_state()


def generate_data(n_eps=50):
    """Generate state-action pairs from random play."""
    states, actions = [], []
    for seed in range(n_eps):
        g = GridGame(seed=seed)
        s = g.reset()
        steps = 0
        while not g.done and steps < 50:
            a = random.randint(0, 3)
            ns, r, d, _ = g.step(a)
            states.append(s)
            actions.append(a)
            s = ns
            steps += 1
    return np.array(states), np.array(actions)


if __name__ == "__main__":
    print("Testing game...")
    g = GridGame(seed=42)
    s = g.reset()
    print(f"State shape: {s.shape}")
    print(f"Playing 20 steps...")
    for _ in range(20):
        s, r, d, _ = g.step(1)  # DOWN
    print(f"Score: {g.score}")
    
    print("\nGenerating data...")
    states, actions = generate_data(30)
    print(f"Generated: {states.shape}")
    
    print("Done!")