#!/usr/bin/env python3
"""Grid Game with Time Pressure - Progressive reward decrease."""
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
        self.turn = 0
        self.reward_counter = 100  # Starts at 100, decreases each turn
        
        # Place boxes randomly
        self.green = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(5)}
        self.red = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(5)}
        self.obstacles = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(30)}
        
        # Ensure no overlap
        all_boxes = list(self.green | self.red)
        while len(set(all_boxes) & self.obstacles) > 0 or len(set(all_boxes)) != len(all_boxes):
            self.obstacles = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(30)}
            all_boxes = list(self.green | self.red)

    def get_state(self):
        """11 features."""
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
    
    def get_full_state(self):
        """1024 features: flattened 32x32 board with box locations."""
        board = np.zeros((self.h, self.w), dtype=np.float32)
        for gx, gy in self.green:
            if (gx, gy) not in self.collected_green:
                board[gy, gx] = 1.0
        for rx, ry in self.red:
            if (rx, ry) not in self.collected_red:
                board[ry, rx] = -1.0
        return np.concatenate([board.flatten(), np.array([self.x/self.w, self.y/self.h])]).astype(np.float32)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        
        # Reward decreases by 1 each turn
        reward = self.reward_counter
        self.reward_counter = max(0, self.reward_counter - 1)
        
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        old_x, old_y = self.x, self.y
        self.x = max(0, min(self.w-1, self.x + dx))
        self.y = max(0, min(self.h-1, self.y + dy))
        
        if (self.x, self.y) in self.obstacles:
            self.x, self.y = old_x, old_y
            return self.get_state(), -0.1, False, {'blocked': True}
        
        if (self.x, self.y) in self.green and (self.x, self.y) not in self.collected_green:
            self.score += reward
            self.collected_green.add((self.x, self.y))
        elif (self.x, self.y) in self.red and (self.x, self.y) not in self.collected_red:
            self.score -= reward
            self.collected_red.add((self.x, self.y))
        
        self.turn += 1
        if self.reward_counter <= 0 or len(self.collected_green) >= len(self.green):
            self.done = True
        
        return self.get_state(), reward - 1, self.done, {'turn': self.turn}

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.__init__(seed=seed if seed else 42)
        return self.get_state()


def generate_data(n_eps=50):
    """Generate state-action pairs from random play."""
    states, actions, rewards = [], [], []
    for seed in range(n_eps):
        g = GridGame(seed=seed)
        s = g.reset()
        while not g.done and g.turn < 100:
            a = random.randint(0, 3)
            ns, r, d, info = g.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = ns
    return np.array(states), np.array(actions), np.array(rewards)

generate_training_data = generate_data

if __name__ == "__main__":
    print("Testing game with progressive rewards...")
    g = GridGame(seed=42)
    s = g.reset()
    print(f"State shape: {s.shape}")
    total = 0
    for _ in range(5):
        s, r, d, info = g.step(0)  # UP
        total += r
        print(f"Reward: {r}, Total: {total}, Turn: {info['turn']}")
    print("Done!")