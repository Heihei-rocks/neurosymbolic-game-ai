#!/usr/bin/env python3
"""Game with improved state representation for NN."""
import random, numpy as np

class GridGame:
    def __init__(self, width=32, height=32, seed=42, red_disabled=True, green_count=50):
        random.seed(seed); np.random.seed(seed)
        self.w, self.h = width, height
        self.x, self.y = width // 2, height // 2
        self.score, self.collected_green, self.collected_red = 0, set(), set()
        self.done, self.turn, self.reward_counter = False, 0, 100
        self.red_enabled = not red_disabled
        self.green = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(green_count)}
        self.red = {(random.randint(0,self.w-1), random.randint(0,self.h-1)) for _ in range(5)} if self.red_enabled else set()
        self.obstacles = set()

    def get_state(self):
        """12 features: pos, nearby boxes, green count, dist to nearest green."""
        gn = int((self.x, self.y-1) in self.green and (self.x, self.y-1) not in self.collected_green)
        gs = int((self.x, self.y+1) in self.green and (self.x, self.y+1) not in self.collected_green)
        ge = int((self.x+1, self.y) in self.green and (self.x+1, self.y) not in self.collected_green)
        gw = int((self.x-1, self.y) in self.green and (self.x-1, self.y) not in self.collected_green)
        rn = int((self.x, self.y-1) in self.red and (self.x, self.y-1) not in self.collected_red) if self.red_enabled else 0
        rs = int((self.x, self.y+1) in self.red and (self.x, self.y+1) not in self.collected_red) if self.red_enabled else 0
        re = int((self.x+1, self.y) in self.red and (self.x+1, self.y) not in self.collected_red) if self.red_enabled else 0
        rw = int((self.x-1, self.y) in self.red and (self.x-1, self.y) not in self.collected_red) if self.red_enabled else 0
        
        greens = [(gx, gy) for gx, gy in self.green if (gx, gy) not in self.collected_green]
        if greens:
            gx, gy = min(greens, key=lambda b: (b[0]-self.x)**2 + (b[1]-self.y)**2)
            dist = ((gx-self.x)**2 + (gy-self.y)**2)**0.5
        else:
            dist = 0
        
        return np.array([self.x/self.w, self.y/self.h, gn, gs, ge, gw, rn, rs, re, rw, len(self.green)-len(self.collected_green), dist/32], dtype=np.float32)

    def step(self, action):
        if self.done or self.reward_counter <= 0: return self.get_state(), 0, True, {}
        reward = self.reward_counter
        self.reward_counter = max(0, self.reward_counter - 1)
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        old_x, old_y = self.x, self.y
        self.x = max(0, min(self.w-1, self.x + dx))
        self.y = max(0, min(self.h-1, self.y + dy))
        if (self.x, self.y) in self.obstacles: self.x, self.y = old_x, old_y
        if (self.x, self.y) in self.green and (self.x, self.y) not in self.collected_green:
            self.score += reward; self.collected_green.add((self.x, self.y))
        elif (self.x, self.y) in self.red and (self.x, self.y) not in self.collected_red:
            self.score -= reward; self.collected_red.add((self.x, self.y))
        self.turn += 1
        if self.reward_counter <= 0 or len(self.collected_green) >= len(self.green): self.done = True
        return self.get_state(), reward-1, self.done, {'turn': self.turn, 'score': self.score}

    def reset(self, seed=None):
        if seed is not None: random.seed(seed); np.random.seed(seed)
        self.__init__(seed=seed if seed else 42)
        return self.get_state()


def generate_training_data(n_eps=100, red_disabled=True, green_count=50):
    """Generate state-action pairs."""
    states, actions = [], []
    for seed in range(n_eps):
        g = GridGame(seed=seed, green_count=green_count, red_disabled=red_disabled)
        s = g.reset()
        while not g.done and g.turn < 60:
            # Greedy toward nearest green
            greens = [(gx, gy) for gx, gy in g.green if (gx, gy) not in g.collected_green]
            a = random.randint(0,3)
            if greens:
                gx, gy = min(greens, key=lambda b: (b[0]-g.x)**2 + (b[1]-g.y)**2)
                a = 2 if gx < g.x else (3 if gx > g.x else (0 if gy < g.y else 1))
            states.append(s.copy())
            actions.append(a)
            s, _, d, _ = g.step(a)
    return np.array(states), np.array(actions)

generate = generate_training_data

if __name__ == "__main__":
    print("Testing improved game...")
    g = GridGame(seed=42, green_count=50, red_disabled=True)
    s = g.reset()
    print(f"State shape: {s.shape} (12 features)")
    print(f"Green: {len(g.green)}, Red: {len(g.red)}")