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
        self.prev_dist_to_nearest = 0  # Track previous distance for reward shaping

    def get_state(self):
        """
        Enhanced state with 22 features including reward counter, distance delta, and multiple targets.

        Returns:
            numpy array with:
            [0-1]   pos_x, pos_y
            [2-5]   green_N/S/E/W (adjacent)
            [6-9]   red_N/S/E/W (adjacent)
            [10]    remaining boxes
            [11]    euclidean distance to nearest
            [12-13] direction vector (dx, dy) to nearest
            [14]    angle to nearest
            [15]    manhattan distance to nearest
            [16]    reward_counter (TIME PRESSURE - CRITICAL!)
            [17]    distance_delta (am I getting closer? - CRITICAL!)
            [18-21] 2nd and 3rd nearest targets (dx, dy)
        """
        gn = int((self.x, self.y-1) in self.green and (self.x, self.y-1) not in self.collected_green)
        gs = int((self.x, self.y+1) in self.green and (self.x, self.y+1) not in self.collected_green)
        ge = int((self.x+1, self.y) in self.green and (self.x+1, self.y) not in self.collected_green)
        gw = int((self.x-1, self.y) in self.green and (self.x-1, self.y) not in self.collected_green)
        rn = int((self.x, self.y-1) in self.red and (self.x, self.y-1) not in self.collected_red) if self.red_enabled else 0
        rs = int((self.x, self.y+1) in self.red and (self.x, self.y+1) not in self.collected_red) if self.red_enabled else 0
        re = int((self.x+1, self.y) in self.red and (self.x+1, self.y) not in self.collected_red) if self.red_enabled else 0
        rw = int((self.x-1, self.y) in self.red and (self.x-1, self.y) not in self.collected_red) if self.red_enabled else 0

        # Get sorted list of uncollected greens by distance
        greens = [(gx, gy) for gx, gy in self.green if (gx, gy) not in self.collected_green]

        if greens:
            # Sort by distance to get nearest 3
            sorted_greens = sorted(greens, key=lambda b: (b[0]-self.x)**2 + (b[1]-self.y)**2)

            # Nearest green
            gx, gy = sorted_greens[0]
            dist = ((gx-self.x)**2 + (gy-self.y)**2)**0.5
            dx = (gx - self.x) / self.w
            dy = (gy - self.y) / self.h
            import math
            angle = math.atan2(dy, dx) / math.pi
            manhattan = (abs(gx - self.x) + abs(gy - self.y)) / (self.w + self.h)

            # Distance delta (am I getting closer?)
            dist_delta = (self.prev_dist_to_nearest - dist) / 32

            # 2nd nearest green (if exists)
            if len(sorted_greens) > 1:
                g2x, g2y = sorted_greens[1]
                dx2 = (g2x - self.x) / self.w
                dy2 = (g2y - self.y) / self.h
            else:
                dx2 = dy2 = 0

            # 3rd nearest green (if exists)
            if len(sorted_greens) > 2:
                g3x, g3y = sorted_greens[2]
                dx3 = (g3x - self.x) / self.w
                dy3 = (g3y - self.y) / self.h
            else:
                dx3 = dy3 = 0

        else:
            dist = dx = dy = angle = manhattan = 0
            dist_delta = 0
            dx2 = dy2 = dx3 = dy3 = 0

        return np.array([
            self.x/self.w, self.y/self.h,                    # [0-1] Position
            gn, gs, ge, gw,                                  # [2-5] Green adjacent
            rn, rs, re, rw,                                  # [6-9] Red adjacent
            len(self.green)-len(self.collected_green),       # [10] Remaining
            dist/32,                                         # [11] Distance to nearest
            dx, dy,                                          # [12-13] Direction to nearest
            angle,                                           # [14] Angle
            manhattan,                                       # [15] Manhattan
            self.reward_counter / 100,                       # [16] ⭐ TIME PRESSURE
            dist_delta,                                      # [17] ⭐ Distance change
            dx2, dy2,                                        # [18-19] 2nd nearest
            dx3, dy3                                         # [20-21] 3rd nearest
        ], dtype=np.float32)

    def step(self, action):
        if self.done or self.reward_counter <= 0:
            return self.get_state(), 0, True, {'turn': self.turn, 'score': self.score}

        # Calculate distance to nearest green BEFORE move
        greens = [(gx, gy) for gx, gy in self.green if (gx, gy) not in self.collected_green]
        if greens:
            gx, gy = min(greens, key=lambda b: (b[0]-self.x)**2 + (b[1]-self.y)**2)
            dist_before = ((gx-self.x)**2 + (gy-self.y)**2)**0.5
        else:
            dist_before = 0

        # Execute move
        reward = self.reward_counter
        self.reward_counter = max(0, self.reward_counter - 1)
        dx, dy = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
        old_x, old_y = self.x, self.y
        self.x = max(0, min(self.w-1, self.x + dx))
        self.y = max(0, min(self.h-1, self.y + dy))
        if (self.x, self.y) in self.obstacles:
            self.x, self.y = old_x, old_y

        # Calculate distance to nearest green AFTER move
        greens = [(gx, gy) for gx, gy in self.green if (gx, gy) not in self.collected_green]
        if greens:
            gx, gy = min(greens, key=lambda b: (b[0]-self.x)**2 + (b[1]-self.y)**2)
            dist_after = ((gx-self.x)**2 + (gy-self.y)**2)**0.5
        else:
            dist_after = 0

        # Store for next state
        self.prev_dist_to_nearest = dist_after

        # REWARD SHAPING: Small reward for moving closer to target
        shaped_reward = 0
        if greens:  # Only shape reward if there are boxes left
            if dist_after < dist_before:
                shaped_reward = 0.1  # Moving closer
            elif dist_after > dist_before:
                shaped_reward = -0.1  # Moving farther

        # Main reward from collecting boxes
        main_reward = 0
        if (self.x, self.y) in self.green and (self.x, self.y) not in self.collected_green:
            main_reward = reward
            self.score += reward
            self.collected_green.add((self.x, self.y))
        elif (self.x, self.y) in self.red and (self.x, self.y) not in self.collected_red:
            main_reward = -reward
            self.score -= reward
            self.collected_red.add((self.x, self.y))

        self.turn += 1
        if self.reward_counter <= 0 or len(self.collected_green) >= len(self.green):
            self.done = True

        # Total reward = main reward + shaped reward
        total_reward = main_reward + shaped_reward

        return self.get_state(), total_reward, self.done, {'turn': self.turn, 'score': self.score}

    def reset(self, seed=None):
        if seed is not None: random.seed(seed); np.random.seed(seed)
        self.__init__(seed=seed if seed else 42)

        # Initialize prev_dist_to_nearest
        greens = [(gx, gy) for gx, gy in self.green if (gx, gy) not in self.collected_green]
        if greens:
            gx, gy = min(greens, key=lambda b: (b[0]-self.x)**2 + (b[1]-self.y)**2)
            self.prev_dist_to_nearest = ((gx-self.x)**2 + (gy-self.y)**2)**0.5
        else:
            self.prev_dist_to_nearest = 0

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