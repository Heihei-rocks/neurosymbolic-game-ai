# Symbolic Rules Extracted from Neural Network

These mathematical formulas were discovered by PySR through symbolic regression.
They approximate the neural network's Q-value function for each action.

## State Features

- `x0`: pos_x
- `x1`: pos_y
- `x2`: green_N
- `x3`: green_S
- `x4`: green_E
- `x5`: green_W
- `x6`: red_N
- `x7`: red_S
- `x8`: red_E
- `x9`: red_W
- `x10`: remaining
- `x11`: dist_nearest
- `x12`: nearest_dx
- `x13`: nearest_dy
- `x14`: nearest_angle
- `x15`: nearest_manhattan
- `x16`: reward_counter
- `x17`: dist_delta
- `x18`: target2_dx
- `x19`: target2_dy
- `x20`: target3_dx
- `x21`: target3_dy

## Extracted Formulas

### UP

**Selected formula (complexity 4):**
```
Q_UP = reward_counter * 347.90
```
- **Loss:** 4216.68
- **Score:** 1.492

**Better formula (complexity 6, lower loss):**
```
Q_UP = (remaining - 24.39)²
```
- **Loss:** 619.89
- **Score:** 0.959

### DOWN

**Selected formula (complexity 4):**
```
Q_DOWN = reward_counter * 351.17
```
- **Loss:** 3414.45
- **Score:** 1.663

**Better formula (complexity 6, lower loss):**
```
Q_DOWN = (remaining - 24.56)²
```
- **Loss:** 946.84
- **Score:** 0.641

### LEFT

**Selected formula (complexity 4):**
```
Q_LEFT = reward_counter * 356.53
```
- **Loss:** 3775.01
- **Score:** 1.615

**Better formula (complexity 6, lower loss):**
```
Q_LEFT = (remaining - 24.20)²
```
- **Loss:** 640.57
- **Score:** 0.887

### RIGHT

**Selected formula (complexity 4):**
```
Q_RIGHT = reward_counter * 358.74
```
- **Loss:** 4428.80
- **Score:** 1.530

**Better formula (complexity 6, lower loss):**
```
Q_RIGHT = (remaining - 24.13)²
```
- **Loss:** 432.86
- **Score:** 1.163

## Analysis

**Complexity 4 formulas:** All actions scale with `reward_counter` (time pressure), with slight directional bias.

**Complexity 6 formulas:** All actions based on `(remaining - offset)²` where offset ≈ 24. This suggests the NN learned that when ~24 boxes remain, Q-values are minimized, and actions are evaluated based on distance from this critical point.

