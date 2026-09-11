# Advantage-Based Symbolic Rules

These formulas predict A(s,a) = Q(s,a) - mean(Q(s,:))

To use: Compute advantage for each action, pick action with highest advantage.

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

## Formulas

### UP

```
A_UP(state) = (x14 * -32.683876) * x16
```

- Loss: 62.99
- Complexity: 5
- Score: 0.239

### DOWN

```
A_DOWN(state) = (x16 * (x13 / x11)) * 23.171993
```

- Loss: 24.35
- Complexity: 7
- Score: 0.430

### LEFT

```
A_LEFT(state) = x12 * -105.0549
```

- Loss: 44.96
- Complexity: 3
- Score: 0.463

### RIGHT

```
A_RIGHT(state) = x12 * 118.336235
```

- Loss: 63.00
- Complexity: 3
- Score: 0.472

