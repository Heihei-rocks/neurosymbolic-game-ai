# Symbolic Rules from Neural Network Distillation (v0.27)

Extracted using PySR advantage-based regression.

## Selected Features

- `x0`: unknown_206
- `x1`: cov_mean
- `x2`: cov_max
- `x3`: wcov_mean
- `x4`: coverage_map_29
- `x5`: unknown_209
- `x6`: robot2_teammate_dist
- `x7`: coverage_map_30
- `x8`: robot1_x
- `x9`: coverage_map_38
- `x10`: robot3_x
- `x11`: unknown_211
- `x12`: coverage_map_39
- `x13`: coverage_map_51
- `x14`: unknown_207
- `x15`: coverage_map_42
- `x16`: coverage_map_47
- `x17`: coverage_map_21
- `x18`: robot2_x
- `x19`: coverage_map_28

## Rules by Robot

### robot_1

#### turn_left

```
A_turn_left(state) = cos(x9) * -388.0985
```

- Loss: 5730.9890
- Complexity: 4

#### straight

```
A_straight(state) = x0 * -422.14233
```

- Loss: 4071.8550
- Complexity: 3

#### turn_right

```
A_turn_right(state) = exp(sin(exp(max(x12, x1))) / 0.16699207)
```

- Loss: 12816.9900
- Complexity: 8

#### speed_up

```
A_speed_up(state) = (x7 - 0.79357696) * 177.53784
```

- Loss: 3374.0444
- Complexity: 5

#### slow_down

```
A_slow_down(state) = -183.38803 / cos(x0)
```

- Loss: 1346.7821
- Complexity: 4

### robot_2

#### turn_left

```
A_turn_left(state) = 128.60725 / square(max(x1, x7))
```

- Loss: 13624.8750
- Complexity: 6

#### straight

```
A_straight(state) = -41.91525 / max(x16, x7)
```

- Loss: 1111.6917
- Complexity: 5

#### turn_right

```
A_turn_right(state) = (x4 + -0.75773156) * 260.92316
```

- Loss: 5599.3354
- Complexity: 5

#### speed_up

```
A_speed_up(state) = x7 * 201.75586
```

- Loss: 3832.6740
- Complexity: 3

#### slow_down

```
A_slow_down(state) = sqrt(x0) * -226.98158
```

- Loss: 584.1365
- Complexity: 4

### robot_3

#### turn_left

```
A_turn_left(state) = 177.132 / max(x7, x12)
```

- Loss: 18275.4380
- Complexity: 5

#### straight

```
A_straight(state) = exp(x12 * 4.428987)
```

- Loss: 6852.1080
- Complexity: 4

#### turn_right

```
A_turn_right(state) = x6 * -186.24332
```

- Loss: 5644.7935
- Complexity: 3

#### speed_up

```
A_speed_up(state) = x7 * 140.02557
```

- Loss: 2546.1814
- Complexity: 3

#### slow_down

```
A_slow_down(state) = (x12 - x6) * -71.959465
```

- Loss: 2886.6902
- Complexity: 5

