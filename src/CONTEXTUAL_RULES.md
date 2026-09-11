# Contextual Symbolic Rules from PySR

These formulas are context-specific, creating a piecewise policy.

## UP

### Adjacent to green
```
square(24.203041 - x10)
```
- Samples: 230
- Loss: 1071.29
- Complexity: 4

### Near target
```
square(x10 + -24.392841)
```
- Samples: 980
- Loss: 632.44
- Complexity: 4

### Early game
```
square(x10 + -24.425756)
```
- Samples: 690
- Loss: 847.49
- Complexity: 4

## DOWN

### Adjacent to green
```
square(x16 / -0.04451029)
```
- Samples: 230
- Loss: 774.13
- Complexity: 4

### Near target
```
square(x16 * 21.570568)
```
- Samples: 980
- Loss: 771.78
- Complexity: 4

### Early game
```
square(x16 / -0.046386044)
```
- Samples: 690
- Loss: 1044.01
- Complexity: 4

## LEFT

### Adjacent to green
```
square(23.874056 - x10)
```
- Samples: 230
- Loss: 513.31
- Complexity: 4

### Near target
```
square(24.199335 - x10)
```
- Samples: 980
- Loss: 653.59
- Complexity: 4

### Early game
```
square(x10 + -24.226412)
```
- Samples: 690
- Loss: 891.74
- Complexity: 4

## RIGHT

### Adjacent to green
```
square(x16 * 22.776546)
```
- Samples: 230
- Loss: 642.33
- Complexity: 4

### Near target
```
square(x16 / 0.045663495)
```
- Samples: 980
- Loss: 836.46
- Complexity: 4

### Early game
```
square(x10 + -24.160254)
```
- Samples: 690
- Loss: 572.28
- Complexity: 4

