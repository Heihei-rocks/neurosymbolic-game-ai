# Neurosymbolic Distillation Research Summary

## Core Papers

### 1. AI Feynman (MIT, 2019)
- **Paper**: "AI Feynman: a Physics-Inspired Method for Symbolic Regression" (Udrescu & Tegmark)
- **ArXiv**: 1905.11481 | **Journal**: Science Advances, 2020
- **Code**: https://github.com/SJ001/AI-Feynman

#### Key Method
Physics-inspired decomposition:
1. Neural network fits the function
2. Separates into additive/multiplicative components via transform tricks
3. Extracts symbolic expression per component

#### Results
- 100 Feynman equations: **100% success** (prev: 71%)
- Hard test set: **90% success** (prev: 15%)

---

### 2. PySR (2023)
- **Paper**: "Interpretable Machine Learning for Science with PySR" (Cranmer)
- **ArXiv**: 2305.01582

#### Key Features
- Python interface to JuliaSymbolicRegression.jl backend
- Evolutionarily optimized symbolic expressions
- Multi-population genetic programming

---

### 3. SymTorch (2026)
- **Paper**: "SymTorch: A Framework for Symbolic Distillation of Deep Neural Networks"
- **ArXiv**: 2602.21307 (Tan, Soubki, Cranmer)

#### For Game AI
```python
import torch
from symtorch import SymbolicModule

# Distill trained MLP to symbolic rules
model = SymbolicModule.from_nn(trained_model, X_train, y_train)
print(model.sym_expr)  # Human-readable equation!
```

#### Results
- LLM MLP replacement: **8.3% throughput improvement**  
- Works with transformers, GNNs, PINNs

---

### 4. Neurosymbolic RL Agents (2024)
- **Paper**: "Interpretable end-to-end Neurosymbolic Reinforcement Learning agents"
- **ArXiv**: 2410.14371

#### Architecture
```
[Observations] → [NN Encoder] → [Concepts] → [Rule Policy] → [Actions]
```

- **Concept bottlenecks** for intermediate interpretability
- **Policy distillation** from NN to symbolic

---

## Research Landscape

### Approaches
| Method | Training | Extraction | Primary Use |
|--------|----------|------------|-------------|
| AI Feynman | Physics-inspired transforms | Component-wise separation | Physics equations |
| PySR | Genetic programming | Multi-population evolution | Science discovery |
| SymTorch | **Distillation from trained NN** | PySR backend | **Game AI, LLM acceleration** |
| Concept Bottlenecks | Joint training | Post-hoc analysis | RL agents |

### Trend: **Distillation-First**
Previous work often trained symbolic models directly. **Newest trend**: Train neural net first, then distill to symbolic (SymTorch, Jacobian regularization).

---

## Python Tools

### PySR (Ready to Use)
```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "*", "-", "/", "^"],
    unary_operators=["sin", "cos", "exp", "log", "sqrt"],
    equations_file="equations.csv"
)
model.fit(X, y)
print(model.sym_expr)
```

### SymTorch (Newest - PyTorch Plugin)
- Direct PyTorch integration
- Automatic data collection and caching
- GPU-CPU transfer handling

### Install Commands
```bash
pip install pysr  # Includes Julia backend
pip install sympy torch scikit-learn
```

---

## Application to Your Game AI

### Your Model: `game_model.joblib`
- **11 features**: position, adjacent boxes, green count, distance
- **Architecture**: MLP with hidden layers (you can check exact size)

### Distillation Workflow
1. Load trained NumPy/Joblib model
2. Generate input-output dataset from game interactions
3. Run PySR/SymTorch to extract symbolic policy
4. Get **human-readable game heuristic**!

### Expected Output
Instead of `action = NN.predict(state)`, get:
```
IF green_nearby AND distance < 3 THEN move_toward_green
ELIF red_adjacent THEN move_away
ELSE random_move
```

---

## Key Citations (BibTeX Ready)

```bibtex
@article{udrescu2020ai,
  title={AI Feynman: a Physics-Inspired Method for Symbolic Regression},
  author={Udrescu, Silviu-Marian and Tegmark, Max},
  journal={Science Advances},
  year={2020},
  publisher={arXiv}
}

@article{cranmer2023pysr,
  title={Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl},
  author={Cranmer, Miles},
  year={2023},
  journal={arXiv},
  url={https://arxiv.org/abs/2305.01582}
}

@article{tan2026symtorch,
  title={SymTorch: A Framework for Symbolic Distillation of Deep Neural Networks},
  author={Tan, Elizabeth SZ and Soubki, Adil and Cranmer, Miles},
  year={2026},
  journal={arXiv},
  url={https://arxiv.org/abs/2602.21307}
}

@article{raj2024interpretable,
  title={Interpretable end-to-end Neurosymbolic Reinforcement Learning agents},
  author={Raj, ...},
  year={2024},
  journal={arXiv},
  url={https://arxiv.org/abs/2410.14371}
}
```

---

## Next Steps

1. **Load your game model** with joblib
2. **Generate dataset**: Run game, record (state, action) pairs
3. **Apply PySR**: Extract symbolic policy
4. **Compare**: Symbolic vs NN performance on game

Files saved to `/tmp/research/notes.md` with full details on each paper.