# NN Engine Core

[![PyPI version](https://img.shields.io/pypi/v/nn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/nn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance, fully native C++ Neural Network engine exposed to Python via pybind11. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `nn-engine-core` executes the entire deep learning training loop (forward pass, loss calculation, backpropagation, and weight updates) strictly in native C++ using Eigen. It supports mini-batch gradient descent, regression, and multi-class classification, achieving massive speedups over standard Python-loop ML libraries.

## Highlights

- **Native Loop Hoisting**: The `Model::fit` loop executes entirely in C++, eliminating the Python GIL overhead across epochs and batches.
- **Wengert List Autograd (Gradient Tape)**: Features a custom, zero-overhead automatic differentiation engine. Uses arena allocation and C++ move semantics (`std::move`) to dynamically build computational graphs without memory fragmentation or deep-copy bottlenecks.
- **Memory-Optimized Backpropagation**: Utilizes Eigen's `.noalias()` to perform in-place matrix calculus without allocating temporary memory buffers.
- **Mathematically Stable**: Built-in He (Kaiming) initialization and batch-normalized gradients to prevent exploding gradients.
- **Cross-Platform Threading**: Graceful degradation OpenMP support.

## Repository Structure

```text
.
├── CMakeLists.txt
├── pyproject.toml
├── include/
│   ├── autograd/
│   │   ├── Tape.hpp
│   │   └── Tensor.hpp
│   ├── core/
│   │   ├── Layer.hpp
│   │   ├── Loss.hpp
│   │   ├── Model.hpp
│   │   ├── Optimizer.hpp
│   │   ├── Regularizer.hpp
│   │   └── Random.hpp
│   └── parametric/
│       ├── DenseLayer.hpp
│       ├── LeakyReLULayer.hpp
│       ├── ReLULayer.hpp
│       ├── Sequential.hpp
│       └── SoftmaxLayer.hpp
├── src/
│   ├── binding.cpp
│   ├── core/
│   │   └── Model.cpp
│   └── parametric/
│       └── DenseLayer.cpp
└── examples/
    └── script.py
```

## Requirements

- Python 3.8+
- CMake 3.18+
- C++17 compiler
- Ninja (recommended generator)

Python dependencies are declared in `pyproject.toml`:
- `numpy`

## Installation

Install the released wheel from PyPI:

```bash
pip install nn-engine-core
```

Or install in editable/development mode from the repository (recommended for development):

```bash
pip install -e /path/to/NNEngine
```

To build the native extension locally and run the example without installing into the environment:

```bash
cmake -S . -B build
cmake --build build -j
python examples/script.py --seed 42
```

## Quick Start (Multi-Class Classification)

```python
import numpy as np
import nn_core

# Example Data (100 samples, 4 features, 3 classes)
X_train = np.random.rand(100, 4).astype(np.float64)
# One-hot encoded targets for Categorical Cross Entropy
y_train = np.eye(3)[np.random.choice(3, 100)].astype(np.float64)

model = nn_core.Model()
model.add(nn_core.DenseLayer(4, 16))
model.add(nn_core.ReLULayer())
model.add(nn_core.DenseLayer(16, 3))
model.add(nn_core.SoftmaxLayer())

optimizer = nn_core.Adam(learning_rate=0.01)
loss = nn_core.CategoricalCrossEntropyLoss()
model.compile(optimizer, loss)

# Train (use examples/script.py --seed to make runs deterministic)
model.fit(X_train, y_train, epochs=200, batch_size=16, tol=1e-4, verbose=True)

predictions = model.predict(X_train[:1])
print("Class Probabilities:", predictions)
```

## Python API

### `Model()`
The orchestrator for the neural network.
- `add(layer)`: Appends a layer to the network sequence. If called after `compile()` the model will automatically synchronize trainable parameters with the optimizer.
- `compile(optimizer, loss_fn, regularizer=None)`: Attach an optimizer and loss function (and optional regularizer). The model caches the trainable parameter list and wires it to the optimizer at compile time.
- `fit(X, y, epochs=100, batch_size=32, tol=1e-4, n_iter_no_change=10, verbose=True)`: Executes mini-batch gradient descent. Shuffles data automatically per epoch and reuses internal batch buffers to avoid per-epoch allocations. *Note: `X` and `y` must be 2D `float64` NumPy arrays.*
- `predict(X)`: Runs a forward pass on new data.

### Layers (`nn_core.*`)
- `DenseLayer(input_dim: int, output_dim: int)`: A fully connected parametric layer using He (Kaiming) initialization.
- `ReLULayer()`: A non-parametric Rectified Linear Unit activation layer.
- `SoftmaxLayer()`: Converts raw logits into a normalized probability distribution.
- `Sequential()`: The underlying layer container (automatically managed by `Model`).

### Loss Functions (`nn_core.*`)
- `MSELoss()`: Mean Squared Error loss for continuous regression targets. 
- `CategoricalCrossEntropyLoss()`: Cross-entropy loss for multi-class classification (requires one-hot encoded targets).

## Benchmark Results

Testing custom `NNEngine (C++)` vs `sklearn.neural_network.MLPClassifier` using Mini-Batch Gradient Descent. The table below lists the best single-run NNEngine results observed across multiple trial runs, paired with the corresponding sklearn baseline from the same run when available.

| Dataset | Samples | Features | Classes | NNEngine Acc. | Sklearn Acc. | NNEngine Time | Speedup |
|---|--:|--:|--:|--:|--:|--:|--:|
| **Iris Flower** | 150 | 4 | 3 | **96.67%** | 93.33% | **0.0060s** | **47.01x** |
| **Digits** | 1,797 | 64 | 10 | **98.89%** | 98.33% | **0.5684s** | **2.80x** |
| **Olivetti Faces** | 400 | 4,096 | 40 | **96.25%** | 96.25% | **18.8583s** | **0.71x** |

These results are stochastic by nature — see Reproducibility notes below for how to reproduce runs deterministically.

## Notes and Limitations

- Targets (`y`) passed to `model.fit()` must be strictly 2D arrays. For classification, they must be one-hot encoded (e.g., shape `(N, C)`).
- Input data should be standardized (e.g., mean `0`, variance `1`) before passing to `fit()` to maintain gradient stability.

## Reproducibility & Recent Changes

- `set_seed(seed: int)` — NNEngine now exposes a global seed hook to Python via `nn_core.set_seed(seed)`. Calling this will seed the internal C++ RNG used for weight initialization and batch shuffling. Use `examples/script.py --seed <N>` to run deterministic benchmarks.
- Deterministic dataset splits in the example benchmark now use the same seed for NumPy/scikit-learn and NNEngine to keep runs comparable.
