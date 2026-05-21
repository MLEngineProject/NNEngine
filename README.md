# NN Engine Core

[![PyPI version](https://img.shields.io/pypi/v/nn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/nn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance, fully native C++ Neural Network engine exposed to Python via pybind11. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `nn-engine-core` executes the entire deep learning training loop (forward pass, loss calculation, backpropagation, and weight updates) strictly in native C++ using Eigen, achieving significant speedups over standard Python-loop ML libraries.

## Highlights

- **Native Loop Hoisting**: The `Model::fit` loop executes entirely in C++, eliminating the Python GIL overhead across epochs.
- **Memory-Optimized Backpropagation**: Utilizes Eigen's `.noalias()` to perform in-place matrix calculus without allocating temporary memory buffers.
- **Mathematically Stable**: Built-in Xavier (Glorot) initialization and batch-normalized gradients to prevent exploding gradients.
- **Cross-Platform Threading**: Graceful degradation OpenMP support. Uses multi-threading where available, gracefully falling back to single-threaded standard C++ on restricted environments (e.g., Apple Clang without `libomp`).
- **Clean Python API**: A familiar Keras/scikit-learn style interface.

## Repository Structure

```text
.
├── CMakeLists.txt
├── pyproject.toml
├── include/
│   ├── core/
│   │   ├── Layer.hpp
│   │   ├── Loss.hpp
│   │   └── Model.hpp
│   └── parametric/
│       ├── DenseLayer.hpp
│       ├── LogisticNeuron.hpp
│       ├── ReLULayer.hpp
│       └── Sequential.hpp
├── src/
│   ├── binding.cpp
│   ├── core/
│   │   └── Model.cpp
│   └── parametric/
│       ├── DenseLayer.cpp
│       ├── LogisticNeuron.cpp
│       └── ReLULayer.cpp
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

### Option 1: Install from PyPI (recommended)

```bash
pip install nn-engine-core
```

### Option 2: Install from source (editable)

From the project root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

## Quick Start

```python
import numpy as np
import nn_core

# Example Data (Scaled)
X_train = np.random.rand(100, 20).astype(np.float64)
y_train = np.random.rand(100, 1).astype(np.float64)

# 1. Initialize the Model
model = nn_core.Model()

# 2. Build Architecture
model.add(nn_core.DenseLayer(20, 64))
model.add(nn_core.ReLULayer())
model.add(nn_core.DenseLayer(64, 1))

# 3. Compile with Loss Function
model.compile(nn_core.MSELoss())

# 4. Train (Executes entirely in C++)
model.fit(X_train, y_train, epochs=200, learning_rate=0.01, verbose=True)

# 5. Predict
sample = np.random.rand(1, 20).astype(np.float64)
predictions = model.predict(sample)
print("Prediction:", predictions)
```

## Python API

### `Model()`
The orchestrator for the neural network.
- `add(layer)`: Appends a layer to the network sequence.
- `compile(loss_fn)`: Attaches a loss function to the model.
- `fit(X, y, epochs=100, learning_rate=0.01, verbose=True)`: Executes full-batch gradient descent. *Note: `X` and `y` must be 2D `float64` NumPy arrays.*
- `predict(X)`: Runs a forward pass on new data.

### Layers (`nn_core.*`)
- `DenseLayer(input_dim: int, output_dim: int)`: A fully connected parametric layer using Xavier initialization.
- `ReLULayer()`: A non-parametric Rectified Linear Unit activation layer.
- `Sequential()`: The underlying layer container (automatically managed by `Model`).

### Loss Functions (`nn_core.*`)
- `MSELoss()`: Mean Squared Error loss. Automatically normalizes gradients by batch size.

## Benchmark Results

The following results were produced using a non-linear regression dataset (`sklearn.datasets.make_regression`, 5000 samples, 20 features, noise=0.1) trained over 200 epochs via Full-Batch Gradient Descent. 

Comparing `NNEngine` against `sklearn.neural_network.MLPRegressor` natively highlights the extreme performance advantage of C++ loop hoisting and `noalias()` matrix optimization.

| Engine | MSE | Time | Speedup |
|---|---:|---:|---:|
| **NNEngine (C++)** | **0.067268** | **0.5593s** | **2.59×** |
| Scikit-Learn | 0.145411 | 1.4470s | 1.00× |

Console output:
```text
--- Testing NNEngine: Non-Linear Regression ---
Training NNEngine...
NNEngine   | MSE: 0.067268 | Time: 0.5593s

Training Scikit-Learn...
Scikit-Learn | MSE: 0.145411 | Time: 1.4470s

Speedup: 2.59x faster than Sklearn!
```

## Notes and Limitations

- Targets (`y`) passed to `model.fit()` must be strictly 2D arrays (e.g., shape `(N, 1)` for regression), unlike scikit-learn which often accepts 1D arrays.
- The optimizer is currently integrated as Vanilla Full-Batch Gradient Descent. 
- Input and Target data should be standardized (e.g., mean `0`, variance `1`) before passing to `fit()` to maintain gradient stability.
