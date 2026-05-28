# NN Engine Core

[![PyPI version](https://img.shields.io/pypi/v/nn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/nn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance, fully native C++ Neural Network engine exposed to Python via pybind11. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `nn-engine-core` executes the entire deep learning training loop (forward pass, loss calculation, backpropagation, and weight updates) strictly in native C++ using Eigen. It utilizes a zero-allocation flat-memory Autograd graph, AVX SIMD vectorization, and dynamically compiled OpenBLAS to achieve PyTorch-level performance.

## Highlights

- **Native Loop Hoisting**: The `Model::fit` loop executes entirely in C++, eliminating the Python GIL overhead across epochs and batches.
- **Zero-Allocation Autograd (Wengert List)**: Features a custom, zero-overhead automatic differentiation engine. Uses arena allocation and flat contiguous memory structs to dynamically build computational graphs without heap allocations or virtual table overhead.
- **Memory-Optimized Backpropagation**: Utilizes Eigen's `.noalias()` to perform in-place matrix calculus without allocating temporary memory buffers.
- **Hardware Maxmimized**: Uses `-ffast-math` flush-to-zero routines, raw SIMD loops, and statically compiled OpenBLAS to hit the absolute limits of the CPU architecture.
- **Mathematically Stable**: Built-in Glorot (Xavier) uniform initialization, and Log-Sum-Exp fusion for numerically stable Softmax gradients.

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
│   │   ├── loss/
│   │   ├── optimizer/
│   │   ├── regularizer/
│   │   ├── Layer.hpp
│   │   ├── Model.hpp
│   │   ├── Random.hpp
│   │   └── Types.hpp
│   └── layers/
│       ├── DenseLayer.hpp
│       ├── LeakyReLULayer.hpp
│       ├── ReLULayer.hpp
│       └── Sequential.hpp
├── src/
│   ├── binding.cpp
│   ├── core/
│   │   └── Model.cpp
│   └── layers/
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
# Note: NNEngine is optimized for 32-bit floats!
X_train = np.random.rand(100, 4).astype(np.float32)
# One-hot encoded targets for Softmax Cross Entropy
y_train = np.eye(3)[np.random.choice(3, 100)].astype(np.float32)

model = nn_core.Model()
model.add(nn_core.DenseLayer(4, 16))
model.add(nn_core.ReLULayer())
model.add(nn_core.DenseLayer(16, 3))
# Note: Do not add a Softmax layer. The SoftmaxCrossEntropyLoss fuses it!

optimizer = nn_core.Adam(learning_rate=0.01)
loss = nn_core.SoftmaxCrossEntropyLoss()
model.compile(optimizer, loss)

# Train (use examples/script.py --seed to make runs deterministic)
model.fit(X_train, y_train, epochs=200, batch_size=16, tol=1e-4, verbose=True)

# Note: predict() outputs raw logits. Use np.argmax for classification!
predictions = model.predict(X_train[:1])
print("Class Predictions:", np.argmax(predictions, axis=1))
```

## Python API

### `Model()`
The orchestrator for the neural network.
- `add(layer)`: Appends a layer to the network sequence. If called after `compile()` the model will automatically synchronize trainable parameters with the optimizer.
- `compile(optimizer, loss_fn, regularizer=None)`: Attach an optimizer and loss function (and optional regularizer). The model caches the trainable parameter list and wires it to the optimizer at compile time.
- `fit(X, y, epochs=100, batch_size=32, tol=1e-4, n_iter_no_change=10, verbose=True)`: Executes mini-batch gradient descent. Shuffles data automatically per epoch and reuses internal batch buffers to avoid per-epoch allocations. *Note: `X` and `y` must be 2D `float32` NumPy arrays.*
- `predict(X)`: Runs a forward pass on new data, returning raw float32 logits.

### Layers (`nn_core.*`)
- `DenseLayer(input_dim: int, output_dim: int)`: A fully connected parametric layer using Glorot (Xavier) uniform initialization.
- `ReLULayer()`: A non-parametric Rectified Linear Unit activation layer.
- `Sequential()`: The underlying layer container (automatically managed by `Model`).

### Loss Functions (`nn_core.*`)
- `MSELoss()`: Mean Squared Error loss for continuous regression targets. 
- `SoftmaxCrossEntropyLoss()`: Fused Log-Sum-Exp Softmax and Cross-entropy loss for numerically stable multi-class classification (requires one-hot encoded targets).

## Benchmark Results

Testing custom `NNEngine (C++)` vs `sklearn.neural_network.MLPClassifier` using Mini-Batch Gradient Descent. The table below lists the single-run NNEngine results on 32-bit floats.

| Dataset | Samples | Features | Classes | NNEngine Acc. | Sklearn Acc. | Speedup |
|---|--:|--:|--:|--:|--:|--:|
| **Iris Flower** | 150 | 4 | 3 | **96.67%** | 93.33% | **~90x** |
| **Digits** | 1,797 | 64 | 10 | **98.33%** | 98.33% | **~7x** |
| **Olivetti Faces** | 400 | 4,096 | 40 | **92.50%** | 90.00% | **~2x** |

These results are stochastic by nature — see Reproducibility notes below for how to reproduce runs deterministically.

## Notes and Limitations

- Targets (`y`) passed to `model.fit()` must be strictly 2D `float32` arrays. For classification, they must be one-hot encoded (e.g., shape `(N, C)`).
- Input data should be standardized (e.g., mean `0`, variance `1`) before passing to `fit()` to maintain gradient stability.