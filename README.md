# NN Engine Core

[![PyPI version](https://img.shields.io/pypi/v/nn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/nn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance, fully native C++ Neural Network engine exposed to Python via pybind11. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `nn-engine-core` executes the entire deep learning training loop (forward pass, validation, loss calculation, backpropagation, and weight updates) strictly in native C++ using Eigen. It utilizes a zero-allocation flat-memory Autograd graph, AVX SIMD vectorization, and dynamically compiled OpenBLAS to achieve massive speedups over Scikit-Learn.

## Highlights

- **Native Loop Hoisting**: The `JITCompiler::fit` loop executes entirely in C++, eliminating the Python GIL overhead across epochs and batches.
- **Pure Python Extensibility**: Easily define custom Autograd operations (`nn.Op`) in pure Python. The engine uses PyBind11 trampolines to dispatch the C++ backward pass dynamically to your Python methods.
- **Zero-Allocation Autograd**: Uses arena allocation (`Tape`) and flat contiguous memory structs to dynamically build computational graphs without heap allocations.
- **Validation Early Stopping**: Features industry-standard early stopping with best-weight restoration evaluated cleanly on an isolated validation set.
- **Native Checkpointing**: Dump and restore raw contiguous memory weights directly to disk via C++ streams (`.nne` files) for blazing fast model saving.
- **Mathematically Stable**: Built-in Glorot (Xavier) initialization, and Log-Sum-Exp fusion for numerically stable Softmax gradients.

## Repository Structure

```text
.
├── CMakeLists.txt
├── pyproject.toml
├── include/
│   ├── autograd/
│   │   ├── ops/          <-- Standalone Operations (ReLU, MatMul, PyOp)
│   │   ├── Tape.hpp
│   │   ├── Tensor.hpp
│   │   └── Op.hpp
│   ├── core/
│   │   ├── JITGraph.hpp
│   │   └── Module.hpp
│   └── layers/
├── src/
│   ├── autograd/ops/     <-- Forward/Backward Implementations
│   └── binding.cpp       <-- PyBind11 Python Mappings
├── nnengine/
│   ├── __init__.py
│   ├── compiler.py
│   └── module.py
└── examples/
    └── script.py
```

## Installation

Install the released wheel from PyPI:

```bash
pip install nn-engine-core
```

Or install in editable/development mode from the repository (requires CMake 3.18+ and a C++17 compiler):

```bash
pip install -e .
```

## Quick Start (Multi-Class Classification)

```python
import numpy as np
import nnengine as nn

# 1. Prepare Data (float32 required)
X_train = np.random.rand(100, 4).astype(np.float32)
y_train = np.eye(3)[np.random.choice(3, 100)].astype(np.float32) # One-hot encoded

# 2. Define Network using PyTorch-like Syntax
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = self.add_module(nn.DenseLayer(4, 16))
        self.relu = self.add_module(nn.ReLULayer())
        self.fc2 = self.add_module(nn.DenseLayer(16, 3))

    def forward(self, tape, x):
        x = self.fc1(tape, x)
        x = self.relu(tape, x)
        return self.fc2(tape, x)

model = MyModel()

# 3. Compile & Train using C++ JIT
optimizer = nn.Adam(learning_rate=0.01)
loss_fn = nn.SoftmaxCrossEntropyLoss()
trainer = nn.JITCompiler(model, optimizer, loss_fn)

dataloader = nn.DataLoader(X_train, y_train, batch_size=16)

# Executes entirely in C++ without the GIL!
trainer.fit(dataloader, epochs=100, tol=1e-4)

# 4. Save and Load C++ Binary Checkpoints
model.save_weights("model.nne")
model.load_weights("model.nne")
```

## Defining Custom Autograd Operations in Python

You can easily extend the C++ engine by defining Custom Operations in Pure Python. The C++ `Tape` will safely map NumPy views to the Eigen memory and correctly reverse the graph!

```python
import numpy as np
import nnengine as nn

class MulOp(nn.Op):
    def __init__(self, tape, a, b):
        super().__init__()
        self.a, self.b = a, b
        # Let the C++ Arena allocate the flat memory
        self.out = tape.alloc_tensor(a.data.shape[0], b.data.shape[1], True)

    def forward(self):
        self.out.data = self.a.data * self.b.data 

    def backward(self):
        # Read/Write directly into the C++ Backend via NumPy views!
        if self.a.requires_grad:
            self.a.grad += self.out.grad * self.b.data
        if self.b.requires_grad:
            self.b.grad += self.out.grad * self.a.data
```

## Benchmark Results

Testing custom `NNEngine (JIT C++)` vs `sklearn.neural_network.MLPClassifier`. Both frameworks use identical splits, Validation-Loss Early Stopping (`validation_fraction=0.1`), Adam optimizer, and `tol=1e-4`. 

*Executed single-threaded on 32-bit floats to demonstrate architectural framework overhead.*

| Dataset | Samples | Features | Classes | NNEngine Acc. | Sklearn Acc. | Speedup |
|---|--:|--:|--:|--:|--:|--:|
| **Iris Flower** | 150 | 4 | 3 | **96.67%** | 80.00% | **~16.4x** |
| **Digits** | 1,797 | 64 | 10 | **98.33%** | 97.50% | **~3.3x** |
| **Olivetti Faces** | 400 | 4,096 | 40 | **87.50%** | 87.50% | **~4.3x** |


## Notes and Limitations

- Targets (`y`) passed to DataLoaders must be strictly 2D `float32` arrays. For classification, they must be one-hot encoded (e.g., shape `(N, C)`).
- Use `nn.set_seed(seed)` alongside `np.random.seed(seed)` to guarantee end-to-end reproducibility.