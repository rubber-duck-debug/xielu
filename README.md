# XIELU

XIELU is a high-performance CUDA implementation of a parameterized activation function designed for deep learning applications. This library provides optimized GPU kernels with PyTorch integration for both training and inference.

## Overview

XIELU implements a custom activation function with learnable parameters `alpha_p` (positive slope), `alpha_n` (negative slope), `beta` (scaling factor), and `eps` (epsilon for numerical stability). The activation function is designed to be differentiable and suitable for gradient-based optimization.

### Features

- **CUDA Accelerated**: Optimized CUDA kernels for maximum performance on NVIDIA GPUs
- **PyTorch Integration**: Seamless integration with PyTorch's autograd system
- **Flexible Precision**: Support for different floating-point precisions including bfloat16 optimizations
- **Memory Efficient**: Optimized memory access patterns for improved throughput
- **Gradient Support**: Full backward pass implementation for training

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA Toolkit (CUDA_HOME environment variable must be set)
- CMake >= 3.30
- NVIDIA GPU with compute capability 6.0+

### Setup

1. Ensure the `CUDA_HOME` environment variable points to your CUDA toolkit directory:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   ```

2. Install the package:
   ```bash
   pip install . --no-build-isolation --no-deps
   ```

   For GH200 or other specialized hardware, install on top of your existing container/uenv/python environment.

## Usage

XIELU provides three implementation variants for different use cases:

- **`XIELU`**: CUDA-accelerated implementation with `torch.compile` support (recommended for production)
- **`XIELUfn`**: Pure PyTorch with custom autograd function
- **`XIELUPy`**: Pure PyTorch implementation (reference implementation)

### Basic Usage

```python
import torch
from xielu.ops.wrappers import XIELU

# Initialize the activation function
device = torch.device("cuda")
xielu = XIELU(
    alpha_p_init=0.8,    # Initial positive slope parameter
    alpha_n_init=0.8,    # Initial negative slope parameter  
    beta=0.5,            # Scaling factor
    eps=1e-6,           # Epsilon for numerical stability
    device=device,
    dtype=torch.float32
)

# Forward pass
input_tensor = torch.randn(32, 128, 512, device=device)
output = xielu(input_tensor)

# The parameters are learnable and will be updated during training
optimizer = torch.optim.Adam(xielu.parameters(), lr=0.001)
```

### Integration with Neural Networks

```python
import torch.nn as nn
from xielu.ops.wrappers import XIELU

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.xielu = XIELU(device=torch.device("cuda"))
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.xielu(x)  # Custom activation
        x = self.linear2(x)
        return x
```

### Performance Options

For maximum performance, you can enable vectorized memory loads:

```python
xielu = XIELU(
    alpha_p_init=0.8,
    alpha_n_init=0.8, 
    beta=0.5,
    eps=1e-6,
    device=device,
    with_vector_loads=True  # Enable optimized memory access
)
```

### torch.compile Compatibility

XIELU supports `torch.compile` for additional performance optimizations and integration with compilable models:

```python
import torch
from xielu.ops.wrappers import XIELU

# Create model with XIELU activation
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xielu = XIELU(device=torch.device("cuda"))
    
    def forward(self, x):
        return self.xielu(x)

# Compile the model for optimized performance
model = MyModel()
compiled_model = torch.compile(model)

# Use as normal - now with compilation optimizations
output = compiled_model(input_tensor)
```

## Development

### Running Tests

The test suite includes correctness tests, gradient checks, and performance benchmarks:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python tests/test_xielu.py
python tests/test_reduced_precision.py

# Run benchmark
python tests/benchmark.py
```

### Test Coverage

The test suite validates:
- **Correctness**: Forward pass agreement between CUDA and PyTorch implementations
- **Gradients**: Gradient correctness using `torch.autograd.gradcheck`
- **Precision**: Reduced precision (bfloat16) functionality
- **Performance**: Throughput benchmarks across different tensor sizes

### Building from Source

The project uses CMake for building the CUDA extensions:

```bash
# Clean build
rm -rf build/

# Build in development mode
pip install -e . --no-build-isolation --no-deps

# For debugging, you can build with verbose output
CMAKE_VERBOSE_MAKEFILE=1 pip install -e . --no-build-isolation --no-deps
```

### Optimization Features

- **Vectorized Memory Access**: Enable `with_vector_loads=True` for improved memory throughput
- **Reduced Precision**: Support for bfloat16 operations for faster inference
- **Fused Operations**: Custom CUDA kernels minimize memory bandwidth requirements
- **Gradient Optimization**: Efficient backward pass implementation

## Mathematical Definition

The XIELU activation function is defined as:

```
f(x) = {
  α_p * x² + β * x,                                    if x > 0
  α_n * (exp(min(x, ε)) - 1) - α_n * x + β * x,       if x ≤ 0
}
```

Where:
- `α_p = softplus(alpha_p)`: Learned positive slope parameter
- `α_n = β + softplus(alpha_n)`: Learned negative slope parameter  
- `β`: Fixed scaling factor
- `ε`: Numerical stability parameter
