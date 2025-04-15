# Platform Detection Framework

A comprehensive framework for detecting hardware, software, and OS-specific capabilities to optimize computational workloads across different platforms.

[![PyPI version](https://badge.fury.io/py/platform-detection.svg)](https://badge.fury.io/py/platform-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **Automatic backend selection**: Optimizes computations based on available hardware and software
- 🔍 **Comprehensive detection**: Hardware, software, and OS-specific capabilities
- 🖥️ **Cross-platform**: Windows, macOS, and Linux support
- 🧮 **Performance-focused**: Ideal for data science, quant finance, and scientific computing
- 🔌 **Simple API**: Easy to integrate with existing code
- 🛠️ **Decorator-based optimization**: Add performance with minimal code changes

## Installation

```bash
pip install platform-detection
```

For additional dependencies:

```bash
pip install 'platform-detection[full]'
```

## Quick Start

### Automatic Detection

```python
from platform_detection import get_detector

# Get the detector
detector = get_detector()

# Print optimal backend
print(f"Optimal backend: {detector.get_optimal_backend()}")

# Save detection results to JSON
detector.json_dump("platform_capabilities.json")
```

### Optimize Functions with Decorators

```python
import numpy as np
import pandas as pd
from platform_detection import optimize

# Automatically use the optimal backend for matrix operations
@optimize(operation_type="matrix")
def calculate_correlation_matrix(df):
    return np.corrcoef(df.values.T)

# Specify data size estimation for more accurate backend selection
@optimize(operation_type="stat", data_size_estimator=lambda df: df.size)
def calculate_portfolio_risk(returns, weights):
    cov_matrix = returns.cov()
    return np.sqrt(weights.T @ cov_matrix @ weights)

# Financial operations use low-latency backends
@optimize(operation_type="finance")
def calculate_moving_averages(prices, windows=[20, 50, 200]):
    result = {}
    for window in windows:
        result[f'MA{window}'] = prices.rolling(window=window).mean()
    return pd.concat(result.values(), axis=1, keys=result.keys())
```

### Context Manager for Backend Selection

```python
import numpy as np
from platform_detection.backends import use_backend
from platform_detection.orchestrator import ComputeBackend

# Large matrix multiplication with CUDA (if available)
with use_backend(ComputeBackend.CUDA):
    result = np.dot(large_matrix1, large_matrix2)
```

## Command-Line Usage

```bash
# Show all detected capabilities
platform-detect

# Output as JSON
platform-detect --json

# Save to file
platform-detect --file=capabilities.json

# Show summary
platform-detect --summary
```

## Use Cases

### Quantitative Finance / Hedge Funds

- Automatically select the fastest array processing backend
- Optimize numerical computations on heterogeneous infrastructure
- Ensure consistent performance across different developer machines

### Data Science and Machine Learning

- Select appropriate backends based on data size and operation type
- Leverage platform-specific optimizations without manual tuning
- Handle transition between development and production environments

### Scientific Computing

- Make use of specialized hardware when available
- Fallback gracefully to CPU optimizations when GPUs unavailable
- Process data at different scales with optimal backends

## Documentation

For full documentation, visit [https://platform-detection.readthedocs.io](https://platform-detection.readthedocs.io).

## License

MIT License - see [LICENSE](LICENSE) for details.