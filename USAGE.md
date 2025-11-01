# sfblib Usage Guide

## Setup

### 1. Environment Setup (First Time)

```bash
# Verify uv is installed
uv --version

# Clone the project
git clone <repository-url>
cd 2025-sfblib_new

# Install dependencies
uv sync
```

### 2. Adding Packages

When you need to add new packages:

```bash
uv add package_name

# Example: Add matplotlib
uv add matplotlib

# Add as development dependency
uv add --dev pytest
```

### 3. Running Scripts

```bash
# Run Python scripts
uv run python your_script.py

# Interactive Python shell
uv run python

# Run as module
uv run python -m your_module
```

## Basic Usage

### Importing sfblib

```python
import sys
sys.path.insert(0, 'src')
import sfblib
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix random seed
sfblib.set_seed(42)
```

### Mutual Information Estimation for Gaussian Input Channel

```python
import numpy as np

# Parameter configuration
n = 4  # Dimension
P = 1.0  # Signal power
t = 0.5  # Noise variance

# Define sampler
def sampler_x(batch_size, device):
    return torch.randn(batch_size, n, device=device) * np.sqrt(P)

# Frontend (identity function)
frontend = torch.nn.Identity()

# Grid configuration
grid_config = sfblib.LogGridConfig(
    t_min=P/200,
    t_max=200*P,
    m_points=10,
    T_lower=P/200
)

# DSM configuration
dsm_config = sfblib.DSMConfig(
    lr=1e-3,
    steps=300,
    batch_size=4096,
    scheme="per_t"
)

# Fisher information configuration
fisher_config = sfblib.FisherConfig(mc_samples=100000)

# Tail correction configuration
tail_config = sfblib.TailConfig(use_tail=True)

# MI estimation
result = sfblib.estimate_mi_forward(
    sampler_x=sampler_x,
    frontend=frontend,
    t_grid=grid_config,
    dsm=dsm_config,
    fisher=fisher_config,
    tail=tail_config,
    device=device
)

print(f"Estimated MI: {result['I_hat']:.4f} nats")
```

### Comparison with Closed-Form Solution (Gaussian Input)

```python
# Theoretical value
closed_form = sfblib.gaussian_awgn_closed_forms(P=P, n=n, t=t)
print(f"Theoretical MI: {closed_form['I']:.4f} nats")
print(f"Fisher information: {closed_form['J']:.4f}")
print(f"MMSE: {closed_form['mmse']:.4f}")
```

## Troubleshooting

### When CUDA is Not Available

```python
# Run on CPU
device = torch.device("cpu")
```

### When Running Out of Memory

```python
# Reduce batch size
dsm_config = sfblib.DSMConfig(batch_size=1024)

# Reduce number of samples
fisher_config = sfblib.FisherConfig(mc_samples=50000)
```

## References

- `doc/main_info_grad.pdf`: Theory of information gradient
- `doc/main_Fisher.pdf`: Mutual information estimation via Fisher integral
