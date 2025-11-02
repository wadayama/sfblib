# sfblib - Score-to-Fisher Bridge Library

PyTorch-based library for mutual information estimation and information gradient computation in nonlinear Gaussian channels.

## Features

- **Mutual Information Estimation** via Score-to-Fisher Bridge (SFB) methodology
- **Information Gradient Computation** using VJP (Vector-Jacobian Product)
- **Denoising Score Matching (DSM)** for score function learning
- **Task-Oriented Extensions** for semantic communication
- **Information Bottleneck** optimization support
- **GPU-Ready**: Automatically uses GPU when available

## Paper References

This library implements the methods described in:

- **SFB/MI Estimation**: [arXiv:2510.05496v2](https://arxiv.org/abs/2510.05496v2)
  *"Mutual Information Estimation via Score-to-Fisher Bridge for Nonlinear Gaussian Noise Channels"*

- **Information Gradient**: [arXiv:2510.20179v1](https://arxiv.org/abs/2510.20179v1)
  *"Information Gradient for Nonlinear Gaussian Channel with Applications to Task-Oriented Communication"*

## Installation

Clone this repository and install dependencies:

```bash
# Verify uv is installed
uv --version

# Clone the repository
git clone <repository-url>
cd sfblib

# Install dependencies using uv
uv sync
```

## Quick Start

### Running Example Scripts

```bash
# Identity channel validation (< 0.5% error vs theory)
uv run python src/comp_MI_identity.py

# Generate MI curve visualization (identity channel)
uv run python src/MI_sfblib.py

# Nonlinear tanh channel: DSM vs KDE-LOO comparison
uv run python src/MI_tanh.py

# Information gradient: reproduce paper Fig.3
uv run python src/IG_sfblib_vjp.py

# Projected gradient ascent: optimize channel matrix A
uv run python src/A_optim_sfblib.py

# Path integral MI reconstruction (parameter integration)
uv run python src/path_integral.py
```

### Running Tests

Quick smoke tests to verify core functionality (< 1 second):

```bash
# Run all tests
uv run pytest tests/test_smoke.py -v

# Or standalone
uv run python tests/test_smoke.py
```

The smoke tests validate:
- Library imports correctly
- Identity channel MI matches theory
- Path integral computation works
- Stein calibration functions properly

### Adding New Packages

```bash
# Add a package
uv add package_name

# Add as development dependency
uv add --dev pytest
```

## Basic Usage

### Importing sfblib

```python
import sys
sys.path.insert(0, 'src')
import sfblib as sfb
import torch
import numpy as np

# Device configuration (automatically uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix random seed for reproducibility
sfb.set_seed(42)
```

### Mutual Information Estimation

```python
# Problem setup
n = 4  # Dimension
P = 1.0  # Signal power
t = 0.5  # Noise variance

# Define input sampler: X ~ N(0, P·I_n)
def sampler_x(batch_size, device):
    return torch.randn(batch_size, n, device=device) * np.sqrt(P)

# Frontend function: f(x) = x (identity)
frontend = torch.nn.Identity()

# Configuration objects
grid_config = sfb.LogGridConfig(
    t_min=P/200,
    t_max=200*P,
    m_points=10,
    T_lower=P/200
)

dsm_config = sfb.DSMConfig(
    lr=1e-3,
    steps=300,
    batch_size=4096,
    scheme="per_t"  # or "noise_cond"
)

fisher_config = sfb.FisherConfig(mc_samples=100000)
tail_config = sfb.TailConfig(use_tail=True)

# Run MI estimation
result = sfb.estimate_mi_forward(
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

### Information Gradient Computation

Compute gradients of mutual information with respect to channel parameters:

```python
# Define a parameterized frontend
class LinearFrontend(torch.nn.Module):
    def __init__(self, A, alpha):
        super().__init__()
        self.register_buffer("A", A)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return self.alpha * (x @ self.A.T)

# Create frontend and score function
frontend = LinearFrontend(A, alpha_init=1.0)
score_fn = ...  # From DSM training or analytical

# Compute information gradient using VJP
grad_dict = sfb.estimate_info_grad(
    frontend=frontend,
    score_eval=score_fn,
    sampler_x=sampler_x,
    t=0.5,
    N=100_000,
    batch_size=8192,
    device=device,
    params=(frontend.alpha,),
    stop_grad_score=True
)

print(f"∂I/∂α = {grad_dict['alpha']:.4f}")
```

## API Quick Reference

> Minimal cheatsheet. See function docstrings for details.

### Core training / gradient

- `set_seed(seed=0, deterministic=True) -> None`
  Reproducibility flags (`cudnn.deterministic` etc).

- `DSMConfig(steps=1000, batch_size=4096, lr=1e-3, hidden=256, layers=2, activation="silu", grad_clip=1.0)`
  Config for DSM training.

- `train_dsm_uncond(sampler_x, frontend, t, dsm: DSMConfig, device, y_dim=None) -> nn.Module`
  Train **unconditional** score `s(y)` at fixed noise variance `t`. Returns `score(y)`.

- `estimate_info_grad(frontend, score_eval, sampler_x, t, N, batch_size, device, params=None, stop_grad_score=True) -> dict[str, float|np.ndarray]`
  VJP-based info gradient w.r.t. `params` (e.g. `(frontend.alpha,)` or `(frontend.A,)`).
  **score_eval** is a callable `s(y)`. Set `stop_grad_score=True`.

- `stein_calibrate_scalar(score: nn.Module, y_sampler: Callable[[int], Tensor], m: int, B=8192) -> float`
  One-scalar Stein calibration: make `E[Y^T (c s(Y))] ≈ -m`.

### Path integral / Fisher integral

- `integrate_along_path(grad_fn, thetas: list, cumulative=True, offset=0.0)`
  Trapezoid rule over a 1D path.
  **Example:** `I = integrate_along_path(lambda a: dIdalpha(a), alphas)  # I[0]=offset`

- `make_log_grid(LogGridConfig) -> np.ndarray`
  Geometric `t` grid; use with Fisher integral.

- `integrate_mi_log_trapz(t_vals, J_vals, dim_y, T_lower, tail: TailConfig, trace_cov_x=None) -> float`
  Log-domain trapezoid + tail correction; use with `estimate_fisher_from_score`.

- `estimate_fisher_from_score(score, sampler_x, frontend, t, fisher: FisherConfig, device, noise_conditional=False, stein_calibrate=False) -> float`
  Monte-Carlo `J(Y_t)=E||s(Y_t)||^2` from a trained score.

### Linear Gaussian baselines (closed-form)

- `mi_linear_gaussian(A, sigma_x2, t, alpha=1.0) -> float`
  I = (1/2) log det(I + (α²σ_x²/t) A A^T) via Cholesky.

### Utilities

- `project_to_frobenius_ball(A: Tensor, radius: float) -> Tensor`
  Scale `A` to keep `||A||_F ≤ radius`. Good for constrained ascent.

- `mi_kde_loo_gaussian_pairs(w, y, t, chunk=None) -> float`
- `estimate_mi_kde_loo(sampler_x, frontend, t, N, device, chunk=None) -> float`
  KDE-LOO MI baseline (Gaussian kernel, LOO). Use for quick checks.

### Shapes / devices

- `sampler_x(B, device) -> X:(B,n)`; `frontend(X) -> W:(B,m)`; `Y = W + sqrt(t)*Z`.
  Pass `device` consistently; large `N` uses internal chunking.

### Quick recipes

- **dI/dα (Fig.3)**: train DSM per α → `c = stein_calibrate_scalar(...)` → `estimate_info_grad(..., params=(front.alpha,))` → path integral.
- **A-optim (Fig.4)**: train DSM per iter → `gA = estimate_info_grad(..., params=(front.A,))` → step & `project_to_frobenius_ball`.
- **MI via SFB**: per-t DSM → `J(Y_t)` → `integrate_mi_log_trapz` (log-grid + tail).

## Theory

This implementation is based on two research papers:

### 1. Information Gradient for Nonlinear Gaussian Channel

**Key formula:**
```
∇_η I(X; Yt) = -E[Df_η(X)^T s_Yt(Yt)]
```

**Features:**
- VJP-based efficient gradient computation
- Task-oriented extensions: `∇_η I(T; Yt)`
- Information bottleneck: `∇_η [I(T; Yt) - βI(X; Yt)]`

### 2. Mutual Information Estimation via Score-to-Fisher Bridge

**Fisher integral representation:**
```
I(X; YT) = (1/2) ∫_T^∞ [n/t - J(Yt)] dt
```

**Implementation details:**
- Log-domain trapezoid integration (Eq. 43-44)
- Tail correction for high noise (Eq. 45)
- Per-t DSM or noise-conditional DSM (Eq. 27, 41)

### Channel Model

```
Yt = f_η(X) + Zt,  Zt ~ N(0, tI_m)
```

where `f_η : R^n → R^m` is a parametric nonlinear front-end function.

## GPU Support

The code is **fully GPU-ready** and automatically uses GPU when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Expected speedup with GPU:**
- DSM training: 10-30× faster
- Fisher estimation: 10-15× faster
- Overall MI estimation: 10-20× faster

To enable GPU support, install PyTorch with CUDA:

```bash
# For CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### Memory Issues

```python
# Reduce batch size
dsm_config = sfb.DSMConfig(batch_size=1024)

# Reduce Monte Carlo samples
fisher_config = sfb.FisherConfig(mc_samples=50000)
```

### Force CPU Usage

```python
device = torch.device("cpu")
```

## Project Structure

```
sfblib/
├── src/
│   ├── sfblib.py                 # Core library with VJP helpers
│   ├── comp_MI_identity.py       # Identity channel validation
│   ├── MI_sfblib.py              # MI curve visualization (identity)
│   ├── MI_tanh.py                # Tanh channel: DSM vs KDE-LOO
│   ├── IG_sfblib_vjp.py          # Information gradient (reproduces paper Fig.3)
│   ├── A_optim_sfblib.py         # Projected gradient ascent for channel matrix A
│   └── path_integral.py          # Path integral MI reconstruction
├── tests/
│   └── test_smoke.py             # Minimal smoke tests (< 1s)
├── README.md                     # This file
├── claude.md                     # Instructions for Claude Code
├── pyproject.toml                # uv project configuration
├── uv.lock                       # Dependency lock file
└── .gitignore                    # Git exclusions
```

## Acknowledgement

This work was supported by JST, CRONOS, Japan Grant Number JPMJCS25N5.

## License

This software is released under the [MIT License](https://opensource.org/licenses/MIT).
Copyright (c) 2025 Tadashi Wadayama

## Author

**Tadashi Wadayama**
Nagoya Institute of Technology
wadayama@nitech.ac.jp 
