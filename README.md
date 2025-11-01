# sfblib - Score-to-Fisher Bridge Library

PyTorch-based library for mutual information estimation and information gradient computation in nonlinear Gaussian channels.

## Features

- **Mutual Information Estimation** via Score-to-Fisher Bridge (SFB) methodology
- **Information Gradient Computation** using VJP (Vector-Jacobian Product)
- **Denoising Score Matching (DSM)** for score function learning
- **Task-Oriented Extensions** for semantic communication
- **Information Bottleneck** optimization support

## Installation

Clone this repository and install dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd 2025-sfblib_new

# Install dependencies using uv
uv sync
```

## Usage

Run Python scripts with uv:

```bash
uv run python src/sfblib.py
```

## Adding Dependencies

To add new packages:

```bash
uv add package_name
```

## Theory

This implementation is based on two papers:

1. **Information Gradient for Nonlinear Gaussian Channel** (`doc/main_info_grad.pdf`)
   - Information gradient formula: `∇_η I(X; Yt) = -E[Df_η(X)^T s_Yt(Yt)]`
   - Task-oriented and information bottleneck extensions

2. **Mutual Information Estimation via Score-to-Fisher Bridge** (`doc/main_Fisher.pdf`)
   - Fisher integral representation: `I(X; YT) = (1/2) ∫_T^∞ [n/t - J(Yt)] dt`
   - Log-domain trapezoid integration

## Channel Model

```
Yt = f_η(X) + Zt,  Zt ~ N(0, tI_m)
```

where `f_η` is a parametric nonlinear front-end function.

## Citation

Please cite the project papers when using this library (see PDF documentation in `doc/` folder).

## License

Copyright (c) 2025. Research use permitted.

## Author

Tadashi Wadayama
Nagoya Institute of Technology
wadayama@nitech.ac.jp
