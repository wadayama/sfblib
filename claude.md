# Claude Code Instructions for sfblib Project

## First Steps When Starting a Conversation

1. **Always read README.md first** to understand:
   - Project structure and purpose
   - How to run scripts (use `uv run python src/script.py`)
   - Dependencies and installation
   - Available example scripts and their purposes

## Running Python Scripts

**IMPORTANT**: This project uses `uv` for package management.

- ✅ Correct: `uv run python src/script_name.py`
- ❌ Wrong: `python src/script_name.py`

The `uv run` command automatically activates the virtual environment and ensures all dependencies are available.

## Project Overview

This is a PyTorch library for mutual information estimation and information gradient computation in nonlinear Gaussian channels using the Score-to-Fisher Bridge (SFB) methodology.

### Key Components

- **Core library**: `src/sfblib.py` (~1135 lines)
- **Example scripts**:
  - `src/comp_MI_identity.py` - Identity channel validation
  - `src/MI_sfblib.py` - MI curve visualization
  - `src/MI_tanh.py` - Tanh channel DSM vs KDE-LOO comparison
  - `src/IG_fig3.py` - Information gradient Figure 3 reproduction
  - `src/IG_fig3_sfblib_vjp.py` - Information gradient with VJP helpers

### Import Style

Always use the standard import pattern:
```python
import sfblib as sfb
```

## Common Tasks

### Running Experiments
```bash
# Identity channel validation
uv run python src/comp_MI_identity.py

# Generate MI curve
uv run python src/MI_sfblib.py

# Tanh channel experiment
uv run python src/MI_tanh.py

# Information gradient experiments
uv run python src/IG_fig3.py
uv run python src/IG_fig3_sfblib_vjp.py
```

### Adding Dependencies
```bash
# Add a package
uv add package_name

# Add as development dependency
uv add --dev pytest
```

## Code Style Guidelines

1. **No emojis** unless explicitly requested by the user
2. **ASCII characters only** in code comments (no Unicode Greek letters)
3. **Remove citation tags** from code (e.g., fileciteturn0file0, fileciteturn0file1)
4. **Standard import style**: `import sfblib as sfb` followed by `sfb.function_name()`

## Paper References

- **SFB/MI Estimation**: [arXiv:2510.05496v2](https://arxiv.org/abs/2510.05496v2)
- **Information Gradient**: [arXiv:2510.20179v1](https://arxiv.org/abs/2510.20179v1)

## Notes

- GPU is automatically used when available
- The library supports both per-t DSM and noise-conditional DSM training schemes
- All MI values are in nats (natural logarithm)
