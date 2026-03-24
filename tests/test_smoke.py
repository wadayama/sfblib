"""
Minimal smoke test for sfblib

Quick sanity checks (< 2 seconds) to detect breaking changes:
1. Import sfblib successfully
2. Basic identity channel validation (super lightweight)

Run with: uv run python -m pytest tests/test_smoke.py -v
Or standalone: uv run python tests/test_smoke.py
"""

import sys
import math
sys.path.insert(0, 'src')

import torch
import sfblib as sfb


def test_import():
    """Test that sfblib can be imported and has expected functions."""
    assert hasattr(sfb, 'set_seed')
    assert hasattr(sfb, 'DSMConfig')
    assert hasattr(sfb, 'estimate_info_grad')
    assert hasattr(sfb, 'train_dsm_uncond')
    assert hasattr(sfb, 'integrate_along_path')
    assert hasattr(sfb, 'stein_calibrate_scalar')
    assert hasattr(sfb, 'vjp_loss')
    assert hasattr(sfb, 'mi_linear_gaussian')
    print("✓ Import test passed")


def test_identity_channel_lightweight():
    """
    Ultra-lightweight identity channel test.
    Verifies basic MI estimation works without extensive computation.
    """
    sfb.set_seed(42)
    device = torch.device("cpu")  # Force CPU for consistency

    # Minimal setup
    n = 2  # Small dimension
    P = 1.0
    t = 0.5

    # Theoretical MI for identity channel: I = 0.5 * n * log(1 + P/t)
    I_theory = 0.5 * n * math.log1p(P / t)

    # Test mi_linear_gaussian helper
    A_identity = torch.eye(n, device=device)
    I_computed = sfb.mi_linear_gaussian(A_identity, sigma_x2=P, t=t)

    # Should match theory within floating point precision
    rel_error = abs(I_computed - I_theory) / I_theory
    assert rel_error < 1e-6, f"Identity channel MI mismatch: {I_computed:.6f} vs {I_theory:.6f}"

    print(f"✓ Identity channel test passed: I={I_computed:.6f} nats (theory={I_theory:.6f})")


def test_integrate_along_path_simple():
    """Test integrate_along_path with a simple known integral."""
    # ∫ 2x dx from 0 to 2 = x^2 |_0^2 = 4
    grad_fn = lambda x: 2 * x
    thetas = [0.0, 1.0, 2.0]

    # Test with offset=0
    result = sfb.integrate_along_path(grad_fn, thetas, cumulative=False, offset=0.0)
    expected = 4.0
    assert abs(result - expected) < 0.01, f"Path integral mismatch: {result} vs {expected}"

    # Test with offset=10
    result_offset = sfb.integrate_along_path(grad_fn, thetas, cumulative=False, offset=10.0)
    expected_offset = 14.0
    assert abs(result_offset - expected_offset) < 0.01, f"Path integral with offset mismatch: {result_offset} vs {expected_offset}"

    print(f"✓ integrate_along_path test passed: result={result:.2f}, with offset={result_offset:.2f}")


def test_stein_calibrate_scalar_basic():
    """Test Stein calibration with a simple score function."""
    device = torch.device("cpu")
    m = 3

    # For Gaussian: true score is -y, so E[y^T (-y)] = -m
    # Stein calibration should return c ≈ 1.0
    score = lambda y: -y
    sampler = lambda B: torch.randn(B, m, device=device)

    c = sfb.stein_calibrate_scalar(score, sampler, m, B=1000)

    # Should be close to 1.0
    assert abs(c - 1.0) < 0.1, f"Stein calibration unexpected: c={c:.4f}"
    print(f"✓ Stein calibration test passed: c={c:.4f}")


def test_fm_imports():
    """Test that all flow matching symbols are exported."""
    for name in [
        "t_to_tau", "t_to_tau_tensor", "velocity_to_score",
        "VelocityNetMLP", "FlowMatchingScoreAdapter", "FMConfig",
        "cfm_loss_per_t", "cfm_loss_noise_cond",
        "train_fm_per_t", "train_fm_noise_cond",
    ]:
        assert hasattr(sfb, name), f"Missing export: {name}"
    print("✓ FM import test passed")


def test_t_to_tau():
    """Test t ↔ τ conversion at known values."""
    # t=1 → √1/(1+√1) = 0.5
    assert abs(sfb.t_to_tau(1.0) - 0.5) < 1e-10
    # t=0 edge → τ=0
    assert abs(sfb.t_to_tau(0.0)) < 1e-10
    # t=4 → 2/(1+2) = 2/3
    assert abs(sfb.t_to_tau(4.0) - 2.0 / 3.0) < 1e-10

    # tensor version
    t_tensor = torch.tensor([1.0, 4.0])
    tau_tensor = sfb.t_to_tau_tensor(t_tensor)
    assert abs(tau_tensor[0].item() - 0.5) < 1e-6
    assert abs(tau_tensor[1].item() - 2.0 / 3.0) < 1e-6
    print("✓ t_to_tau test passed")


def test_cfm_loss_runs():
    """Test that CFM loss functions return scalar tensors."""
    dim = 4
    B = 32
    net = sfb.VelocityNetMLP(dim=dim, hidden=16, layers=2)
    w = torch.randn(B, dim)

    # per-t loss
    loss_pt = sfb.cfm_loss_per_t(net, w, t=1.0)
    assert loss_pt.dim() == 0, f"Expected scalar, got shape {loss_pt.shape}"
    assert loss_pt.item() > 0

    # noise-cond loss
    t_vec = torch.rand(B) * 2.0 + 0.1
    loss_nc = sfb.cfm_loss_noise_cond(net, w, t_vec)
    assert loss_nc.dim() == 0, f"Expected scalar, got shape {loss_nc.shape}"
    assert loss_nc.item() > 0
    print("✓ CFM loss test passed")


def test_fm_adapter_shape():
    """Test that FlowMatchingScoreAdapter outputs the correct shape."""
    dim = 4
    B = 16
    net = sfb.VelocityNetMLP(dim=dim, hidden=16, layers=2)
    adapter = sfb.FlowMatchingScoreAdapter(net, tau=0.5)
    y = torch.randn(B, dim)
    s = adapter(y)
    assert s.shape == (B, dim), f"Expected ({B}, {dim}), got {s.shape}"
    print("✓ FM adapter shape test passed")


if __name__ == "__main__":
    print("Running sfblib smoke tests...\n")

    test_import()
    test_identity_channel_lightweight()
    test_integrate_along_path_simple()
    test_stein_calibrate_scalar_basic()
    test_fm_imports()
    test_t_to_tau()
    test_cfm_loss_runs()
    test_fm_adapter_shape()

    print("\n✅ All smoke tests passed!")
