# path_integral.py
# ------------------------------------------------------------
# Path-integral reconstruction of MI for a linear vector Gaussian channel:
#   Y_t = α A X + Z_t,  X ~ N(0, σ_x^2 I_n),  Z_t ~ N(0, t I_m).
# We compare:
#   (1) Analytic MI (log-det closed form),
#   (2) Path integral of dI/dα via VJP + TRUE score,
#   (3) Path integral of dI/dα via VJP + DSM score (+ Stein scalar calibration).
#
# Output: a single PDF figure "path_integral.pdf" with the three curves.
#
# Requires: sfblib v0.2.x with the following functions available:
# - set_seed, DSMConfig, train_dsm_uncond
# - estimate_info_grad, stein_calibrate_scalar
# - integrate_along_path
# - grad_linear_gaussian_dIdalpha, mi_linear_gaussian
#
# Notes:
# - The spectrum of A matches the "paper version": s = geomspace(3.0, 0.3, n).
# - The anchor α=0 gives I=0, so no offset is needed in the path integral.
# ------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sfblib as sfb

# ------------------------------
# Local helper functions
# ------------------------------
@torch.no_grad()
def mi_linear_gaussian(A, sigma_x2: float, t: float, alpha: float = 1.0) -> float:
    """
    I(X;Y_t) for Y = alpha * A X + Z_t,  X ~ N(0, sigma_x2 I),  Z_t ~ N(0, t I).
    Closed form:
        I = 0.5 * log det( I_m + (alpha^2 * sigma_x2 / t) * A A^T ).
    Computed via Cholesky for numerical stability.

    Args:
        A: torch.Tensor, shape (m, n)
        sigma_x2: float
        t: float
        alpha: float (default 1.0)

    Returns:
        float  (nats)
    """
    m = A.shape[0]
    scale = (alpha * alpha) * (sigma_x2 / t)
    M = torch.eye(m, device=A.device, dtype=A.dtype) + scale * (A @ A.T)
    L = torch.linalg.cholesky(M)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    return float(0.5 * logdet.item())


@torch.no_grad()
def grad_linear_gaussian_dIdalpha(A, sigma_x2: float, alpha: float, t: float) -> float:
    """
    Analytic gradient dI/dalpha for Y = alpha * A X + Z_t.

    Formula:
        dI/dalpha = alpha * sigma_x2 * trace[ (Sigma_Y)^{-1} A A^T ]
    where Sigma_Y = (alpha^2 sigma_x2) A A^T + t I.

    Args:
        A: torch.Tensor, shape (m, n)
        sigma_x2: float
        alpha: float
        t: float

    Returns:
        float  (gradient value)
    """
    m = A.shape[0]
    AA = A @ A.T
    Sigma_Y = (alpha * alpha * sigma_x2) * AA + t * torch.eye(m, device=A.device, dtype=A.dtype)
    # Solve Sigma_Y^{-1} A A^T via Cholesky
    L = torch.linalg.cholesky(Sigma_Y)
    inv_AA = torch.cholesky_solve(AA, L, upper=False)
    grad = alpha * sigma_x2 * torch.trace(inv_AA)
    return float(grad.item())


# ------------------------------
# Problem setup
# ------------------------------
SEED = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sfb.set_seed(SEED, deterministic=True)

n = m = 8
sigma_x2 = 1.0
t = 0.5
alphas = np.linspace(0.0, 3.0, 31)  # α ∈ [0,3]

def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(sigma_x2)

@torch.no_grad()
def orthogonal_matrix(dim, dev):
    q, r = torch.linalg.qr(torch.randn(dim, dim, device=dev))
    d = torch.sign(torch.diag(r))
    return q @ torch.diag(d)

# Paper-style spectrum: s ∈ geomspace(3.0, 0.3, n)
def make_A_paper(n, dev):
    U = orthogonal_matrix(n, dev)
    V = orthogonal_matrix(n, dev)
    s = torch.tensor(np.geomspace(3.0, 0.3, n), dtype=torch.float32, device=dev)
    A = U @ torch.diag(s) @ V.T
    return A, s

A, s = make_A_paper(n, device)

# ------------------------------
# Front-ends
# ------------------------------
class FrontAlphaConst(nn.Module):
    def __init__(self, A: torch.Tensor, alpha: float):
        super().__init__()
        self.register_buffer("A", A)
        self.alpha = float(alpha)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A.T)

class FrontAlphaParam(nn.Module):
    def __init__(self, A: torch.Tensor, alpha_init: float):
        super().__init__()
        self.register_buffer("A", A)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A.T)

# ------------------------------
# True score for linear Gaussian
# ------------------------------
def make_true_score(alpha: float):
    # s_Y(y) = -SigmaY^{-1} y with SigmaY = α^2 σ_x^2 A A^T + t I
    AX = A @ A.T
    SigmaY = (alpha**2) * sigma_x2 * AX + t * torch.eye(m, device=device)
    L = torch.linalg.cholesky(SigmaY)
    def s_true(y: torch.Tensor) -> torch.Tensor:
        v = torch.cholesky_solve(y.T, L, upper=False)  # (m, B)
        return -v.T
    return s_true

# ------------------------------
# Gradients dI/dα (three methods)
# ------------------------------
def grad_analytic(alpha: float) -> float:
    # Use the library helper for the closed-form gradient
    return float(grad_linear_gaussian_dIdalpha(A, sigma_x2, alpha, t))

def grad_true_vjp(alpha: float, N: int = 100_000, batch: int = 8192) -> float:
    frontend = FrontAlphaParam(A, alpha).to(device)
    out = sfb.estimate_info_grad(
        frontend=frontend,
        score_eval=make_true_score(alpha),
        sampler_x=sampler_x,
        t=t, N=N, batch_size=batch,
        device=device, params=(frontend.alpha,), stop_grad_score=True,
    )
    return float(out.get("alpha", list(out.values())[0]))

dsm_cfg = sfb.DSMConfig(
    steps=1000, batch_size=4096, lr=1e-3,
    hidden=256, layers=2, activation="silu", grad_clip=1.0
)

def grad_dsm_vjp(alpha: float, N: int = 100_000, batch: int = 8192) -> float:
    # Train per-α unconditional DSM
    frontend_train = FrontAlphaConst(A, alpha).to(device)
    score = sfb.train_dsm_uncond(sampler_x, frontend_train, t, dsm_cfg, device, y_dim=m)

    # Parameterized front-end for gradient extraction
    frontend = FrontAlphaParam(A, alpha).to(device)

    @torch.no_grad()
    def sample_y(B: int) -> torch.Tensor:
        x = sampler_x(B, device); z = torch.randn(B, m, device=device)
        return frontend(x) + math.sqrt(t) * z

    # Stein scalar calibration (Gaussian Stein identity)
    c = sfb.stein_calibrate_scalar(score, sample_y, m, B=8192)

    out = sfb.estimate_info_grad(
        frontend=frontend,
        score_eval=lambda y: c * score(y),
        sampler_x=sampler_x,
        t=t, N=N, batch_size=batch,
        device=device, params=(frontend.alpha,), stop_grad_score=True,
    )
    return float(out.get("alpha", list(out.values())[0]))

# ------------------------------
# Analytic MI (log-det) for baseline
# ------------------------------
@torch.no_grad()
def mi_analytic(alpha: float) -> float:
    # I = 0.5 * log det( I + (α^2 σ_x^2 / t) A A^T )
    AA = (alpha**2) * (sigma_x2 / t) * (A @ A.T)
    I = torch.eye(m, device=device, dtype=A.dtype)
    L = torch.linalg.cholesky(I + AA)
    return float(torch.log(torch.diag(L)).sum().mul(2.0).mul(0.5).item())

# ------------------------------
# Run
# ------------------------------
def main():
    # Diagnostics: scale check
    fro2 = float((s**2).sum().item())
    slope0 = (sigma_x2 / t) * fro2
    print(f"||A||_F^2 = {fro2:.4f},  initial slope dI/dα|α→0 ≈ {slope0:.4f}")

    # Path integrals using the library helper (trapezoid in α, anchor I(0)=0)
    I_an_path  = sfb.integrate_along_path(lambda a: grad_analytic(a),   alphas.tolist())
    I_tr_path  = sfb.integrate_along_path(lambda a: grad_true_vjp(a),   alphas.tolist())
    I_dsm_path = sfb.integrate_along_path(lambda a: grad_dsm_vjp(a),    alphas.tolist())

    # Closed-form MI (ground truth)
    I_true = [mi_analytic(a) for a in alphas]

    # Errors (DSM path vs analytic)
    import numpy as np
    I_an_path = np.asarray(I_an_path, dtype=float)
    I_tr_path = np.asarray(I_tr_path, dtype=float)
    I_dsm_path = np.asarray(I_dsm_path, dtype=float)
    I_true_np  = np.asarray(I_true, dtype=float)

    mae_tr  = float(np.mean(np.abs(I_tr_path - I_true_np)))
    mae_dsm = float(np.mean(np.abs(I_dsm_path - I_true_np)))
    mre_tr  = float(np.max(np.abs((I_tr_path - I_true_np) / np.maximum(1e-9, I_true_np))))
    mre_dsm = float(np.max(np.abs((I_dsm_path - I_true_np) / np.maximum(1e-9, I_true_np))))
    print(f"[VJP TRUE]  MAE={mae_tr:.4e},  MaxRelErr={mre_tr*100:.2f}%")
    print(f"[VJP DSM ]  MAE={mae_dsm:.4e}, MaxRelErr={mre_dsm*100:.2f}%")

    # Plot
    plt.figure()
    plt.plot(alphas, I_true_np, label="Analytic MI (log-det)", linewidth=2.0)
    plt.plot(alphas, I_tr_path, linestyle="--", label="Path integral (VJP + TRUE score)")
    plt.plot(alphas, I_dsm_path, linestyle="-.", label="Path integral (VJP + DSM)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$I(X;Y_t)$  [nats]")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    out = "path_integral.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
