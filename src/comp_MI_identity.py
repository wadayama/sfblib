
import math, torch
import numpy as np
from sfblib import (
    estimate_mi_forward, DSMConfig, LogGridConfig, FisherConfig, TailConfig
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Problem size and input power
n, P = 4, 1.0

# X ~ N(0, P I_n)
def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(P)

# Identity front-end: f(x) = x
class Front(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# Log grid and estimator configs (as used in the papers)
# T_lower is the lower integration limit in the Fisher integral
T_lower = P / 200.0
t_grid = LogGridConfig(t_min=T_lower, t_max=50.0 * P, m_points=12, T_lower=T_lower)
dsm = DSMConfig(steps=300, batch_size=8192, lr=1e-3, grad_clip=1.0, activation="silu")
fisher = FisherConfig(mc_samples=100_000)
tail = TailConfig(use_tail=True, cov_trace_est_samples=50_000)

# Run the pipeline
frontend = Front().to(device)
out = estimate_mi_forward(
    sampler_x=sampler_x,
    frontend=frontend,
    t_grid=t_grid,
    dsm=dsm,
    fisher=fisher,
    tail=tail,
    device=device,
    conditional="per_t",  # "noise_cond" is also available
)

# Theoretical value for Gaussian input at T_lower:
# I(X; Y_T) = (n/2) * log(1 + P/T)
I_theory = (n / 2.0) * math.log(1.0 + P / T_lower)

print("=" * 60)
print("Identity function: f(x) = x")
print("=" * 60)
print(f"Estimated I(X;Y_T) [nats] = {out['I_hat']:.6f}")
print(f"Theoretical I(X;Y_T) [nats] = {I_theory:.6f}")
rel = abs(out["I_hat"] - I_theory) / I_theory * 100.0
print(f"Relative error = {rel:.3f}%")
print("=" * 60)
