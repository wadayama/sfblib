
# sfblib.py (refactored) - Score-to-Fisher & Information-Gradient Toolkit
# Single-file PyTorch library for MI estimation, score learning, and VJP-based information gradients.
# Version: 0.2.0 (breaking changes allowed)
#
# Theory mapping:
#   - Information gradient: nabla_eta I(X; Yt) = -E[Df_eta(X)^T s_Yt(Yt)]
#   - Fisher integral: I(X; YT) = (1/2) integral_T^inf [n/t - J(Yt)] dt
#   - Score function learning via Denoising Score Matching (DSM)
#
# Paper References:
#   - SFB/MI Estimation: arXiv:2510.05496v2
#     "Mutual Information Estimation via Score-to-Fisher Bridge for Nonlinear Gaussian Noise Channels"
#   - Information Gradient: arXiv:2510.20179v1
#     "Information Gradient for Nonlinear Gaussian Channel with Applications to Task-Oriented Communication"
#
# MIT License - Copyright (c) 2025 Tadashi Wadayama
# See LICENSE file for full license text.

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities: device & RNG
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """
    Fix RNG for reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_device(x: torch.Tensor | nn.Module, device: torch.device) -> torch.Tensor | nn.Module:
    """
    Move a tensor or module to device.
    """
    if isinstance(x, nn.Module):
        return x.to(device)
    return x.to(device)


def chunked_apply(fn: Callable[[torch.Tensor], torch.Tensor], data: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """
    Apply `fn` to `data` in chunks along the first dim; returns concatenated result.
    """
    outs = []
    for i in range(0, data.shape[0], chunk):
        outs.append(fn(data[i:i+chunk]))
    return torch.cat(outs, dim=0)


# -----------------------------------------------------------------------------
# Channel & sampling
# -----------------------------------------------------------------------------

@dataclass
class ChannelSpec:
    """
    Unified description for a (possibly nonlinear) AWGN channel:

        Y_t = f(X) + sqrt(t) * Z,  Z ~ N(0, I)

    Attributes
    ----------
    sampler_x : Callable[[int, torch.device], torch.Tensor]
        Function to draw X samples with shape (B, n).
    frontend : nn.Module
        Deterministic mapping f: R^n -> R^m (can be nonlinear).
    y_dim : Optional[int]
        Output dimension m. If None, inferred from a one-sample forward pass.
    """
    sampler_x: Callable[[int, torch.device], torch.Tensor]
    frontend: nn.Module
    y_dim: Optional[int] = None


@torch.no_grad()
def simulate_y(channel: ChannelSpec, t: float, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a mini-batch for the channel: returns (x, w=f(x), y).
    Shapes: x:(B,n), w:(B,m), y:(B,m).

    This function is the atomic sampling primitive used throughout the library.
    """
    x = channel.sampler_x(batch_size, device)
    w = channel.frontend(x)
    if channel.y_dim is None:
        channel.y_dim = w.shape[-1]
    z = torch.randn(batch_size, channel.y_dim, device=device)
    y = w + math.sqrt(t) * z
    return x, w, y


# -----------------------------------------------------------------------------
# Score networks (MLPs) & t-embedding
# -----------------------------------------------------------------------------

def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unknown activation: {name}")


class FourierTEmbedding(nn.Module):
    """
    Gaussian Fourier features for scalar t (noise variance).
    Produces a fixed-dim embedding phi(t) in R^{2*D}.

    Reference: common in diffusion models; here used for conditional DSM (main Eq.(41)). 
    """
    def __init__(self, embed_dim: int = 32, scale: float = 10.0):
        super().__init__()
        D = embed_dim // 2
        self.B = nn.Parameter(torch.randn(D) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=self.B.device)
        t = t.view(-1, 1)  # (B,1)
        xb = t * self.B.view(1, -1)
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)


class ScoreNetMLP(nn.Module):
    """
    Unconditional score s(y) ~ nabla__y log p_{Y_t}(y). DSM loss: main Eq.(27). 
    """
    def __init__(self, dim: int, hidden: int = 128, layers: int = 3, activation: str = "silu"):
        super().__init__()
        act = _activation(activation)
        mods = []
        in_dim = dim
        for _ in range(layers):
            mods += [nn.Linear(in_dim, hidden), nn.SiLU() if act == F.silu else nn.GELU() if act == F.gelu else nn.ReLU()]
            in_dim = hidden
        mods += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class NoiseCondScoreNet(nn.Module):
    """
    Noise-conditional score s(y, t) (main Eq.(41)).  t is passed via Fourier embedding. 
    """
    def __init__(self, dim: int, hidden: int = 128, layers: int = 3, activation: str = "silu", t_embed_dim: int = 32):
        super().__init__()
        self.temb = FourierTEmbedding(t_embed_dim)
        act = _activation(activation)
        mods = []
        in_dim = dim + t_embed_dim
        for _ in range(layers):
            mods += [nn.Linear(in_dim, hidden), nn.SiLU() if act == F.silu else nn.GELU() if act == F.gelu else nn.ReLU()]
            in_dim = hidden
        mods += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.full((y.shape[0],), float(t), dtype=y.dtype, device=y.device)
        emb = self.temb(t)  # (B, D)
        if emb.shape[0] != y.shape[0]:
            emb = emb.expand(y.shape[0], -1)
        h = torch.cat([y, emb], dim=-1)
        return self.net(h)


class CondTaskScoreNet(nn.Module):
    """
    Conditional score s(y, tau) used for task-oriented objectives (info_grad Eq.(62)). 
    """
    def __init__(self, y_dim: int, tau_dim: int, hidden: int = 128, layers: int = 3, activation: str = "silu"):
        super().__init__()
        act = _activation(activation)
        mods = []
        in_dim = y_dim + tau_dim
        for _ in range(layers):
            mods += [nn.Linear(in_dim, hidden), nn.SiLU() if act == F.silu else nn.GELU() if act == F.gelu else nn.ReLU()]
            in_dim = hidden
        mods += [nn.Linear(in_dim, y_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.dim() == 1:
            tau = tau.view(-1, 1)
        h = torch.cat([y, tau], dim=-1)
        return self.net(h)


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DSMConfig:
    lr: float = 1e-3
    steps: int = 300
    batch_size: int = 4096
    hidden: int = 128
    layers: int = 3
    activation: str = "silu"
    grad_clip: float = 1.0
    scheme: str = "per_t"       # "per_t" or "noise_cond"  (main Eq.(41)) 
    t_embed_dim: int = 32
    weight_decay: float = 0.0
    stein_calibrate: bool = False   # Optional scale calibration at evaluation (info_grad Sec.VII). 

@dataclass
class LogGridConfig:
    t_min: float
    t_max: float
    m_points: int
    T_lower: float             # lower limit T for MI integral (Eq.(24)); controls slice index in Eq.(44). 

@dataclass
class FisherConfig:
    mc_samples: int = 100_000

@dataclass
class TailConfig:
    use_tail: bool = True
    cov_trace_est_samples: int = 50_000


# -----------------------------------------------------------------------------
# DSM losses (core)
# -----------------------------------------------------------------------------

def dsm_loss_uncond(score: Callable[[torch.Tensor], torch.Tensor],
                    w: torch.Tensor, t: float, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    DSM loss at fixed t (main Eq.(27)):  E || s(y) + eps/sqrtt ||^2  with  y = w + sqrtt eps.  
    Equivalent target: s(y) ~ -(y - w)/t.
    """
    B, m = w.shape
    if eps is None:
        eps = torch.randn_like(w)
    y = w + math.sqrt(t) * eps
    target = (y - w) / t
    pred = score(y)
    return F.mse_loss(pred + target, torch.zeros_like(pred))


def dsm_loss_noise_cond(score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        w: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Noise-conditional DSM (main Eq.(41)):  E_t E || s(y,t) + eps/sqrtt ||^2 .  t is vector-shaped.  
    """
    if eps is None:
        eps = torch.randn_like(w)
    y = w + (t.view(-1, 1).sqrt()) * eps
    target = (y - w) / t.view(-1, 1)
    pred = score(y, t)
    return F.mse_loss(pred + target, torch.zeros_like(pred))


def dsm_loss_conditional(score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         w: torch.Tensor, tau: torch.Tensor, t: float,
                         eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Conditional DSM for s(y, tau) (info_grad Eq.(62)): noise is added only to y, not tau.  
    """
    if eps is None:
        eps = torch.randn_like(w)
    y = w + math.sqrt(t) * eps
    target = (y - w) / t
    pred = score(y, tau)
    return F.mse_loss(pred + target, torch.zeros_like(pred))


# -----------------------------------------------------------------------------
# Generic training loop
# -----------------------------------------------------------------------------

def train_score_generic(
    make_model: Callable[[], nn.Module],
    loss_fn: Callable[..., torch.Tensor],
    channel: ChannelSpec,
    device: torch.device,
    t: float | Tuple[float, float],
    steps: int,
    batch_size: int,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    weight_decay: float = 0.0,
    sampler_cond: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> nn.Module:
    """
    Universal training loop for score models.
    - If t is float   fixed-t DSM (per-t)
    - If t is (t_min, t_max)  noise-conditional DSM with log-uniform t

    This function underlies the specialized wrappers below.
    """
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_uniform = None
    if isinstance(t, tuple):
        t_min, t_max = float(t[0]), float(t[1])
        if t_min <= 0 or t_max <= 0 or t_max <= t_min:
            raise ValueError("Require 0 < t_min < t_max for conditional training.")
        u_min, u_max = math.log(t_min), math.log(t_max)
        def _sample_t(B: int) -> torch.Tensor:
            u = torch.rand(B, device=device) * (u_max - u_min) + u_min
            return torch.exp(u)
        log_uniform = _sample_t

    for _ in range(steps):
        x, w, _ = simulate_y(channel, t if not log_uniform else float(1.0), batch_size, device)  # y not used here
        if log_uniform is None:
            # per-t
            loss = loss_fn(model, w, float(t))
        else:
            # conditional
            t_vec = log_uniform(batch_size)
            loss = loss_fn(model, w, t_vec)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    return model


# -----------------------------------------------------------------------------
# Monte Carlo expectations & Fisher/MMSE
# -----------------------------------------------------------------------------

@torch.no_grad()
def mc_expect(fn: Callable[[int], torch.Tensor], N: int, reduce: str = "mean", chunk: int = 65536) -> torch.Tensor:
    """
    Compute E[ fn(B) ] by repeated chunk evaluation. fn returns a vector of size (B,) or (B,d).

    MC error decays ~ O(N^{-1/2}).  (main Sec.III-C; CLT remark after Eq.(29))  
    """
    acc = []
    done = 0
    while done < N:
        B = min(chunk, N - done)
        acc.append(fn(B))
        done += B
    out = torch.cat(acc, dim=0)
    return out.mean(dim=0) if reduce == "mean" else out.sum(dim=0)


def fisher_from_score(
    score_eval: Callable[[torch.Tensor], torch.Tensor],
    y_sampler: Callable[[int], torch.Tensor],
    N: int = 100_000,
    chunk: int = 65536
) -> float:
    """
    J(Y_t) = E || s(Y_t) ||^2, estimated by Monte Carlo.  (main Eq.(8),(29))  
    """
    @torch.no_grad()
    def _fn(B: int) -> torch.Tensor:
        y = y_sampler(B)
        s = score_eval(y)
        return (s.pow(2).sum(dim=-1)).view(-1, 1)
    val = mc_expect(_fn, N=N, reduce="mean", chunk=chunk).item()
    return float(val)


def mmse_from_fisher(J: float, m: int, t: float) -> float:
    """
    mmse(t) = m t - t^2 J(Y_t)  (main Eq.(23))  
    """
    return m * t - (t ** 2) * J


def fisher_from_mmse(mmse: float, m: int, t: float) -> float:
    """
    J(Y_t) = (m t - mmse(t)) / t^2  (rearranged main Eq.(23))  
    """
    return (m * t - mmse) / (t ** 2)


# -----------------------------------------------------------------------------
# Log-domain integration for MI (core integrators)
# -----------------------------------------------------------------------------

def make_log_grid(cfg: LogGridConfig) -> np.ndarray:
    """
    Geometric grid over [t_min, t_max] with M points (main Eq.(42)).  
    """
    t_min, t_max, M = cfg.t_min, cfg.t_max, cfg.m_points
    if t_min <= 0 or t_max <= 0 or t_max <= t_min:
        raise ValueError("Require 0 < t_min < t_max.")
    return np.geomspace(t_min, t_max, num=M).astype(np.float64)


def log_trapz(u: np.ndarray, f_u: np.ndarray) -> float:
    """
    Trapezoid rule on a *uniform u-grid*.  Used after u = log t change (main Eq.(43)(44)).  
    """
    if len(u) != len(f_u):
        raise ValueError("u and f(u) must have same length.")
    if len(u) < 2:
        return 0.0
    du = float(u[1] - u[0])
    return float(0.5 * du * (f_u[0] + 2.0 * f_u[1:-1].sum() + f_u[-1]))


def tail_correction_covtrace(trace_cov_x: float, t_max: float) -> float:
    """
    Universal tail ~ 0.5 * tr Cov(X) / t_max (main Eq.(45)).  
    """
    return 0.5 * (trace_cov_x / t_max)


def integrate_mi_log_trapz(
    t_vals: np.ndarray,
    J_vals: np.ndarray,
    dim_y: int,
    T_lower: float,
    tail: TailConfig,
    trace_cov_x: Optional[float] = None
) -> float:
    """
    MI via Fisher integral, evaluated in the *log domain* (main Eq.(43)(44)), plus tail (Eq.(45)).  
    """
    t = np.asarray(t_vals, dtype=np.float64)
    J = np.asarray(J_vals, dtype=np.float64)
    if t.shape != J.shape:
        raise ValueError("t and J must have same shape.")

    # select indices >= T_lower
    mask = t >= T_lower
    t, J = t[mask], J[mask]

    u = np.log(t)
    # l(u) = m - e^u J(e^u)    (derivation from Eq.(43))
    ell = (dim_y - np.exp(u) * J)

    I_num = 0.5 * log_trapz(u, ell)

    if tail.use_tail:
        if trace_cov_x is None:
            raise ValueError("trace_cov_x is required for tail correction.")
        I_num += tail_correction_covtrace(trace_cov_x, t_max=float(t[-1]))

    return float(I_num)


# -----------------------------------------------------------------------------
# High-level MI pipeline (kept for convenience)
# -----------------------------------------------------------------------------

def estimate_trace_cov_x(sampler_x: Callable[[int, torch.device], torch.Tensor],
                         device: torch.device, samples: int = 50_000) -> float:
    """
    Estimate tr Cov(X) from samples for tail correction (main Eq.(45)).  
    """
    @torch.no_grad()
    def _fn(B: int) -> torch.Tensor:
        x = sampler_x(B, device)
        return (x ** 2).sum(dim=-1, keepdim=True)

    tr_cov = mc_expect(_fn, N=samples, reduce="mean").item()
    return float(tr_cov)


def estimate_fisher_from_score(
    score: nn.Module,
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    frontend: nn.Module,
    t: float,
    fisher: FisherConfig,
    device: torch.device,
    noise_conditional: bool = False,
    stein_calibrate: bool = False
) -> float:
    """
    Wrapper for common pattern: J(Y_t) from a trained score model.

    If noise_conditional=True, `score(y, t)` is assumed. Otherwise `score(y)`.
    Optionally apply Stein calibration scale to reduce global bias (info_grad Sec.VII). 
    """
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend)
    channel.frontend = channel.frontend.to(device)

    @torch.no_grad()
    def _y_sampler(B: int) -> torch.Tensor:
        _, _, y = simulate_y(channel, t, B, device)
        return y

    if stein_calibrate:
        c = stein_calibrate_scalar(lambda yy: score(yy, t) if noise_conditional else score(yy),
                                   _y_sampler, B=8192)
        def _score_eval(y):  # scaled score
            return c * (score(y, t) if noise_conditional else score(y))
    else:
        def _score_eval(y):
            return score(y, t) if noise_conditional else score(y)

    return fisher_from_score(_score_eval, _y_sampler, N=fisher.mc_samples)


def estimate_mi_forward(
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    frontend: nn.Module,
    t_grid: LogGridConfig,
    dsm: DSMConfig,
    fisher: FisherConfig,
    tail: TailConfig,
    device: torch.device,
    conditional: str = "per_t"
) -> Dict[str, Any]:
    """
    Full pipeline: DSM  Fisher  MI (main Eq.(24),(27),(29),(41),(43)-(45)).  
    """
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend.to(device))
    t_vals = make_log_grid(t_grid)
    m = None

    J_list = []

    if conditional == "per_t":
        # train separate score per t
        for t in t_vals:
            # model
            model = ScoreNetMLP(dim=channel.y_dim or sampler_x(1, device).shape[1],
                                hidden=dsm.hidden, layers=dsm.layers, activation=dsm.activation).to(device)
            # train
            model = train_score_generic(
                make_model=lambda: model,
                loss_fn=lambda mdl, w, tt: dsm_loss_uncond(mdl, w, float(tt)),
                channel=channel, device=device, t=float(t),
                steps=dsm.steps, batch_size=dsm.batch_size,
                lr=dsm.lr, grad_clip=dsm.grad_clip, weight_decay=dsm.weight_decay
            )
            m = channel.y_dim or sampler_x(1, device).shape[1]
            # fisher
            Jt = estimate_fisher_from_score(model, sampler_x, frontend, float(t), fisher, device,
                                            noise_conditional=False, stein_calibrate=dsm.stein_calibrate)
            J_list.append(Jt)
    elif conditional == "noise_cond":
        # one conditional model for all t
        model = NoiseCondScoreNet(dim=channel.y_dim or sampler_x(1, device).shape[1],
                                  hidden=dsm.hidden, layers=dsm.layers,
                                  activation=dsm.activation, t_embed_dim=dsm.t_embed_dim).to(device)
        model = train_score_generic(
            make_model=lambda: model,
            loss_fn=lambda mdl, w, t_vec: dsm_loss_noise_cond(mdl, w, t_vec),
            channel=channel, device=device, t=(float(t_grid.t_min), float(t_grid.t_max)),
            steps=dsm.steps, batch_size=dsm.batch_size,
            lr=dsm.lr, grad_clip=dsm.grad_clip, weight_decay=dsm.weight_decay
        )
        m = channel.y_dim or sampler_x(1, device).shape[1]
        for t in t_vals:
            Jt = estimate_fisher_from_score(model, sampler_x, frontend, float(t), fisher, device,
                                            noise_conditional=True, stein_calibrate=dsm.stein_calibrate)
            J_list.append(Jt)
    else:
        raise ValueError("conditional must be 'per_t' or 'noise_cond'.")

    J_arr = np.asarray(J_list, dtype=np.float64)
    tr_cov_x = estimate_trace_cov_x(sampler_x, device, samples=tail.cov_trace_est_samples) if tail.use_tail else None
    I_hat = integrate_mi_log_trapz(t_vals, J_arr, dim_y=m, T_lower=t_grid.T_lower, tail=tail, trace_cov_x=tr_cov_x)

    return {
        "I_hat": I_hat,
        "t": t_vals,
        "J_hat": J_arr,
        "meta": {
            "dim_y": m,
            "conditional": conditional,
            "stein_calibrate": dsm.stein_calibrate,
        }
    }


# -----------------------------------------------------------------------------
# Information gradients via VJP (core) and alternation
# -----------------------------------------------------------------------------

def info_gradient(
    frontend: nn.Module,
    score_model: nn.Module,
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    t: float,
    batch_size: int,
    device: torch.device,
    noise_conditional: bool = False,
    stein_calibrate: bool = False
) -> Tuple[float, Tuple[torch.Tensor, ...]]:
    """
    nabla__eta I(X;Y_t) = -E[ Df_eta(X)^T s_Y(Y) ] via VJP (info_grad Eq.(10),(23),(25)).  
    Returns (Lvjp, grads) where -nabla__eta Lvjp = nabla__eta I.
    """
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend.to(device))
    x, w, y = simulate_y(channel, t, batch_size, device)

    if stein_calibrate:
        # calibrate on a separate batch
        def _y_sampler(B: int) -> torch.Tensor:
            _, _, yy = simulate_y(channel, t, B, device)
            return yy
        c = stein_calibrate_scalar(lambda yy: score_model(yy, t) if noise_conditional else score_model(yy), _y_sampler, B=batch_size)
    else:
        c = 1.0

    s_y = c * (score_model(y, t) if noise_conditional else score_model(y))
    loss = vjp_loss(frontend, s_y, x, stop_grad=True)
    # Compute grads w.r.t. frontend params
    for p in frontend.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    grads = tuple(p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in frontend.parameters())
    return float(loss.item()), grads


def task_info_gradient(
    frontend: nn.Module,
    score_uncond: nn.Module,
    score_cond: nn.Module,
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    task_fn: Callable[[torch.Tensor], torch.Tensor],
    t: float,
    batch_size: int,
    device: torch.device,
    noise_conditional: bool = False,
    stein_calibrate: bool = False
) -> Tuple[float, Tuple[torch.Tensor, ...]]:
    """
    nabla__eta I(T;Y_t) = E[ Df_eta^T ( s_{Y|T} - s_Y ) ] via VJP (info_grad Eq.(55)). 
    """
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend.to(device))
    x, w, y = simulate_y(channel, t, batch_size, device)
    tau = task_fn(x)

    # Optional calibration (same scalar applied to both for simplicity)
    if stein_calibrate:
        def _y_sampler(B: int) -> torch.Tensor:
            _, _, yy = simulate_y(channel, t, B, device)
            return yy
        c = stein_calibrate_scalar(lambda yy: score_uncond(yy, t) if noise_conditional else score_uncond(yy), _y_sampler, B=batch_size)
    else:
        c = 1.0

    sY = c * (score_uncond(y, t) if noise_conditional else score_uncond(y))
    sY_T = c * score_cond(y, tau)

    loss = vjp_loss(frontend, (sY_T - sY), x, stop_grad=True)
    for p in frontend.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    grads = tuple(p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in frontend.parameters())
    return float(loss.item()), grads


def ib_gradient(
    frontend: nn.Module,
    score_uncond: nn.Module,
    score_cond: nn.Module,
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    task_fn: Callable[[torch.Tensor], torch.Tensor],
    t: float,
    beta: float,
    batch_size: int,
    device: torch.device,
    noise_conditional: bool = False,
    stein_calibrate: bool = False
) -> Tuple[float, Tuple[torch.Tensor, ...]]:
    """
    nabla__eta [ I(T;Y_t) - beta I(X;Y_t) ] (info_grad Eq.(67)). 
    """
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend.to(device))
    x, w, y = simulate_y(channel, t, batch_size, device)
    tau = task_fn(x)

    if stein_calibrate:
        def _y_sampler(B: int) -> torch.Tensor:
            _, _, yy = simulate_y(channel, t, B, device)
            return yy
        c = stein_calibrate_scalar(lambda yy: score_uncond(yy, t) if noise_conditional else score_uncond(yy), _y_sampler, B=batch_size)
    else:
        c = 1.0

    sY = c * (score_uncond(y, t) if noise_conditional else score_uncond(y))
    sY_T = c * score_cond(y, tau)

    vec = sY_T + (beta - 1.0) * sY
    loss = vjp_loss(frontend, vec, x, stop_grad=True)
    for p in frontend.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    grads = tuple(p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in frontend.parameters())
    return float(loss.item()), grads


# -----------------------------------------------------------------------------
# Path-integral (eta-direction) - optional advanced utility
# -----------------------------------------------------------------------------

def integrate_along_path(grad_fn: Callable, thetas: list, *, cumulative: bool = True):
    """
    Numerically integrate a scalar (or dict of scalars) gradient along a 1D path {theta_k}.

    Parameters
    ----------
    grad_fn : Callable
        Function grad_fn(theta) -> float | torch.Tensor(0d/1d of size 1) | dict[str, float-like]
        Returns dI/dtheta evaluated at 'theta'.
    thetas : list
        Sequence of theta values in the desired order (e.g., np.linspace(...).tolist()).
    cumulative : bool, optional
        If True, return a list (or dict[str, list]) of cumulative integrals from the first theta.
        The first value is 0.0 (anchor at the first theta). Default: True.

    Returns
    -------
    list | float | dict
        If cumulative=True: list of cumulative integrals (or dict[str, list]).
        If cumulative=False: final integral value (or dict[str, float]).

    Notes
    -----
    Uses trapezoidal rule on a potentially non-uniform grid.
    """
    def _to_float(x):
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, torch.Tensor):
            return float(x.detach().mean().item())
        return float(x)

    # Evaluate gradients on the grid
    gvals = [grad_fn(th) for th in thetas]
    # Scalar or dict?
    is_dict = isinstance(gvals[0], dict)

    def _trapz_scalar(gs, xs):
        acc = [0.0]
        for i in range(1, len(xs)):
            h = float(xs[i] - xs[i-1])
            acc.append(acc[-1] + 0.5 * (_to_float(gs[i-1]) + _to_float(gs[i])) * h)
        return acc

    if not is_dict:
        return _trapz_scalar(gvals, thetas) if cumulative else _trapz_scalar(gvals, thetas)[-1]

    # dict[str, scalar] case
    keys = list(gvals[0].keys())
    series = {k: [] for k in keys}
    for k in keys:
        series[k] = _trapz_scalar([gv[k] for gv in gvals], thetas)
    return series if cumulative else {k: v[-1] for k, v in series.items()}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def project_to_frobenius_ball(A: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Project a matrix to Frobenius ball of given radius.
    If ||A||_F > radius, scale A to ||A||_F = radius.

    Parameters
    ----------
    A : torch.Tensor
        Matrix to project (modified in-place if needed).
    radius : float
        Frobenius norm constraint.

    Returns
    -------
    torch.Tensor
        Projected matrix (same object as input, modified in-place).
    """
    with torch.no_grad():
        fro = torch.linalg.norm(A, ord='fro')
        if fro > radius:
            A = A * (radius / fro)
    return A


def mi_linear_gaussian(A: torch.Tensor, sigma_x2: float, t: float) -> float:
    """
    Mutual information for linear vector Gaussian channel via log-det formula.

    Channel: Y = A X + Z_t,  X ~ N(0, sigma_x2 I),  Z_t ~ N(0, t I)

    Formula: I(A) = 0.5 * log det(I + (sigma_x2/t) A A^T)  [nats]

    Parameters
    ----------
    A : torch.Tensor
        Channel matrix (m x n).
    sigma_x2 : float
        Input variance.
    t : float
        Noise variance.

    Returns
    -------
    float
        Mutual information [nats].
    """
    with torch.no_grad():
        m = A.shape[0]
        AA = (sigma_x2 / t) * (A @ A.T)
        I = torch.eye(m, device=A.device, dtype=A.dtype)
        M = I + AA
        # Use Cholesky for numerical stability: logdet(M) = 2 sum log diag(L)
        L = torch.linalg.cholesky(M)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        return float(0.5 * logdet.item())


# -----------------------------------------------------------------------------
# Backward-compat thin wrappers (optional). These call the generic trainer.
# -----------------------------------------------------------------------------

def train_dsm_uncond(
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    frontend: nn.Module,
    t: float,
    dsm: DSMConfig,
    device: torch.device,
    y_dim: Optional[int] = None
) -> nn.Module:
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend.to(device), y_dim=y_dim)
    def _make(): 
        return ScoreNetMLP(dim=channel.y_dim or sampler_x(1, device).shape[1],
                           hidden=dsm.hidden, layers=dsm.layers, activation=dsm.activation).to(device)
    model = train_score_generic(
        make_model=_make,
        loss_fn=lambda mdl, w, tt: dsm_loss_uncond(mdl, w, float(tt)),
        channel=channel, device=device, t=float(t),
        steps=dsm.steps, batch_size=dsm.batch_size,
        lr=dsm.lr, grad_clip=dsm.grad_clip, weight_decay=dsm.weight_decay
    )
    return model


def train_dsm_noise_cond(
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    frontend: nn.Module,
    t_min: float,
    t_max: float,
    dsm: DSMConfig,
    device: torch.device,
    y_dim: Optional[int] = None
) -> nn.Module:
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend.to(device), y_dim=y_dim)
    def _make():
        return NoiseCondScoreNet(dim=channel.y_dim or sampler_x(1, device).shape[1],
                                 hidden=dsm.hidden, layers=dsm.layers,
                                 activation=dsm.activation, t_embed_dim=dsm.t_embed_dim).to(device)
    model = train_score_generic(
        make_model=_make,
        loss_fn=lambda mdl, w, t_vec: dsm_loss_noise_cond(mdl, w, t_vec),
        channel=channel, device=device, t=(float(t_min), float(t_max)),
        steps=dsm.steps, batch_size=dsm.batch_size,
        lr=dsm.lr, grad_clip=dsm.grad_clip, weight_decay=dsm.weight_decay
    )
    return model


def train_dsm_conditional_task(
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    frontend: nn.Module,
    task_fn: Callable[[torch.Tensor], torch.Tensor],
    t: float,
    dsm: DSMConfig,
    device: torch.device,
    y_dim: Optional[int] = None,
    tau_dim: Optional[int] = None
) -> nn.Module:
    channel = ChannelSpec(sampler_x=sampler_x, frontend=frontend.to(device), y_dim=y_dim)
    if tau_dim is None:
        with torch.no_grad():
            x0 = sampler_x(1, device)
            tau_dim = int(task_fn(x0).shape[1])
    def _make():
        return CondTaskScoreNet(y_dim=channel.y_dim or sampler_x(1, device).shape[1],
                                tau_dim=tau_dim, hidden=dsm.hidden, layers=dsm.layers,
                                activation=dsm.activation).to(device)
    # create a closure to fetch tau=g(X) freshly every step
    def _loss_fn(mdl, w, tt):
        x_tmp = channel.sampler_x(w.shape[0], device)
        tau = task_fn(x_tmp)
        return dsm_loss_conditional(mdl, w, tau, float(t))
    model = train_score_generic(
        make_model=_make,
        loss_fn=_loss_fn,
        channel=channel, device=device, t=float(t),
        steps=dsm.steps, batch_size=dsm.batch_size,
        lr=dsm.lr, grad_clip=dsm.grad_clip, weight_decay=dsm.weight_decay
    )
    return model
# -----------------------------------------------------------------------------
# KDE-LOO Mutual Information (Gaussian kernel) - reusable baseline
# Implements Eq. (50) for Y_t = W + sqrt(t) Z with a Gaussian kernel.
# This baseline is used in Fig.6 (tanh channel).
# -----------------------------------------------------------------------------

@torch.no_grad()
def _pairwise_sq_dists(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Full pairwise squared distances ||y_i - w_j||^2, shape (N,N).
    """
    y2 = (y**2).sum(dim=1, keepdim=True)
    w2 = (w**2).sum(dim=1, keepdim=True).T
    return y2 + w2 - 2.0 * (y @ w.T)


@torch.no_grad()
def mi_kde_loo_gaussian_pairs(
    w: torch.Tensor, y: torch.Tensor, t: float, *, chunk: Optional[int] = None
) -> float:
    """
    KDE-LOO estimator of I(W; Y_t) with a Gaussian kernel (Eq. (50)).

    Parameters
    ----------
    w : (N, m)  clean outputs W = f(X)
    y : (N, m)  noisy outputs Y_t = W + sqrt(t) * Z
    t : float    noise variance
    chunk : Optional[int]
        If None or chunk >= N: builds a full (N x N) log-kernel matrix and
        uses logsumexp (fast but O(N^2) memory).
        If a positive int: runs a memory-safe streaming log-sum-exp across
        column blocks of size `chunk` (O(N^2) time, low memory).

    Returns
    -------
    float : estimated MI [nats]
    """
    device = y.device
    N, m = y.shape
    assert w.shape == y.shape
    t = float(t)

    # term1 = -||y_i - w_i||^2 / (2t)
    d_ii = (y - w).pow(2).sum(dim=1).double()
    term1 = - d_ii / (2.0 * t)

    # Full matrix route
    if chunk is None or (chunk >= N):
        D = _pairwise_sq_dists(y.double(), w.double())     # (N,N)
        S = - D / (2.0 * t)                                # log kernel
        idx = torch.arange(N, device=device)
        S[idx, idx] = -float("inf")                        # LOO
        logsum = torch.logsumexp(S, dim=1) - math.log(N - 1)
        term2 = - logsum
        return float((term1 + term2).mean().item())

    # Streaming route (rows in blocks, columns in chunks)
    Iblock = min(1024, N)
    vals = []
    for i0 in range(0, N, Iblock):
        i1 = min(i0 + Iblock, N)
        y_blk = y[i0:i1].double()
        Bi = y_blk.shape[0]
        m_row = torch.full((Bi,), -float("inf"), dtype=torch.float64, device=device)
        s_row = torch.zeros(Bi, dtype=torch.float64, device=device)

        for j0 in range(0, N, chunk):
            j1 = min(j0 + chunk, N)
            w_blk = w[j0:j1].double()
            D = (y_blk.unsqueeze(1) - w_blk.unsqueeze(0)).pow(2).sum(dim=2)  # (Bi, Bj)
            S_blk = - D / (2.0 * t)

            # mask diagonal when row/col blocks overlap
            if (i0 <= j1 - 1) and (j0 <= i1 - 1):
                diag = torch.arange(max(i0, j0), min(i1, j1), device=device)
                if len(diag) > 0:
                    S_blk[(diag - i0), (diag - j0)] = -float("inf")

            m_blk = torch.max(S_blk, dim=1).values
            new_m = torch.maximum(m_row, m_blk)
            s_row = s_row * torch.exp(m_row - new_m) + torch.sum(torch.exp(S_blk - new_m.unsqueeze(1)), dim=1)
            m_row = new_m

        logsum_blk = torch.log(s_row + 1e-300) + m_row - math.log(N - 1)
        term2_blk = - logsum_blk
        vals.append(term1[i0:i1] + term2_blk)

    return float(torch.cat(vals, dim=0).mean().item())


@torch.no_grad()
def estimate_mi_kde_loo(
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    frontend: nn.Module,
    t: float,
    N: int,
    device: torch.device,
    *, chunk: Optional[int] = None
) -> float:
    """
    High-level wrapper for KDE-LOO MI (Eq. (50)).  Samples (W,Y_t) forward and calls
    `mi_kde_loo_gaussian_pairs`.  Intended for reuse in experiments (e.g., Fig.6 tanh channel).
    """
    x = sampler_x(N, device)
    w = frontend.to(device)(x)
    y = w + math.sqrt(float(t)) * torch.randn_like(w)
    return mi_kde_loo_gaussian_pairs(w, y, float(t), chunk=chunk)

# -----------------------------------------------------------------------------
# Information-gradient helpers (VJP)
# -----------------------------------------------------------------------------

def vjp_loss(frontend: nn.Module,
             score_eval: Callable[[torch.Tensor], torch.Tensor],
             x: torch.Tensor,
             z: torch.Tensor,
             t: float,
             stop_grad: bool = True) -> torch.Tensor:
    f = frontend(x)
    y = f + math.sqrt(float(t)) * z
    s = score_eval(y)
    if stop_grad:
        s = s.detach()           # stop-gradient on the score side (Eq.(25))
    return (f * s).sum(dim=1).mean()

@torch.no_grad()
def stein_calibrate_scalar(score: nn.Module,
                           y_sampler: Callable[[int], torch.Tensor],
                           m: int,
                           B: int = 8192) -> float:
    y = y_sampler(B); s = score(y)
    den = (y * s).sum(dim=1).mean().item()
    return float(-m / (den + 1e-12))

def estimate_info_grad(
    frontend: nn.Module,
    score_eval: Callable[[torch.Tensor], torch.Tensor],
    sampler_x: Callable[[int, torch.device], torch.Tensor],
    t: float,
    N: int,
    batch_size: int,
    device: torch.device,
    params: Optional[Tuple[nn.Parameter, ...]] = None,
    stop_grad_score: bool = True,
) -> Dict[str, float]:
    if params is None:
        params = tuple(p for p in frontend.parameters() if p.requires_grad)
    for p in params:
        if p.grad is not None:
            p.grad = None

    done = 0
    while done < N:
        B = min(batch_size, N - done)
        x = sampler_x(B, device)
        with torch.no_grad():
            m = int(frontend(x[:1]).shape[1])
        z = torch.randn(B, m, device=device)
        L = vjp_loss(frontend, score_eval, x, z, t, stop_grad=stop_grad_score)
        (B / float(N) * L).backward()
        done += B

    out = {}
    for name, p in frontend.named_parameters():
        if p in params and p.grad is not None:
            g = -p.grad.detach().clone()
            out[name] = float(g.item()) if g.numel() == 1 else g.cpu().numpy()
            p.grad = None
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # utils
    "set_seed", "ensure_device", "chunked_apply",
    # channel
    "ChannelSpec", "simulate_y",
    # models
    "ScoreNetMLP", "NoiseCondScoreNet", "CondTaskScoreNet",
    # configs
    "DSMConfig", "LogGridConfig", "FisherConfig", "TailConfig",
    # DSM core & train
    "dsm_loss_uncond", "dsm_loss_noise_cond", "dsm_loss_conditional", "train_score_generic",
    "train_dsm_uncond", "train_dsm_noise_cond", "train_dsm_conditional_task",
    # MC & Fisher/MMSE
    "mc_expect", "fisher_from_score", "mmse_from_fisher", "fisher_from_mmse",
    # integration
    "make_log_grid", "log_trapz", "tail_correction_covtrace", "integrate_mi_log_trapz",
    # high-level
    "estimate_trace_cov_x", "estimate_fisher_from_score", "estimate_mi_forward",
    # Stein & VJP & gradients
    "stein_calibrate_scalar", "vjp_loss", "estimate_info_grad",
    "info_gradient", "task_info_gradient", "ib_gradient",
    # advanced
    "integrate_along_path",
    # utilities
    "project_to_frobenius_ball", "mi_linear_gaussian",
    # KDE-LOO
    "mi_kde_loo_gaussian_pairs", "estimate_mi_kde_loo",
]
