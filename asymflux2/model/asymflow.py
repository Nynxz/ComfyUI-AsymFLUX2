"""AsymFlow calibration + velocity math.

Port of ``lakonlab/models/architectures/asymflow/common.py`` (``AsymFlowMixin``)
working directly on ComfyUI's standard ``(B, 3, H, W)`` pixel-space tensors.
The transformer's *raw* output (a velocity prediction in pixel space) is
post-processed through an orthogonal decomposition using
``proj_buffer ∈ R^(768x128)`` so the low-rank subspace and its complement
are blended with the current ``x_t`` to produce the AsymFlow velocity.

For ``num_timesteps == 1`` (the published config), ``sigma == timestep``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from einops import rearrange


PATCH_DIM = 768       # 3 channels * 16 * 16 patch
BASE_RANK = 128       # AsymFlow projection rank
PATCH_SIZE = 16


class Calibration(NamedTuple):
    s: torch.Tensor         # ()       — scalar
    k: torch.Tensor         # (B, 1)   — per-batch scale
    timestep: torch.Tensor  # (B,)     — cal_timestep
    sigma: torch.Tensor     # (B, 1)   — original sigma


def asymflow_calibration(
    timestep: torch.Tensor,
    scale_buffer: torch.Tensor,
    num_timesteps: int = 1,
) -> Calibration:
    """``timestep`` shape ``(B,)``. Returns scalars / per-batch vectors ready
    to broadcast over packed-token shape ``(B, T, D)``.
    """
    t = timestep.float()
    s = scale_buffer.float()
    sigma = t / num_timesteps                # for num_timesteps=1, sigma == t
    k = 1.0 / (s + (1.0 - s) * sigma)        # (B,)
    cal_t = t * k
    bs = t.shape[0]
    return Calibration(
        s=s,
        k=k.reshape(bs, 1),
        timestep=cal_t,
        sigma=sigma.reshape(bs, 1),
    )


def _pack(img: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """``(B, 3, H, W) -> (B, T, 3*ph*pw)``."""
    return rearrange(
        img, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch_size, pw=patch_size,
    )


def _unpack(tokens: torch.Tensor, h: int, w: int, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Inverse of ``_pack``. ``h``, ``w`` are token-grid dimensions."""
    return rearrange(
        tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h, w=w, ph=patch_size, pw=patch_size,
    )


def asymflow_velocity(
    u_a_image: torch.Tensor,   # (B, 3, H, W) raw transformer output
    x_t_image: torch.Tensor,   # (B, 3, H, W) original noisy input
    cal: Calibration,
    proj_buffer: torch.Tensor, # (768, 128)
    sigma_min: float = 1e-4,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Compose the AsymFlow velocity from the raw transformer output and
    the current ``x_t``. Returns a pixel-space tensor of the same shape.

    The math mirrors ``AsymFlowMixin.asymflow_velocity``:
    decompose both ``u_a`` and ``x_t`` along the low-rank subspace defined
    by ``proj_buffer`` and its complement, then blend each component using
    ``s``, ``k``, and the clamped sigma.
    """
    bs, _, h_px, w_px = x_t_image.shape
    h_tok = h_px // patch_size
    w_tok = w_px // patch_size
    target_device = u_a_image.device

    u_a = _pack(u_a_image, patch_size).float()
    x_t = _pack(x_t_image, patch_size).float()
    # proj_buffer / cal tensors may be staged on cpu (the base model's
    # img_in.weight was on meta when surgery ran). Migrate to the live
    # forward device on first use; this is essentially free for the small
    # 768x128 projection.
    proj = proj_buffer.to(device=target_device, dtype=torch.float32)

    # orthogonal decomposition
    u_a_sub = u_a @ proj @ proj.T
    u_a_comp = u_a - u_a_sub
    x_t_sub = x_t @ proj @ proj.T
    x_t_comp = x_t - x_t_sub

    s = cal.s.to(device=target_device, dtype=torch.float32)
    k_b = cal.k.to(device=target_device, dtype=torch.float32).unsqueeze(-1)
    sigma_b = cal.sigma.to(device=target_device, dtype=torch.float32).unsqueeze(-1)
    sigma_clamped = sigma_b.clamp(min=sigma_min)
    sk = s * k_b

    u_sub = sk * u_a_sub + (1.0 - sk) / sigma_clamped * x_t_sub
    u_comp = (x_t_comp + s * u_a_comp) / sigma_clamped
    u = u_sub + u_comp

    return _unpack(u.to(u_a_image.dtype), h_tok, w_tok, patch_size)
