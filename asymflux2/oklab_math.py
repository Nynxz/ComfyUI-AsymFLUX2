# Math derived (with modifications) from:
#   https://github.com/Lakonik/LakonLab/blob/main/lakonlab/models/architectures/autoencoders/color_encoders.py
#   Copyright (c) 2026 Hansheng Chen, Apache License 2.0
# Oklab matrices: Bjorn Ottosson's public spec (https://bottosson.github.io/posts/oklab/)
# sRGB conversion: IEC 61966-2-1
"""Pure-tensor port of LakonLab's OklabColorEncoder, used as the AsymFLUX2
"VAE" — a deterministic perceptual-color transform between sRGB images and
the 3-channel Oklab pixel-latent space the transformer denoises in.

Constants and matrices match
``lakonlab/models/architectures/autoencoders/color_encoders.py`` line for
line, so a roundtrip ``decode(encode(img))`` is identical to upstream.
"""

from __future__ import annotations

import torch


# Linear-RGB -> LMS  (Bjorn Ottosson, Oklab spec)
_LRGB_TO_LMS = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=torch.float32)

# LMS^(1/3) -> Oklab
_LMS_TO_OKLAB = torch.tensor([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=torch.float32)


def srgb_to_lrgb(srgb: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)


def lrgb_to_srgb(lrgb: torch.Tensor) -> torch.Tensor:
    lrgb = lrgb.clamp(min=0)
    a = 0.055
    return torch.where(lrgb <= 0.0031308, lrgb * 12.92, (1 + a) * (lrgb ** (1 / 2.4)) - a)


def lrgb_to_oklab(lrgb: torch.Tensor) -> torch.Tensor:
    """``lrgb``: (N, 3, *)."""
    m1 = _LRGB_TO_LMS.to(device=lrgb.device, dtype=lrgb.dtype)
    m2 = _LMS_TO_OKLAB.to(device=lrgb.device, dtype=lrgb.dtype)
    lms = torch.einsum("ij,bj...->bi...", m1, lrgb).clamp(min=0)
    return torch.einsum("ij,bj...->bi...", m2, lms.pow(1 / 3))


def oklab_to_lrgb(oklab: torch.Tensor) -> torch.Tensor:
    """``oklab``: (N, 3, *)."""
    m2_inv = torch.linalg.inv(_LMS_TO_OKLAB).to(device=oklab.device, dtype=oklab.dtype)
    m1_inv = torch.linalg.inv(_LRGB_TO_LMS).to(device=oklab.device, dtype=oklab.dtype)
    lms = torch.einsum("ij,bj...->bi...", m2_inv, oklab).pow(3)
    lrgb = torch.einsum("ij,bj...->bi...", m1_inv, lms)
    return lrgb.clamp(0, 1)


def encode_image_to_oklab(
    img_pm1: torch.Tensor,
    affine_mean: tuple[float, float, float] = (0.56, 0.0, 0.01),
    affine_std: float | tuple[float, float, float] = 0.16,
) -> torch.Tensor:
    """Match ``OklabColorEncoder.encode``.
    ``img_pm1`` is in [-1, 1] with shape (N, 3, *).
    Returns affine-normalized Oklab in (N, 3, *).
    """
    rgb_01 = img_pm1 * 0.5 + 0.5
    lrgb = srgb_to_lrgb(rgb_01)
    oklab = lrgb_to_oklab(lrgb)

    mean = torch.tensor(affine_mean, dtype=oklab.dtype, device=oklab.device)
    std_vec = torch.tensor(
        affine_std if isinstance(affine_std, tuple) else (affine_std,) * 3,
        dtype=oklab.dtype, device=oklab.device,
    )
    n_dim = oklab.dim() - 2
    mean = mean.reshape(-1, *([1] * n_dim))
    std_vec = std_vec.reshape(-1, *([1] * n_dim))
    return (oklab - mean) / std_vec


def decode_oklab_to_image(
    oklab_norm: torch.Tensor,
    affine_mean: tuple[float, float, float] = (0.56, 0.0, 0.01),
    affine_std: float | tuple[float, float, float] = 0.16,
) -> torch.Tensor:
    """Match ``OklabColorEncoder.decode``.
    Returns an image in [-1, 1] with shape (N, 3, *).
    """
    mean = torch.tensor(affine_mean, dtype=oklab_norm.dtype, device=oklab_norm.device)
    std_vec = torch.tensor(
        affine_std if isinstance(affine_std, tuple) else (affine_std,) * 3,
        dtype=oklab_norm.dtype, device=oklab_norm.device,
    )
    n_dim = oklab_norm.dim() - 2
    mean = mean.reshape(-1, *([1] * n_dim))
    std_vec = std_vec.reshape(-1, *([1] * n_dim))
    oklab = oklab_norm * std_vec + mean

    lrgb = oklab_to_lrgb(oklab)
    rgb_01 = lrgb_to_srgb(lrgb)
    return rgb_01 * 2 - 1
