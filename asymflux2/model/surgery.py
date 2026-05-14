"""Model surgery — turn a ComfyUI ``Flux2`` model into an AsymFLUX.2 model
*without* mutating the underlying torch module.

Critical correctness rule
-------------------------
``ModelPatcher.clone()`` only clones the patcher wrapper — the underlying
``model`` (and ``model.diffusion_model``) is shared by reference. If we
``setattr`` directly we corrupt the user's base FLUX.2-klein model so any
other workflow holding the same MODEL socket will explode at the first
forward.

Everything goes through ``model_patcher.add_object_patch(name, value)``,
which uses ``comfy.utils.set_attr`` — that resolves dotted nested paths
(including ``nn.Sequential[index]`` via ``Sequential._modules['1']``) and
backs the original up to ``object_patches_backup`` for clean restore on
``unpatch_model``. So our patches apply when this MODEL socket is loaded
and revert when it's unloaded — the base model is never mutated.

What we change relative to the stock FLUX.2-klein-base-9B:

1. ``patch_size`` 1 → 16. ``Flux.process_img`` reads ``self.patch_size``
   every forward.
2. ``img_in`` swapped to ``Linear(768, 4096)`` (3-channel × 16×16).
3. ``final_layer.linear`` swapped to ``Linear(4096, 768)``.
4. ``final_layer.adaLN_modulation[1]`` re-weighted from the adapter.
5. ``guidance_in`` replaced with ``nn.Identity()`` because the adapter
   was trained with ``guidance_embeds=False``.
6. ``in_channels`` / ``out_channels`` updated.
7. ``proj_buffer`` and ``scale_buffer`` exposed as plain attributes (not
   ``register_buffer``, since we can't cleanly un-register a buffer on
   unpatch — ``add_object_patch`` handles regular attributes fine).
8. ``diffusion_model.forward`` wrapped with the AsymFlow calibration +
   velocity. The original ``forward`` is preserved and called inside.
9. ``model_sampling`` set to a flux-shift of 17.0.
10. ``latent_format`` set to a 3-channel pass-through so KSampler does
    not inflate our pixel latent up to the stock Flux2 128 channels.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Optional

import torch
import torch.nn as nn

import comfy.latent_formats
import comfy.model_sampling
import comfy.utils

from .asymflow import (
    PATCH_DIM,
    PATCH_SIZE,
    asymflow_calibration,
    asymflow_velocity,
)
from ..oklab_math import decode_oklab_to_image, encode_image_to_oklab


# Lakonik AsymFLUX.2-klein default Oklab encoder settings (see demo).
_OKLAB_MEAN = (0.56, 0.0, 0.01)
_OKLAB_STD = 0.16


class AsymFlux2PixelLatent(comfy.latent_formats.LatentFormat):
    """3-channel, 1:1 pixel-space latent format. No-op VAE pair so KSampler
    leaves our 3-channel empty latent alone instead of repeating it up to
    the stock Flux2 128 channels."""

    latent_channels = 3
    spacial_downscale_ratio = 1

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


def _log(msg: str) -> None:
    print(f"[AsymFLUX2] {msg}", flush=True)


def _safe_dtype_device(weight: torch.Tensor) -> tuple[torch.dtype, torch.device]:
    dev = weight.device
    if dev.type == "meta":
        dev = torch.device("cpu")
    return weight.dtype, dev


def _new_linear_from(
    weight: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    bias_tensor: Optional[torch.Tensor] = None,
) -> nn.Linear:
    out_f, in_f = weight.shape
    new = nn.Linear(in_f, out_f, bias=bias_tensor is not None, dtype=dtype, device=device)
    with torch.no_grad():
        new.weight.copy_(weight.to(dtype=dtype, device=device))
        if bias_tensor is not None:
            new.bias.copy_(bias_tensor.to(dtype=dtype, device=device))
    return new


_ASYMFLUX2_FWD_MARK = "__asymflux2_wrapped_forward__"


def _make_asymflux2_forward(
    diffusion_model,
    proj_buffer: torch.Tensor,
    scale_buffer: torch.Tensor,
    sigma_min: float = 1e-4,
):
    """Build the wrapped forward closure that runs AsymFlow calibration
    around the original ``Flux.forward``.

    CRITICAL: capture the *class-level* ``forward`` (bound to this
    instance), NOT ``diffusion_model.forward``. If a previous Apply
    Adapter run's patch is still installed at the moment surgery is
    called again, ``diffusion_model.forward`` would be the PREVIOUS
    wrapper -- closing over it would stack wrappers on every re-run,
    inflate memory by ~80 MB per generation, and degrade output quality
    each iteration. The class attribute is always the pristine
    ``Flux.forward`` (which itself dispatches through
    ``comfy.patcher_extension.WrapperExecutor`` to ``_forward``, so
    we don't lose the wrapper-system features either).
    """
    cls = type(diffusion_model)
    original_forward = cls.forward.__get__(diffusion_model, cls)

    def asymflux2_forward(
        self, x, timestep, context,
        y=None, guidance=None, ref_latents=None, control=None,
        transformer_options=None, **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}
        # scale_buffer was cached on cpu during surgery; move to live device
        # so the resulting `k` lands on the same device as `x`.
        s_dev = scale_buffer.to(device=x.device)
        cal = asymflow_calibration(timestep.float(), s_dev)
        k_x = cal.k.reshape(-1, 1, 1, 1).to(dtype=x.dtype, device=x.device)
        x_scaled = x * k_x
        cal_t = cal.timestep.to(dtype=timestep.dtype, device=timestep.device)

        # Image-editing path: upstream divides reference tokens by `s`
        # before they go through x_embedder. We do the equivalent in image
        # space (linearity of patchify makes pre-scaling commutative with
        # the rearrange). Comfy's flux2 `_forward` will then patchify and
        # concat ref tokens at index-offset positions (`ref_index_scale=10`
        # matches upstream's `_prepare_condition_latent_ids(scale=10)`).
        if ref_latents:
            s_val = float(s_dev.detach().to(torch.float32).item())
            ref_latents_scaled = [
                r.to(device=x.device, dtype=x.dtype) / s_val for r in ref_latents
            ]
        else:
            ref_latents_scaled = ref_latents

        u_a = original_forward(
            x_scaled, cal_t, context,
            y=y, guidance=guidance, ref_latents=ref_latents_scaled,
            control=control, transformer_options=transformer_options, **kwargs,
        )
        v = asymflow_velocity(
            u_a, x, cal,
            proj_buffer=proj_buffer,
            sigma_min=sigma_min,
            patch_size=PATCH_SIZE,
        )
        return v

    return asymflux2_forward


def apply_asymflux2_surgery(
    model_patcher,
    overwrites: dict[str, torch.Tensor],
) -> None:
    """Register all AsymFLUX.2-specific changes as object patches on
    ``model_patcher``. The base ``model.diffusion_model`` is NOT mutated —
    everything is reverted automatically when ``unpatch_model`` runs."""
    diffusion_model = model_patcher.model.diffusion_model
    sample_w = diffusion_model.img_in.weight
    dtype, device = _safe_dtype_device(sample_w)
    _log(f"surgery: base dtype={dtype}, target device={device}")

    p = model_patcher.add_object_patch
    dm = "diffusion_model"

    # 1. patch_size + channel counts (plain int attrs)
    p(f"{dm}.patch_size", PATCH_SIZE)
    p(f"{dm}.params.patch_size", PATCH_SIZE)
    p(f"{dm}.params.in_channels", 3)
    p(f"{dm}.params.out_channels", 3)
    p(f"{dm}.in_channels", PATCH_DIM)
    p(f"{dm}.out_channels", PATCH_DIM)
    _log("surgery: patch_size=16, in/out_channels=3 set")

    # 2. img_in
    x_emb = overwrites["x_embedder.weight"]
    p(f"{dm}.img_in", _new_linear_from(x_emb, dtype, device))
    _log(f"surgery: img_in -> Linear({x_emb.shape[1]}, {x_emb.shape[0]})")

    # 3. final_layer.linear
    proj_out = overwrites["proj_out.weight"]
    p(f"{dm}.final_layer.linear", _new_linear_from(proj_out, dtype, device))
    _log(f"surgery: final_layer.linear -> Linear({proj_out.shape[1]}, {proj_out.shape[0]})")

    # 4. final_layer.adaLN_modulation[1]. Sequential children resolve via
    # `_modules['1']`; comfy.utils.set_attr handles the dotted path.
    #
    # CRITICAL: the adapter's ``norm_out.linear.weight`` is in diffusers'
    # AdaLayerNormContinuous convention -- ``scale, shift = chunk(emb, 2)``
    # -- but comfy's LastLayer.forward reads ``shift, scale = chunk(...)``.
    # We have to ``swap_scale_shift`` the two halves before installing it
    # or every patch's adaLN ends up with shift and scale swapped, which
    # produces output that has roughly the right colors but distorted
    # patch-by-patch (== the symptom we were chasing). Comfy itself uses
    # this exact swap when converting diffusers MMDIT -> BFL/comfy layout
    # in MMDIT_MAP_BASIC.
    norm_w = overwrites.get("norm_out.linear.weight")
    if norm_w is not None:
        norm_w_comfy = comfy.utils.swap_scale_shift(norm_w)
        p(f"{dm}.final_layer.adaLN_modulation.1", _new_linear_from(norm_w_comfy, dtype, device))
        _log(
            f"surgery: final_layer.adaLN_modulation[1] -> Linear({norm_w_comfy.shape[1]}, "
            f"{norm_w_comfy.shape[0]}) (scale/shift halves swapped: diffusers -> BFL/comfy)"
        )

    # 5. disable guidance branch
    p(f"{dm}.guidance_in", nn.Identity())
    p(f"{dm}.params.guidance_embed", False)
    _log("surgery: guidance_in disabled")

    # 6. AsymFlow tensors as plain attributes (not buffers — add_object_patch
    # round-trips regular attrs cleanly via set_attr).
    proj_buf = overwrites["proj_buffer"].to(dtype=torch.float32, device=device)
    scale_buf = overwrites["scale_buffer"].to(dtype=torch.float32, device=device)
    p(f"{dm}.asymflow_proj_buffer", proj_buf)
    p(f"{dm}.asymflow_scale_buffer", scale_buf)
    p(f"{dm}.sigma_min", 1e-4)
    _log(f"surgery: proj_buffer {tuple(proj_buf.shape)} + scale_buffer staged")

    # 7. wrapped forward. The closure captures `cls.forward.__get__(...)`
    # (the class-level Flux.forward), so even if a previous patcher is
    # still installed at this moment, we don't stack wrappers.
    fwd = _make_asymflux2_forward(diffusion_model, proj_buf, scale_buf, sigma_min=1e-4)
    setattr(fwd, _ASYMFLUX2_FWD_MARK, True)
    p(f"{dm}.forward", MethodType(fwd, diffusion_model))
    _log("surgery: forward wrapped with AsymFlow calibration + velocity")


def patch_orthogonal_cfg(model_patcher, orthogonal_guidance: float) -> None:
    """Register the AsymFLUX.2 orthogonal CFG bias as a post-CFG hook.

    This is a direct port of upstream ``guidance_jit`` operating in
    *velocity* space (where AsymFlow trained the formula). We reconstruct
    v_cond, v_uncond from comfy's x0 estimates via ``v = (x - x0)/sigma``,
    apply the formula, then convert back. Operating in velocity space
    matters numerically: at small sigma the x0-space bias collapses to
    near-zero while the velocity-space components are well-conditioned —
    doing the projection in x0 space causes accumulated numeric error
    that diverges to NaN over 38 steps.

    ``orthogonal_guidance=0`` is a no-op (returns standard CFG). ``=1``
    matches the upstream demo default.
    """
    if orthogonal_guidance <= 0.0:
        return

    def orthog_cfg(args):
        denoised = args["denoised"]        # standard-CFG result, x0 space
        cond = args["cond_denoised"]
        uncond = args["uncond_denoised"]
        sigma = args["sigma"]
        x_in = args["input"]
        cfg = args["cond_scale"]

        # Bail out if comfy hasn't produced sensible inputs (no uncond at
        # cfg=1.0, etc.) -- preserve standard behavior in that case.
        if uncond is None or cond is None:
            return denoised
        if cfg <= 1.0:
            return denoised

        sigma_b = sigma
        while sigma_b.dim() < x_in.dim():
            sigma_b = sigma_b.unsqueeze(-1)
        sigma_b = sigma_b.clamp(min=1e-4)

        # Reconstruct velocities (model space) from x0 estimates
        v_cond = (x_in - cond) / sigma_b
        v_uncond = (x_in - uncond) / sigma_b
        bias_v = (v_cond - v_uncond) * (cfg - 1.0)
        parallel = cond  # AsymFlux2 paper: parallel_dir = x_t - v_cond*sigma

        dims = list(range(1, bias_v.dim()))
        num = (bias_v * parallel).mean(dim=dims, keepdim=True)
        den = (parallel * parallel).mean(dim=dims, keepdim=True).clamp(min=1e-6)
        bias_v_orthog = bias_v - (num / den) * parallel * orthogonal_guidance

        v_final = v_cond + bias_v_orthog
        out = x_in - v_final * sigma_b

        # Defensive: if anything went NaN/Inf (e.g. early-step instability
        # combined with bf16 noise), fall back to the standard-CFG denoised.
        if not torch.isfinite(out).all():
            return denoised
        return out

    model_patcher.set_model_sampler_post_cfg_function(orthog_cfg)


def patch_clamp_denoised(model_patcher) -> None:
    """Per-step Oklab gamut clamp on the x0 estimate.

    Upstream ``clamp_denoised=True`` default: at every sampling step the
    predicted x0 is decoded to sRGB, clipped to ``[-1, 1]`` to stay
    in-gamut, then re-encoded back to Oklab. Without this the predicted
    x0 can drift out of the valid color space, and small per-step errors
    compound over 38 steps into oversaturated / washed-out output.

    Implemented as a post-CFG hook, so it composes correctly with our
    orthogonal CFG hook (both operate on the same ``denoised`` x0 the
    sampler then converts back to velocity for the integration step).
    """
    def clamp_hook(args):
        denoised = args["denoised"]
        # Cast to fp32 for the color math; the Oklab matrices and pow(1/3)
        # are sensitive to bf16 roundoff and the cost is trivial.
        d_f32 = denoised.float()
        img = decode_oklab_to_image(d_f32, affine_mean=_OKLAB_MEAN, affine_std=_OKLAB_STD)
        img = img.clamp(-1.0, 1.0)
        clamped = encode_image_to_oklab(img, affine_mean=_OKLAB_MEAN, affine_std=_OKLAB_STD)
        clamped = clamped.to(denoised.dtype)
        if not torch.isfinite(clamped).all():
            return denoised
        return clamped

    model_patcher.set_model_sampler_post_cfg_function(clamp_hook)


def patch_model_sampling(model_patcher, shift: float = 17.0) -> None:
    """Set the model's ``model_sampling`` to a flow shift of ``shift``.

    IMPORTANT comfy convention: ``ModelSamplingFlux.set_parameters(shift=...)``
    expects ``mu = log(shift_multiplier)``, *not* the shift multiplier
    itself. ``flux_time_shift(mu, 1, t) = exp(mu) / (exp(mu) + (1/t - 1))``.
    Passing the raw shift (e.g. 17) gives ``exp(17) ~ 24M`` which makes
    every sigma collapse to ~1.0 — the schedule becomes
    ``[1, 1, 1, ..., 1, 0]`` and you get one-step denoising.

    Users pass the shift in the AsymFLUX paper's convention (17.0 default,
    matching ``FlowAdapterScheduler(shift=17.0)`` in lakonlab); we convert
    to comfy's ``mu`` here.
    """
    sampling_base = comfy.model_sampling.ModelSamplingFlux
    sampling_type = comfy.model_sampling.CONST

    class AsymFluxModelSampling(sampling_base, sampling_type):
        pass

    ms = AsymFluxModelSampling(model_patcher.model.model_config)
    mu = math.log(max(shift, 1.0001))
    ms.set_parameters(shift=mu)
    _log(f"model_sampling: shift={shift} (paper) -> mu={mu:.4f} (comfy)")
    model_patcher.add_object_patch("model_sampling", ms)


def make_latent_passthrough(model_patcher) -> None:
    """Override ``latent_format`` and ``process_latent_in/out`` to be a no-op
    pixel-space pair."""
    model_patcher.add_object_patch("latent_format", AsymFlux2PixelLatent())

    def passthrough(self, latent):
        return latent

    model_patcher.add_object_patch(
        "process_latent_in", MethodType(passthrough, model_patcher.model),
    )
    model_patcher.add_object_patch(
        "process_latent_out", MethodType(passthrough, model_patcher.model),
    )
