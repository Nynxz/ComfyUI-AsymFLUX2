"""Model surgery — turn a ComfyUI ``Flux2`` model into an AsymFLUX.2
model *without* mutating the underlying torch module.

Correctness rule
----------------
``ModelPatcher.clone()`` only clones the patcher wrapper — the underlying
``model.diffusion_model`` is shared by reference. If we ``setattr``
directly we corrupt the user's base FLUX.2-klein model so any other
workflow holding the same MODEL socket will explode at the first forward.

Everything goes through ``model_patcher.add_object_patch(name, value)``,
which uses ``comfy.utils.set_attr`` — that resolves dotted nested paths
(including ``nn.Sequential[index]`` via ``Sequential._modules['1']``) and
backs up the previous value to ``object_patches_backup`` for clean
restore on ``unpatch_model``. So our patches apply when this MODEL socket
is loaded and revert when it's unloaded — the base model is never mutated.

What we change relative to the stock FLUX.2-klein-base-9B
---------------------------------------------------------
``apply_asymflux2_surgery`` registers, as object patches on the cloned
``ModelPatcher``:

- ``patch_size`` 1 → 16; ``in/out_channels`` 16 → 3 (params + cached attrs)
- ``img_in`` swapped to ``Linear(768, 4096)`` (3-channel × 16×16)
- ``final_layer.linear`` swapped to ``Linear(4096, 768)``
- ``final_layer.adaLN_modulation[1]`` re-weighted with
  ``swap_scale_shift`` applied (diffusers stores ``(scale, shift)`` halves,
  comfy reads ``(shift, scale)``)
- ``guidance_in`` → ``nn.Identity``; ``guidance_embed`` → ``False``
- ``proj_buffer``, ``scale_buffer``, ``sigma_min`` exposed as plain attrs
- ``forward`` wrapped with AsymFlow calibration + velocity, capturing
  ``cls.forward.__get__(...)`` (the *class-level* method) so re-runs
  never stack wrappers

In addition:

- ``patch_model_sampling`` installs ``(CONST, ModelSamplingFlux)`` with
  ``mu = log(shift)`` (comfy's "shift" param is mu, not the real shift)
- ``make_latent_passthrough`` overrides ``latent_format`` to a 3-channel
  pass-through (so KSampler's ``fix_empty_latent_channels`` doesn't
  inflate our 3-channel pixel latent up to 128) and stubs
  ``process_latent_in/out`` to identity
- ``patch_orthogonal_cfg`` (optional) — upstream's velocity-space
  ``guidance_jit`` as a post-CFG hook
- ``patch_clamp_denoised`` (optional) — upstream's per-step Oklab gamut
  clamp on the x0 estimate, also a post-CFG hook
"""

from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn as nn

import comfy.latent_formats
import comfy.model_sampling
import comfy.utils

from .. import log
from .asymflow import PATCH_DIM, PATCH_SIZE, asymflow_calibration, asymflow_velocity
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


def _safe_dtype_device(
    weight: torch.Tensor,
    prefer_device: torch.device | None = None,
    prefer_dtype: torch.dtype | None = None,
) -> tuple[torch.dtype, torch.device]:
    """Pick the dtype/device for the new Linear modules we install.

    Adjustments:

    - ``prefer_device`` (typically ``model_patcher.load_device``) takes
      priority when supplied. Under GGUF + staged / dynamic-VRAM loading
      the base weight can be on CPU at surgery time even though the
      inference device is CUDA; pre-allocating on the load device avoids
      the mismatch.
    - ``device='meta'`` falls back to ``cpu`` (allocating directly on
      meta crashes on ``copy_``); the forward wrapper migrates on the
      first call as a safety net.
    - For the dtype: when the base weight's dtype is non-floating
      (e.g. ``uint8`` for GGUF-quantized bases), we have to substitute
      a floating dtype because ``nn.Parameter`` only accepts those.
      ``prefer_dtype`` (typically ``model.get_dtype_inference()`` — the
      dtype comfy will cast inputs to during forward) is the right
      substitute: it matches what the rest of the model computes in, so
      our new Linears mesh with the surrounding quantized layers'
      dequantized output without further casts. Falls back to bf16 if
      no preference is supplied.
    """
    dev = prefer_device if prefer_device is not None else weight.device
    if dev.type == "meta":
        dev = torch.device("cpu")

    dtype = weight.dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        if prefer_dtype is not None and (prefer_dtype.is_floating_point or prefer_dtype.is_complex):
            dtype = prefer_dtype
        else:
            dtype = torch.bfloat16

    return dtype, dev


def _new_linear_from(weight: torch.Tensor, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    """Build a fresh ``nn.Linear`` carrying the supplied weight (bias=False
    — the AsymFLUX.2 adapter never ships biases for the layers we swap)."""
    out_f, in_f = weight.shape
    new = nn.Linear(in_f, out_f, bias=False, dtype=dtype, device=device)
    with torch.no_grad():
        new.weight.copy_(weight.to(dtype=dtype, device=device))
    return new


def _make_asymflux2_forward(proj_buffer: torch.Tensor, scale_buffer: torch.Tensor, sigma_min: float = 1e-4):
    """Build the wrapped forward closure that runs AsymFlow calibration
    around the original ``Flux.forward``.

    The wrapper does NOT capture the diffusion_model instance — it reads
    ``type(self).forward.__get__(self, ...)`` per call. This way:

    1. Re-runs of surgery never wrap a wrapper (capture is always the
       pristine class-level method).
    2. The closure is instance-agnostic; can be safely bound to any
       compatible diffusion_model.
    3. ``Flux.forward`` itself dispatches to ``_forward`` via
       ``comfy.patcher_extension.WrapperExecutor``, so we don't lose the
       wrapper-system features either.
    """

    def asymflux2_forward(
        self, x, timestep, context,
        y=None, guidance=None, ref_latents=None, control=None,
        transformer_options=None, **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}

        # Defensive: under GGUF + staged loading the patcher's weight-
        # migration path doesn't always pull our installed Linears onto
        # the inference device. Migrate any that are out of place. This
        # is a no-op on the second+ call -- nn.Module.to() is in-place.
        target_dev = x.device
        if self.img_in.weight.device != target_dev:
            self.img_in.to(target_dev)
            self.final_layer.linear.to(target_dev)
            self.final_layer.adaLN_modulation[1].to(target_dev)

        # Re-resolve the original Flux.forward from the class on every
        # call — robust to surgery re-runs, idempotent under patch stacks.
        cls = self.__class__
        original_forward = cls.forward.__get__(self, cls)

        # scale_buffer was allocated on cpu during surgery; migrate to the
        # live forward device so the computed `k` lands with `x`.
        s_dev = scale_buffer.to(device=x.device)
        cal = asymflow_calibration(timestep.float(), s_dev)
        k_x = cal.k.reshape(-1, 1, 1, 1).to(dtype=x.dtype, device=x.device)
        x_scaled = x * k_x
        cal_t = cal.timestep.to(dtype=timestep.dtype, device=timestep.device)

        u_a = original_forward(
            x_scaled, cal_t, context,
            y=y, guidance=guidance, ref_latents=ref_latents,
            control=control, transformer_options=transformer_options, **kwargs,
        )
        return asymflow_velocity(
            u_a, x, cal,
            proj_buffer=proj_buffer,
            sigma_min=sigma_min,
            patch_size=PATCH_SIZE,
        )

    return asymflux2_forward


def apply_asymflux2_surgery(
    model_patcher,
    overwrites: dict[str, torch.Tensor],
) -> None:
    """Register all AsymFLUX.2-specific changes as object patches on
    ``model_patcher``. The base ``model.diffusion_model`` is NOT mutated —
    everything is reverted automatically when ``unpatch_model`` runs."""
    diffusion_model = model_patcher.model.diffusion_model
    base_w = diffusion_model.img_in.weight
    load_device = getattr(model_patcher, "load_device", None)
    # Ask the model what dtype it computes in -- matters when the base is
    # quantized (uint8 GGUF, fp8, etc.) and our new Linears need to mesh
    # with the inference cast dtype, not blindly fall back to bf16.
    get_inf = getattr(model_patcher.model, "get_dtype_inference", None)
    inference_dtype = get_inf() if callable(get_inf) else None
    dtype, device = _safe_dtype_device(
        base_w, prefer_device=load_device, prefer_dtype=inference_dtype,
    )
    if dtype != base_w.dtype:
        log(
            f"surgery: base dtype={base_w.dtype} not usable for nn.Linear "
            f"(quantized?); installing new Linears as {dtype} "
            f"(model inference dtype={inference_dtype})"
        )
    log(f"surgery: target dtype={dtype}, device={device}")

    p = model_patcher.add_object_patch
    dm = "diffusion_model"

    # patch_size + channel counts (plain int attrs read inside Flux.forward)
    p(f"{dm}.patch_size", PATCH_SIZE)
    p(f"{dm}.params.patch_size", PATCH_SIZE)
    p(f"{dm}.params.in_channels", 3)
    p(f"{dm}.params.out_channels", 3)
    p(f"{dm}.in_channels", PATCH_DIM)
    p(f"{dm}.out_channels", PATCH_DIM)
    log("surgery: patch_size=16, in/out_channels=3")

    # img_in: 3*16*16 -> hidden
    x_emb = overwrites["x_embedder.weight"]
    p(f"{dm}.img_in", _new_linear_from(x_emb, dtype, device))
    log(f"surgery: img_in -> Linear({x_emb.shape[1]}, {x_emb.shape[0]})")

    # final_layer.linear: hidden -> 3*16*16
    proj_out = overwrites["proj_out.weight"]
    p(f"{dm}.final_layer.linear", _new_linear_from(proj_out, dtype, device))
    log(f"surgery: final_layer.linear -> Linear({proj_out.shape[1]}, {proj_out.shape[0]})")

    # final_layer.adaLN_modulation[1]: the adapter is in diffusers'
    # (scale, shift) half-order, comfy's LastLayer reads (shift, scale).
    # swap_scale_shift exchanges the halves so comfy's `chunk(2)` gives
    # the right roles. Same conversion comfy uses in MMDIT_MAP_BASIC for
    # diffusers FLUX safetensors.
    norm_w = overwrites.get("norm_out.linear.weight")
    if norm_w is not None:
        norm_w_comfy = comfy.utils.swap_scale_shift(norm_w)
        p(f"{dm}.final_layer.adaLN_modulation.1", _new_linear_from(norm_w_comfy, dtype, device))
        log(
            f"surgery: final_layer.adaLN_modulation[1] -> Linear({norm_w_comfy.shape[1]}, "
            f"{norm_w_comfy.shape[0]}) (scale/shift halves swapped)"
        )

    # Guidance is disabled in AsymFLUX.2 — the adapter trains with
    # guidance_embed=False.
    p(f"{dm}.guidance_in", nn.Identity())
    p(f"{dm}.params.guidance_embed", False)
    log("surgery: guidance_in disabled")

    # AsymFlow tensors as plain attributes (not buffers — add_object_patch
    # round-trips regular attrs cleanly via set_attr; un-registering a
    # buffer on unpatch is messier).
    proj_buf = overwrites["proj_buffer"].to(dtype=torch.float32, device=device)
    scale_buf = overwrites["scale_buffer"].to(dtype=torch.float32, device=device)
    p(f"{dm}.asymflow_proj_buffer", proj_buf)
    p(f"{dm}.asymflow_scale_buffer", scale_buf)
    p(f"{dm}.sigma_min", 1e-4)
    log(f"surgery: proj_buffer {tuple(proj_buf.shape)} + scale_buffer staged")

    # Wrapped forward. Closure does not capture the diffusion_model
    # instance — see _make_asymflux2_forward.
    fwd = _make_asymflux2_forward(proj_buf, scale_buf, sigma_min=1e-4)
    p(f"{dm}.forward", MethodType(fwd, diffusion_model))
    log("surgery: forward wrapped with AsymFlow calibration + velocity")


def patch_orthogonal_cfg(model_patcher, orthogonal_guidance: float) -> None:
    """Upstream's velocity-space ``guidance_jit`` as a post-CFG hook.

    Comfy gives us x0-space ``cond_denoised`` / ``uncond_denoised``. We
    reconstruct velocities via ``v = (x - x0) / sigma``, apply the
    upstream formula (subtract the projection of the CFG bias onto the
    current x0 direction, weighted by ``orthogonal_guidance``), then
    convert back to x0. Operating in velocity space matters numerically:
    at small sigma the x0-space bias collapses near-zero while the
    velocity-space components are well-conditioned.

    ``orthogonal_guidance == 0`` short-circuits (no hook installed).
    """
    if orthogonal_guidance <= 0.0:
        return

    def orthog_cfg(args):
        denoised = args["denoised"]
        cond = args["cond_denoised"]
        uncond = args["uncond_denoised"]
        sigma = args["sigma"]
        x_in = args["input"]
        cfg = args["cond_scale"]

        if uncond is None or cond is None or cfg <= 1.0:
            return denoised

        sigma_b = sigma
        while sigma_b.dim() < x_in.dim():
            sigma_b = sigma_b.unsqueeze(-1)
        sigma_b = sigma_b.clamp(min=1e-4)

        v_cond = (x_in - cond) / sigma_b
        v_uncond = (x_in - uncond) / sigma_b
        bias_v = (v_cond - v_uncond) * (cfg - 1.0)
        parallel = cond  # paper's `parallel_dir = x_t - v_cond * sigma`

        dims = list(range(1, bias_v.dim()))
        num = (bias_v * parallel).mean(dim=dims, keepdim=True)
        den = (parallel * parallel).mean(dim=dims, keepdim=True).clamp(min=1e-6)
        bias_v_orthog = bias_v - (num / den) * parallel * orthogonal_guidance

        out = x_in - (v_cond + bias_v_orthog) * sigma_b
        if not torch.isfinite(out).all():
            return denoised
        return out

    model_patcher.set_model_sampler_post_cfg_function(orthog_cfg)


def patch_clamp_denoised(model_patcher) -> None:
    """Per-step Oklab gamut clamp on the x0 estimate (upstream
    ``clamp_denoised=True``). Decodes x0 to sRGB, clips to ``[-1, 1]``,
    re-encodes back to Oklab. Prevents x0 drift out of valid color space
    across the 38 steps. Installed as a post-CFG hook so it composes with
    ``patch_orthogonal_cfg`` (orthog applies first, then this clamps the
    result before the sampler converts denoised -> velocity)."""

    def clamp_hook(args):
        denoised = args["denoised"]
        # fp32 for the color math: Oklab matrices + pow(1/3) are bf16-sensitive.
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
    """Install a (CONST, ModelSamplingFlux) model_sampling with the AsymFLUX
    paper's shift.

    Comfy convention: ``ModelSamplingFlux.set_parameters(shift=...)``
    expects ``mu = log(shift_multiplier)``, NOT the multiplier itself. The
    underlying ``flux_time_shift(mu, 1, t) = exp(mu) / (exp(mu) + (1/t - 1))``
    formula means passing 17 directly gives ``exp(17) ~ 24M`` and the
    schedule collapses to all ones. We accept the paper-convention shift
    on the node (17.0 default, matches LakonLab's ``FlowAdapterScheduler``)
    and convert here.
    """
    sampling_base = comfy.model_sampling.ModelSamplingFlux
    sampling_type = comfy.model_sampling.CONST

    class AsymFluxModelSampling(sampling_base, sampling_type):
        pass

    ms = AsymFluxModelSampling(model_patcher.model.model_config)
    mu = math.log(max(shift, 1.0001))
    ms.set_parameters(shift=mu)
    log(f"model_sampling: shift={shift} (paper) -> mu={mu:.4f} (comfy)")
    model_patcher.add_object_patch("model_sampling", ms)


def make_latent_passthrough(model_patcher) -> None:
    """Override ``latent_format`` to a 3-channel pass-through and stub
    ``process_latent_in/out`` to identity. Without this comfy's
    ``fix_empty_latent_channels`` would inflate our 3-channel pixel latent
    up to the stock Flux2 128 channels (which collapses the model's input
    expectations)."""

    def passthrough(self, latent):
        return latent

    model_patcher.add_object_patch("latent_format", AsymFlux2PixelLatent())
    model_patcher.add_object_patch(
        "process_latent_in", MethodType(passthrough, model_patcher.model),
    )
    model_patcher.add_object_patch(
        "process_latent_out", MethodType(passthrough, model_patcher.model),
    )
