"""AsymFLUX2 Apply Adapter — takes a ``MODEL`` loaded from a stock
``Load Diffusion Model`` (a FLUX.2-klein-base-9B safetensors) plus an
adapter ``.safetensors`` from ``models/loras/`` (the AsymFLUX.2-klein
adapter), and returns a patched ``MODEL`` that can be driven by a stock
``KSampler`` against an AsymFLUX2 Empty Pixel Latent.

Everything is installed via ``ModelPatcher.add_object_patch`` so the
base FLUX.2-klein model is never mutated; the patches apply when the
returned MODEL is loaded and revert when it is unloaded.
"""

from __future__ import annotations

import os

import torch
from comfy_api.latest import io
from safetensors import safe_open

import comfy.lora
import comfy.sd
import folder_paths

from .. import log
from ..model.surgery import (
    apply_asymflux2_surgery,
    make_latent_passthrough,
    patch_clamp_denoised,
    patch_model_sampling,
    patch_orthogonal_cfg,
)


# Adapter state-dict prefix used by the upstream lakonlab diffusers pipeline.
_ADAPTER_PREFIX = "transformer."

# AsymFLUX.2 ships its timestep-embedder LoRAs under
# ``time_guidance_embed.timestep_embedder.*`` (LakonLab-specific module
# name), but comfy's ``flux_to_diffusers`` key map only knows about
# ``time_text_embed.timestep_embedder.*`` (the stock diffusers-flux name
# for the same underlying ``time_in`` MLP). We rewrite the adapter's
# diffusers-side names so comfy can match them.
_DIFFUSERS_KEY_RENAMES = {
    "time_guidance_embed.timestep_embedder.": "time_text_embed.timestep_embedder.",
}


# --- Module-level caches ---------------------------------------------
# Both keyed by (path, mtime) so editing/replacing the adapter file
# invalidates. Adapter files are immutable in practice, so once cached
# subsequent Apply Adapter executions in the same Comfy session pay only
# the cost of cloning the patcher + registering object_patches.

# Raw adapter state dict (skip the 707 MB disk read on re-execute).
_ADAPTER_SD_CACHE: dict[tuple[str, float], dict[str, torch.Tensor]] = {}

# Split state dict (skip the 121-key walk + rename pass on re-execute).
_ADAPTER_SPLIT_CACHE: dict[
    tuple[str, float],
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
] = {}


def _cache_key(path: str) -> tuple[str, float]:
    try:
        return (path, os.path.getmtime(path))
    except OSError:
        return (path, 0.0)


def _load_adapter_safetensors(path: str) -> tuple[dict[str, torch.Tensor], bool]:
    """Read a safetensors file fully into CPU memory.

    We open it ourselves rather than going through ``comfy.utils.load_torch_file``
    because that dispatches through the forked memory-management loader on
    aimdo-enabled builds and uses mmap on stock builds — both have been
    observed to segfault when the file sits on an external USB filesystem.
    ``.clone()`` on every tensor detaches from any underlying mmap.

    Returns ``(state_dict, cache_hit)``.
    """
    key = _cache_key(path)
    cached = _ADAPTER_SD_CACHE.get(key)
    if cached is not None:
        return cached, True

    sd: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).clone()
    _ADAPTER_SD_CACHE[key] = sd
    return sd, False


def _remap_diffusers_keys(k: str) -> str:
    for old, new in _DIFFUSERS_KEY_RENAMES.items():
        if old in k:
            return k.replace(old, new)
    return k


def _split_adapter_state_dict(
    sd: dict[str, torch.Tensor], cache_key: tuple[str, float],
) -> tuple[dict, dict]:
    """Split adapter state dict into (overwrites, lora). Strips
    ``transformer.`` prefix and rewrites the timestep-embedder LoRA names
    so comfy's diffusers-flux key map can match them.

    Cached by adapter file identity.
    """
    cached = _ADAPTER_SPLIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    overwrites: dict[str, torch.Tensor] = {}
    lora: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        kk = k.removeprefix(_ADAPTER_PREFIX) if k.startswith(_ADAPTER_PREFIX) else k
        if "lora" in kk.lower():
            kk = _remap_diffusers_keys(kk)
            # diffusers-style lora keys carry the `transformer.` prefix;
            # comfy's diffusers->flux key map expects it.
            lora[f"{_ADAPTER_PREFIX}{kk}"] = v
        else:
            overwrites[kk] = v

    _ADAPTER_SPLIT_CACHE[cache_key] = (overwrites, lora)
    return overwrites, lora


def _check_lora_resolves(model_patcher, lora_sd: dict[str, torch.Tensor]) -> None:
    """Diagnostic: warn if any LoRA stems won't match comfy's key map.
    Silent on the happy path (all match)."""
    key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, {})
    stems: set[str] = set()
    for k in lora_sd:
        if k.endswith(".lora_A.weight"):
            stems.add(k[: -len(".lora_A.weight")])
        elif k.endswith(".lora_B.weight"):
            stems.add(k[: -len(".lora_B.weight")])
    unmatched = sorted(stems - set(key_map.keys()))
    if unmatched:
        log(
            f"WARNING: {len(unmatched)}/{len(stems)} LoRA stems will not match "
            f"comfy's key map and will silently no-op. First 5: {unmatched[:5]}"
        )


class AsymFlux2ApplyAdapter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        adapter_choices = folder_paths.get_filename_list("loras") or [
            "<put AsymFLUX.2-klein adapter in ComfyUI/models/loras/>",
        ]
        return io.Schema(
            node_id="asymflux2.ApplyAdapter",
            display_name="AsymFLUX2 Apply Adapter",
            category="AsymFLUX2/loaders",
            description=(
                "Turns a stock FLUX.2-klein-base-9B MODEL into AsymFLUX.2. "
                "Drop the adapter (Lakonik/AsymFLUX.2-klein-9B "
                "diffusion_pytorch_model.safetensors, ~707 MB) into "
                "ComfyUI/models/loras/. Chain: Load Diffusion Model -> "
                "this node -> KSampler. Use AsymFLUX2 Empty Pixel Latent "
                "for the latent input, AsymFLUX2 Oklab Decode after the "
                "sampler."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="FLUX.2-klein-base-9B from a stock Load Diffusion Model node.",
                ),
                io.Combo.Input(
                    "adapter",
                    options=adapter_choices,
                    tooltip="AsymFLUX.2-klein adapter safetensors (in models/loras/).",
                ),
                io.Float.Input(
                    "shift",
                    default=17.0, min=0.1, max=100.0, step=0.1,
                    tooltip="Flow shift (paper convention; converted to comfy mu = log(shift) internally).",
                ),
                io.Float.Input(
                    "adapter_strength",
                    default=1.0, min=-2.0, max=2.0, step=0.01,
                    tooltip="LoRA strength applied to the rank-256 LoRA. 1.0 = full strength.",
                ),
                io.Float.Input(
                    "orthogonal_guidance",
                    default=1.0, min=0.0, max=2.0, step=0.05,
                    tooltip=(
                        "AsymFlow orthogonal CFG bias strength. 1.0 = "
                        "upstream demo default. 0.0 = standard CFG."
                    ),
                ),
                io.Boolean.Input(
                    "clamp_denoised",
                    default=True,
                    tooltip=(
                        "Per-step Oklab gamut clamp on the x0 estimate. "
                        "Prevents color drift across the sampling loop. "
                        "Disable if using uni_pc — see README."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        adapter: str,
        shift: float,
        adapter_strength: float,
        orthogonal_guidance: float,
        clamp_denoised: bool,
    ) -> io.NodeOutput:
        adapter_path = folder_paths.get_full_path_or_raise("loras", adapter)
        key = _cache_key(adapter_path)

        try:
            adapter_sd, cache_hit = _load_adapter_safetensors(adapter_path)
        except Exception as e:
            raise RuntimeError(
                f"AsymFLUX2: failed to read adapter safetensors at {adapter_path}: {e!r}. "
                "If the file lives on an external drive, copy it to local disk and retry."
            ) from e
        log(f"adapter {adapter_path}{' [CACHED]' if cache_hit else ''} ({len(adapter_sd)} tensors)")

        overwrites, lora_sd = _split_adapter_state_dict(adapter_sd, key)

        required = {"x_embedder.weight", "proj_out.weight", "proj_buffer", "scale_buffer"}
        missing = required - set(overwrites.keys())
        if missing:
            raise RuntimeError(
                f"AsymFLUX2 adapter is missing required tensors: {sorted(missing)}. "
                "Make sure you picked the AsymFLUX.2-klein-9B adapter."
            )

        m = model.clone()
        apply_asymflux2_surgery(m, overwrites)

        if lora_sd:
            _check_lora_resolves(m, lora_sd)
            m, _ = comfy.sd.load_lora_for_models(m, None, lora_sd, float(adapter_strength), 0.0)
            log(f"LoRA applied ({len(lora_sd) // 2} pairs at strength {adapter_strength})")

        patch_model_sampling(m, shift=shift)
        make_latent_passthrough(m)
        patch_orthogonal_cfg(m, orthogonal_guidance=orthogonal_guidance)
        if clamp_denoised:
            patch_clamp_denoised(m)
        log("apply-adapter complete")

        return io.NodeOutput(m)
