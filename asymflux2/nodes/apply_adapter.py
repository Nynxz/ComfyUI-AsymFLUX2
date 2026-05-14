"""AsymFLUX2 Apply Adapter — takes a ``MODEL`` loaded from a stock
``Load Diffusion Model`` (a FLUX.2-klein-base-9B safetensors) plus an
adapter ``.safetensors`` from ``models/loras/`` (the AsymFLUX.2-klein
adapter), and returns a patched ``MODEL`` that can be driven by a
stock ``KSampler`` against an AsymFLUX2 Empty Pixel Latent.

The node performs the input/output projection swap, registers the
``proj_buffer`` / ``scale_buffer``, wraps the diffusion model with the
AsymFlow calibration + velocity, applies the rank-256 LoRA, sets a
flux-shift of 17, and disables the per-VAE latent rescale (pixel-space).
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

import comfy.lora
import comfy.sd
import comfy.utils
import folder_paths
from safetensors import safe_open

from ..model.surgery import (
    apply_asymflux2_surgery,
    make_latent_passthrough,
    patch_clamp_denoised,
    patch_model_sampling,
    patch_orthogonal_cfg,
)


_ADAPTER_PREFIX = "transformer."

# AsymFLUX.2 ships its timestep-embedder LoRAs under
# ``time_guidance_embed.timestep_embedder.*`` (a LakonLab-specific module
# name), but comfy's ``flux_to_diffusers`` key map only knows about
# ``time_text_embed.timestep_embedder.*`` (the stock diffusers-flux name
# for the SAME underlying ``time_in`` MLP in comfy). We rewrite the
# adapter's diffusers-side names so comfy can match them.
_DIFFUSERS_KEY_RENAMES = {
    "time_guidance_embed.timestep_embedder.": "time_text_embed.timestep_embedder.",
}


def _remap_diffusers_keys(k: str) -> str:
    for old, new in _DIFFUSERS_KEY_RENAMES.items():
        if old in k:
            return k.replace(old, new)
    return k


def _log(msg: str) -> None:
    print(f"[AsymFLUX2] {msg}", flush=True)


def _load_adapter_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Read a safetensors file fully into CPU memory, *without* relying on
    comfy.utils.load_torch_file. That function dispatches through the
    forked memory-management loader on aimdo-enabled builds and uses mmap
    on stock builds — both have been observed to segfault when the file
    sits on an external USB filesystem. We open it ourselves and call
    ``.clone()`` on every tensor to detach from any underlying mmap.
    """
    sd: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).clone()
    return sd


def _split_adapter_state_dict(sd: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Split adapter state dict into (overwrites, lora). Strips
    ``transformer.`` prefix from both, matching how lakonlab's
    ``load_lakonlab_adapter`` interprets them. Also rewrites diffusers-side
    LoRA names where AsymFLUX2 disagrees with stock diffusers (currently:
    ``time_guidance_embed -> time_text_embed`` on the timestep MLP).
    """
    overwrites: dict[str, torch.Tensor] = {}
    lora: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        kk = k.removeprefix(_ADAPTER_PREFIX) if k.startswith(_ADAPTER_PREFIX) else k
        if "lora" in kk.lower():
            kk = _remap_diffusers_keys(kk)
            # diffusers-style lora keys come pre-prefixed with `transformer.`
            # comfy's diffusers->flux key map expects that prefix back when
            # the lora is matched via the model's hidden_size.
            lora_key = f"{_ADAPTER_PREFIX}{kk}"
            lora[lora_key] = v
        else:
            overwrites[kk] = v
    return overwrites, lora


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
                "ComfyUI/models/loras/. Then chain: Load Diffusion Model -> "
                "this node -> KSampler. Use AsymFLUX2 Empty Pixel Latent for "
                "the latent input, and AsymFLUX2 Oklab Decode for the output."
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
                    tooltip=(
                        "Flux-style time shift. 17.0 matches the static "
                        "shift in the upstream FlowAdapterScheduler "
                        "default."
                    ),
                ),
                io.Float.Input(
                    "adapter_strength",
                    default=1.0, min=-2.0, max=2.0, step=0.01,
                    tooltip=(
                        "LoRA strength applied to the AsymFlow rank-256 "
                        "LoRA. 1.0 = full strength."
                    ),
                ),
                io.Float.Input(
                    "orthogonal_guidance",
                    default=1.0, min=0.0, max=2.0, step=0.05,
                    tooltip=(
                        "AsymFlow orthogonal CFG bias strength. 1.0 "
                        "matches the upstream demo default (removes the "
                        "component of the CFG bias parallel to the "
                        "current x0 estimate -- significantly sharpens "
                        "detail). 0.0 = standard CFG. Try 0.5 if output "
                        "looks oversharpened."
                    ),
                ),
                io.Boolean.Input(
                    "clamp_denoised",
                    default=True,
                    tooltip=(
                        "Per-step Oklab gamut clamp on the x0 estimate "
                        "(upstream `clamp_denoised=True` default). At "
                        "every step the predicted x0 is decoded to RGB, "
                        "clipped to [-1, 1], and re-encoded back to "
                        "Oklab. Prevents x0 drift out of valid color "
                        "space; expect noticeably better color "
                        "stability and slightly sharper output."
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
        _log(f"loading adapter from {adapter_path}")
        try:
            adapter_sd = _load_adapter_safetensors(adapter_path)
        except Exception as e:
            raise RuntimeError(
                f"AsymFLUX2: failed to read adapter safetensors at {adapter_path}: {e!r}. "
                "If the file lives on an external drive, copy it to your local disk "
                "and retry."
            ) from e
        _log(f"adapter loaded: {len(adapter_sd)} tensors")

        overwrites, lora_sd = _split_adapter_state_dict(adapter_sd)
        _log(f"split adapter into {len(overwrites)} overwrites + {len(lora_sd)} lora tensors")

        required = {
            "x_embedder.weight",
            "proj_out.weight",
            "proj_buffer",
            "scale_buffer",
        }
        missing = required - set(overwrites.keys())
        if missing:
            raise RuntimeError(
                f"AsymFLUX2 adapter is missing required tensors: {sorted(missing)}. "
                "Make sure you picked the AsymFLUX.2-klein-9B adapter."
            )

        _log("cloning model patcher")
        m = model.clone()

        _log("applying model surgery (object_patches; base model unchanged)")
        apply_asymflux2_surgery(m, overwrites)
        _log("surgery complete")

        # Apply the LoRA via comfy's standard machinery. comfy.sd.load_lora_for_models
        # routes through model_lora_keys_unet, which recognises a Flux model and
        # converts diffusers-naming keys ("transformer_blocks.X.attn.to_out.lora_A...")
        # to comfy's internal keys.
        if lora_sd:
            _log(f"applying {len(lora_sd) // 2} LoRA pairs at strength {adapter_strength}")

            # Sniff-check: build the same key_map comfy would, and see how many of
            # our LoRA keys actually resolve to a comfy-side parameter. If this is
            # 0 the LoRA silently no-ops and we ship vanilla FLUX.2 attention/ff
            # behaviour with an AsymFLUX2 input/output projection — produces
            # "rainbow grid" / black output.
            key_map = comfy.lora.model_lora_keys_unet(m.model, {})
            lora_stems = set()
            for k in lora_sd.keys():
                if k.endswith(".lora_A.weight"):
                    lora_stems.add(k[: -len(".lora_A.weight")])
                elif k.endswith(".lora_B.weight"):
                    lora_stems.add(k[: -len(".lora_B.weight")])
            matched = sum(1 for s in lora_stems if s in key_map)
            unmatched = sorted(lora_stems - set(key_map.keys()))
            _log(f"LoRA key map sniff: {matched}/{len(lora_stems)} stems resolvable")
            if unmatched:
                _log(f"LoRA stems comfy won't match (first 5): {unmatched[:5]}")

            m, _ = comfy.sd.load_lora_for_models(m, None, lora_sd, float(adapter_strength), 0.0)
            _log("LoRA applied")

        _log(f"patching model_sampling shift={shift}, latent passthrough")
        patch_model_sampling(m, shift=shift)
        make_latent_passthrough(m)
        if orthogonal_guidance > 0.0:
            patch_orthogonal_cfg(m, orthogonal_guidance=orthogonal_guidance)
            _log(f"orthogonal CFG hook installed (strength={orthogonal_guidance})")
        if clamp_denoised:
            patch_clamp_denoised(m)
            _log("clamp_denoised hook installed (per-step Oklab gamut clamp on x0)")
        _log("apply-adapter complete")

        return io.NodeOutput(m)
