"""AsymFLUX2 Empty Pixel Latent — emits a 3-channel pixel-space latent
at 1:1 spatial resolution (no /8 VAE scaling). Mirrors the shape the
upstream ``PixelFlux2KleinPipeline.prepare_latents`` produces and
matches HiDream-O1's ``EmptyHiDreamO1LatentImage`` design from
ComfyUI PR #13817.

Width and height are snapped down to the nearest multiple of 16 so the
patchified token grid is integral.
"""

from __future__ import annotations

import torch
import comfy.model_management
from comfy_api.latest import io


_PATCH_MULTIPLE = 16


def _snap(v: int, m: int) -> int:
    return max(m, (int(v) // m) * m)


class AsymFlux2EmptyPixelLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="asymflux2.EmptyPixelLatent",
            display_name="AsymFLUX2 Empty Pixel Latent",
            category="AsymFLUX2",
            description=(
                "Empty 3-channel pixel-space latent for AsymFLUX2. "
                "Spatial dims are 1:1 with the output image (no VAE "
                "downscale) and snapped to a multiple of 16."
            ),
            inputs=[
                io.Int.Input(
                    "width",
                    default=960, min=128, max=4096, step=16,
                    tooltip="Output width. Snapped down to a multiple of 16.",
                ),
                io.Int.Input(
                    "height",
                    default=1280, min=128, max=4096, step=16,
                    tooltip="Output height. Snapped down to a multiple of 16.",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1, min=1, max=64,
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        width: int,
        height: int,
        batch_size: int,
    ) -> io.NodeOutput:
        w = _snap(width, _PATCH_MULTIPLE)
        h = _snap(height, _PATCH_MULTIPLE)
        latent = torch.zeros(
            [batch_size, 3, h, w],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        return io.NodeOutput({"samples": latent})
