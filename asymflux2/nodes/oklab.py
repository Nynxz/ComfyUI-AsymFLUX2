"""Oklab Encode/Decode nodes — the AsymFLUX2 "VAE" pair.

These wrap the deterministic Oklab color transform that AsymFLUX.2 uses
in place of a learned VAE. They convert between ComfyUI's standard
``IMAGE`` (BHWC float32 in [0, 1]) and a 3-channel ``LATENT`` carrying
affine-normalized Oklab values at the same spatial resolution.

The defaults match LakonLab's ``OklabColorEncoder`` settings used by
AsymFLUX.2-klein: mean=(0.56, 0.0, 0.01), std=0.16.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ..oklab_math import decode_oklab_to_image, encode_image_to_oklab


def _bhwc_to_bchw_pm1(image: torch.Tensor) -> torch.Tensor:
    return (image.float().permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0)


def _bchw_pm1_to_bhwc(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(-1.0, 1.0).mul(0.5).add(0.5)
    return image.permute(0, 2, 3, 1).contiguous()


class AsymFlux2OklabEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="asymflux2.OklabEncode",
            display_name="AsymFLUX2 Oklab Encode",
            category="AsymFLUX2",
            description=(
                "Encodes a ComfyUI IMAGE into the 3-channel Oklab pixel "
                "latent that the AsymFLUX2 transformer denoises in. This "
                "is a deterministic perceptual-color transform — there is "
                "no learned VAE."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Float.Input(
                    "mean_l", default=0.56, min=-1.0, max=1.0, step=0.001),
                io.Float.Input(
                    "mean_a", default=0.0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input(
                    "mean_b", default=0.01, min=-1.0, max=1.0, step=0.001),
                io.Float.Input(
                    "std", default=0.16, min=0.001, max=1.0, step=0.001),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mean_l: float,
        mean_a: float,
        mean_b: float,
        std: float,
    ) -> io.NodeOutput:
        x = _bhwc_to_bchw_pm1(image)
        oklab = encode_image_to_oklab(
            x,
            affine_mean=(mean_l, mean_a, mean_b),
            affine_std=std,
        )
        return io.NodeOutput({"samples": oklab})


class AsymFlux2OklabDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="asymflux2.OklabDecode",
            display_name="AsymFLUX2 Oklab Decode",
            category="AsymFLUX2",
            description=(
                "Decodes a 3-channel Oklab pixel latent (output of the "
                "AsymFLUX2 sampler or Oklab Encode) back to a ComfyUI "
                "IMAGE. Drop straight into Save Image / Preview Image."
            ),
            inputs=[
                io.Latent.Input("latent"),
                io.Float.Input(
                    "mean_l", default=0.56, min=-1.0, max=1.0, step=0.001),
                io.Float.Input(
                    "mean_a", default=0.0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input(
                    "mean_b", default=0.01, min=-1.0, max=1.0, step=0.001),
                io.Float.Input(
                    "std", default=0.16, min=0.001, max=1.0, step=0.001),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: dict,
        mean_l: float,
        mean_a: float,
        mean_b: float,
        std: float,
    ) -> io.NodeOutput:
        samples = latent["samples"]
        if samples.dim() != 4 or samples.shape[1] != 3:
            raise ValueError(
                "AsymFLUX2 Oklab Decode: expected 3-channel BCHW latent, "
                f"got shape {tuple(samples.shape)}."
            )
        img = decode_oklab_to_image(
            samples.float(),
            affine_mean=(mean_l, mean_a, mean_b),
            affine_std=std,
        )
        return io.NodeOutput(_bchw_pm1_to_bhwc(img).cpu())
