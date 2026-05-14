"""ComfyUI extension entrypoint for ComfyUI-AsymFLUX2."""

from comfy_api.latest import ComfyExtension, io

from .asymflux2.nodes import (
    AsymFlux2ApplyAdapter,
    AsymFlux2EmptyPixelLatent,
    AsymFlux2OklabDecode,
    AsymFlux2OklabEncode,
)


class AsymFlux2Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AsymFlux2ApplyAdapter,
            AsymFlux2EmptyPixelLatent,
            AsymFlux2OklabEncode,
            AsymFlux2OklabDecode,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return AsymFlux2Extension()
