"""ComfyUI-AsymFLUX2 — native ComfyUI nodes for the pixel-space
AsymFLUX.2-klein 9B model from LakonLab (Asymmetric Flow Models,
arXiv 2605.12964).

Layout
------
- ``nodes/``    — V3 ``io.ComfyNode`` definitions exposed in the menu.
- ``model/``    — AsymFLUX2 model surgery + AsymFlow calibration/velocity.
- ``oklab_math.py`` — pure-tensor Oklab encode/decode (the "VAE" pair).

The node pack does NOT depend on the upstream ``lakonlab`` package — all
math is reimplemented against ComfyUI primitives.
"""


def log(msg: str) -> None:
    """Console log helper used by every module. Prefixes [AsymFLUX2] so
    lines are easy to spot in ComfyUI's stdout. Single source of truth so
    the prefix can be changed in one place."""
    print(f"[AsymFLUX2] {msg}", flush=True)
