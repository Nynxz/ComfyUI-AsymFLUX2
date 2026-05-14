"""ComfyUI-AsymFLUX2 — native ComfyUI nodes for the pixel-space
AsymFLUX.2-klein 9B model from LakonLab (Asymmetric Flow Models,
arXiv 2605.12964).

Layout
------
- ``nodes/``    — V3 ``io.ComfyNode`` definitions exposed in the menu.
- ``model/``    — AsymFLUX2 ``BaseModel`` subclass and any architecture
                  overrides (input/output projection swap, asymflow
                  calibration + velocity wrapper).
- ``sampler/``  — Custom denoising loop with orthogonal CFG and the
                  optional per-step Oklab clamp round-trip.

The node pack does NOT depend on the upstream ``lakonlab`` package — all
math is reimplemented against ComfyUI primitives.
"""
