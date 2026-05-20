# ComfyUI-AsymFLUX2

> [!IMPORTANT]
> **The original authors have released an official node pack:
> [Lakonik/ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow).**
> If you're just looking to run AsymFLUX.2 in ComfyUI, you should probably
> install that instead — it's maintained by the people who trained the
> model. This repo predates AsymFLUX.2 support landing in piFlow and stays
> up as an alternative.

ComfyUI nodes for running [AsymFLUX.2-klein](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B)
in pixel space on top of a stock FLUX.2-klein-base-9B safetensors.

AsymFLUX.2 is from the *Asymmetric Flow Models* paper
([arXiv 2605.12964](https://arxiv.org/abs/2605.12964), Chen et al.) and
ships as an adapter on top of FLUX.2-klein — the transformer denoises a
3-channel Oklab image directly instead of going through a VAE latent.

![AsymFlux2 example](.github/assets/example.jpg)

## Workflow

```
Load Diffusion Model (FLUX.2-klein-base-9B) ─┐
                                             ├─►  AsymFLUX2 Apply Adapter  ─►  KSampler  ─►  AsymFLUX2 Oklab Decode  ─►  Save Image
CLIPLoader (Qwen3 8B, type=flux2)  ─►  CLIPTextEncode (pos / neg)  ────┘                ▲
                                                                                        │
                                 AsymFLUX2 Empty Pixel Latent  ───────────────────────┘
```

Example: [`example_workflows/asymflux2_t2i.json`](example_workflows/asymflux2_t2i.json).

## Nodes

- **AsymFLUX2 Apply Adapter** — takes a FLUX.2-klein MODEL + the adapter
  filename, returns a patched MODEL ready for KSampler.
- **AsymFLUX2 Empty Pixel Latent** — 3-channel pixel-space empty latent
  (no /8 VAE downscale).
- **AsymFLUX2 Oklab Encode** — `IMAGE` → 3-channel Oklab `LATENT`.
- **AsymFLUX2 Oklab Decode** — 3-channel Oklab `LATENT` → `IMAGE`. Goes
  after KSampler.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-AsymFLUX2
```

No extra Python deps.

## Models you need

1. **FLUX.2-klein-base-9B** safetensors in `ComfyUI/models/diffusion_models/`,
   loaded via `Load Diffusion Model` (`UNETLoader`).
2. **AsymFLUX.2-klein adapter** —
   [`diffusion_pytorch_model.safetensors`](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B/blob/main/diffusion_pytorch_model.safetensors)
   from the HF repo (~707 MB). Drop into `ComfyUI/models/loras/`.
3. **Qwen3 8B text encoder** for FLUX.2-klein, loaded via single
   `CLIPLoader` with `type=flux2`.

You'll need to accept the
[FLUX.2 klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md)
and AsymFLUX.2-klein licenses on Hugging Face.

## Settings

**Apply Adapter:**

| Input | Default | What it does |
| --- | --- | --- |
| `shift` | `17.0` | Flow shift, paper convention. Converted to comfy's `mu = log(shift)` internally. |
| `adapter_strength` | `1.0` | LoRA strength. |
| `orthogonal_guidance` | `1.0` | Upstream `guidance_jit` strength. `0.0` = standard CFG. |
| `clamp_denoised` | `True` | Per-step Oklab gamut clamp on the x0 estimate. |

**KSampler:** the example workflow ships `uni_pc` + `simple`, **20 steps**,
CFG 4.0, at 960 × 1280. UniPC is a multistep solver — its per-step
quality is high enough that 15-25 steps generally matches what a
single-step solver does at 30-50. Pushing it much past ~25 steps tends
to degrade output because the per-step `clamp_denoised` non-linearity
compounds in the multistep polynomial extrapolation. If you want to
try other samplers: `dpmpp_2m` and `dpmpp_2m_sde` at 30-50 steps,
`deis` at 20-30, all with `simple` scheduler.

## Known limitations

- Output is close to the [HF Space](https://huggingface.co/spaces/Lakonik/AsymFLUX.2-klein)
  but not identical. Known differences from upstream:
    - Static flow shift (we use `17` from the paper; upstream uses a
      resolution-dependent dynamic shift between `log(17)` and `log(34)`).
      For 960 × 1280 the upstream value lands around `20`. The gap
      widens at higher resolutions.
    - Sampler. Upstream uses `UniPCMultistep` integrated by its
      `FlowAdapterScheduler`. We default to `uni_pc` at 20 steps —
      comfy's port of the same algorithm — which should track upstream
      closely. The remaining difference is mostly that comfy's
      `uni_pc` runs through a generic-denoised path rather than
      diffusers' `prediction_type='flow_prediction'` route; they're
      mathematically equivalent for our model_sampling but the code
      path is different.
- Image-editing / reference-image conditioning isn't supported. The
  upstream `image=` kwarg on `PixelFlux2KleinPipeline.__call__` is
  plumbing inherited from `Flux2KleinPipeline`; output quality on
  reference-conditioned generation didn't match the t2i output in
  our testing, so it isn't shipped here.
- No Qwen3-VL prompt rewriter (the demo's optional prompt-quality
  preprocessor — separate model).

## How it's wired together

Read the code under `asymflux2/` for the details. The short version:

- `apply_adapter.py` clones the input ModelPatcher, splits the adapter
  state dict into overwrites and LoRA, and registers everything via
  `ModelPatcher.add_object_patch` so the base model is never mutated.
- `model/surgery.py` is where the patches live: replacement Linears for
  `img_in` / `final_layer` / `final_layer.adaLN_modulation[1]`, the
  AsymFlow `proj_buffer` / `scale_buffer`, a wrapped `forward` that
  applies AsymFlow calibration + velocity around the original
  `Flux.forward`, and the two optional post-CFG hooks (orthogonal CFG
  and Oklab gamut clamp).
- `model/asymflow.py` is the AsymFlow math: calibration (`k = 1/(s + (1-s)·σ)`)
  and velocity (orthogonal decomposition along `proj_buffer`).
- `oklab_math.py` is the Oklab transform pair, used by the encode/decode
  nodes and the per-step clamp.

The AsymFlow math, the orthogonal CFG, and the Oklab transform each
have a synthetic-input unit test verifying they match the upstream
LakonLab functions when run on the same tensors. The end-to-end output
still depends on sampler choice, schedule discretization, and the static
vs dynamic shift, so it's close to but not identical to the HF Space.

## Citation & credits

- [LakonLab](https://huggingface.co/Lakonik) — trained the AsymFLUX.2-klein
  adapter and authored the reference [`lakonlab`](https://github.com/Lakonik/lakonlab)
  implementation that this node pack reimplements against ComfyUI primitives.
- [Black Forest Labs](https://huggingface.co/black-forest-labs) —
  FLUX.2-klein base model.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — host runtime.

Original paper:

```bibtex
@article{chen2026asymmetric,
  title={Asymmetric Flow Models},
  author={Hansheng Chen and Jan Ackermann and Minseo Kim and Gordon Wetzstein and Leonidas Guibas},
  journal={arXiv preprint arXiv:2605.12964},
  url={https://arxiv.org/abs/2605.12964},
  year={2026},
}
```
