# ComfyUI-AsymFLUX2

Native ComfyUI v3 nodes for the **pixel-space** AsymFLUX.2-klein 9B model
from LakonLab (Chen et al., *Asymmetric Flow Models*, arXiv 2605.12964).

AsymFLUX.2 is an adapter that turns FLUX.2-klein-base-9B into a pixel-space
generator: instead of denoising in a VAE latent, the transformer operates
directly on a 3-channel Oklab image representation. This pack runs it
**natively in ComfyUI** — no diffusers, no `lakonlab` install — using
the standard `Load Diffusion Model`, `DualCLIPLoader`, and `KSampler`
nodes alongside four small AsymFLUX2-specific nodes.

## Workflow

```
Load Diffusion Model (FLUX.2-klein-base-9B)  ─┐
                                              ├─►  AsymFLUX2 Apply Adapter  ─►  KSampler  ─►  AsymFLUX2 Oklab Decode  ─►  Save Image
CLIPLoader (Qwen3 8B, type=flux2)  ─►  CLIPTextEncode (pos / neg)  ─────┘                ▲
                                                                                         │
                                  AsymFLUX2 Empty Pixel Latent  ───────────────────────┘
```

A working workflow lives at
[`example_workflows/asymflux2_t2i.json`](example_workflows/asymflux2_t2i.json).

## Nodes

| Node | Role |
| --- | --- |
| **AsymFLUX2 Apply Adapter** | Mutates a stock FLUX.2-klein MODEL into AsymFLUX.2 — swaps `img_in` / `final_layer` to 3ch×16×16 patches, registers `proj_buffer` / `scale_buffer`, applies the rank-256 LoRA, wraps the diffusion forward with AsymFlow calibration + velocity, sets flux-shift to 17, disables guidance embed, and makes the latent path pass-through (no VAE rescale). |
| **AsymFLUX2 Empty Pixel Latent** | Empty 3-channel pixel latent at 1:1 spatial resolution. Drop-in replacement for `EmptyLatentImage`. |
| **AsymFLUX2 Oklab Encode** | `IMAGE` → 3-channel Oklab `LATENT`. Pure-math port of LakonLab's `OklabColorEncoder`. Use this for img2img. |
| **AsymFLUX2 Oklab Decode** | 3-channel Oklab `LATENT` → `IMAGE`. Used after KSampler. |

## Install

Clone into ComfyUI's custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-AsymFLUX2
```

No extra Python dependencies — everything we need (`torch`, `einops`,
ComfyUI internals) is already in your ComfyUI env.

## Models you need

1. **FLUX.2-klein-base-9B** — already in `ComfyUI/models/diffusion_models/`.
   Use whichever single-file safetensors repackaging you have. Loaded via
   the stock `Load Diffusion Model` (a.k.a. `UNETLoader`) node.
2. **AsymFLUX.2-klein adapter** — download
   [`Lakonik/AsymFLUX.2-klein-9B/diffusion_pytorch_model.safetensors`](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B/blob/main/diffusion_pytorch_model.safetensors)
   (~707 MB). Drop it into `ComfyUI/models/loras/` (rename to something
   like `asymflux2-klein-9b.safetensors` if you want).
3. **Qwen3 8B text encoder** — the FLUX.2-klein TE (single file). Loaded
   via `CLIPLoader` (not Dual) with `type=flux2`.

You'll need to accept the
[FLUX.2 klein license](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md)
and the AsymFLUX.2-klein license on Hugging Face first.

## KSampler settings

Mirroring the upstream Gradio defaults:

- **Sampler**: `euler` (most reliable; see note below)
- **Scheduler**: `simple`
- **Steps**: 38 (range 4–50)
- **CFG**: 4.0
- **Resolution**: 960 × 1280 (snapped to multiples of 16)

> **Sampler choice:** the upstream pipeline uses `UniPCMultistep`, but
> ComfyUI's stock `uni_pc` builds its multistep polynomial directly from
> the sigma schedule and can go singular with the high static shift
> (17.0) AsymFLUX.2 uses — `torch.linalg.solve` raises `singular matrix`.
> Use `euler` until we ship a custom sampler that mirrors LakonLab's
> `FlowAdapterScheduler` wrapping. `dpmpp_2m` and `deis` also work.

Note: the Apply Adapter node currently sets a static `flux-shift = 17`.
The upstream pipeline uses a **dynamic** shift between `ln(17)` and `ln(34)`
keyed on resolution; for 960×1280 the dynamic shift lands close to 17, so
the static value is a safe first cut. A future loader option will add the
dynamic schedule.

## What this pack does *not* yet do

- **Orthogonal CFG bias** — the upstream `guidance_jit` removes the
  component of the CFG bias parallel to the current x0 estimate. KSampler
  uses standard CFG, so output may differ slightly from the upstream
  Gradio demo. With `orthogonal_guidance=0`, the upstream formula reduces
  to standard CFG, so this only affects fidelity at higher CFG values.
- **Per-step Oklab clamp round-trip** — upstream optionally re-projects
  the predicted x0 through Oklab decode → clamp → re-encode each step to
  keep colors in-gamut. Not yet implemented; expect mild color drift in
  some prompts.
- **Resolution-dependent dynamic shift** — fixed at 17 for now.
- **Qwen3-VL prompt rewriter** — the demo's optional prompt-quality
  preprocessor.

## How it works (the surgery)

The AsymFLUX.2-klein adapter (707 MB) is *not* a standalone model — it's a
delta on top of FLUX.2-klein-base-9B containing:

- 116 LoRA tensors (rank 256) on `attn.to_out`, `ff.linear_in/out`,
  `ff_context.linear_in/out` for all 8 double + 24 single transformer
  blocks.
- Full overwrites for `x_embedder.weight` (3·16·16 = 768 → 4096),
  `proj_out.weight` (4096 → 768), and `norm_out.linear.weight` (4096 → 8192).
- Two extra buffers: `proj_buffer` (768, 128) and `scale_buffer` ()
  used by the AsymFlow velocity formula.

The Apply Adapter node mutates a cloned `ModelPatcher` so the underlying
diffusion model's `img_in` / `final_layer.linear` / `final_layer.adaLN_modulation`
are physically swapped to the new shapes, then wraps `forward` with the
AsymFlow calibration (per-batch `k = 1 / (s + (1-s)·σ)` scaling on the
input + replacing the timestep with `cal_t = t·k`) and the AsymFlow
velocity (orthogonal decomposition of the raw transformer output along the
`proj_buffer` subspace, blended with the current x_t). The math is a
bit-exact port of `lakonlab.models.architectures.asymflow.common`.

The `Sampler = uni_pc` + `Scheduler = simple` + `model_sampling` patched
to `(CONST, ModelSamplingFlux, shift=17)` reproduces the upstream
`FlowAdapterScheduler(base_scheduler='UniPCMultistep')` flow exactly:
ComfyUI's `CONST.calculate_denoised(σ, v, x) = x − v·σ` is precisely the
x₀ estimate the AsymFLUX.2 sampling loop computes per step.

See [`PORTING.md`](PORTING.md) for the full architecture diff.
