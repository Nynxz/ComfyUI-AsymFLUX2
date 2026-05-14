# ComfyUI-AsymFLUX2

Native ComfyUI v3 nodes for the **pixel-space** AsymFLUX.2-klein 9B model
from LakonLab (Chen et al., *Asymmetric Flow Models*, [arXiv 2605.12964](https://arxiv.org/abs/2605.12964)).

AsymFLUX.2 turns FLUX.2-klein-base-9B into a pixel-space generator —
instead of denoising in a VAE latent, the transformer operates directly
on a 3-channel Oklab image representation. This pack runs it **natively
in ComfyUI** (no diffusers, no `lakonlab` install) using the standard
`Load Diffusion Model`, `CLIPLoader`, and `KSampler` nodes alongside
four small AsymFLUX2-specific nodes.

## Workflow

```
Load Diffusion Model (FLUX.2-klein-base-9B)  ─┐
                                              ├─►  AsymFLUX2 Apply Adapter  ─►  KSampler  ─►  AsymFLUX2 Oklab Decode  ─►  Save Image
CLIPLoader (Qwen3 8B, type=flux2)  ─►  CLIPTextEncode (pos / neg)  ─────┘                ▲
                                                                                         │
                                  AsymFLUX2 Empty Pixel Latent  ───────────────────────┘
```

A working text-to-image workflow lives at
[`example_workflows/asymflux2_t2i.json`](example_workflows/asymflux2_t2i.json).

## Nodes

| Node | Role |
| --- | --- |
| **AsymFLUX2 Apply Adapter** | Patches a stock FLUX.2-klein MODEL into AsymFLUX.2 via `ModelPatcher.add_object_patch` (base model is never mutated). Swaps `img_in` / `final_layer` to 3 ch × 16 × 16 patches, registers `proj_buffer` / `scale_buffer`, wraps the diffusion forward with AsymFlow calibration + velocity, applies the rank-256 LoRA, sets the flow shift, makes the latent path pass-through, and (optionally) installs the orthogonal CFG bias and the per-step Oklab gamut clamp on x0. |
| **AsymFLUX2 Empty Pixel Latent** | Empty 3-channel pixel latent at 1:1 spatial resolution. Drop-in replacement for `EmptyLatentImage`. |
| **AsymFLUX2 Oklab Encode** | `IMAGE` → 3-channel Oklab `LATENT`. Pure-math port of LakonLab's `OklabColorEncoder`. |
| **AsymFLUX2 Oklab Decode** | 3-channel Oklab `LATENT` → `IMAGE`. Goes after KSampler. |

## Install

Clone into ComfyUI's custom nodes directory and restart:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-AsymFLUX2
```

No extra Python dependencies — `torch`, `einops`, and ComfyUI internals
are already in your ComfyUI env.

## Models you need

1. **FLUX.2-klein-base-9B** in `ComfyUI/models/diffusion_models/`. Use
   whichever single-file safetensors repackaging you have. Loaded via
   `Load Diffusion Model` (a.k.a. `UNETLoader`).
2. **AsymFLUX.2-klein adapter** —
   [`Lakonik/AsymFLUX.2-klein-9B/diffusion_pytorch_model.safetensors`](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B/blob/main/diffusion_pytorch_model.safetensors)
   (~707 MB). Drop into `ComfyUI/models/loras/` (rename to e.g.
   `asymflux2-klein-9b.safetensors` if you want).
3. **Qwen3 8B text encoder** — the FLUX.2-klein TE (single file).
   Loaded via `CLIPLoader` (single, not Dual) with `type=flux2`.

You'll need to accept the
[FLUX.2 klein license](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md)
and the AsymFLUX.2-klein license on Hugging Face first.

## Apply Adapter settings

| Input | Default | Notes |
| --- | --- | --- |
| `shift` | `17.0` | Paper convention; converted to comfy's `mu = log(shift)` internally. |
| `adapter_strength` | `1.0` | LoRA strength applied to the rank-256 LoRA. |
| `orthogonal_guidance` | `1.0` | Upstream `guidance_jit` strength. `0.0` = standard CFG. |
| `clamp_denoised` | `True` | Per-step Oklab gamut clamp on the x0 estimate. Prevents color drift; mostly a quality win. Turn off if you want to use `uni_pc` (see below). |

## KSampler settings

Mirroring the upstream Gradio defaults:

- **Sampler**: `dpmpp_2m_sde`
- **Scheduler**: `simple`
- **Steps**: 38 (range 4–50)
- **CFG**: 4.0
- **Resolution**: 960 × 1280 (snapped to multiples of 16)

> **Why not `uni_pc`?** Upstream uses `UniPCMultistep`, which is
> algorithmically the same as comfy's `uni_pc`. But our pipeline ships
> two post-CFG hooks (orthogonal CFG and the Oklab gamut clamp). The
> clamp is a hard non-linearity (`x.clamp(-1, 1)` in RGB space) that
> puts a kink in `denoised(sigma)`. UniPC's multistep polynomial
> extrapolation chokes on that kink: history points stop being valid
> predictors and the BH update overshoots → bad output. SDE samplers
> (`dpmpp_2m_sde`, `dpmpp_sde`) inject noise per step which averages
> over the kink. Use `dpmpp_2m_sde + simple`. If you want to A/B vs
> `uni_pc`, turn off `clamp_denoised` on the Apply Adapter node so the
> kink goes away.

## How it works

The AsymFLUX.2-klein adapter (707 MB, 121 tensors) is *not* a standalone
model — it's a delta on top of FLUX.2-klein-base-9B containing:

- **116 LoRA tensors** (rank 256) on `attn.to_out`, `ff.linear_in/out`,
  `ff_context.linear_in/out` across all 8 double + 24 single transformer
  blocks, plus the timestep MLP.
- **Full-tensor overwrites** for `x_embedder.weight` (3·16·16 → 4096),
  `proj_out.weight` (4096 → 3·16·16), and `norm_out.linear.weight`
  (the `LastLayer.adaLN_modulation[1]`).
- **Two extra buffers** used by the AsymFlow velocity formula:
  `proj_buffer` `(768, 128)` and `scale_buffer` (scalar `s ≈ 2.4375`).

The Apply Adapter node clones the input `ModelPatcher` and registers all
of the above as object patches (`ModelPatcher.add_object_patch`). The
base FLUX.2-klein-base-9B module itself is never mutated — patches apply
when the cloned patcher is loaded and revert when it is unloaded.

The wrapped `diffusion_model.forward` performs the AsymFlow per-step
recipe: scale input tokens by `k = 1 / (s + (1-s)·σ)`, replace the
timestep with `cal_t = t·k`, run the original `Flux.forward`, then
wrap the raw transformer output with the AsymFlow velocity (orthogonal
decomposition along `proj_buffer` blended with the current `x_t`). The
closure resolves `cls.forward.__get__(self)` per call, so it never
wraps a previous wrapper across re-runs.

Two optional post-CFG hooks (registered via
`set_model_sampler_post_cfg_function`) match the upstream pipeline:

- **Orthogonal CFG** — port of `lakonlab.models.diffusions.gaussian_flow.guidance_jit`,
  done in velocity space (numerically more stable at small sigma than the
  algebraically-equivalent x0-space formulation).
- **clamp_denoised** — every step, decode the predicted x0 to sRGB,
  clip to `[-1, 1]`, re-encode to Oklab. Prevents x0 drift out of
  valid color space.

The model is registered as `(CONST, ModelSamplingFlux)` with the user's
shift converted to comfy's `mu = log(shift)`. ComfyUI's
`CONST.calculate_denoised(σ, v, x) = x − v·σ` is exactly the AsymFLUX.2
sampling-loop formula for x0, so KSampler steps integrate the AsymFlow
velocity correctly.

The math (AsymFlow velocity, orthogonal CFG, Oklab encode/decode) was
verified bit-exact against `lakonlab.models.architectures.asymflow.common`
and `lakonlab.models.diffusions.gaussian_flow.guidance_jit` on synthetic
inputs.

## Not yet implemented

- **Resolution-dependent dynamic shift** — currently static `shift=17`.
  Upstream interpolates between `log(17)` and `log(34)` keyed on the
  pixel sequence length. For 960 × 1280 the dynamic value lands at ~20;
  the static 17 is a safe first cut. Matters more at higher
  resolutions.
- **Qwen3-VL prompt rewriter** — the demo's optional prompt-quality
  preprocessor.
