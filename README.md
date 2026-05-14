# ComfyUI-AsymFLUX2

ComfyUI nodes for running [AsymFLUX.2-klein](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B)
in pixel space on top of a stock FLUX.2-klein-base-9B safetensors.

AsymFLUX.2 is from the *Asymmetric Flow Models* paper
([arXiv 2605.12964](https://arxiv.org/abs/2605.12964), Chen et al.) and
ships as an adapter on top of FLUX.2-klein — the transformer denoises a
3-channel Oklab image directly instead of going through a VAE latent.

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

**KSampler:** demo defaults are `dpmpp_2m` + `simple`, 38 steps, CFG 4.0,
at 960 × 1280. Other samplers worth trying: `dpmpp_2m_sde`, `dpmpp_sde`,
`deis`. `uni_pc` is unstable with `clamp_denoised=True` (the gamut clip
is a hard non-linearity that the multistep polynomial doesn't like) —
turn the clamp off if you want to compare against `uni_pc`.

## Known limitations

- Output is close to the [HF Space](https://huggingface.co/spaces/Lakonik/AsymFLUX.2-klein)
  but not identical. The base model + adapter + Oklab encode are the same;
  the parts that diverge from upstream are:
    - Static flow shift (we use `17` from the paper; upstream uses a
      resolution-dependent dynamic shift between `log(17)` and `log(34)`).
      For 960 × 1280 the upstream value lands around `20`. Bigger gap at
      higher resolutions.
    - Sampler: upstream uses `UniPCMultistep` integrated by their
      `FlowAdapterScheduler`. We use ComfyUI's stock samplers — comfy's
      `uni_pc` is the same algorithm but interacts badly with our
      post-CFG hooks, so we route around it.
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
