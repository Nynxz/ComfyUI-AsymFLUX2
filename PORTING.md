# AsymFLUX.2-klein → ComfyUI native port — design notes

This file captures what the AsymFLUX.2-klein adapter actually contains and
the concrete design for the ComfyUI loader, model wrapper, and sampler.
Read before adding more code.

## What the adapter file actually is

I read the safetensors header of `Lakonik/AsymFLUX.2-klein-9B/diffusion_pytorch_model.safetensors`
(707 MB total, 121 tensors) directly via an HF range request. It is **not**
a standalone model — it is an *adapter on top of* `black-forest-labs/FLUX.2-klein-base-9B`:

| Tensor | Shape | Role |
| --- | --- | --- |
| `proj_buffer` | `[768, 128]` (bf16) | AsymFlow low-rank projection (`patch_dim=3·16·16=768`, `base_rank=128`) |
| `scale_buffer` | `[]` (bf16) | AsymFlow per-model scalar `s` |
| `x_embedder.weight` | `[4096, 768]` (fp16) | **Full overwrite** of input projection (3-channel × 16×16 patch → inner_dim 4096) |
| `proj_out.weight` | `[768, 4096]` (fp16) | **Full overwrite** of output projection (inner_dim → 3-channel × 16×16) |
| `norm_out.linear.weight` | `[8192, 4096]` (fp16) | **Full overwrite** of final adaLN out |
| `*.lora_A.weight` / `*.lora_B.weight` | rank 256 | 116 LoRA tensors (58 pairs) on `attn.to_out`, `ff.linear_in/out`, `ff_context.linear_in/out` across all 8 double + 24 single transformer blocks |

The base FLUX.2-klein-base-9B transformer body (modulation, attention, ff)
is reused verbatim — only the input/output projections, the final norm,
and the per-layer attention/ff projections (via LoRA) are touched.

The adapter's published config (`config.json` on the same repo):

```json
{
  "_class_name": "AsymFlux2Transformer2DModel",
  "in_channels": 3,
  "patch_size": 16,
  "base_rank": 128,
  "num_layers": 8,
  "num_single_layers": 24,
  "num_attention_heads": 32,
  "attention_head_dim": 128,
  "joint_attention_dim": 12288,
  "guidance_embeds": false,
  "axes_dims_rope": [32, 32, 32, 32],
  "rope_theta": 2000,
  "mlp_ratio": 3.0,
  "num_timesteps": 1
}
```

Implications:

- Inner dim is `32 * 128 = 4096` — same as the FLUX.2 klein base.
- Block counts (8 + 24) match the base.
- Text-encoder-side dim is **12288**, which is the lakonlab pipeline's
  Qwen3 layer concat `(layer 9 ‖ layer 18 ‖ layer 27)` at 4096 each.
  ComfyUI's `flux2_te` text encoder needs to be inspected to confirm it
  produces this exact 12288-dim concat — if not, we will need a custom
  text-encoder wrapper.
- The base transformer's spatial behaviour (`patch_size=2`,
  `in_channels=16`) is **replaced** by the adapter
  (`patch_size=16`, `in_channels=3`). This is not just LoRA fine-tuning,
  it changes the model's input/output contract.
- `guidance_embeds=False` → AsymFLUX2 does *not* use FLUX-style
  distilled-guidance embeds; CFG is handled in the sampler instead.

## What the sampler does that KSampler can't

From `lakonlab/pipelines/pipeline_pixelflux2_klein.py`, the per-step loop
adds three things on top of vanilla flow-matching:

1. **Orthogonal CFG bias** (`guidance_jit`) — not the standard
   `cond + scale * (cond - uncond)` formula; mixes orthogonal/parallel
   components against the current `x_t - denoised * t` direction.
2. **Optional per-step Oklab clamp round-trip** — predict `x0`, decode it
   to RGB via Oklab, clamp to gamut, re-encode back to Oklab, recompute
   the velocity from the clamped estimate. Keeps colors in-gamut during
   sampling.
3. **`FlowAdapterScheduler`** — shift-warped sigmas + a UniPCMultistep
   ODE solver underneath. Dynamic shift is keyed on `image_seq_len`.

KSampler's `set_model_sampler_cfg_function` handles (1). KSampler has no
hook for (2). (3) is the easiest to bolt on as a Comfy `Sampler` subclass
with our own sigma schedule. Verdict: **own sampler node**.

## Architecture port plan

```
ComfyUI-AsymFLUX2/asymflux2/
├── nodes/
│   ├── empty_latent.py         ✅ done — 3ch pixel-latent emitter
│   ├── oklab.py                ✅ done — IMAGE ↔ LATENT pair
│   ├── loader.py               ⬜ AsymFlux2Loader (base safetensors + adapter safetensors → MODEL)
│   └── sampler.py              ⬜ AsymFlux2Sampler (own KSampler)
├── model/
│   ├── arch.py                 ⬜ AsymFlux2DiT — subclass of comfy.ldm.flux.model.Flux,
│   │                                overrides patch_size + projections + asymflow forward wrap
│   ├── base.py                 ⬜ AsymFlux2BaseModel — subclass of comfy.model_base.Flux2,
│   │                                holds proj_buffer/scale_buffer
│   ├── config.py               ⬜ AsymFlux2Config — supported_models.BASE entry,
│   │                                detection via 3ch x_embedder shape + proj_buffer key
│   └── load.py                 ⬜ key remapping (diffusers asymflux2 layer names → comfy flux2),
│                                    layer-shape surgery (img_in/final_layer.linear/norm),
│                                    LoRA fusion via comfy.sd.load_lora_for_models
├── sampler/
│   ├── orthogonal_cfg.py       ⬜ port lakonlab.models.diffusions.gaussian_flow.guidance_jit
│   ├── flow_adapter.py         ⬜ shift-warped sigma schedule (UniPC under the hood)
│   └── loop.py                 ⬜ denoising loop with optional Oklab clamp round-trip
└── oklab_math.py               ✅ done — pure Oklab transform
```

Phased delivery (testable at each phase):

- **Phase 1 (DONE)** — Empty Pixel Latent + Oklab Encode/Decode.
  Lets us verify the perceptual color round-trip in isolation.
- **Phase 2** — Loader. Build the model, load weights, run an opaque
  forward pass on noise. Visual output will be garbage; just confirms the
  shapes line up and there are no missing/unexpected keys.
- **Phase 3** — Custom sampler with vanilla CFG (no orthogonal, no
  clamp). Should produce valid-but-wrong images.
- **Phase 4** — Orthogonal CFG. Compare against upstream reference
  outputs (same prompt + seed, run upstream pipeline once locally).
- **Phase 5** — Per-step Oklab clamp round-trip. Final fidelity match.

## Open questions that need a real ComfyUI run to answer

1. **Text encoder compat** — does `comfy.text_encoders.flux.flux2_te` produce
   a `[B, L, 12288]` tensor at the layer indices AsymFLUX2 expects? Read
   `/home/user/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/text_encoders/flux.py`
   end-to-end during phase 2 implementation.
2. **FLUX.2 base weights format** — does `Comfy-Org/FLUX.2-klein` ship a
   single-file safetensors with `img_in.weight` and `final_layer.linear.weight`
   keys (Comfy layout), or does the user have to convert from diffusers?
   Loader needs to pick a path here.
3. **Patch-size mismatch tolerance** — comfy's `comfy.ldm.flux.model.Flux`
   reads `patch_size` from `params` at `__init__`. Our loader has to pass
   `patch_size=16` in the unet_config so the rearrange ops work. Verify
   no other code path assumes 2.
4. **Scheduler integration** — does it cleaner to ship our own
   `Sampler`/sigma-schedule than to extend `comfy.model_sampling.CONST`
   the way ComfyUI-piFlow does? Probably yes, given UniPC-under-flow-shift
   is non-trivial.

## Out of scope for v0.2

- Qwen3-VL prompt rewriter (the demo's optional second model).
- GGUF / fp8 quantized adapter loading.
- LoRA stacking on top of the AsymFLUX2 adapter.
