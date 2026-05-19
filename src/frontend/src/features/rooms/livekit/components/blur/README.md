# Background Segmentation & Compositing Pipeline

## 1. Introduction

This folder implements the real-time background processing pipeline used by the
Meet video pipeline. Two user-facing effects are supported through a single
shared engine:

- **Background blur** — the camera frame is composited on top of a blurred copy
  of itself so the participant stays sharp while their surroundings melt away.
- **Virtual background** — the camera frame is composited on top of an
  arbitrary image so the participant appears in a different scene.

Both effects share the same problem: given a live video frame, recover a clean
alpha matte that separates the person from the background, then composite the
two layers as fast as the camera can deliver frames. In practice this is
constrained by several hard requirements that drive the entire design:

- **Cross-browser** — must run in Chromium, Firefox and Safari. Safari has
  historically broken `ctx.filter` and several Canvas2D paths, so all
  per-pixel work is moved to GLSL shaders.
- **Real time** — target ≥30 fps on commodity laptops, never blocking the
  camera's native cadence.
- **Resilience** — when anything fails (model load, WebGL2 init, segmenter
  timeout), the user must still see *something* — never a frozen frame.
- **Quality** — the segmenter operates at low resolution (256×144 or 256×256)
  but the output runs at the camera's native resolution (typically 1280×720),
  so the mask has to be upsampled and refined without producing visible
  halos or seams.

The pipeline is orchestrated by [AdvancedMattingProcessor.ts](AdvancedMattingProcessor.ts),
which implements the LiveKit `TrackProcessor` interface and exposes a
processed `MediaStreamTrack` to the rest of the application.

---

## 2. Pipeline Diagram

The two loops on either side of the diagram run independently. They meet at
the shared `_latestMask` slot: the segmenter loop writes it, the render loop
reads it. After that point, the rest of the pipeline runs entirely inside the
render loop on the GPU.

```
                   ┌────────────────────────────────────────┐
                   │      <video> element (camera)          │
                   │       full-res RGBA frame              │
                   └────────────────────┬───────────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       │                                 │
                  SEGMENTER LOOP                    RENDER LOOP
              (async while, ~50 Hz)             (requestAnimationFrame)
                       │                                 │
                       ▼                                 ▼
      ┌───────────────────────────────┐  ┌───────────────────────────────┐
      │   PREPROCESSING               │  │   VIDEO UPLOAD                │
      │   RoiCropper.getNextCropBbox  │  │   videoEl → videoTex          │
      │   sizeSource(bbox) drawImage  │  │   (full-res RGBA, per frame)  │
      │   → 256×N ImageData (RGBA)    │  └───────────────┬───────────────┘
      └───────────────┬───────────────┘                  │
                      ▼                                  │
      ┌───────────────────────────────┐                  │
      │   SEGMENTATION (Mediapipe)    │                  │
      │   GPU delegate probed 1×/sess │                  │
      │   AUTO benchmark @ 30 ms:     │                  │
      │    pass → Multiclass 256×256  │                  │
      │      (6 cls, fg = 1 − bg)     │                  │
      │    fail → Landscape 256×144   │                  │
      │      (binary, fg = mask[0])   │                  │
      └───────────────┬───────────────┘                  │
                      ▼                                  │
      ┌───────────────────────────────┐                  │
      │   CPU POSTPROCESS             │                  │
      │   applyAfterInference():      │                  │
      │    resize crop-space mask     │                  │
      │    paste into full-frame buf  │                  │
      │   RoiCropper.updateWithMask:  │                  │
      │    bbox dead-zone + EMA       │                  │
      └───────────────┬───────────────┘                  │
                      │ writes                           │ reads
                      │                                  │
                      └──────┐                    ┌──────┘
                             │                    │
                             ▼                    ▼
                       ┌──────────────────────────────────┐
                       │           _latestMask            │
                       │       Float32Array  [0, 1]       │
                       │          (shared slot)           │
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────────┐
                       │   GPU MASK POSTPROCESS           │
                       │   uploadMask: Float32 → R8       │
                       │   closing shader (dil + ero)     │
                       │   EMA   α·cur + (1 − α)·prev     │
                       │    (persistent emaTex)           │
                       └────────────────┬─────────────────┘
                                        ▼
                       ┌──────────────────────────────────┐
                       │   UPSAMPLING (proc → out res)    │
                       │   bilinear: free LINEAR sample   │
                       │   guided filter: GpuGuidedFilter │
                       │    (He et al. 2013, RGB —        │
                       │     7 box passes + apply)        │
                       └────────────────┬─────────────────┘
                                        ▼
                       ┌──────────────────────────────────┐
                       │   BACKGROUND CONSTRUCTION        │
                       │   BLUR (half-res):               │
                       │    1. masked downsample          │
                       │       (weight-normalised — no    │
                       │        halo)                     │
                       │    2. mask-weighted gauss H      │
                       │    3. mask-weighted gauss V      │
                       │   VIRTUAL:                       │
                       │    virtualBgTex (uploaded once   │
                       │    from HTMLImageElement)        │
                       └────────────────┬─────────────────┘
                                        ▼
                       ┌──────────────────────────────────┐
                       │   COMPOSITE                      │
                       │   BLUR / fallback:               │
                       │    pComposite                    │
                       │     mix(bg, video, eroded mask)  │
                       │   VIRTUAL (segmo path):          │
                       │    pSegmoEdgeFeather             │
                       │    pMaskedFg + pFgColorCast      │
                       │    pCompositeSegmo               │
                       │     edge-adaptive sharpen +      │
                       │     closed-form alpha matting +  │
                       │     I + (B_new − B_old)(1 − α)   │
                       │    pLightWrap (optional)         │
                       └────────────────┬─────────────────┘
                                        ▼
                       ┌──────────────────────────────────┐
                       │   output <canvas>                │
                       │   captureStream(30) →            │
                       │   processedTrack (to LiveKit)    │
                       └──────────────────────────────────┘
```

Inputs to each GPU stage that are not drawn explicitly: `videoTex` (refreshed
every render tick from the `<video>` element) is implicitly available to the
background-construction and composite stages — both sample it directly.

The two loops are intentionally decoupled. The render loop never blocks on
inference; if no fresh mask is available it either reuses the last one or
falls back to a passthrough mask filled with 1.0 (which composites the camera
unchanged). This is what guarantees the user always sees a live frame even if
the segmenter stalls.

---

## 3. Technique Details

### 3.1 Two-loop engine — [AdvancedMattingProcessor.ts](AdvancedMattingProcessor.ts)

**Problem.** Mediapipe inference is asynchronous and can spike beyond one
frame period on slower devices. If the render path waits on it, the camera
output stutters or freezes.

**How it works.** Two independent loops share a single `_latestMask`
reference:

- The **segmenter loop** is a free-running `async while` capped at ~50 Hz that
  pulls a frame, runs preprocessing + inference + postprocessing, and
  publishes the resulting mask. If inference exceeds 20 ms, the loop simply
  yields with `setTimeout(0)` and starts the next iteration immediately.
- The **render loop** is a `requestAnimationFrame` callback that reads
  `_latestMask` and composites unconditionally. It never awaits the
  segmenter.

The output `<canvas>` is wrapped with `captureStream(30)` to produce the
`MediaStreamTrack` that LiveKit publishes.

### 3.2 ROI cropping — [preprocessing/RoiCropper.ts](preprocessing/RoiCropper.ts)

**Problem.** Sending the full 1280×720 frame to a 256×256 model wastes
resolution on regions that do not contain the person. The model is
disproportionately accurate when the person fills the input.

**How it works.**
1. On each inference, scan the previous full-frame mask, compute the tight
   bounding box of pixels above 0.5, expand it by 5 % padding, and clamp to
   `[0, 1]`.
2. Apply a dead zone (3 % of frame on position, 1.5 % on size) before
   blending the new bbox with the current one via an EMA with factor 0.5.
   This kills small-motion jitter that would otherwise make the crop
   "breathe".
3. Every 45 frames, force a full-frame inference. This is what lets a newly
   arriving second person be detected — without it the bbox would lock onto
   the first person and never see the rest of the scene again.
4. After inference, [PreProcessingPipeline.applyAfterInference()](preprocessing/PreProcessingPipeline.ts)
   bilinearly resizes the crop-space mask and pastes it back into a
   zero-filled full-frame Float32Array so the rest of the pipeline always
   sees a mask in full-frame coordinates.

### 3.3 Segmenter selection & GPU delegate probing — [segmenters/](segmenters/)

**Problem.** The multiclass model (256×256, 6 classes) gives noticeably
cleaner edges than the landscape model (256×144, binary), but is heavier.
On CPU-only Mediapipe delegates it spends 80–150 ms per frame, which would
saturate the inference queue and effectively look like a frozen mask.

**How it works.**
- [probeMediapipeDelegate()](segmenters/Segmenter.ts) tries to instantiate an
  `ImageSegmenter` with `delegate: 'GPU'` once per session and memoises the
  result. If it throws, the session falls back to CPU. This replaces an
  older user-agent sniff that incorrectly disabled GPU on Safari 17+.
- In `SegmentationModel.AUTO` mode, [AdvancedMattingProcessor._benchmarkSegmenter()](AdvancedMattingProcessor.ts)
  runs 4 warmup-then-timed inferences on the multiclass model with a dummy
  frame. If the average latency ≤ 30 ms the multiclass model is kept,
  otherwise the segmenter is destroyed and replaced with the landscape
  model. If the GPU delegate probe came back CPU, benchmarking is skipped
  entirely and landscape is used directly.
- Both segmenters return a `Float32Array` mask in `[0, 1]` where 1 = person.
  For multiclass the foreground probability is computed as `1 − bg_prob`
  rather than summing the five "person" classes (faster, equivalent).
- Each `segment()` call races inference against a 2 s timeout — if the model
  hangs, the loop catches and continues rather than wedging the entire
  pipeline.

### 3.4 GPU postprocessing — [renderers/WebGl2Renderer.ts](renderers/WebGl2Renderer.ts)

**Closing (dilation then erosion).** Small holes inside the person mask
(e.g. between fingers or against a similarly-coloured background) become
visible as flickering background showing through the body. Closing fills
holes smaller than the kernel radius without growing the silhouette.
Implemented as two passes of a 1D min/max shader.

**Temporal EMA.** Inference is non-deterministic frame to frame. Even on a
static subject, the silhouette wiggles by ±1–2 pixels each frame, which the
eye reads as boiling. The EMA pass runs `out = α·cur + (1−α)·prev` on a
persistent `emaTex` and damps that high-frequency noise. The first frame
after a config change uses α = 1.0 to avoid the mask appearing to fade in.

The CPU equivalents in [postprocessing/Morphology.ts](postprocessing/Morphology.ts)
and [postprocessing/TemporalEMA.ts](postprocessing/TemporalEMA.ts) are kept
for reference and for paths that need to operate on the host-side mask
(e.g. RoiCropper feedback).

### 3.5 Mask upsampling

The segmenter mask lives at processing resolution (256×144 or 256×256),
which is roughly 10× smaller than the camera frame on either axis. How we
upsample directly drives the perceived quality of the silhouette.

- **Bilinear.** Free, performed implicitly when the composite shader samples
  the small mask texture with `LINEAR` filtering. Good enough at the
  default blur radius — the blur itself hides the soft edges.
- **Guided filter.** [GpuGuidedFilter.ts](renderers/GpuGuidedFilter.ts)
  implements the He et al. (2013) RGB-guided filter entirely in GLSL.
  Conceptually: for each pixel of the high-res output, take the
  bilinearly-upsampled low-res mask `p`, learn an affine model
  `q = a·I + b` that explains `p` from the high-res RGB guide `I` within a
  local window, then box-blur the per-window `(a, b)` and apply.
  In practice, that means seven RGBA32F box-filter passes (statistics for
  `I`, `I·I`, `I·p`), a 3×3 covariance solve, then a final apply pass.
  The effect is that the mask edge snaps to actual RGB edges in the video —
  hair, shoulders, glasses — instead of cutting through them at the
  low-res sample grid. Requires `EXT_color_buffer_float`.

### 3.6 Background construction (blur path)

A naive "downsample then gaussian blur then composite" produces a visible
dark halo around the silhouette: when the small downsample pass samples a
3×3 neighbourhood near the person, the person's pixels get averaged into
the "background" output. The halo is the colour of the *person* bleeding
out, not of the background bleeding in.

The fix used here is **mask-weighted blur with weight normalisation**:

1. **Masked downsample** — each output pixel samples a 3×3 area of the
   full-res video, weights every sample by `(1 − mask)`, sums both the
   weighted RGB and the weights, and divides at the end. Pixels that fall
   entirely inside the person contribute zero weight and zero color, so the
   output is whatever the *background* pixels in the neighbourhood looked
   like.
2. **Mask-weighted horizontal gaussian** at half resolution.
3. **Mask-weighted vertical gaussian** at half resolution.

The blur runs at half output resolution because the result is going to be
behind the (soft-edged) person anyway — full-res blur is invisible and
costs 4× the work.

### 3.7 Composite — blur and fallback path

The standard composite shader implements `mix(background, video, mask)`,
with the mask first eroded by `postCfg.erosion.pixels` (sampled per-pixel
via a min over a small neighbourhood directly inside the shader). The
erosion shrinks the visible silhouette by a couple of pixels, hiding any
remaining halo from the mask not perfectly aligning with the RGB edge.

### 3.8 Composite — segmo path (virtual background)

When the mode is `virtual` *and* a virtual background image has finished
uploading to a GPU texture, the renderer switches to a more elaborate
compositor designed to handle the harder case of pasting the person onto a
new scene. The reason this case is harder than blur: the new background has
arbitrary colors, so any leftover halo or color contamination from the old
background becomes obvious instead of being hidden behind a similar-tone
blur of the same frame.

Stages:

1. **Edge feather** (`pSegmoEdgeFeather`) — gaussian-blurs the mask only in
   a narrow band near the silhouette. Widens the transition zone so the
   closed-form matting in the next step has more pixels to operate on,
   without affecting the interior or exterior of the mask.
2. **Foreground masked extraction** (`pMaskedFg`) + **mean colour cast**
   (`pFgColorCast`) — extract the person, then optionally tint the camera
   frame slightly toward the mean colour of the virtual background. The
   means are recovered for free by reading the top mip level of mipmapped
   textures, so no CPU readback is needed.
3. **Segmo composite** (`pCompositeSegmo`) — runs the
   foreground-recovery shader: edge-adaptive sharpening driven by the
   camera gradient, closed-form alpha matting on a 13-tap cross pattern
   inside the feathered transition band, a chroma-aware colour-separation
   gate, and the VFX decontamination equation `output = I + (B_new − B_old) × (1 − α)`
   to subtract the contribution of the original (unknown) background from
   contaminated edge pixels. This intentionally subsumes the erosion step
   from the standard composite.
4. **Light wrap** (`pLightWrap`, optional) — mixes a small fraction of the
   background colour into the foreground edge band so the subject reads as
   lit by the new scene rather than pasted onto it. Skipped when the
   strength is zero.

The blur path and the virtual-no-image fallback never enter this codepath
and run the standard `pComposite` shader instead.

### 3.9 Error handling & resilience — [errors/MattingErrorStore.ts](errors/MattingErrorStore.ts)

Every failure mode that is recoverable converts to an entry in the matting
error store rather than a thrown exception:

- WebGL2 context creation failure → fall back to passing the raw track
  through unchanged.
- Mediapipe init failure → continue rendering with a passthrough mask
  (all-ones) so the user sees their camera; the user can retry by toggling
  the effect off and on.
- Segmenter `segment()` timeout (2 s) → drop the frame, sleep 100 ms,
  continue.
- Virtual background image load failure → keep the blur fallback active.

The render loop is designed so that it can always proceed: if there is no
mask yet, it composites against the passthrough mask, which `mix()`-es to
the raw camera frame.

---

## Key files at a glance

| File | Role |
|---|---|
| [AdvancedMattingProcessor.ts](AdvancedMattingProcessor.ts) | Orchestrator, two-loop engine, segmenter selection, lifecycle |
| [index.ts](index.ts) | Public API: `ProcessorConfig`, `SegmentationModel`, factory |
| [preprocessing/PreProcessingPipeline.ts](preprocessing/PreProcessingPipeline.ts) | Pre/post-inference orchestration |
| [preprocessing/RoiCropper.ts](preprocessing/RoiCropper.ts) | Person-tracking bbox with dead zone + EMA, 45-frame full-frame refresh |
| [segmenters/Segmenter.ts](segmenters/Segmenter.ts) | Shared interface, GPU delegate probe, fileset cache |
| [segmenters/LandscapeSegmenter.ts](segmenters/LandscapeSegmenter.ts) | 256×144 binary selfie segmenter |
| [segmenters/MulticlassSegmenter.ts](segmenters/MulticlassSegmenter.ts) | 256×256 6-class selfie segmenter |
| [postprocessing/Morphology.ts](postprocessing/Morphology.ts) | CPU separable min/max morphology (erosion/dilation/opening/closing) |
| [postprocessing/TemporalEMA.ts](postprocessing/TemporalEMA.ts) | CPU temporal EMA reference |
| [renderers/GpuRenderer.ts](renderers/GpuRenderer.ts) | Backend-agnostic renderer interface |
| [renderers/WebGl2Renderer.ts](renderers/WebGl2Renderer.ts) | WebGL2 compositor: postprocessing, blur path, segmo path |
| [renderers/GpuGuidedFilter.ts](renderers/GpuGuidedFilter.ts) | GPU implementation of guided-filter mask upsampling |
| [errors/MattingErrorStore.ts](errors/MattingErrorStore.ts) | Centralised, non-fatal error reporting |
