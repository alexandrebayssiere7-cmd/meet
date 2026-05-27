import { ProcessorOptions, Track } from 'livekit-client'
import {
  BackgroundProcessorInterface,
  LatencyMode,
  MaskBlendMode,
  ProcessorConfig,
  ProcessorType,
  SegmentationModel,
  PostProcessingConfig,
  UpsamplingConfig,
  PreProcessingConfig,
} from '.'
import { PreProcessingPipeline } from './preprocessing/PreProcessingPipeline'
import { MaskMotionTracker } from './preprocessing/MaskMotionTracker'
import { BBox } from './preprocessing/RoiCropper'
import {
  Segmenter,
  createSegmenter,
  RVMSegmenter,
  probeMediapipeDelegate,
} from './segmenters'
import { WebGl2Renderer } from './renderers/WebGl2Renderer'
import { Canvas2dRenderer } from './renderers/Canvas2dRenderer'
import { GpuRenderer, GpuRendererInitOpts } from './renderers/GpuRenderer'
import {
  pushMattingError,
  dismissMattingError,
} from './errors/MattingErrorStore'
import { debugLog, debugWarn, debugInfo } from './debug'
import {
  pushGapSample,
  pushInferenceSample,
  pushLatencySample,
  resetMattingStats,
  setCameraSettings,
  setEffectiveLatencyMode,
  setMaskOffset,
  setMattingStatsActive,
  setMattingStatsModel,
  setMotionScore,
  setPredictionActive,
  setSegmenterFrameSkip,
  tickCameraFrame,
  tickRenderFrame,
  tickSegmenterFrame,
} from './stats/MattingStatsStore'

const SEGMENTATION_MASK_CANVAS_ID = 'background-blur-local-segmentation'
const BLUR_CANVAS_ID = 'background-blur-local'
const DEFAULT_BLUR = 10
const DEFAULT_LATENCY_MODE: LatencyMode = 2

// Auto-tuning thresholds for the latency/halo trade-off (uv per second).
// Hysteresis prevents the mode from flapping between frameLock/blend/live at
// the boundaries. Tweak these if the auto mode feels jittery in real use.
const AUTO_LOCK_THRESHOLD = 0.1
const AUTO_LIVE_THRESHOLD = 0.6
const AUTO_HYSTERESIS = 0.05
// When Prediction is opted-in, the user has explicitly accepted the halo
// trade-off in exchange for lower latency. We then bias the auto-tuner toward
// the live side: never stay fully frame-locked, and use a non-zero blend
// baseline even at zero motion (the prediction warp is a no-op then anyway).
const AUTO_PRED_BLEND_BASELINE = 0.5

// Hard caps for the mask warp prediction so a noisy velocity never produces
// visible halos. 0.08 uv ≈ 8% of the frame width.
const MAX_PREDICTION_OFFSET_UV = 0.08
// Frame-budget used to size the blend cross-fade ramp at mode "Équilibré".
const FRAME_MS = 1000 / 30
const BLEND_MODE_MAX_AGE_MS = FRAME_MS * 4

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

function clampOffset(v: number): number {
  if (v > MAX_PREDICTION_OFFSET_UV) return MAX_PREDICTION_OFFSET_UV
  if (v < -MAX_PREDICTION_OFFSET_UV) return -MAX_PREDICTION_OFFSET_UV
  return v
}

// Mapping from the user-facing LatencyMode (0..4) to the internal effective
// blend mode + prediction gain. Used when `latencyAuto` is false.
const STATIC_MODE_TABLE: ReadonlyArray<{
  mode: MaskBlendMode
  predictionGain: number
}> = [
  { mode: 'frameLock', predictionGain: 0 }, // 0 Lock
  { mode: 'frameLock', predictionGain: 0 }, // 1 Stable (handled separately by EMA boost)
  { mode: 'blend', predictionGain: 0 }, // 2 Équilibré
  { mode: 'live', predictionGain: 0.5 }, // 3 Réactif
  { mode: 'live', predictionGain: 1.0 }, // 4 Live
]

/**
 * Pair of mask + the exact source frame that produced it. Stored together to
 * allow frame-locked compositing (mask applied to its own source frame, no
 * spatial mismatch).
 */
interface FrameMaskPair {
  mask: Float32Array
  source: ImageBitmap
  captureTime: number
  // Approx. moment the camera shutter captured this frame (DOMHighResTimeStamp,
  // same clock as performance.now()). Sourced from rVFC metadata.captureTime
  // when supported; falls back to `captureTime` (snapshot time) otherwise.
  cameraCaptureTime: number
  procW: number
  procH: number
}

interface VideoFrameMeta {
  captureTime: number
  presentationTime: number
  mediaTime: number
  receivedAt: number
}

/**
 * Unified background processor using WebGL2 for compositing.
 *
 * Two independent loops:
 *   Segmenter loop  — free-running async, pulls frames, runs inference, then
 *                     publishes a (mask, captured frame) pair as
 *                     _latestPair as fast as the GPU allows.
 *   Render loop     — requestVideoFrameCallback (fallback: rAF), fires at the
 *                     camera's native framerate, composites using the pair so
 *                     the mask is always applied to the frame it was computed
 *                     from (frame-locked). When maxFrameOffset > 0, the render
 *                     loop may instead apply the mask to the live <video>
 *                     frame as long as the pair is younger than N frames
 *                     (latency vs. halo trade-off, user-controlled).
 *
 * Decoupling prevents RVM's ~20-50ms inference from introducing render jitter.
 * All blur passes run in GLSL shaders — no ctx.filter (unreliable on Safari).
 */
export class AdvancedMattingProcessor implements BackgroundProcessorInterface {
  options: ProcessorConfig
  name: string
  type: ProcessorType
  processedTrack?: MediaStreamTrack

  source?: MediaStreamTrack
  sourceSettings?: MediaTrackSettings
  videoElement?: HTMLVideoElement
  videoElementLoaded?: boolean

  outputCanvas?: HTMLCanvasElement

  segmentationMaskCanvas?: HTMLCanvasElement
  segmentationMaskCanvasCtx?: CanvasRenderingContext2D
  sourceImageData?: ImageData

  // Full-res snapshot canvas: a single sync drawImage(videoElement) per
  // segmenter iteration, used to derive both the segmenter input AND the
  // ImageBitmap consumed by the renderer — guaranteeing both come from the
  // exact same video instant (no drift while createImageBitmap awaits).
  private _snapshotCanvas?: HTMLCanvasElement
  private _snapshotCanvasCtx?: CanvasRenderingContext2D

  private _motionCanvas?: HTMLCanvasElement
  private _motionCanvasCtx?: CanvasRenderingContext2D
  private static readonly MOTION_W = 128
  private static readonly MOTION_H = 72

  segmenter?: Segmenter
  private gpuRenderer?: GpuRenderer
  private _passthroughMask?: Float32Array

  // Two-loop state
  private _segLoopActive = false
  private _latestPair: FrameMaskPair | null = null
  private _renderLoopHandle: number | null = null
  // Last frame metadata delivered by requestVideoFrameCallback on the source
  // <video>. Used to derive end-to-end capture→display latency. When the API
  // isn't supported (older Firefox/Safari), this stays undefined and the stats
  // fall back to snapshot-time approximation.
  private _latestVideoFrameMeta?: VideoFrameMeta
  private _rvfcHandle: number | null = null
  // Monotonic counter of rVFC ticks on the source <video>. The segmenter loop
  // reads this to drive its frame-skip logic (run inference once every N
  // camera frames) and to wake on actual camera ticks rather than a fixed
  // timer. Only meaningful when rVFC is available.
  private _videoFrameSeq = 0
  // One-shot promise resolved by the rVFC tick to wake the segmenter loop.
  // Recreated lazily on each await so multiple ticks between two awaits don't
  // queue spurious wakes.
  private _frameAwaiter: {
    promise: Promise<void>
    resolve: () => void
  } | null = null
  // Last `_videoFrameSeq` value that produced an inference, used by the
  // frame-skip gate. Init to -INF so the very first frame always runs.
  private _lastInferenceSeq = -1
  // Number of camera frames to skip between inferences. Set by the benchmark
  // at init time: 1 = 30fps segmenter (fast GPU), 2 = 15fps segmenter (mid GPU).
  private _segmenterFrameSkip = 2
  // User-facing latency/halo controls. `latencyAuto` resolves the effective
  // mode each frame from the motion tracker; `latencyMode` is the manual
  // preset used otherwise. `maskPrediction` toggles the velocity-driven mask
  // warp (off by default — opt-in).
  private _latencyMode: LatencyMode = DEFAULT_LATENCY_MODE
  private _latencyAuto = true
  private _maskPrediction = false
  // Last effective mode resolved by `_renderFrame`, used by the auto-tuning
  // hysteresis on the next frame to decide whether to switch.
  private _lastEffectiveMode: MaskBlendMode = 'frameLock'
  // Per-frame motion tracker fed from the segmenter loop with the latest
  // stabilised RoiCropper bbox. Stays unfed (and `valid() === false`) when
  // ROI cropping is disabled, which the auto/prediction logic checks before
  // engaging anything.
  private _motionTracker = new MaskMotionTracker()

  virtualBackgroundImage?: HTMLImageElement

  private _configuredModel?: SegmentationModel
  currentModel?: SegmentationModel
  private processingWidth = 256
  private processingHeight = 144
  private _pendingModel?: SegmentationModel
  private _readyResolvers: Array<() => void> = []
  private _destroyed = false
  private _preProcessingPipeline?: PreProcessingPipeline
  private _lastMask?: Float32Array
  // Tracks video.currentTime of the last processed frame to detect duplicate
  // frames (camera delivering slower than the segmenter loop cadence).
  private _lastVideoTime = -1

  constructor(opts: ProcessorConfig) {
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this.options = opts
    this.type = opts.type
    const cfg = this._readLatencyConfig(opts)
    this._latencyMode = cfg.mode
    this._latencyAuto = cfg.auto
    this._maskPrediction = cfg.prediction
  }

  /**
   * Resolve the latency/halo controls from the processor config, applying the
   * "auto + prediction require ROI cropping" guard rail. When ROI cropping is
   * off the auto-tuning and prediction features have no motion signal to act on,
   * so we force-disable them at the source rather than rely on runtime checks.
   */
  private _readLatencyConfig(opts: ProcessorConfig): {
    mode: LatencyMode
    auto: boolean
    prediction: boolean
  } {
    if (
      opts.type !== ProcessorType.BLUR &&
      opts.type !== ProcessorType.VIRTUAL
    ) {
      return { mode: DEFAULT_LATENCY_MODE, auto: false, prediction: false }
    }
    const rawMode = opts.latencyMode
    const mode: LatencyMode =
      rawMode === 0 ||
      rawMode === 1 ||
      rawMode === 2 ||
      rawMode === 3 ||
      rawMode === 4
        ? rawMode
        : DEFAULT_LATENCY_MODE
    const roiEnabled = opts.preProcessing?.roiCropping?.enabled === true
    return {
      mode,
      auto: roiEnabled && opts.latencyAuto !== false,
      prediction: roiEnabled && opts.maskPrediction === true,
    }
  }

  /** Resolves once the active segmenter is loaded and producing frames. */
  waitForReady(): Promise<void> {
    if (this.segmenter || this._destroyed) return Promise.resolve()
    return new Promise((resolve) => this._readyResolvers.push(resolve))
  }

  private _resolveReady() {
    this._readyResolvers.splice(0).forEach((r) => r())
  }

  async init(opts: ProcessorOptions<Track.Kind>) {
    this._destroyed = false
    if (!opts.element) {
      throw new Error('Element is required for processing')
    }
    this.source = opts.track as MediaStreamTrack
    this.sourceSettings = this.source!.getSettings()
    this.videoElement = opts.element as HTMLVideoElement
    this._publishCameraSettings()
    const video = this.videoElement

    try {
      if (this._destroyed) return

      if (video.videoWidth === 0 || video.readyState < 2) {
        await new Promise<void>((resolve) => {
          const handleLoaded = () => {
            if (video.videoWidth > 0) {
              cleanup()
              resolve()
            }
          }
          const cleanup = () => {
            video.removeEventListener('loadedmetadata', handleLoaded)
            video.removeEventListener('loadeddata', handleLoaded)
            video.removeEventListener('canplay', handleLoaded)
            video.removeEventListener('playing', handleLoaded)
          }
          video.addEventListener('loadedmetadata', handleLoaded)
          video.addEventListener('loadeddata', handleLoaded)
          video.addEventListener('canplay', handleLoaded)
          video.addEventListener('playing', handleLoaded)
          setTimeout(() => {
            cleanup()
            resolve()
          }, 1000)
        })
      }

      if (this._destroyed) return

      const realW = video.videoWidth || this.sourceSettings!.width || 1280
      const realH = video.videoHeight || this.sourceSettings!.height || 720

      this._initVirtualBackgroundImage()
      this._createMainCanvasWithSize(realW, realH)
      this._createMaskCanvas()

      if (this._destroyed) return

      const rendererOpts: GpuRendererInitOpts = {
        outW: realW,
        outH: realH,
        processingW: this.processingWidth,
        processingH: this.processingHeight,
        postProcessing: this._getPostProcessingConfig(),
        upsampling: this._getUpsamplingConfig(),
      }
      this.gpuRenderer = await this._initRendererWithFallback(rendererOpts)
      this._applyRendererConfig()

      if (this._destroyed) return

      if (!this.outputCanvas!.captureStream) {
        throw new Error('captureStream not supported on this browser')
      }
      const stream = this.outputCanvas!.captureStream(30)
      const tracks = stream.getVideoTracks()
      if (tracks.length === 0) {
        throw new Error('No tracks found in captureStream()')
      }
      this.processedTrack = tracks[0]

      if (this._destroyed) {
        if (this.processedTrack && this.processedTrack !== this.source) {
          try {
            this.processedTrack.stop()
          } catch {
            // best-effort cleanup; track may already be stopped
          }
        }
        this.processedTrack = undefined
        return
      }

      this._startLoops()

      this._configuredModel = this._getModel(this.options)
      // Initialize segmenter in background — passthrough renders until it's ready.
      this._initSegmenterBackground(this._configuredModel)
    } catch (e) {
      debugWarn(
        '%c┌────────────────────────────────────────────────────────────┐\n' +
          '│ [AMP INIT] INITIALIZATION FAILED                           │\n' +
          '├────────────────────────────────────────────────────────────┤\n' +
          `│  Error: ${e instanceof Error ? e.message.padEnd(50).slice(0, 50) : String(e).padEnd(50).slice(0, 50)} │\n` +
          '│  Falling back transparently to passthrough raw track.     │\n' +
          '└────────────────────────────────────────────────────────────┘',
        'color: #ef4444; font-weight: bold; font-family: monospace; font-size: 11px;'
      )
      pushMattingError({
        code: 'WEBGL2_INIT_FAILED',
        level: 'warn',
        detail: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
      })
      this.processedTrack = this.source
    }
  }

  async update(opts: ProcessorConfig): Promise<void> {
    this.options = opts
    this.type = opts.type
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    const cfg = this._readLatencyConfig(opts)
    const autoChanged = cfg.auto !== this._latencyAuto
    const predictionChanged = cfg.prediction !== this._maskPrediction
    this._latencyMode = cfg.mode
    this._latencyAuto = cfg.auto
    this._maskPrediction = cfg.prediction
    if (autoChanged || predictionChanged) {
      this._motionTracker.reset()
      this._lastEffectiveMode = 'frameLock'
    }

    if (!this.gpuRenderer) {
      debugInfo(
        '[AMP] Update called in passthrough fallback mode; ignoring background processor updates.'
      )
      return
    }

    const prevConfigured = this._configuredModel
    const newModel = this._getModel(opts)
    this._configuredModel = newModel
    const prevRvmRatio = this._getRvmRatio(this.options)
    const nextRvmRatio = this._getRvmRatio(opts)

    this._initVirtualBackgroundImage()

    if (newModel !== prevConfigured) {
      if (this.segmenter) {
        this._switchSegmenterBackground(newModel)
      } else {
        this._initSegmenterBackground(newModel)
      }
    } else if (
      newModel === SegmentationModel.RVM &&
      nextRvmRatio !== prevRvmRatio &&
      this.segmenter instanceof RVMSegmenter
    ) {
      this.segmenter.setDownsampleRatio(nextRvmRatio ?? this._autoRvmRatio())
    }
    this._applyRendererConfig()
  }

  private _getModel(opts: ProcessorConfig): SegmentationModel {
    if (
      opts.type === ProcessorType.BLUR ||
      opts.type === ProcessorType.VIRTUAL
    ) {
      return opts.model ?? SegmentationModel.AUTO
    }
    return SegmentationModel.AUTO
  }

  /**
   * Common measurement protocol: 5 warm-up runs (displayed but not timed) +
   * 15 timed runs on fresh video frames. Returns p75 latency in ms, or null
   * if a warm-up run throws (caller decides how to handle).
   * Each result is published to _latestPair so the render loop shows the
   * effect building up during the benchmark.
   */
  private async _measureInferenceP75(seg: Segmenter): Promise<number | null> {
    const width = seg.inputSize.width
    const height = seg.inputSize.height

    const benchCanvas = document.createElement('canvas')
    benchCanvas.width = width
    benchCanvas.height = height
    const ctx = benchCanvas.getContext('2d')
    if (!ctx) return null

    const hasRealFrame = (): boolean =>
      !!(
        this.videoElement &&
        this.videoElement.readyState >= 2 &&
        this.videoElement.videoWidth > 0
      )

    const captureFrame = (): ImageData => {
      if (hasRealFrame()) ctx.drawImage(this.videoElement!, 0, 0, width, height)
      return ctx.getImageData(0, 0, width, height)
    }

    const publishFrame = async (mask: Float32Array): Promise<void> => {
      if (!hasRealFrame()) return
      const now = performance.now()
      let bitmap: ImageBitmap
      try {
        bitmap = await createImageBitmap(benchCanvas, {
          imageOrientation: 'flipY',
        })
      } catch {
        return
      }
      if (this._destroyed) {
        bitmap.close()
        return
      }
      const prev = this._latestPair
      this._latestPair = {
        mask,
        source: bitmap,
        captureTime: now,
        cameraCaptureTime: now,
        procW: width,
        procH: height,
      }
      prev?.source.close()
    }

    const WARMUP = 5
    for (let i = 0; i < WARMUP; i++) {
      if (this._destroyed) return null
      const frame = captureFrame()
      const mask = await seg.segment(frame, performance.now()) // throws → caller handles
      await publishFrame(mask)
    }

    const RUNS = 15
    const samples: number[] = []
    for (let i = 0; i < RUNS; i++) {
      if (this._destroyed) return null
      const frame = captureFrame()
      const start = performance.now()
      const mask = await seg.segment(frame, performance.now())
      samples.push(performance.now() - start)
      await publishFrame(mask)
    }

    samples.sort((a, b) => a - b)
    return samples[Math.floor(RUNS * 0.75)] // p75: index 11 of 15
  }

  private async _benchmarkSegmenter(
    seg: Segmenter
  ): Promise<'landscape' | 'multiclass_skip1' | 'multiclass_skip2'> {
    const B =
      'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;'
    const W = 'color: #e2e8f0; font-family: monospace; font-size: 11px;'
    try {
      const probe = await probeMediapipeDelegate()
      if (probe === 'CPU') {
        debugWarn(
          '%c┌────────────────────────────────────────────────────────────┐\n' +
            '│ [AMP BENCHMARK] SKIPPED: CPU DELEGATE DETECTED             │\n' +
            '├────────────────────────────────────────────────────────────┤\n' +
            '│  Device WebGL Delegate is CPU.                             │\n' +
            '│  To prevent performance degradation, benchmarking is       │\n' +
            '│  skipped and Landscape fallback is automatically used.     │\n' +
            '└────────────────────────────────────────────────────────────┘',
          B
        )
        return 'landscape'
      }

      let p75: number | null
      try {
        p75 = await this._measureInferenceP75(seg)
      } catch {
        debugWarn(
          '%c┌────────────────────────────────────────────────────────────┐\n' +
            '│ [AMP BENCHMARK] WARM-UP FAILED                             │\n' +
            '├────────────────────────────────────────────────────────────┤\n' +
            '│  Warm-up run threw — safe fallback to Landscape.           │\n' +
            '└────────────────────────────────────────────────────────────┘',
          B
        )
        return 'landscape'
      }
      if (p75 === null || this._destroyed) return 'landscape'

      let result: 'landscape' | 'multiclass_skip1' | 'multiclass_skip2'
      let resultVal: string
      let resultColor: string
      if (p75 < 25) {
        result = 'multiclass_skip1'
        resultVal = 'PASS — Multiclass 30fps (skip=1)'
        resultColor =
          'color: #10b981; font-weight: bold; font-family: monospace; font-size: 11px;'
      } else if (p75 <= 50) {
        result = 'multiclass_skip2'
        resultVal = 'PASS — Multiclass 15fps (skip=2)'
        resultColor =
          'color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px;'
      } else {
        result = 'landscape'
        resultVal = 'FAIL — Landscape fallback'
        resultColor =
          'color: #ef4444; font-weight: bold; font-family: monospace; font-size: 11px;'
      }

      const pad = (s: string, n: number) =>
        s + ' '.repeat(Math.max(0, n - s.length))
      const W60 = 60
      const p75Str = `${p75.toFixed(2)} ms`
      const resStr = resultVal
      debugLog(
        `%c┌────────────────────────────────────────────────────────────┐\n` +
          `%c│%c${pad('  [AMP BENCHMARK] MULTICLASS PERFORMANCE', W60)}%c│\n` +
          `%c├────────────────────────────────────────────────────────────┤\n` +
          `%c│%c  Protocol: 5 warm-up + 15 timed runs, fresh frames${' '.repeat(W60 - 50)}%c│\n` +
          `%c│%c  P75 Inference Latency:     %c${p75Str}${' '.repeat(Math.max(0, W60 - 28 - p75Str.length))}%c│\n` +
          `%c├────────────────────────────────────────────────────────────┤\n` +
          `%c│%c${pad('  Tier 1 (skip=1, 30fps):   < 25.00 ms', W60)}%c│\n` +
          `%c│%c${pad('  Tier 2 (skip=2, 15fps):  25–50.00 ms', W60)}%c│\n` +
          `%c│%c${pad('  Tier 3 (Landscape):        > 50.00 ms', W60)}%c│\n` +
          `%c├────────────────────────────────────────────────────────────┤\n` +
          `%c│%c  Evaluation Result:         %c${resStr}${' '.repeat(Math.max(0, W60 - 28 - resStr.length))}%c│\n` +
          `%c└────────────────────────────────────────────────────────────┘`,
        B,
        B,
        'color: #60a5fa; font-weight: bold; font-family: monospace; font-size: 11px;',
        B,
        B,
        B,
        W,
        B,
        B,
        W,
        'color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px;',
        B,
        B,
        B,
        W,
        B,
        B,
        W,
        B,
        B,
        W,
        B,
        B,
        B,
        W,
        resultColor,
        B,
        B
      )

      return result
    } catch {
      debugWarn(
        '%c┌────────────────────────────────────────────────────────────┐\n' +
          '│ [AMP BENCHMARK] ERROR — falling back to Landscape.         │\n' +
          '└────────────────────────────────────────────────────────────┘',
        B
      )
      return 'landscape'
    }
  }

  private async _benchmarkLandscapeSkip(
    seg: Segmenter
  ): Promise<'skip1' | 'skip2'> {
    const B =
      'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;'
    const W = 'color: #e2e8f0; font-family: monospace; font-size: 11px;'
    try {
      let p75: number | null
      try {
        p75 = await this._measureInferenceP75(seg)
      } catch {
        debugWarn(
          '%c┌────────────────────────────────────────────────────────────┐\n' +
            '│ [AMP BENCHMARK] LANDSCAPE WARM-UP FAILED → skip=2         │\n' +
            '└────────────────────────────────────────────────────────────┘',
          B
        )
        return 'skip2'
      }
      if (p75 === null || this._destroyed) return 'skip2'

      // Two-tier: no further model fallback possible, skip=2 is the floor.
      //   < 25ms  → skip=1 (30fps, ~8ms GPU margin/frame)
      //   ≥ 25ms  → skip=2 (15fps, fits in 66.7ms window)
      const result: 'skip1' | 'skip2' = p75 < 25 ? 'skip1' : 'skip2'
      const resultVal =
        result === 'skip1'
          ? 'PASS — 30fps (skip=1)'
          : 'FALLBACK — 15fps (skip=2)'
      const resultColor =
        result === 'skip1'
          ? 'color: #10b981; font-weight: bold; font-family: monospace; font-size: 11px;'
          : 'color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px;'

      const pad = (s: string, n: number) =>
        s + ' '.repeat(Math.max(0, n - s.length))
      const W60 = 60
      const p75Str = `${p75.toFixed(2)} ms`
      const resStr = resultVal
      debugLog(
        `%c┌────────────────────────────────────────────────────────────┐\n` +
          `%c│%c${pad('  [AMP BENCHMARK] LANDSCAPE SKIP SELECTION', W60)}%c│\n` +
          `%c├────────────────────────────────────────────────────────────┤\n` +
          `%c│%c  Protocol: 5 warm-up + 15 timed runs, fresh frames${' '.repeat(W60 - 50)}%c│\n` +
          `%c│%c  P75 Inference Latency:     %c${p75Str}${' '.repeat(Math.max(0, W60 - 28 - p75Str.length))}%c│\n` +
          `%c├────────────────────────────────────────────────────────────┤\n` +
          `%c│%c${pad('  Tier 1 (skip=1, 30fps):   < 25.00 ms', W60)}%c│\n` +
          `%c│%c${pad('  Tier 2 (skip=2, 15fps):   ≥ 25.00 ms', W60)}%c│\n` +
          `%c├────────────────────────────────────────────────────────────┤\n` +
          `%c│%c  Evaluation Result:         %c${resStr}${' '.repeat(Math.max(0, W60 - 28 - resStr.length))}%c│\n` +
          `%c└────────────────────────────────────────────────────────────┘`,
        B,
        B,
        'color: #34d399; font-weight: bold; font-family: monospace; font-size: 11px;',
        B,
        B,
        B,
        W,
        B,
        B,
        W,
        'color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px;',
        B,
        B,
        B,
        W,
        B,
        B,
        W,
        B,
        B,
        B,
        W,
        resultColor,
        B,
        B
      )

      return result
    } catch {
      debugWarn(
        '%c┌────────────────────────────────────────────────────────────┐\n' +
          '│ [AMP BENCHMARK] LANDSCAPE ERROR → skip=2 (safe default)   │\n' +
          '└────────────────────────────────────────────────────────────┘',
        B
      )
      return 'skip2'
    }
  }

  private _getRvmRatio(opts: ProcessorConfig): number | undefined {
    if (
      opts.type === ProcessorType.BLUR ||
      opts.type === ProcessorType.VIRTUAL
    ) {
      return opts.rvmDownsampleRatio
    }
    return undefined
  }

  private _autoRvmRatio(): number {
    const w = this.sourceSettings?.width ?? 1280
    if (w > 1920) return 0.125
    if (w >= 720) return 0.25
    return 0.5
  }

  private _getPostProcessingConfig(): PostProcessingConfig {
    if (
      this.options.type === ProcessorType.BLUR ||
      this.options.type === ProcessorType.VIRTUAL
    ) {
      return this.options.postProcessing ?? {}
    }
    return {}
  }

  private _getUpsamplingConfig(): UpsamplingConfig {
    if (
      this.options.type === ProcessorType.BLUR ||
      this.options.type === ProcessorType.VIRTUAL
    ) {
      return this.options.upsampling ?? {}
    }
    return {}
  }

  private _getPreProcessingConfig(): PreProcessingConfig | undefined {
    if (
      this.options.type === ProcessorType.BLUR ||
      this.options.type === ProcessorType.VIRTUAL
    ) {
      return this.options.preProcessing
    }
    return undefined
  }

  /**
   * Try WebGL2; if unavailable, fall back silently to Canvas2D so matting
   * stays functional (degraded quality) on machines without GPU support.
   */
  private async _initRendererWithFallback(
    opts: GpuRendererInitOpts
  ): Promise<GpuRenderer> {
    const webgl2 = new WebGl2Renderer()
    try {
      await webgl2.init(this.outputCanvas!, opts)
      return webgl2
    } catch (e) {
      pushMattingError({
        code: 'CANVAS2D_FALLBACK',
        level: 'info',
        detail: `WebGL2 unavailable, using Canvas2D fallback: ${e instanceof Error ? e.message : String(e)}`,
      })
      const c2d = new Canvas2dRenderer()
      await c2d.init(this.outputCanvas!, opts)
      dismissMattingError('WEBGL2_INIT_FAILED')
      return c2d
    }
  }

  private _applyRendererConfig() {
    if (!this.gpuRenderer) return
    const mode =
      this.options.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this.gpuRenderer.setMode(mode)
    if (this.options.type === ProcessorType.BLUR) {
      this.gpuRenderer.setBlurRadius(this.options.blurRadius ?? DEFAULT_BLUR)
    }
    this.gpuRenderer.setPostProcessing(this._getPostProcessingConfig())
    this.gpuRenderer.setUpsampling(this._getUpsamplingConfig())
    this.gpuRenderer.setVirtualBackground(
      this.options.type === ProcessorType.VIRTUAL
        ? (this.virtualBackgroundImage ?? null)
        : null
    )

    const preCfg = this._getPreProcessingConfig()
    this._preProcessingPipeline = preCfg?.roiCropping?.enabled
      ? new PreProcessingPipeline(preCfg)
      : undefined
  }

  private async _initSegmenterBackground(model: SegmentationModel) {
    if (this._destroyed) return
    this._pendingModel = model
    try {
      let targetModel = model
      if (model === SegmentationModel.AUTO) {
        targetModel = SegmentationModel.MULTICLASS
      }

      let seg = createSegmenter(targetModel, {
        rvmDownsampleRatio:
          this._getRvmRatio(this.options) ?? this._autoRvmRatio(),
      })
      await seg.init()

      if (this._destroyed || this._pendingModel !== model) {
        seg.destroy()
        return
      }

      // Benchmark Multiclass whenever it is the candidate (AUTO or explicit).
      // AUTO: 'landscape' result triggers a real fallback; explicit MULTICLASS:
      // 'landscape' result is ignored but skip defaults to 2 (conservative).
      if (targetModel === SegmentationModel.MULTICLASS) {
        const benchResult = await this._benchmarkSegmenter(seg)
        if (this._destroyed || this._pendingModel !== model) {
          seg.destroy()
          return
        }
        if (benchResult === 'landscape' && model === SegmentationModel.AUTO) {
          seg.destroy()
          targetModel = SegmentationModel.LANDSCAPE
          seg = createSegmenter(targetModel)
          await seg.init()
          if (this._destroyed || this._pendingModel !== model) {
            seg.destroy()
            return
          }
          // Skip for this Landscape fallback is determined below.
        } else {
          this._segmenterFrameSkip = benchResult === 'multiclass_skip1' ? 1 : 2
        }
      }

      // Benchmark Landscape skip when it is the active model — whether selected
      // explicitly or as a fallback from the Multiclass benchmark above.
      if (targetModel === SegmentationModel.LANDSCAPE) {
        const skipResult = await this._benchmarkLandscapeSkip(seg)
        if (this._destroyed || this._pendingModel !== model) {
          seg.destroy()
          return
        }
        this._segmenterFrameSkip = skipResult === 'skip1' ? 1 : 2
      }

      if (this._destroyed) {
        seg.destroy()
        return
      }

      this.segmenter = seg
      this.currentModel = targetModel
      setMattingStatsModel(model, targetModel)
      setSegmenterFrameSkip(this._segmenterFrameSkip)
      this.processingWidth = seg.inputSize.width
      this.processingHeight = seg.inputSize.height
      this._resizeMaskIfNeeded()
      this._resolveReady()
    } catch (e) {
      if (!this._destroyed && this._pendingModel === model) {
        console.error(
          '[AMP] segmenter init failed — running in passthrough mode',
          e
        )
        this.segmenter = undefined
        setMattingStatsModel(model, null)
        this._resolveReady()
      }
    }
  }

  private async _switchSegmenterBackground(model: SegmentationModel) {
    if (this._destroyed) return
    this._pendingModel = model
    try {
      let targetModel = model
      if (model === SegmentationModel.AUTO) {
        targetModel = SegmentationModel.MULTICLASS
      }

      let seg = createSegmenter(targetModel, {
        rvmDownsampleRatio:
          this._getRvmRatio(this.options) ?? this._autoRvmRatio(),
      })
      await seg.init()

      if (this._destroyed || this._pendingModel !== model) {
        seg.destroy()
        return
      }

      if (targetModel === SegmentationModel.MULTICLASS) {
        const benchResult = await this._benchmarkSegmenter(seg)
        if (this._destroyed || this._pendingModel !== model) {
          seg.destroy()
          return
        }
        if (benchResult === 'landscape' && model === SegmentationModel.AUTO) {
          seg.destroy()
          targetModel = SegmentationModel.LANDSCAPE
          seg = createSegmenter(targetModel)
          await seg.init()
          if (this._destroyed || this._pendingModel !== model) {
            seg.destroy()
            return
          }
        } else {
          this._segmenterFrameSkip = benchResult === 'multiclass_skip1' ? 1 : 2
        }
      }

      if (targetModel === SegmentationModel.LANDSCAPE) {
        const skipResult = await this._benchmarkLandscapeSkip(seg)
        if (this._destroyed || this._pendingModel !== model) {
          seg.destroy()
          return
        }
        this._segmenterFrameSkip = skipResult === 'skip1' ? 1 : 2
      }

      if (this._destroyed) {
        seg.destroy()
        return
      }

      const old = this.segmenter
      this.segmenter = seg
      this.currentModel = targetModel
      setMattingStatsModel(model, targetModel)
      setSegmenterFrameSkip(this._segmenterFrameSkip)
      this.processingWidth = seg.inputSize.width
      this.processingHeight = seg.inputSize.height
      old?.destroy()
      this._resizeMaskIfNeeded()
      this._resolveReady()
    } catch (e) {
      if (!this._destroyed && this._pendingModel === model) {
        console.error('[AMP] segmenter switch failed', e)
        this._resolveReady()
      }
    }
  }

  private _resizeMaskIfNeeded() {
    this.segmentationMaskCanvas?.setAttribute(
      'width',
      '' + this.processingWidth
    )
    this.segmentationMaskCanvas?.setAttribute(
      'height',
      '' + this.processingHeight
    )
    this.gpuRenderer?.resizeProcessing(
      this.processingWidth,
      this.processingHeight
    )
    this._passthroughMask = undefined
    // Invalidate stale pair from old dimensions — render loop falls back to
    // passthrough until the segmenter produces a mask at the new size.
    if (this._latestPair) {
      try {
        this._latestPair.source.close()
      } catch {
        /* ImageBitmap.close() — best-effort */
      }
      this._latestPair = null
    }
  }

  private _initVirtualBackgroundImage() {
    if (this.options.type !== ProcessorType.VIRTUAL) {
      this.virtualBackgroundImage = undefined
      return
    }
    const path = this.options.imagePath
    const currentPath = this.virtualBackgroundImage?.dataset.srcPath
    if (currentPath === path) return

    const img = document.createElement('img')
    img.crossOrigin = 'anonymous'
    img.dataset.srcPath = path
    img.onerror = () => {
      pushMattingError({
        code: 'VIRTUAL_BG_LOAD_FAILED',
        level: 'warn',
        detail: `Failed to load background image: ${path}`,
      })
    }
    img.src = path
    this.virtualBackgroundImage = img
  }

  // ─── Two-loop engine ────────────────────────────────────────────────────────

  private _startLoops(): void {
    if (this._destroyed) return
    if (this.videoElementLoaded || this.videoElement!.readyState >= 2) {
      this._launch()
    } else {
      this.videoElement!.onloadeddata = () => {
        if (this._destroyed) return
        this._launch()
      }
    }
  }

  private _launch(): void {
    if (this._destroyed) return
    this.videoElementLoaded = true
    this._segLoopActive = true
    this._startVideoFrameMetaTracking()
    setMattingStatsActive(true)
    this._runSegmenterLoop() // fire-and-forget
    this._scheduleRender()
  }

  /**
   * Subscribe to requestVideoFrameCallback on the source <video> to keep a
   * fresh frame metadata snapshot (capture/presentation/media times). Used
   * by the stats pipeline to compute end-to-end capture→display latency.
   * Silently no-op when the browser doesn't expose the API.
   */
  private _startVideoFrameMetaTracking(): void {
    const video = this.videoElement
    if (!video) return
    const anyVideo = video as unknown as {
      requestVideoFrameCallback?: (
        cb: (
          now: number,
          meta: {
            captureTime?: number
            presentationTime: number
            mediaTime: number
            expectedDisplayTime?: number
          }
        ) => void
      ) => number
      cancelVideoFrameCallback?: (handle: number) => void
    }
    if (typeof anyVideo.requestVideoFrameCallback !== 'function') return
    const tick = (
      now: number,
      meta: {
        captureTime?: number
        presentationTime: number
        mediaTime: number
      }
    ) => {
      if (this._destroyed || !this._segLoopActive) return
      this._latestVideoFrameMeta = {
        captureTime:
          typeof meta.captureTime === 'number' ? meta.captureTime : now,
        presentationTime: meta.presentationTime,
        mediaTime: meta.mediaTime,
        receivedAt: performance.now(),
      }
      this._videoFrameSeq++
      tickCameraFrame()
      if (this._frameAwaiter) {
        const a = this._frameAwaiter
        this._frameAwaiter = null
        a.resolve()
      }
      this._rvfcHandle = anyVideo.requestVideoFrameCallback!(tick)
    }
    this._rvfcHandle = anyVideo.requestVideoFrameCallback(tick)
  }

  /**
   * Promise resolved by the next rVFC tick. The segmenter loop awaits this to
   * align with the camera's native cadence instead of polling on a timer.
   * Multiple ticks between two awaits collapse into a single wake (the loop
   * always reads `_videoFrameSeq` after wake to know the freshest frame index).
   */
  private _waitNextVideoFrame(): Promise<void> {
    if (!this._frameAwaiter) {
      let resolve!: () => void
      const promise = new Promise<void>((r) => {
        resolve = r
      })
      this._frameAwaiter = { promise, resolve }
    }
    return this._frameAwaiter.promise
  }

  private _publishCameraSettings(): void {
    const s = this.sourceSettings
    const track = this.source as MediaStreamTrack & {
      getCapabilities?: () => MediaTrackCapabilities
    }
    const cap = track.getCapabilities?.()
    setCameraSettings({
      frameRateRequested: typeof s?.frameRate === 'number' ? s.frameRate : null,
      frameRateActual: typeof s?.frameRate === 'number' ? s.frameRate : null,
      frameRateMax:
        typeof cap?.frameRate?.max === 'number' ? cap.frameRate.max : null,
      width: typeof s?.width === 'number' ? s.width : null,
      height: typeof s?.height === 'number' ? s.height : null,
    })
  }

  private _stopVideoFrameMetaTracking(): void {
    const video = this.videoElement
    if (!video || this._rvfcHandle === null) return
    const anyVideo = video as unknown as {
      cancelVideoFrameCallback?: (handle: number) => void
    }
    try {
      anyVideo.cancelVideoFrameCallback?.(this._rvfcHandle)
    } catch {
      /* best-effort */
    }
    this._rvfcHandle = null
    this._latestVideoFrameMeta = undefined
    // Wake the segmenter loop if it's waiting on a frame — it will see
    // `_segLoopActive === false` and exit cleanly.
    if (this._frameAwaiter) {
      const a = this._frameAwaiter
      this._frameAwaiter = null
      a.resolve()
    }
  }

  /**
   * Segmenter loop: driven by `requestVideoFrameCallback` on the source video
   * when available — the loop wakes on each real camera tick instead of a
   * fixed timer, then runs inference only every N camera frames (frame skip).
   * This frees the GPU between inferences so the camera capture path can
   * deliver its full native framerate (e.g. 30fps) instead of being throttled
   * by back-to-back inference work.
   *
   * Fallback (no rVFC, e.g. Firefox <132): paces on `setTimeout(16.67ms)`
   * exactly like before.
   *
   * Each iteration captures the current <video> frame as an ImageBitmap
   * (GPU-backed snapshot), runs inference, and publishes the (mask, frame)
   * pair atomically. The previous bitmap is closed to keep memory bounded to
   * a single in-flight pair.
   */
  private async _runSegmenterLoop(): Promise<void> {
    const FALLBACK_MS = 1000 / 60
    while (this._segLoopActive && !this._destroyed) {
      const hasRvfc = this._rvfcHandle !== null
      // Gate: wake on real camera frame (rVFC) or timer (fallback).
      if (hasRvfc) {
        await this._waitNextVideoFrame()
        if (!this._segLoopActive || this._destroyed) return
        // Frame skip: only run inference once every _segmenterFrameSkip camera frames.
        const seq = this._videoFrameSeq
        if (seq - this._lastInferenceSeq < this._segmenterFrameSkip) continue
        this._lastInferenceSeq = seq
      }
      const t0 = performance.now()
      const seg = this.segmenter
      if (!seg || !this.videoElement || this.videoElement.videoWidth === 0) {
        await new Promise<void>((r) => setTimeout(r, FALLBACK_MS))
        continue
      }
      let capturedSource: ImageBitmap | null = null
      try {
        // Atomic snapshot: a single sync drawImage(videoElement) defines the
        // frame instant. Both the segmenter input (downsampled from this
        // snapshot) and the renderer bitmap (createImageBitmap of this
        // snapshot) are derived from the SAME static canvas, so the bitmap
        // cannot drift to a newer frame while createImageBitmap awaits.
        const snapshot = this._captureSnapshot()
        if (!snapshot) {
          await new Promise<void>((r) => setTimeout(r, FALLBACK_MS))
          continue
        }
        // Record the camera shutter time for this snapshot, derived from the
        // last rVFC tick. Fallback: the snapshot wall-clock t0.
        const cameraCaptureTime = this._latestVideoFrameMeta?.captureTime ?? t0
        const motionRgba = this._preProcessingPipeline
          ? (this._getMotionFrameRgba() ?? undefined)
          : undefined
        const cropBbox =
          this._preProcessingPipeline?.getNextCropBbox(
            motionRgba,
            AdvancedMattingProcessor.MOTION_W,
            AdvancedMattingProcessor.MOTION_H
          ) ?? null
        this.sizeSource(snapshot, cropBbox)
        // Pre-flip the bitmap on Y; the renderer disables UNPACK_FLIP_Y_WEBGL
        // for ImageBitmap uploads. The flip is moved into the bitmap because
        // UNPACK_FLIP_Y_WEBGL is unreliable for ImageBitmap across browsers.
        capturedSource = await createImageBitmap(snapshot, {
          imageOrientation: 'flipY',
        })
        if (!this._segLoopActive) {
          capturedSource.close()
          return
        }
        const frameToSegment = this._preProcessingPipeline
          ? this._preProcessingPipeline.apply(
              this.sourceImageData!,
              this._lastMask
            )
          : this.sourceImageData!
        const inferStart = performance.now()
        const rawMask = await seg.segment(frameToSegment, inferStart)
        pushInferenceSample(performance.now() - inferStart)
        tickSegmenterFrame()
        if (!this._segLoopActive) {
          capturedSource.close()
          return
        }
        if (this.segmenter === seg) {
          const mask = this._preProcessingPipeline
            ? this._preProcessingPipeline.applyAfterInference(
                rawMask,
                this.processingWidth,
                this.processingHeight,
                cropBbox
              )
            : rawMask
          this._lastMask = mask
          const previous = this._latestPair
          this._latestPair = {
            mask,
            source: capturedSource,
            captureTime: t0,
            cameraCaptureTime,
            procW: this.processingWidth,
            procH: this.processingHeight,
          }
          capturedSource = null // ownership transferred to _latestPair
          previous?.source.close()
          // Feed the motion tracker with the latest stabilised bbox so the
          // render loop can resolve the auto mode and compute the prediction
          // offset. When ROI cropping is off, `getCurrentBbox()` returns null
          // and the tracker stays invalid (auto + prediction stay disabled).
          if (this._latencyAuto || this._maskPrediction) {
            this._motionTracker.update(
              this._preProcessingPipeline?.getCurrentBbox() ?? null,
              cameraCaptureTime
            )
          }
        } else {
          capturedSource.close()
          capturedSource = null
        }
      } catch (e) {
        if (capturedSource) {
          try {
            capturedSource.close()
          } catch {
            /* ImageBitmap.close() — best-effort */
          }
          capturedSource = null
        }
        if (!this._segLoopActive) return
        console.error('[AMP] segmenter loop error', e)
        pushMattingError({
          code: 'SEGMENTER_TIMEOUT_PASSTHROUGH',
          level: 'warn',
          detail: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
        })
        await new Promise<void>((r) => setTimeout(r, 100))
        continue
      }
      // Fallback pacing only — when rVFC is active the next iteration awaits
      // the rVFC tick at the top and we must not double-sleep here.
      if (this._rvfcHandle === null) {
        const elapsed = performance.now() - t0
        await new Promise<void>((r) =>
          setTimeout(r, Math.max(0, FALLBACK_MS - elapsed))
        )
      }
    }
  }

  // Last _videoFrameSeq value that triggered a render. Compared to
  // _videoFrameSeq each rAF tick: render fires only when a new camera frame
  // has actually arrived, bounding render FPS to camera FPS.
  private _lastRenderedSeq = -1

  private _scheduleRender(): void {
    if (!this._segLoopActive) return
    this._renderLoopHandle = requestAnimationFrame(() => {
      if (this._rvfcHandle !== null) {
        // rVFC active: _videoFrameSeq increments on every real camera frame.
        // Render exactly once per new frame — no duplicate composites.
        const seq = this._videoFrameSeq
        if (seq > this._lastRenderedSeq) {
          this._lastRenderedSeq = seq
          this._renderFrame()
        }
      } else {
        // No rVFC (older browsers): detect new frames via currentTime change.
        // Also tick the camera FPS counter since the rVFC callback isn't firing.
        if (this.videoElement) {
          const t = this.videoElement.currentTime
          if (t !== this._lastVideoTime) {
            this._lastVideoTime = t
            tickCameraFrame()
            this._renderFrame()
          }
        }
      }
      this._scheduleRender()
    })
  }

  private _cancelRender(): void {
    if (this._renderLoopHandle === null) return
    cancelAnimationFrame(this._renderLoopHandle)
    this._renderLoopHandle = null
  }

  private _renderFrame(): void {
    if (
      !this.gpuRenderer ||
      !this.videoElement ||
      this.videoElement.videoWidth === 0
    )
      return
    const pair = this._latestPair
    if (!pair) {
      // First mask not ready yet — passthrough with a uniform-1 mask; no halo
      // to worry about since the mask is constant.
      const vw = this.videoElement.videoWidth
      const vh = this.videoElement.videoHeight
      if (vw !== this.gpuRenderer.outW || vh !== this.gpuRenderer.outH) {
        this.gpuRenderer.resizeOutput(vw, vh)
      }
      this.gpuRenderer.setMaskOffset(0, 0)
      this.gpuRenderer.setBlendMix(0)
      setEffectiveLatencyMode(null)
      setMotionScore(0)
      setMaskOffset(0, 0)
      setPredictionActive(false)
      this._drawPassthrough()
      return
    }
    this.gpuRenderer.uploadMask(pair.mask, pair.procW, pair.procH)

    const motionScore = this._motionTracker.isValid()
      ? this._motionTracker.getMotionScore()
      : 0
    setMotionScore(motionScore)

    // Resolve the effective blend mode + prediction gain. Auto path requires
    // a valid motion signal (i.e. ROI cropping is on and at least one bbox
    // has been observed). Manual path uses the static mapping table.
    let effectiveMode: MaskBlendMode
    let predictionGain: number
    let blendT = 0
    if (this._latencyAuto && this._motionTracker.isValid()) {
      effectiveMode = this._resolveAutoMode(motionScore)
      // Prediction is an explicit opt-in to the halo trade-off. In that case
      // we never stay fully frame-locked — the whole point is to see a latency
      // reduction. We upgrade frameLock to blend so the cross-fade kicks in.
      if (this._maskPrediction && effectiveMode === 'frameLock') {
        effectiveMode = 'blend'
      }
      predictionGain =
        effectiveMode === 'live' ? 1.0 : effectiveMode === 'blend' ? 0.3 : 0
      if (effectiveMode === 'blend') {
        const span = AUTO_LIVE_THRESHOLD - AUTO_LOCK_THRESHOLD
        const motionBlend = clamp01((motionScore - AUTO_LOCK_THRESHOLD) / span)
        // With prediction on, mix a baseline blend so even zero motion gives
        // a visible latency drop. Without prediction, fall back to a pure
        // motion-driven blend (no halo risk when subject is still).
        const baseline = this._maskPrediction ? AUTO_PRED_BLEND_BASELINE : 0
        blendT = clamp01(baseline + (1 - baseline) * motionBlend)
      }
    } else {
      const entry = STATIC_MODE_TABLE[this._latencyMode]
      effectiveMode = entry.mode
      predictionGain = entry.predictionGain
      if (effectiveMode === 'blend') {
        const ageMs = performance.now() - pair.captureTime
        blendT = clamp01(ageMs / BLEND_MODE_MAX_AGE_MS)
      }
    }
    this._lastEffectiveMode = effectiveMode
    setEffectiveLatencyMode(effectiveMode)

    // Compute prediction offset (uv coords). Only applied when the user has
    // enabled it AND we are reading from a live frame (frame-locked composite
    // doesn't benefit — the mask already matches the displayed pixels).
    let offsetU = 0
    let offsetV = 0
    const predictionWillRun =
      this._maskPrediction &&
      predictionGain > 0 &&
      this._motionTracker.isValid() &&
      effectiveMode !== 'frameLock'
    if (predictionWillRun) {
      const v = this._motionTracker.getVelocityUv()
      const predictionDt_s = (performance.now() - pair.cameraCaptureTime) / 1000
      offsetU = clampOffset(v.vx * predictionDt_s * predictionGain)
      offsetV = clampOffset(v.vy * predictionDt_s * predictionGain)
    }
    this.gpuRenderer.setMaskOffset(offsetU, offsetV)
    setMaskOffset(offsetU, offsetV)
    setPredictionActive(predictionWillRun)

    this.gpuRenderer.setBlendMix(effectiveMode === 'blend' ? blendT : 0)

    // Pick the primary source. Frame-locked uses the pair's ImageBitmap. Live
    // and blend both use the live <video> as the primary; blend additionally
    // uploads the pair's bitmap as the second (frame-locked) source via the
    // optional liveSource arg, so the shader can cross-fade.
    if (effectiveMode === 'frameLock') {
      const sw = pair.source.width
      const sh = pair.source.height
      if (sw !== this.gpuRenderer.outW || sh !== this.gpuRenderer.outH) {
        this.gpuRenderer.resizeOutput(sw, sh)
      }
      this.gpuRenderer.render(pair.source)
    } else if (effectiveMode === 'live') {
      const vw = this.videoElement.videoWidth
      const vh = this.videoElement.videoHeight
      if (vw !== this.gpuRenderer.outW || vh !== this.gpuRenderer.outH) {
        this.gpuRenderer.resizeOutput(vw, vh)
      }
      this.gpuRenderer.render(this.videoElement)
    } else {
      // blend: render(frame-locked bitmap, live video) — the shader mixes them
      // with blendMix as the cross-fade weight.
      const sw = pair.source.width
      const sh = pair.source.height
      if (sw !== this.gpuRenderer.outW || sh !== this.gpuRenderer.outH) {
        this.gpuRenderer.resizeOutput(sw, sh)
      }
      this.gpuRenderer.render(pair.source, this.videoElement)
    }

    // Stats: capture→display latency reflects the camera shutter time of the
    // pixels actually shown. For blend, we report the interpolated time
    // proportional to blendT.
    const now = performance.now()
    const liveCameraTime = this._latestVideoFrameMeta?.captureTime ?? now
    let appliedCameraTime: number
    if (effectiveMode === 'live') {
      appliedCameraTime = liveCameraTime
    } else if (effectiveMode === 'frameLock') {
      appliedCameraTime = pair.cameraCaptureTime
    } else {
      appliedCameraTime =
        pair.cameraCaptureTime +
        (liveCameraTime - pair.cameraCaptureTime) * blendT
    }
    pushLatencySample(now - appliedCameraTime)
    pushGapSample(Math.max(0, appliedCameraTime - pair.cameraCaptureTime))
    tickRenderFrame()
  }

  /**
   * Apply the auto-tuning thresholds with hysteresis around `_lastEffectiveMode`
   * so the resolved mode doesn't flap when the motion score sits right on a
   * boundary. The hysteresis band is asymmetric — leaving a mode requires
   * crossing a slightly stricter threshold than entering it.
   */
  private _resolveAutoMode(motionScore: number): MaskBlendMode {
    const lock = AUTO_LOCK_THRESHOLD
    const live = AUTO_LIVE_THRESHOLD
    const h = AUTO_HYSTERESIS
    const prev = this._lastEffectiveMode
    if (motionScore < lock - h) return 'frameLock'
    if (motionScore > live + h) return 'live'
    if (motionScore < lock + h && prev === 'frameLock') return 'frameLock'
    if (motionScore > live - h && prev === 'live') return 'live'
    return 'blend'
  }

  // ────────────────────────────────────────────────────────────────────────────

  /**
   * Downsample a source image (canvas or video) into the proc-res segmentation
   * canvas and read it back as ImageData. The source's natural dimensions are
   * read from `width`/`height` (canvas) or `videoWidth`/`videoHeight` (video).
   */
  private sizeSource(
    source: HTMLCanvasElement | HTMLVideoElement,
    cropBbox?: BBox | null
  ) {
    const vw =
      (source as HTMLVideoElement).videoWidth ??
      (source as HTMLCanvasElement).width
    const vh =
      (source as HTMLVideoElement).videoHeight ??
      (source as HTMLCanvasElement).height
    const sx = cropBbox ? Math.round(cropBbox.x * vw) : 0
    const sy = cropBbox ? Math.round(cropBbox.y * vh) : 0
    const sw = cropBbox ? Math.round(cropBbox.width * vw) : vw
    const sh = cropBbox ? Math.round(cropBbox.height * vh) : vh
    this.segmentationMaskCanvasCtx!.drawImage(
      source,
      sx,
      sy,
      sw,
      sh,
      0,
      0,
      this.processingWidth,
      this.processingHeight
    )
    this.sourceImageData = this.segmentationMaskCanvasCtx!.getImageData(
      0,
      0,
      this.processingWidth,
      this.processingHeight
    )
  }

  /**
   * Ensures the snapshot canvas exists and matches the current video size,
   * then draws the current video frame into it. Synchronous → defines the
   * atomic instant from which the segmenter input and the renderer bitmap
   * are both derived.
   */
  private _captureSnapshot(): HTMLCanvasElement | null {
    if (!this.videoElement || this.videoElement.videoWidth === 0) return null
    const vw = this.videoElement.videoWidth
    const vh = this.videoElement.videoHeight
    if (
      !this._snapshotCanvas ||
      this._snapshotCanvas.width !== vw ||
      this._snapshotCanvas.height !== vh
    ) {
      const canvas = this._snapshotCanvas ?? document.createElement('canvas')
      canvas.width = vw
      canvas.height = vh
      this._snapshotCanvas = canvas
      this._snapshotCanvasCtx = canvas.getContext('2d', {
        willReadFrequently: false,
      }) as CanvasRenderingContext2D
    }
    this._snapshotCanvasCtx!.drawImage(this.videoElement, 0, 0, vw, vh)
    return this._snapshotCanvas
  }

  private _getMotionFrameRgba(): Uint8ClampedArray | null {
    if (!this._snapshotCanvas) return null
    const mw = AdvancedMattingProcessor.MOTION_W
    const mh = AdvancedMattingProcessor.MOTION_H
    if (!this._motionCanvas) {
      const canvas = document.createElement('canvas')
      canvas.width = mw
      canvas.height = mh
      this._motionCanvas = canvas
      this._motionCanvasCtx = canvas.getContext('2d', {
        willReadFrequently: true,
      }) as CanvasRenderingContext2D
    }
    this._motionCanvasCtx!.drawImage(this._snapshotCanvas, 0, 0, mw, mh)
    return this._motionCanvasCtx!.getImageData(0, 0, mw, mh).data
  }

  private _drawPassthrough() {
    if (!this.gpuRenderer || !this.videoElement) return
    const w = this.processingWidth
    const h = this.processingHeight
    if (!this._passthroughMask || this._passthroughMask.length !== w * h) {
      this._passthroughMask = new Float32Array(w * h).fill(1)
    }
    this.gpuRenderer.uploadMask(this._passthroughMask, w, h)
    this.gpuRenderer.render(this.videoElement)
  }

  private _createMainCanvasWithSize(w: number, h: number) {
    let canvas = document.querySelector(
      `canvas#${BLUR_CANVAS_ID}`
    ) as HTMLCanvasElement | null
    if (!canvas) {
      canvas = this._createCanvas(BLUR_CANVAS_ID, w, h)
    } else {
      canvas.setAttribute('width', '' + w)
      canvas.setAttribute('height', '' + h)
    }
    this.outputCanvas = canvas
  }

  private _createMaskCanvas() {
    let canvas = document.querySelector(
      `#${SEGMENTATION_MASK_CANVAS_ID}`
    ) as HTMLCanvasElement | null
    if (!canvas) {
      canvas = this._createCanvas(
        SEGMENTATION_MASK_CANVAS_ID,
        this.processingWidth,
        this.processingHeight
      )
    } else {
      canvas.setAttribute('width', '' + this.processingWidth)
      canvas.setAttribute('height', '' + this.processingHeight)
    }
    this.segmentationMaskCanvas = canvas
    this.segmentationMaskCanvasCtx = canvas.getContext('2d', {
      willReadFrequently: true,
    })!
  }

  private _createCanvas(id: string, width: number, height: number) {
    const el = document.createElement('canvas')
    el.setAttribute('id', id)
    el.setAttribute('width', '' + width)
    el.setAttribute('height', '' + height)
    return el
  }

  async restart(opts: ProcessorOptions<Track.Kind>) {
    await this.destroy()
    return this.init(opts)
  }

  async destroy() {
    this._destroyed = true
    this._pendingModel = undefined
    this._configuredModel = undefined
    this._segLoopActive = false
    this._lastVideoTime = -1
    this._videoFrameSeq = 0
    this._lastInferenceSeq = -1
    this._lastRenderedSeq = -1
    this._motionTracker.reset()
    this._lastEffectiveMode = 'frameLock'
    this._cancelRender()
    this._stopVideoFrameMetaTracking()
    resetMattingStats()
    if (this.videoElement) {
      this.videoElement.onloadeddata = null
    }
    this.segmenter?.destroy()
    this.segmenter = undefined
    this.gpuRenderer?.destroy()
    this.gpuRenderer = undefined
    this._preProcessingPipeline = undefined
    this._lastMask = undefined
    this._motionCanvas = undefined
    this._motionCanvasCtx = undefined
    if (this._latestPair) {
      try {
        this._latestPair.source.close()
      } catch {
        /* ImageBitmap.close() — best-effort */
      }
      this._latestPair = null
    }
    this._resolveReady()

    if (this.processedTrack && this.processedTrack !== this.source) {
      try {
        this.processedTrack.stop()
      } catch (e) {
        debugWarn(
          '[AMP] Failed to stop canvas capture track during destroy:',
          e
        )
      }
    }
    this.processedTrack = undefined
  }
}
