import { ProcessorOptions, Track } from 'livekit-client'
import {
  BackgroundProcessorInterface,
  ProcessorConfig,
  ProcessorType,
  SegmentationModel,
  PostProcessingConfig,
  UpsamplingConfig,
  PreProcessingConfig,
} from '.'
import { PreProcessingPipeline } from './preprocessing/PreProcessingPipeline'
import { BBox } from './preprocessing/RoiCropper'
import { Segmenter, createSegmenter, probeMediapipeDelegate } from './segmenters'
import { WebGl2Renderer } from './renderers/WebGl2Renderer'
import { pushMattingError } from './errors/MattingErrorStore'

const SEGMENTATION_MASK_CANVAS_ID = 'background-blur-local-segmentation'
const BLUR_CANVAS_ID = 'background-blur-local'
const DEFAULT_BLUR = 10

/**
 * Pair of mask + the exact source frame that produced it. Stored together to
 * allow frame-locked compositing (mask applied to its own source frame, no
 * spatial mismatch).
 */
interface FrameMaskPair {
  mask: Float32Array
  source: ImageBitmap
  captureTime: number
  procW: number
  procH: number
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
  /** Configured processor options (blur radius, virtual image, models). */
  options: ProcessorConfig
  /** Unique name of the processor ('blur' or 'virtual'). */
  name: string
  /** Discriminated processor type. */
  type: ProcessorType
  /** Output MediaStreamTrack containing the WebGL2 composition stream. */
  processedTrack?: MediaStreamTrack

  /** Original camera input video track. */
  source?: MediaStreamTrack
  /** MediaTrackSettings of the source track (contains active width/height). */
  sourceSettings?: MediaTrackSettings
  /** The source HTML5 <video> element. */
  videoElement?: HTMLVideoElement
  /** True when the video element has loaded and started playing. */
  videoElementLoaded?: boolean

  /** Offscreen or visible canvas displaying the final composted stream. */
  outputCanvas?: HTMLCanvasElement

  /** Canvas used as a scratchpad for drawing resized segmenter frames. */
  segmentationMaskCanvas?: HTMLCanvasElement
  /** 2D rendering context of the segmentation scratchpad. */
  segmentationMaskCanvasCtx?: CanvasRenderingContext2D
  /** Extracted raw RGBA ImageData frame of the segmenter input. */
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

  /** The active segmentation model (e.g. MulticlassSegmenter or LandscapeSegmenter). */
  segmenter?: Segmenter
  private gpuRenderer?: WebGl2Renderer
  private _passthroughMask?: Float32Array

  // Two-loop state
  private _segLoopActive = false
  private _latestPair: FrameMaskPair | null = null
  private _renderLoopHandle: number | null = null
  // Max allowed offset (in frames @ 30fps) between the frame that produced the
  // mask and the frame the mask is applied to. 0 = strict frame-lock (no halo,
  // ~inference-time latency). Higher = lower latency, halo bounded by N frames.
  private _maxFrameOffset = 0

  /** Virtual background image element (preloaded when type is VIRTUAL). */
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

  constructor(opts: ProcessorConfig) {
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this.options = opts
    this.type = opts.type
    this._maxFrameOffset = this._readMaxFrameOffset(opts)
  }

  /**
   * Extract and sanitise the `maxFrameOffset` value from a processor config.
   * Non-finite, negative, or missing values default to 0 (strict frame-lock).
   * The value is capped to 60 frames to avoid unreasonably stale compositing.
   *
   * @param opts Processor configuration to read.
   * @returns    Validated frame-offset value in [0, 60].
   */
  private _readMaxFrameOffset(opts: ProcessorConfig): number {
    if (opts.type === ProcessorType.BLUR || opts.type === ProcessorType.VIRTUAL) {
      const v = opts.maxFrameOffset
      if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) return 0
      return Math.min(Math.floor(v), 60)
    }
    return 0
  }

  /** Resolves once the active segmenter is loaded and producing frames. */
  waitForReady(): Promise<void> {
    if (this.segmenter || this._destroyed) return Promise.resolve()
    return new Promise(resolve => this._readyResolvers.push(resolve))
  }

  /**
   * Resolve all pending `waitForReady()` promises and clear the resolver queue.
   * Called once the first segmenter finishes initialising (or on failure/destroy).
   */
  private _resolveReady() {
    this._readyResolvers.splice(0).forEach(r => r())
  }

  /**
   * Initializes the background matting processor. Sets up original video track,
   * configures offscreen canvases, instantiates WebGL2 graphics renderers,
   * starts segmenter/render loops, and starts pre-loading segmentation models in the background.
   */
  async init(opts: ProcessorOptions<Track.Kind>) {
    this._destroyed = false
    if (!opts.element) {
      throw new Error('Element is required for processing')
    }
    this.source = opts.track as MediaStreamTrack
    this.sourceSettings = this.source!.getSettings()
    this.videoElement = opts.element as HTMLVideoElement
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

      this.gpuRenderer = new WebGl2Renderer()
      await this.gpuRenderer.init(this.outputCanvas!, {
        outW: realW,
        outH: realH,
        processingW: this.processingWidth,
        processingH: this.processingHeight,
        postProcessing: this._getPostProcessingConfig(),
        upsampling: this._getUpsamplingConfig(),
      })
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
          } catch { }
        }
        this.processedTrack = undefined
        return
      }

      this._startLoops()

      this._configuredModel = this._getModel(this.options)
      // Initialize segmenter in background — passthrough renders until it's ready.
      this._initSegmenterBackground(this._configuredModel)
    } catch (e) {
      console.warn(
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

  /**
   * Updates the processor dynamically with a new configuration.
   * Enables seamless transitions between blur and virtual backgrounds, changes in
   * blur intensity, post-processing options, or switching models on the fly.
   */
  async update(opts: ProcessorConfig): Promise<void> {
    this.options = opts
    this.type = opts.type
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this._maxFrameOffset = this._readMaxFrameOffset(opts)

    if (!this.gpuRenderer) {
      console.info('[AMP] Update called in passthrough fallback mode; ignoring background processor updates.')
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
      this.segmenter.setDownsampleRatio(
        nextRvmRatio ?? this._autoRvmRatio()
      )
    }
    this._applyRendererConfig()
  }


  /**
   * Extract the segmentation model requested by a config, defaulting to AUTO.
   *
   * @param opts Processor configuration.
   * @returns    Requested `SegmentationModel`.
   */
  private _getModel(opts: ProcessorConfig): SegmentationModel {
    if (opts.type === ProcessorType.BLUR || opts.type === ProcessorType.VIRTUAL) {
      return opts.model ?? SegmentationModel.AUTO
    }
    return SegmentationModel.AUTO
  }

  /**
   * Benchmark a freshly-initialised segmenter to decide whether it is fast enough
   * for the AUTO model selection path.
   *
   * Runs 4 inference calls on a blank canvas and measures the average duration.
   * The segmenter is considered "fast enough" if the average is ≤ 30 ms/frame.
   * Skipped entirely when the GPU delegate is unavailable (CPU path would be too slow).
   *
   * @param seg An already-initialised `Segmenter` instance to benchmark.
   * @returns   `true` if the model should be kept; `false` to fall back to Landscape.
   */
  private async _benchmarkSegmenter(seg: Segmenter): Promise<boolean> {
    try {
      const probe = await probeMediapipeDelegate()
      if (probe === 'CPU') {
        console.warn(
          '%c┌────────────────────────────────────────────────────────────┐\n' +
          '│ [AMP BENCHMARK] SKIPPED: CPU DELEGATE DETECTED             │\n' +
          '├────────────────────────────────────────────────────────────┤\n' +
          '│  Device WebGL Delegate is CPU.                             │\n' +
          '│  To prevent performance degradation, benchmarking is       │\n' +
          '│  skipped and Landscape fallback is automatically used.     │\n' +
          '└────────────────────────────────────────────────────────────┘',
          'color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px;'
        )
        return false
      }

      const width = seg.inputSize.width
      const height = seg.inputSize.height

      const dummyCanvas = document.createElement('canvas')
      dummyCanvas.width = width
      dummyCanvas.height = height
      const ctx = dummyCanvas.getContext('2d')
      if (!ctx) return false
      const dummyData = ctx.createImageData(width, height)

      try {
        await seg.segment(dummyData, performance.now())
      } catch (warmupErr) {
        console.warn(
          '%c┌────────────────────────────────────────────────────────────┐\n' +
          '│ [AMP BENCHMARK] WARM-UP FAILED                             │\n' +
          '├────────────────────────────────────────────────────────────┤\n' +
          '│  Warm-up run threw an exception.                           │\n' +
          '│  Safe fallback to Landscape mode initiated.                │\n' +
          '└────────────────────────────────────────────────────────────┘',
          'color: #ef4444; font-weight: bold; font-family: monospace; font-size: 11px;'
        )
        return false
      }

      const runs = 4
      let totalTime = 0
      for (let i = 0; i < runs; i++) {
        if (this._destroyed) return false
        const start = performance.now()
        await seg.segment(dummyData, performance.now())
        totalTime += performance.now() - start
      }

      const avg = totalTime / runs
      const success = avg <= 30

      const widthCard = 60
      const padRight = (str: string, len: number) => str + ' '.repeat(Math.max(0, len - str.length))

      const titleLine = padRight(`  [AMP BENCHMARK] MULTICLASS PERFORMANCE`, widthCard)
      const latencyLabel = `  Average Inference Latency: `
      const latencyVal = `${avg.toFixed(2)} ms`
      const latencyPadding = ' '.repeat(Math.max(0, widthCard - latencyLabel.length - latencyVal.length))

      const thresholdLine = padRight(`  Target Threshold:          30.00 ms`, widthCard)
      const delegateLine = padRight(`  Device WebGL Delegate:     GPU`, widthCard)
      const resultLabel = `  Evaluation Result:         `
      const resultVal = success ? 'PASS (Use Multiclass)' : 'FAIL (Fallback to Landscape)'
      const resultPadding = ' '.repeat(Math.max(0, widthCard - resultLabel.length - resultVal.length))

      console.log(
        `%c┌────────────────────────────────────────────────────────────┐\n` +
        `%c│%c${titleLine}%c│\n` +
        `%c├────────────────────────────────────────────────────────────┤\n` +
        `%c│%c${latencyLabel}%c${latencyVal}%c${latencyPadding}│\n` +
        `%c│%c${thresholdLine}%c│\n` +
        `%c│%c${delegateLine}%c│\n` +
        `%c├────────────────────────────────────────────────────────────┤\n` +
        `%c│%c${resultLabel}%c${resultVal}%c${resultPadding}│\n` +
        `%c└────────────────────────────────────────────────────────────┘`,
        // Line 1: top border
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 2: start, title, end
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #60a5fa; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 3: middle border
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 4: start, latency label, latency val, end
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #e2e8f0; font-family: monospace; font-size: 11px;',
        'color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 5: start, threshold, end
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #e2e8f0; font-family: monospace; font-size: 11px;',
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 6: start, delegate, end
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #e2e8f0; font-family: monospace; font-size: 11px;',
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 7: middle border
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 8: start, result label, result val, end
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #e2e8f0; font-family: monospace; font-size: 11px;',
        success ? 'color: #10b981; font-weight: bold; font-family: monospace; font-size: 11px;' : 'color: #ef4444; font-weight: bold; font-family: monospace; font-size: 11px;',
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;',
        // Line 9: bottom border
        'color: #3b82f6; font-weight: bold; font-family: monospace; font-size: 11px;'
      )

      return success
    } catch (e) {
      console.warn(
        '%c┌────────────────────────────────────────────────────────────┐\n' +
        '│ [AMP BENCHMARK] ERROR ENCOUNTERED                          │\n' +
        '├────────────────────────────────────────────────────────────┤\n' +
        '│  Benchmark execution failed.                               │\n' +
        '│  Falling back safely to Landscape mode.                    │\n' +
        '└────────────────────────────────────────────────────────────┘',
        'color: #ef4444; font-weight: bold; font-family: monospace; font-size: 11px;'
      )
      return false
    }
  }

  /**
   * Extract the `rvmDownsampleRatio` from a processor config, or `undefined` if
   * the config type does not support it (e.g. FACE_LANDMARKS).
   *
   * @param opts Processor configuration.
   * @returns    Downsample ratio or `undefined`.
   */
  private _getRvmRatio(opts: ProcessorConfig): number | undefined {
    if (opts.type === ProcessorType.BLUR || opts.type === ProcessorType.VIRTUAL) {
      return opts.rvmDownsampleRatio
    }
    return undefined
  }

  /**
   * Compute an automatic RVM downsample ratio based on the camera resolution.
   * Higher resolution → lower ratio → smaller model input → faster inference.
   *
   * @returns Ratio in {0.125, 0.25, 0.5}.
   */
  private _autoRvmRatio(): number {
    const w = this.sourceSettings?.width ?? 1280
    if (w > 1920) return 0.125
    if (w >= 720) return 0.25
    return 0.5
  }

  /**
   * Extract the post-processing configuration from the current options.
   * Returns `{}` for processor types that do not support post-processing.
   */
  private _getPostProcessingConfig(): PostProcessingConfig {
    if (
      this.options.type === ProcessorType.BLUR ||
      this.options.type === ProcessorType.VIRTUAL
    ) {
      return this.options.postProcessing ?? {}
    }
    return {}
  }

  /**
   * Extract the upsampling configuration from the current options.
   * Returns `{}` for processor types that do not support upsampling.
   */
  private _getUpsamplingConfig(): UpsamplingConfig {
    if (
      this.options.type === ProcessorType.BLUR ||
      this.options.type === ProcessorType.VIRTUAL
    ) {
      return this.options.upsampling ?? {}
    }
    return {}
  }

  /**
   * Extract the pre-processing configuration from the current options.
   * Returns `undefined` for processor types that do not support pre-processing.
   */
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
   * Push the current options (mode, blur radius, post-processing, upsampling,
   * virtual background, pre-processing pipeline) to the GPU renderer.
   * Also rebuilds the `PreProcessingPipeline` when ROI cropping is enabled.
   * No-op if no GPU renderer is active (passthrough mode).
   */
  private _applyRendererConfig() {
    if (!this.gpuRenderer) return
    const mode = this.options.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
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
    this._preProcessingPipeline =
      preCfg?.roiCropping?.enabled ? new PreProcessingPipeline(preCfg) : undefined
  }

  /**
   * Asynchronously initialise the segmenter for the first time.
   * For `AUTO`, starts with Multiclass and runs `_benchmarkSegmenter`; falls
   * back to Landscape if the benchmark fails or if the GPU delegate is CPU.
   * Stale initialisations are cancelled via `_pendingModel` guard.
   * Sets `this.segmenter` and resolves `waitForReady()` on completion.
   *
   * @param model Requested segmentation model (may be AUTO).
   */
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

      if (model === SegmentationModel.AUTO) {
        const isFastEnough = await this._benchmarkSegmenter(seg)
        if (this._destroyed || this._pendingModel !== model) {
          seg.destroy()
          return
        }
        if (!isFastEnough) {
          seg.destroy()
          targetModel = SegmentationModel.LANDSCAPE
          seg = createSegmenter(targetModel)
          await seg.init()
          if (this._destroyed || this._pendingModel !== model) {
            seg.destroy()
            return
          }
        }
      }

      if (this._destroyed) {
        seg.destroy()
        return
      }

      this.segmenter = seg
      this.currentModel = targetModel
      this.processingWidth = seg.inputSize.width
      this.processingHeight = seg.inputSize.height
      this._resizeMaskIfNeeded()
      this._resolveReady()
    } catch (e) {
      if (!this._destroyed && this._pendingModel === model) {
        console.error('[AMP] segmenter init failed — running in passthrough mode', e)
        this.segmenter = undefined
        this._resolveReady()
      }
    }
  }

  /**
   * Hot-swap the active segmenter for a new model while the render loop keeps running.
   * Initialises the new segmenter in the background; once ready it atomically replaces
   * `this.segmenter` and destroys the old one. Stale switches are cancelled via the
   * `_pendingModel` guard.
   *
   * @param model New target segmentation model.
   */
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

      if (model === SegmentationModel.AUTO) {
        const isFastEnough = await this._benchmarkSegmenter(seg)
        if (this._destroyed || this._pendingModel !== model) {
          seg.destroy()
          return
        }
        if (!isFastEnough) {
          seg.destroy()
          targetModel = SegmentationModel.LANDSCAPE
          seg = createSegmenter(targetModel)
          await seg.init()
          if (this._destroyed || this._pendingModel !== model) {
            seg.destroy()
            return
          }
        }
      }

      if (this._destroyed) {
        seg.destroy()
        return
      }

      const old = this.segmenter
      this.segmenter = seg
      this.currentModel = targetModel
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

  /**
   * Resize the segmentation canvas and GPU processing textures to match the new
   * segmenter's input dimensions. Also invalidates the stale `_latestPair` so the
   * render loop falls back to passthrough until the first mask at the new size arrives.
   */
  private _resizeMaskIfNeeded() {
    this.segmentationMaskCanvas?.setAttribute('width', '' + this.processingWidth)
    this.segmentationMaskCanvas?.setAttribute('height', '' + this.processingHeight)
    this.gpuRenderer?.resizeProcessing(this.processingWidth, this.processingHeight)
    this._passthroughMask = undefined
    // Invalidate stale pair from old dimensions — render loop falls back to
    // passthrough until the segmenter produces a mask at the new size.
    if (this._latestPair) {
      try { this._latestPair.source.close() } catch { /* ImageBitmap.close() — best-effort */ }
      this._latestPair = null
    }
  }

  /**
   * Create (or reuse) an `<img>` element for the virtual background.
   * The image is loaded asynchronously; the renderer uploads it lazily on the
   * next `render()` call once `img.complete` is true. If the image fails to load,
   * a `VIRTUAL_BG_LOAD_FAILED` error is pushed to the error store.
   * Clears the image when the processor type is not VIRTUAL.
   */
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

  /**
   * Start the segmenter loop and render loop once the output canvas stream is ready.
   * If the video element already has data, launches immediately; otherwise waits
   * for the `loadeddata` event.
   */
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

  /**
   * Actually launch both processing loops. Must only be called once per processor
   * lifecycle (subsequent calls are no-ops due to `_destroyed` guard).
   */
  private _launch(): void {
    if (this._destroyed) return
    this.videoElementLoaded = true
    this._segLoopActive = true
    this._runSegmenterLoop()   // fire-and-forget
    this._scheduleRender()
  }

  /**
   * Segmenter loop: runs at most 50fps. Each iteration captures the current
   * <video> frame as an ImageBitmap (GPU-backed snapshot), runs inference, and
   * publishes the (mask, frame) pair atomically. The previous bitmap is closed
   * to keep memory bounded to a single in-flight pair.
   */
  private async _runSegmenterLoop(): Promise<void> {
    const TARGET_MS = 1000 / 50
    while (this._segLoopActive && !this._destroyed) {
      const t0 = performance.now()
      const seg = this.segmenter
      if (!seg || !this.videoElement || this.videoElement.videoWidth === 0) {
        await new Promise<void>(r => setTimeout(r, TARGET_MS))
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
          await new Promise<void>(r => setTimeout(r, TARGET_MS))
          continue
        }
        const motionRgba = this._preProcessingPipeline ? this._getMotionFrameRgba() ?? undefined : undefined
        const cropBbox = this._preProcessingPipeline?.getNextCropBbox(
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
          ? this._preProcessingPipeline.apply(this.sourceImageData!, this._lastMask)
          : this.sourceImageData!
        const rawMask = await seg.segment(frameToSegment, performance.now())
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
            procW: this.processingWidth,
            procH: this.processingHeight,
          }
          capturedSource = null // ownership transferred to _latestPair
          previous?.source.close()
        } else {
          capturedSource.close()
          capturedSource = null
        }
      } catch (e) {
        if (capturedSource) {
          try { capturedSource.close() } catch { /* ImageBitmap.close() — best-effort */ }
          capturedSource = null
        }
        if (!this._segLoopActive) return
        console.error('[AMP] segmenter loop error', e)
        pushMattingError({
          code: 'SEGMENTER_TIMEOUT_PASSTHROUGH',
          level: 'warn',
          detail: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
        })
        await new Promise<void>(r => setTimeout(r, 100))
        continue
      }
      // Always yield to the event loop; sleep for whatever is left of 20ms.
      // If inference took longer than one frame period, setTimeout(0) still
      // lets the browser process render callbacks and input before looping.
      const elapsed = performance.now() - t0
      await new Promise<void>(r => setTimeout(r, Math.max(0, TARGET_MS - elapsed)))
    }
  }

  // Render loop target: 50fps = 20ms per frame.
  private static readonly RENDER_TARGET_MS = 1000 / 50
  private _lastRenderTime = 0

  /**
   * Schedule the next render frame via `requestAnimationFrame`.
   * Self-reschedules recursively as long as `_segLoopActive` is true.
   * Throttled to `RENDER_TARGET_MS` (~20 ms / 50 fps) to avoid redundant renders
   * when the browser fires rAF at a higher rate.
   */
  private _scheduleRender(): void {
    if (!this._segLoopActive) return
    this._renderLoopHandle = requestAnimationFrame((now) => {
      if (now - this._lastRenderTime >= AdvancedMattingProcessor.RENDER_TARGET_MS) {
        this._lastRenderTime = now
        this._renderFrame()
      }
      this._scheduleRender()
    })
  }

  /**
   * Cancel any pending `requestAnimationFrame` callback and clear the handle.
   */
  private _cancelRender(): void {
    if (this._renderLoopHandle === null) return
    cancelAnimationFrame(this._renderLoopHandle)
    this._renderLoopHandle = null
  }

  /**
   * Composite one output frame. Called by the render loop at ~50 fps.
   *
   * - No mask yet → render passthrough (uniform-1 mask, no halo possible).
   * - `_maxFrameOffset > 0` and mask is fresh → use live `<video>` frame
   *   (lower latency, bounded halo risk).
   * - Default → frame-locked mode: apply the mask to the exact `ImageBitmap` it
   *   was computed from (zero halo, latency = last inference time).
   * Resizes the output canvas automatically if the camera resolution changes.
   */
  private _renderFrame(): void {
    if (!this.gpuRenderer || !this.videoElement || this.videoElement.videoWidth === 0) return
    const pair = this._latestPair
    if (!pair) {
      // First mask not ready yet — passthrough with a uniform-1 mask; no halo
      // to worry about since the mask is constant.
      const vw = this.videoElement.videoWidth
      const vh = this.videoElement.videoHeight
      if (vw !== this.gpuRenderer.outW || vh !== this.gpuRenderer.outH) {
        this.gpuRenderer.resizeOutput(vw, vh)
      }
      this._drawPassthrough()
      return
    }
    this.gpuRenderer.uploadMask(pair.mask, pair.procW, pair.procH)

    // Decide whether to apply the mask to the live <video> frame (low latency,
    // up to maxFrameOffset frames of halo) or to the captured frame the mask
    // was computed from (frame-locked, zero halo, ~inference-time latency).
    // 1 "frame" ≈ 1000/30 ms (UI convention; actual camera framerate may differ).
    const FRAME_MS = 1000 / 30
    const maxAgeMs = this._maxFrameOffset * FRAME_MS
    const ageMs = performance.now() - pair.captureTime
    const useLive = this._maxFrameOffset > 0 && ageMs <= maxAgeMs

    if (useLive) {
      const vw = this.videoElement.videoWidth
      const vh = this.videoElement.videoHeight
      if (vw !== this.gpuRenderer.outW || vh !== this.gpuRenderer.outH) {
        this.gpuRenderer.resizeOutput(vw, vh)
      }
      this.gpuRenderer.render(this.videoElement)
    } else {
      const sw = pair.source.width
      const sh = pair.source.height
      if (sw !== this.gpuRenderer.outW || sh !== this.gpuRenderer.outH) {
        this.gpuRenderer.resizeOutput(sw, sh)
      }
      this.gpuRenderer.render(pair.source)
    }
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
      sx, sy, sw, sh,
      0, 0, this.processingWidth, this.processingHeight
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

  /**
   * Downscale the current snapshot to 128×72 and read its RGBA pixels.
   * This low-resolution frame is fed to `RoiCropper._hasMotionOutsideBbox()`
   * for cheap luma-based motion detection.
   *
   * @returns RGBA pixel data (length = 128*72*4) or `null` if no snapshot exists.
   */
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

  /**
   * Render a fully transparent (passthrough) frame by uploading a uniform-1 mask.
   * Used when no segmentation mask is available yet (during initialisation) or
   * when the segmenter is running in passthrough fallback mode.
   */
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

  /**
   * Find or create the main output canvas at the given dimensions.
   * Re-uses the existing DOM canvas (by ID) to avoid detaching a live
   * `captureStream()` track on hot-reloads or processor restarts.
   *
   * @param w Canvas width in pixels (= camera frame width).
   * @param h Canvas height in pixels (= camera frame height).
   */
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

  /**
   * Find or create the segmentation scratchpad canvas at the current processing
   * resolution. Uses `willReadFrequently: true` so the browser optimises for
   * repeated `getImageData()` calls.
   */
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

  /**
   * Create a new `<canvas>` element with the given ID and dimensions.
   * The element is not attached to the DOM.
   *
   * @param id     HTML `id` attribute to assign (used for re-use lookups).
   * @param width  Canvas width in pixels.
   * @param height Canvas height in pixels.
   * @returns      The newly created `<canvas>` element.
   */
  private _createCanvas(id: string, width: number, height: number) {
    const el = document.createElement('canvas')
    el.setAttribute('id', id)
    el.setAttribute('width', '' + width)
    el.setAttribute('height', '' + height)
    return el
  }

  /**
   * Destroy the current processor instance and re-initialise it with new options.
   * Useful when the source track changes (e.g. camera switch).
   *
   * @param opts New processor options (track + element).
   */
  async restart(opts: ProcessorOptions<Track.Kind>) {
    await this.destroy()
    return this.init(opts)
  }

  /**
   * Destroys the background processor. Stops the render loop, terminates the
   * timing web worker, closes WebGL2 renderers and active segmenters, and releases
   * captured tracks.
   */
  async destroy() {
    this._destroyed = true
    this._pendingModel = undefined
    this._configuredModel = undefined
    this._segLoopActive = false
    this._cancelRender()
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
      try { this._latestPair.source.close() } catch { /* ImageBitmap.close() — best-effort */ }
      this._latestPair = null
    }
    this._resolveReady()

    if (this.processedTrack && this.processedTrack !== this.source) {
      try {
        this.processedTrack.stop()
      } catch (e) {
        console.warn('[AMP] Failed to stop canvas capture track during destroy:', e)
      }
    }
    this.processedTrack = undefined
  }
}
