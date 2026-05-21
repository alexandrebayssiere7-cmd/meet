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
import { Segmenter, createSegmenter, RVMSegmenter, probeMediapipeDelegate } from './segmenters'
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

  private _resolveReady() {
    this._readyResolvers.splice(0).forEach(r => r())
  }

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
          } catch {}
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


  private _getModel(opts: ProcessorConfig): SegmentationModel {
    if (opts.type === ProcessorType.BLUR || opts.type === ProcessorType.VIRTUAL) {
      return opts.model ?? SegmentationModel.AUTO
    }
    return SegmentationModel.AUTO
  }

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

  private _getRvmRatio(opts: ProcessorConfig): number | undefined {
    if (opts.type === ProcessorType.BLUR || opts.type === ProcessorType.VIRTUAL) {
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
        const cropBbox = this._preProcessingPipeline?.getNextCropBbox() ?? null
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

  private _cancelRender(): void {
    if (this._renderLoopHandle === null) return
    cancelAnimationFrame(this._renderLoopHandle)
    this._renderLoopHandle = null
  }

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
