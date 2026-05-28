import { ProcessorOptions, Track } from 'livekit-client'
import {
  BackgroundProcessorInterface,
  LatencyMode,
  ProcessorConfig,
  ProcessorType,
  SegmentationModel,
  PostProcessingConfig,
  UpsamplingConfig,
  PreProcessingConfig,
} from '.'
import { PreProcessingPipeline } from './preprocessing/PreProcessingPipeline'
import { MaskMotionTracker } from './preprocessing/MaskMotionTracker'
import {
  Segmenter,
  createSegmenter,
  DepthAnythingSegmenter,
} from './segmenters'
import { GpuRenderer, GpuRendererInitOpts } from './renderers/GpuRenderer'
import { WebGl2Renderer } from './renderers/WebGl2Renderer'
import { Canvas2dRenderer } from './renderers/Canvas2dRenderer'
import {
  pushMattingError,
  dismissMattingError,
} from './errors/MattingErrorStore'
import { debugWarn } from './debug'
import {
  resetMattingStats,
  setCameraSettings,
  setMattingStatsActive,
  setMattingStatsModel,
  setSegmenterFrameSkip,
} from './stats/MattingStatsStore'

import { MattingCanvasManager } from './preprocessing/MattingCanvasManager'
import { SegmenterBenchmarker } from './segmenters/SegmenterBenchmarker'
import { DynamicLatencyEngine } from './stats/DynamicLatencyEngine'
import { VideoFrameTracker } from './preprocessing/VideoFrameTracker'
import { SegmenterLoopRunner, FrameMaskPair } from './segmenters/SegmenterLoopRunner'
import { RenderLoopRunner } from './renderers/RenderLoopRunner'

const BLUR_CANVAS_ID = 'background-blur-local'
const DEFAULT_BLUR = 10

/**
 * Unified background processor using WebGL2 for compositing.
 * Orchestrates high-performance background blur and virtual backgrounds.
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

  // Refactored helpers & architectural sub-modules
  private _canvasManager = new MattingCanvasManager()
  private _latencyEngine = new DynamicLatencyEngine()
  private _frameTracker = new VideoFrameTracker()
  private _segmenterRunner!: SegmenterLoopRunner
  private _renderRunner!: RenderLoopRunner

  segmenter?: Segmenter
  gpuRenderer?: GpuRenderer

  // Shared two-loop states
  private _latestPair: FrameMaskPair | null = null
  private _segmenterFrameSkip = 2

  private _latencyMode: LatencyMode = 0

  private _motionTracker = new MaskMotionTracker()

  virtualBackgroundImage?: HTMLImageElement

  private _configuredModel?: SegmentationModel
  currentModel?: SegmentationModel
  processingWidth = 256
  processingHeight = 144
  private _pendingModel?: SegmentationModel
  private _readyResolvers: Array<() => void> = []
  private _destroyed = false
  private _preProcessingPipeline?: PreProcessingPipeline

  constructor(opts: ProcessorConfig) {
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this.options = opts
    this.type = opts.type
    const cfg = DynamicLatencyEngine.getLatencyConfig(opts)
    this._latencyMode = cfg.mode

    this._initRunners()
  }

  private _initRunners() {
    this._segmenterRunner = new SegmenterLoopRunner(
      () => this.segmenter,
      () => this._preProcessingPipeline,
      () => this._canvasManager,
      () => this._frameTracker,
      () => this._segmenterFrameSkip,
      () => ({ w: this.processingWidth, h: this.processingHeight }),
      (pair) => this._onPairProduced(pair)
    )

    this._renderRunner = new RenderLoopRunner(
      () => this.gpuRenderer,
      () => this._latencyEngine,
      () => this._motionTracker,
      () => this._frameTracker,
      () => this._latestPair,
      (w, h) => this._canvasManager.getPassthroughMask(w, h),
      () => this.getLatencyParams(),
      () => ({ w: this.processingWidth, h: this.processingHeight })
    )
  }

  // Getters for external and testing compatibility
  get segmentationMaskCanvas() {
    return this._canvasManager.segmentationMaskCanvas
  }

  get segmentationMaskCanvasCtx() {
    return this._canvasManager.segmentationMaskCanvasCtx
  }

  get sourceImageData() {
    return undefined
  }

  getLatencyParams() {
    return { latencyMode: this._latencyMode }
  }

  private _onPairProduced(pair: FrameMaskPair) {
    const prev = this._latestPair
    this._latestPair = pair
    prev?.source.close()
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
      this._canvasManager.ensureMaskCanvas(this.processingWidth, this.processingHeight)

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
        this._stopTrackCleanup()
        return
      }

      this._startLoops()

      this._configuredModel = this._getModel(this.options)
      this._initSegmenterBackground(this._configuredModel)
    } catch (e) {
      debugWarn('[AMP INIT] Initialization failed, falling back to passthrough track.', e)
      pushMattingError({
        code: 'WEBGL2_INIT_FAILED',
        level: 'warn',
        detail: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
      })
      this.processedTrack = this.source
    }
  }

  private _stopTrackCleanup() {
    if (this.processedTrack && this.processedTrack !== this.source) {
      try {
        this.processedTrack.stop()
      } catch {
        // best-effort
      }
    }
    this.processedTrack = undefined
  }

  async update(opts: ProcessorConfig): Promise<void> {
    this.options = opts
    this.type = opts.type
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    const cfg = DynamicLatencyEngine.getLatencyConfig(opts)
    this._latencyMode = cfg.mode

    if (!this.gpuRenderer) {
      return
    }

    const prevConfigured = this._configuredModel
    const newModel = this._getModel(opts)
    this._configuredModel = newModel

    this._initVirtualBackgroundImage()

    if (newModel !== prevConfigured) {
      if (this.segmenter) {
        this._switchSegmenterBackground(newModel)
      } else {
        this._initSegmenterBackground(newModel)
      }
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

  private _publishBenchmarkPair(mask: Float32Array, source: ImageBitmap, captureTime: number) {
    this._onPairProduced({
      mask,
      source,
      captureTime,
      cameraCaptureTime: captureTime,
      procW: this.processingWidth,
      procH: this.processingHeight,
    })
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

  private async _createAndCalibrateSegmenter(model: SegmentationModel): Promise<{
    seg: Segmenter
    targetModel: SegmentationModel
  } | undefined> {
    // ── Tier 1 probe: DepthAnything via WebGPU (AUTO mode only) ────────────
    if (model === SegmentationModel.AUTO) {
      console.log('[AMP] Tier 1 probe: checking WebGPU availability...')
      const webgpuAvailable = await SegmenterBenchmarker.probeWebGPU()
      console.log(`[AMP] Tier 1 probe: WebGPU available = ${webgpuAvailable}`)

      if (webgpuAvailable && !this._destroyed && this._pendingModel === model) {
        const t1Seg = new DepthAnythingSegmenter()
        try {
          await t1Seg.init()
          if (this._destroyed || this._pendingModel !== model) {
            t1Seg.destroy()
            return undefined
          }
          console.log('[AMP] Tier 1: model loaded, running benchmark (15 warm-up + timed runs)...')
          const t1P75 = await SegmenterBenchmarker.measureInferenceP75(
            t1Seg,
            this.videoElement,
            (mask, source, time) => this._publishBenchmarkPair(mask, source, time),
            () => this._destroyed || this._pendingModel !== model
          )
          if (!this._destroyed && this._pendingModel === model && t1P75 !== null && t1P75 < 80) {
            console.log(`[AMP] Tier 1 SELECTED — Depth Anything V2 (WebGPU) | P75=${t1P75.toFixed(1)}ms | skip=2 (15 FPS inference / 30 FPS render)`)
            this._segmenterFrameSkip = 2
            return { seg: t1Seg, targetModel: SegmentationModel.DEPTH_ANYTHING }
          }
          console.warn(
            t1P75 === null
              ? '[AMP] Tier 1 rejected: benchmark aborted — falling back to Tier 2 (Multiclass MediaPipe)'
              : `[AMP] Tier 1 rejected: P75=${t1P75.toFixed(1)}ms ≥ 80ms threshold — falling back to Tier 2 (Multiclass MediaPipe)`
          )
        } catch (e) {
          console.warn('[AMP] Tier 1 rejected: init failed — falling back to Tier 2 (Multiclass MediaPipe)', e)
        }
        t1Seg.destroy()
        if (this._destroyed || this._pendingModel !== model) return undefined
      } else if (!webgpuAvailable) {
        console.log('[AMP] Tier 1 skipped: no WebGPU — using Tier 2 (Multiclass MediaPipe)')
      }
    }

    // ── Tier 2 / Tier 3: existing MediaPipe logic ───────────────────────────
    let targetModel = model
    if (model === SegmentationModel.AUTO) {
      targetModel = SegmentationModel.MULTICLASS
    }

    let seg = createSegmenter(targetModel)
    await seg.init()

    if (this._destroyed || this._pendingModel !== model) {
      seg.destroy()
      return undefined
    }

    if (targetModel === SegmentationModel.MULTICLASS) {
      const benchResult = await SegmenterBenchmarker.benchmarkSegmenter(
        seg,
        this.videoElement,
        (mask, source, time) => this._publishBenchmarkPair(mask, source, time),
        () => this._destroyed || this._pendingModel !== model
      )
      if (this._destroyed || this._pendingModel !== model) {
        seg.destroy()
        return undefined
      }
      if (benchResult === 'landscape' && model === SegmentationModel.AUTO) {
        seg.destroy()
        targetModel = SegmentationModel.LANDSCAPE
        seg = createSegmenter(targetModel)
        await seg.init()
        if (this._destroyed || this._pendingModel !== model) {
          seg.destroy()
          return undefined
        }
      } else {
        this._segmenterFrameSkip = benchResult === 'multiclass_skip1' ? 1 : 2
      }
    }

    if (targetModel === SegmentationModel.LANDSCAPE) {
      const skipResult = await SegmenterBenchmarker.benchmarkLandscapeSkip(
        seg,
        this.videoElement,
        (mask, source, time) => this._publishBenchmarkPair(mask, source, time),
        () => this._destroyed || this._pendingModel !== model
      )
      if (this._destroyed || this._pendingModel !== model) {
        seg.destroy()
        return undefined
      }
      this._segmenterFrameSkip = skipResult === 'skip1' ? 1 : 2
    }

    return { seg, targetModel }
  }

  private _applyDepthBokehMode(targetModel: SegmentationModel) {
    const isDepth = targetModel === SegmentationModel.DEPTH_ANYTHING
    console.log(`[AMP] renderer mode: ${isDepth ? 'DEPTH BOKEH (True Bokeh shader — FS_DEPTH_BOKEH)' : 'STANDARD BLUR/VIRTUAL'}`)
    this.gpuRenderer?.setDepthBokehMode?.(isDepth)
  }

  private async _initSegmenterBackground(model: SegmentationModel) {
    if (this._destroyed) return
    this._pendingModel = model
    try {
      const result = await this._createAndCalibrateSegmenter(model)
      if (!result || this._destroyed || this._pendingModel !== model) {
        return
      }

      this.segmenter = result.seg
      this.currentModel = result.targetModel
      setMattingStatsModel(model, result.targetModel)
      setSegmenterFrameSkip(this._segmenterFrameSkip)
      this.processingWidth = result.seg.inputSize.width
      this.processingHeight = result.seg.inputSize.height
      this._applyDepthBokehMode(result.targetModel)
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
      const result = await this._createAndCalibrateSegmenter(model)
      if (!result || this._destroyed || this._pendingModel !== model) {
        return
      }

      const old = this.segmenter
      this.segmenter = result.seg
      this.currentModel = result.targetModel
      setMattingStatsModel(model, result.targetModel)
      setSegmenterFrameSkip(this._segmenterFrameSkip)
      this.processingWidth = result.seg.inputSize.width
      this.processingHeight = result.seg.inputSize.height
      this._applyDepthBokehMode(result.targetModel)
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
    this._canvasManager.ensureMaskCanvas(this.processingWidth, this.processingHeight)
    this.gpuRenderer?.resizeProcessing(
      this.processingWidth,
      this.processingHeight
    )
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
    setMattingStatsActive(true)
    this._frameTracker.start(this.videoElement!)
    this._segmenterRunner.start(this.videoElement!)
    this._renderRunner.start(this.videoElement!, this.outputCanvas!)
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

  private _createMainCanvasWithSize(w: number, h: number) {
    let canvas = document.querySelector(
      `canvas#${BLUR_CANVAS_ID}`
    ) as HTMLCanvasElement | null
    if (!canvas) {
      canvas = this._canvasManager.createCanvas(BLUR_CANVAS_ID, w, h)
    } else {
      canvas.setAttribute('width', '' + w)
      canvas.setAttribute('height', '' + h)
    }
    this.outputCanvas = canvas
  }

  async restart(opts: ProcessorOptions<Track.Kind>) {
    await this.destroy()
    return this.init(opts)
  }

  async destroy() {
    this._destroyed = true
    this._pendingModel = undefined
    this._configuredModel = undefined
    this.videoElementLoaded = false
    this._motionTracker.reset()
    this._latencyEngine.reset()

    this._segmenterRunner.stop()
    this._renderRunner.stop()
    this._frameTracker.stop()

    resetMattingStats()
    if (this.videoElement) {
      this.videoElement.onloadeddata = null
    }
    this.segmenter?.destroy()
    this.segmenter = undefined
    this.gpuRenderer?.destroy()
    this.gpuRenderer = undefined
    this._preProcessingPipeline = undefined
    this._canvasManager.destroy()
    if (this._latestPair) {
      try {
        this._latestPair.source.close()
      } catch {
        /* ImageBitmap.close() — best-effort */
      }
      this._latestPair = null
    }
    this._resolveReady()
    this._stopTrackCleanup()
  }
}