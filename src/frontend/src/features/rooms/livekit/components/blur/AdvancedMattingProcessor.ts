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
import { Segmenter, createSegmenter, RVMSegmenter } from './segmenters'
import { WebGl2Renderer } from './renderers/WebGl2Renderer'
import { pushMattingError } from './errors/MattingErrorStore'

const SEGMENTATION_MASK_CANVAS_ID = 'background-blur-local-segmentation'
const BLUR_CANVAS_ID = 'background-blur-local'
const DEFAULT_BLUR = 10

/**
 * Unified background processor using WebGL2 for compositing.
 *
 * Two independent loops:
 *   Segmenter loop  — free-running async, pulls frames, runs inference, writes
 *                     to _latestMask as fast as the GPU allows.
 *   Render loop     — requestVideoFrameCallback (fallback: rAF), fires at the
 *                     camera's native framerate, composites the latest available
 *                     mask without ever blocking on inference.
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

  segmenter?: Segmenter
  private gpuRenderer?: WebGl2Renderer
  private _passthroughMask?: Float32Array

  // Two-loop state
  private _segLoopActive = false
  private _latestMask: Float32Array | null = null
  private _renderLoopHandle: number | null = null

  virtualBackgroundImage?: HTMLImageElement

  private currentModel?: SegmentationModel
  private processingWidth = 256
  private processingHeight = 144
  private _pendingModel?: SegmentationModel
  private _readyResolvers: Array<() => void> = []
  private _preProcessingPipeline?: PreProcessingPipeline
  private _lastMask?: Float32Array

  constructor(opts: ProcessorConfig) {
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this.options = opts
    this.type = opts.type
  }

  /** Resolves once the active segmenter is loaded and producing frames. */
  waitForReady(): Promise<void> {
    if (this.segmenter) return Promise.resolve()
    return new Promise(resolve => this._readyResolvers.push(resolve))
  }

  private _resolveReady() {
    this._readyResolvers.splice(0).forEach(r => r())
  }

  async init(opts: ProcessorOptions<Track.Kind>) {
    if (!opts.element) {
      throw new Error('Element is required for processing')
    }
    this.source = opts.track as MediaStreamTrack
    this.sourceSettings = this.source!.getSettings()
    this.videoElement = opts.element as HTMLVideoElement

    this._initVirtualBackgroundImage()
    this._createMainCanvas()
    this._createMaskCanvas()

    this.gpuRenderer = new WebGl2Renderer()
    await this.gpuRenderer.init(this.outputCanvas!, {
      outW: this.sourceSettings!.width || 1280,
      outH: this.sourceSettings!.height || 720,
      processingW: this.processingWidth,
      processingH: this.processingHeight,
      postProcessing: this._getPostProcessingConfig(),
      upsampling: this._getUpsamplingConfig(),
    })
    this._applyRendererConfig()

    if (!this.outputCanvas!.captureStream) {
      pushMattingError({
        code: 'CAPTURESTREAM_UNSUPPORTED',
        level: 'error',
        detail: 'captureStream not supported on this browser',
      })
      throw new Error('[AMP] captureStream not supported on this browser')
    }
    const stream = this.outputCanvas!.captureStream(30)
    const tracks = stream.getVideoTracks()
    if (tracks.length === 0) {
      throw new Error('[AMP] No tracks found in captureStream()')
    }
    this.processedTrack = tracks[0]

    this._startLoops()

    // Initialize segmenter in background — passthrough renders until it's ready.
    this._initSegmenterBackground(this._getModel(this.options))
  }

  async update(opts: ProcessorConfig): Promise<void> {
    const prevModel = this.currentModel
    const newModel = this._getModel(opts)
    const prevRvmRatio = this._getRvmRatio(this.options)
    const nextRvmRatio = this._getRvmRatio(opts)
    this.options = opts
    this.type = opts.type
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'

    this._initVirtualBackgroundImage()

    if (newModel !== prevModel) {
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
      return opts.model ?? SegmentationModel.LANDSCAPE
    }
    return SegmentationModel.LANDSCAPE
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
    this._pendingModel = model
    try {
      const seg = createSegmenter(model, {
        rvmDownsampleRatio:
          this._getRvmRatio(this.options) ?? this._autoRvmRatio(),
      })
      await seg.init()
      if (this._pendingModel !== model) {
        seg.destroy()
        return
      }
      this.segmenter = seg
      this.currentModel = model
      this.processingWidth = seg.inputSize.width
      this.processingHeight = seg.inputSize.height
      this._resizeMaskIfNeeded()
      this._resolveReady()
    } catch (e) {
      if (this._pendingModel === model) {
        console.error('[AMP] segmenter init failed — running in passthrough mode', e)
        this.segmenter = undefined
        this._resolveReady()
      }
    }
  }

  private async _switchSegmenterBackground(model: SegmentationModel) {
    this._pendingModel = model
    try {
      const seg = createSegmenter(model, {
        rvmDownsampleRatio:
          this._getRvmRatio(this.options) ?? this._autoRvmRatio(),
      })
      await seg.init()
      if (this._pendingModel !== model) {
        seg.destroy()
        return
      }
      const old = this.segmenter
      this.segmenter = seg
      this.currentModel = model
      this.processingWidth = seg.inputSize.width
      this.processingHeight = seg.inputSize.height
      old?.destroy()
      this._resizeMaskIfNeeded()
      this._resolveReady()
    } catch (e) {
      if (this._pendingModel === model) {
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
    // Invalidate stale mask from old dimensions — render loop falls back to
    // passthrough until the segmenter produces a mask at the new size.
    this._latestMask = null
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
    if (this.videoElementLoaded || this.videoElement!.readyState >= 2) {
      this._launch()
    } else {
      this.videoElement!.onloadeddata = () => this._launch()
    }
  }

  private _launch(): void {
    this.videoElementLoaded = true
    this._segLoopActive = true
    this._runSegmenterLoop()   // fire-and-forget
    this._scheduleRender()
  }

  /**
   * Segmenter loop: runs at most 30fps, writes the latest alpha mask to
   * _latestMask. Capped so it does not starve the render loop or saturate the
   * GPU when inference is faster than one frame period.
   */
  private async _runSegmenterLoop(): Promise<void> {
    const TARGET_MS = 1000 / 50
    while (this._segLoopActive) {
      const t0 = performance.now()
      const seg = this.segmenter
      if (!seg || !this.videoElement || this.videoElement.videoWidth === 0) {
        await new Promise<void>(r => setTimeout(r, TARGET_MS))
        continue
      }
      try {
        const cropBbox = this._preProcessingPipeline?.getNextCropBbox() ?? null
        this.sizeSource(cropBbox)
        const frameToSegment = this._preProcessingPipeline
          ? this._preProcessingPipeline.apply(this.sourceImageData!, this._lastMask)
          : this.sourceImageData!
        const rawMask = await seg.segment(frameToSegment, performance.now())
        if (!this._segLoopActive) return
        if (this.segmenter === seg) {
          const refinedMask = this._maybeApplyGuidedFilter(rawMask)
          const mask = this._preProcessingPipeline
            ? this._preProcessingPipeline.applyAfterInference(
                refinedMask,
                this.processingWidth,
                this.processingHeight,
                cropBbox
              )
            : refinedMask
          this._lastMask = mask
          this._latestMask = mask
        }
      } catch (e) {
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
      // Always yield to the event loop; sleep for whatever is left of 33ms.
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
    const mask = this._latestMask
    if (mask) {
      this.gpuRenderer.uploadMask(mask, this.processingWidth, this.processingHeight)
      this.gpuRenderer.render(this.videoElement)
    } else {
      this._drawPassthrough()
    }
  }

  // ────────────────────────────────────────────────────────────────────────────

  private sizeSource(cropBbox?: BBox | null) {
    const vw = this.videoElement!.videoWidth
    const vh = this.videoElement!.videoHeight
    const sx = cropBbox ? Math.round(cropBbox.x * vw) : 0
    const sy = cropBbox ? Math.round(cropBbox.y * vh) : 0
    const sw = cropBbox ? Math.round(cropBbox.width * vw) : vw
    const sh = cropBbox ? Math.round(cropBbox.height * vh) : vh
    this.segmentationMaskCanvasCtx!.drawImage(
      this.videoElement!,
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

  private _maybeApplyGuidedFilter(rawMask: Float32Array): Float32Array {
    if (
      this.options.type !== ProcessorType.BLUR &&
      this.options.type !== ProcessorType.VIRTUAL
    ) {
      return rawMask
    }
    const gf = this.options.postProcessing?.guidedFilter
    if (!gf || !this.sourceImageData) return rawMask

    return applyGuidedFilter(rawMask, this.sourceImageData, gf.radius, gf.eps)
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

  private _createMainCanvas() {
    let canvas = document.querySelector(
      `canvas#${BLUR_CANVAS_ID}`
    ) as HTMLCanvasElement | null
    if (!canvas) {
      canvas = this._createCanvas(
        BLUR_CANVAS_ID,
        this.sourceSettings!.width || 1280,
        this.sourceSettings!.height || 720
      )
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
    this._pendingModel = undefined
    this._segLoopActive = false
    this._cancelRender()
    this.segmenter?.destroy()
    this.segmenter = undefined
    this.gpuRenderer?.destroy()
    this.gpuRenderer = undefined
    this._preProcessingPipeline = undefined
    this._lastMask = undefined
    this._latestMask = null
    this._resolveReady()
  }
}
