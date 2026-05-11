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
import { Segmenter, createSegmenter } from './segmenters'
import {
  CLEAR_TIMEOUT,
  SET_TIMEOUT,
  TIMEOUT_TICK,
  timerWorkerScript,
} from './TimerWorker'
import { WebGl2Renderer } from './renderers/WebGl2Renderer'
import { pushMattingError } from './errors/MattingErrorStore'

const SEGMENTATION_MASK_CANVAS_ID = 'background-blur-local-segmentation'
const BLUR_CANVAS_ID = 'background-blur-local'
const DEFAULT_BLUR = 10

/**
 * Unified background processor using WebGL2 for compositing.
 *
 * Pipeline per frame:
 *   videoElement
 *     → resize to segmenter input → ImageData (RGBA)
 *     → Segmenter.segment → Float32Array mask in [0, 1]
 *     → WebGl2Renderer.uploadMask + render → GPU post-processing + composite
 *
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
  // outputCanvasCtx removed — WebGl2Renderer owns the WebGL2 context

  segmentationMaskCanvas?: HTMLCanvasElement
  segmentationMaskCanvasCtx?: CanvasRenderingContext2D
  sourceImageData?: ImageData

  segmenter?: Segmenter
  private gpuRenderer?: WebGl2Renderer
  private _passthroughMask?: Float32Array

  timerWorker?: Worker
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

    // Initialize GPU renderer — throws and pushes to MattingErrorStore on failure.
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

    this._initWorker()

    // Initialize MediaPipe in the background so init() returns immediately.
    // _drawPassthrough() keeps the video visible until the segmenter is ready.
    this._initSegmenterBackground(this._getModel(this.options))
  }

  async update(opts: ProcessorConfig): Promise<void> {
    const prevModel = this.currentModel
    const newModel = this._getModel(opts)
    this.options = opts
    this.type = opts.type
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'

    this._initVirtualBackgroundImage()

    if (newModel !== prevModel) {
      if (this.segmenter) {
        // Active segmenter: keep it running during load, swap atomically when ready.
        this._switchSegmenterBackground(newModel)
      } else {
        // No active segmenter (still loading or failed): restart loading.
        this._initSegmenterBackground(newModel)
      }
    }
    this._applyRendererConfig()
  }

  private _getModel(opts: ProcessorConfig): SegmentationModel {
    if (opts.type === ProcessorType.BLUR || opts.type === ProcessorType.VIRTUAL) {
      return opts.model ?? SegmentationModel.LANDSCAPE
    }
    return SegmentationModel.LANDSCAPE
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

  /** Push current options (mode, blur, virtual bg, pre/post-processing) to the renderer. */
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

  /** Initial load: no existing segmenter, shows passthrough until ready. */
  private async _initSegmenterBackground(model: SegmentationModel) {
    this._pendingModel = model
    try {
      const seg = createSegmenter(model)
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
        // pushMattingError already called by the segmenter implementation.
        console.error('[AMP] segmenter init failed — running in passthrough mode', e)
        this.segmenter = undefined
        this._resolveReady()
      }
    }
  }

  /** Model switch: keep old segmenter running, atomically replace when new one is ready. */
  private async _switchSegmenterBackground(model: SegmentationModel) {
    this._pendingModel = model
    try {
      const seg = createSegmenter(model)
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
        // pushMattingError already called by the segmenter implementation.
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
  }

  private _initVirtualBackgroundImage() {
    if (this.options.type !== ProcessorType.VIRTUAL) {
      this.virtualBackgroundImage = undefined
      return
    }
    const path = this.options.imagePath
    // Use a data attribute to compare paths without the absolute-URL issue
    // (img.src always returns the absolute URL, but path is relative).
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

  private _initWorker() {
    this.timerWorker = new Worker(timerWorkerScript, { name: 'AdvancedMatting' })
    this.timerWorker.onmessage = (data) => this.onTimerMessage(data)
    // readyState >= 2 (HAVE_CURRENT_DATA): video already loaded — start immediately.
    // onloadeddata won't fire again (e.g. on Safari when the camera was already running).
    if (this.videoElementLoaded || this.videoElement!.readyState >= 2) {
      this.videoElementLoaded = true
      this.timerWorker.postMessage({ id: SET_TIMEOUT, timeMs: 1000 / 30 })
    } else {
      this.videoElement!.onloadeddata = () => {
        this.videoElementLoaded = true
        this.timerWorker!.postMessage({ id: SET_TIMEOUT, timeMs: 1000 / 30 })
      }
    }
  }

  private onTimerMessage(response: { data: { id: number } }) {
    if (response.data.id === TIMEOUT_TICK) {
      this.process()
    }
  }

  private sizeSource(cropBbox?: BBox | null) {
    const vw = this.videoElement!.videoWidth
    const vh = this.videoElement!.videoHeight
    // When a crop bbox is provided, extract that region of the full-res video and
    // scale it to the model input size — giving the segmenter ~2× effective resolution
    // on the person's region compared to a full-frame downscale.
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

  async process() {
    if (!this.videoElement || this.videoElement.videoWidth === 0) {
      this.timerWorker?.postMessage({ id: SET_TIMEOUT, timeMs: 1000 / 30 })
      return
    }

    // Segmenter not yet ready (still loading) — passthrough to keep video visible.
    if (!this.segmenter) {
      this._drawPassthrough()
      this.timerWorker?.postMessage({ id: SET_TIMEOUT, timeMs: 1000 / 30 })
      return
    }

    try {
      // Ask the preprocessing pipeline for a spatial crop bbox (from last frame's mask).
      const cropBbox = this._preProcessingPipeline?.getNextCropBbox() ?? null
      this.sizeSource(cropBbox)

      const frameToSegment = this._preProcessingPipeline
        ? this._preProcessingPipeline.apply(this.sourceImageData!, this._lastMask)
        : this.sourceImageData!

      // Inference is in crop-bbox space when ROI cropping is active.
      const rawMask = await this.segmenter!.segment(frameToSegment, performance.now())

      // Remap from crop space → full-frame space and update RoiCropper state.
      const mask = this._preProcessingPipeline
        ? this._preProcessingPipeline.applyAfterInference(
            rawMask,
            this.processingWidth,
            this.processingHeight,
            cropBbox
          )
        : rawMask

      this._lastMask = mask
      this.gpuRenderer!.uploadMask(mask, this.processingWidth, this.processingHeight)
      this.gpuRenderer!.render(this.videoElement!)
    } catch (e) {
      console.error('[AMP] process error', e)
      pushMattingError({
        code: 'SEGMENTER_TIMEOUT_PASSTHROUGH',
        level: 'warn',
        detail: e instanceof Error ? e.message : String(e),
      })
      this._drawPassthrough()
    }
    this.timerWorker?.postMessage({ id: SET_TIMEOUT, timeMs: 1000 / 30 })
  }

  private _drawPassthrough() {
    if (!this.gpuRenderer || !this.videoElement) return
    const w = this.processingWidth
    const h = this.processingHeight
    // Upload an all-ones mask so the composite shader outputs fg = video frame.
    // mix(bg, fg, 1.0) = fg — regardless of what the background mode is.
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
    // No Canvas2D context here — WebGl2Renderer.init() calls getContext('webgl2').
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
    this.timerWorker?.postMessage({ id: CLEAR_TIMEOUT })
    this.timerWorker?.terminate()
    this.timerWorker = undefined
    this.segmenter?.destroy()
    this.segmenter = undefined
    this.gpuRenderer?.destroy()
    this.gpuRenderer = undefined
    this._preProcessingPipeline = undefined
    this._lastMask = undefined
    this._resolveReady()
  }
}
