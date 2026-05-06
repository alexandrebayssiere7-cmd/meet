import { ProcessorOptions, Track } from 'livekit-client'
import {
  BackgroundProcessorInterface,
  ProcessorConfig,
  ProcessorType,
  SegmentationModel,
  PostProcessingConfig,
} from '.'
import { Segmenter, createSegmenter } from './segmenters'
import { PostProcessingPipeline } from './postprocessing/PostProcessingPipeline'
import {
  CLEAR_TIMEOUT,
  SET_TIMEOUT,
  TIMEOUT_TICK,
  timerWorkerScript,
} from './TimerWorker'

const SEGMENTATION_MASK_CANVAS_ID = 'background-blur-local-segmentation'
const BLUR_CANVAS_ID = 'background-blur-local'
const DEFAULT_BLUR = 10

/**
 * Unified background processor running on every browser via canvas + captureStream().
 * Replaces the previous (Unified | Custom) split.
 *
 * Pipeline per frame:
 *   videoElement
 *     → resize to segmenter input → ImageData (RGBA)
 *     → Segmenter.segment → Float32Array mask in [0, 1]
 *     → PostProcessingPipeline.apply → refined Float32 mask
 *     → write mask as alpha into segmentationMask ImageData (Float32 → 0..255)
 *     → composite: blur OR virtual background using globalCompositeOperation.
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
  outputCanvasCtx?: CanvasRenderingContext2D

  segmentationMaskCanvas?: HTMLCanvasElement
  segmentationMaskCanvasCtx?: CanvasRenderingContext2D
  segmentationMask?: ImageData
  sourceImageData?: ImageData

  segmenter?: Segmenter
  postPipeline?: PostProcessingPipeline

  timerWorker?: Worker
  virtualBackgroundImage?: HTMLImageElement

  private currentModel?: SegmentationModel
  private processingWidth = 256
  private processingHeight = 144

  constructor(opts: ProcessorConfig) {
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this.options = opts
    this.type = opts.type
  }

  async init(opts: ProcessorOptions<Track.Kind>) {
    console.log('[AMP] init start', opts)
    if (!opts.element) {
      throw new Error('Element is required for processing')
    }
    this.source = opts.track as MediaStreamTrack
    this.sourceSettings = this.source!.getSettings()
    console.log('[AMP] sourceSettings', this.sourceSettings)
    this.videoElement = opts.element as HTMLVideoElement

    this._initVirtualBackgroundImage()
    console.log('[AMP] initializing segmenter...')
    await this._initSegmenter()
    console.log('[AMP] segmenter ready, inputSize=', this.processingWidth, 'x', this.processingHeight)
    this._initPipeline()
    this._createMainCanvas()
    this._createMaskCanvas()

    if (!this.outputCanvas!.captureStream) {
      throw new Error('[AMP] captureStream not supported on this browser')
    }
    const stream = this.outputCanvas!.captureStream(30)
    const tracks = stream.getVideoTracks()
    console.log('[AMP] captureStream tracks:', tracks.length, tracks)
    if (tracks.length === 0) {
      throw new Error('[AMP] No tracks found in captureStream()')
    }
    this.processedTrack = tracks[0]

    this.segmentationMask = new ImageData(
      this.processingWidth,
      this.processingHeight
    )
    this._initWorker()
    console.log('[AMP] init complete, processedTrack=', this.processedTrack)
  }

  async update(opts: ProcessorConfig): Promise<void> {
    const prevModel = this.currentModel
    const newModel = this._getModel(opts)
    this.options = opts
    this.type = opts.type
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'

    this._initVirtualBackgroundImage()

    if (newModel !== prevModel) {
      this.segmenter?.destroy()
      await this._initSegmenter()
      // input size may have changed → recreate ImageData and mask canvas
      this.segmentationMask = new ImageData(
        this.processingWidth,
        this.processingHeight
      )
      this.segmentationMaskCanvas?.setAttribute('width', '' + this.processingWidth)
      this.segmentationMaskCanvas?.setAttribute(
        'height',
        '' + this.processingHeight
      )
    }
    this._initPipeline()
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

  private async _initSegmenter() {
    const model = this._getModel(this.options)
    this.currentModel = model
    this.segmenter = createSegmenter(model)
    await this.segmenter.init()
    this.processingWidth = this.segmenter.inputSize.width
    this.processingHeight = this.segmenter.inputSize.height
  }

  private _initPipeline() {
    this.postPipeline = new PostProcessingPipeline(this._getPostProcessingConfig())
  }

  private _initVirtualBackgroundImage() {
    if (this.options.type !== ProcessorType.VIRTUAL) {
      this.virtualBackgroundImage = undefined
      return
    }
    const path = this.options.imagePath
    const needsUpdate =
      !this.virtualBackgroundImage || this.virtualBackgroundImage.src !== path
    if (needsUpdate && path) {
      this.virtualBackgroundImage = document.createElement('img')
      this.virtualBackgroundImage.crossOrigin = 'anonymous'
      this.virtualBackgroundImage.src = path
    }
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

  private sizeSource() {
    this.segmentationMaskCanvasCtx!.drawImage(
      this.videoElement!,
      0,
      0,
      this.videoElement!.videoWidth,
      this.videoElement!.videoHeight,
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

  private writeMaskAlpha(mask: Float32Array) {
    const data = this.segmentationMask!.data
    for (let i = 0; i < mask.length; i++) {
      const v = mask[i]
      const a = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255)
      data[i * 4 + 3] = a
    }
  }

  private composite() {
    const w = this.outputCanvas!.width
    const h = this.outputCanvas!.height
    this.segmentationMaskCanvasCtx!.putImageData(this.segmentationMask!, 0, 0)

    // 1) draw the slightly-blurred mask scaled to output size
    this.outputCanvasCtx!.globalCompositeOperation = 'copy'
    this.outputCanvasCtx!.filter = 'blur(8px)'
    this.outputCanvasCtx!.drawImage(
      this.segmentationMaskCanvas!,
      0,
      0,
      this.processingWidth,
      this.processingHeight,
      0,
      0,
      w,
      h
    )

    // 2) draw clear body (only where mask alpha > 0)
    this.outputCanvasCtx!.globalCompositeOperation = 'source-in'
    this.outputCanvasCtx!.filter = 'none'
    this.outputCanvasCtx!.drawImage(this.videoElement!, 0, 0, w, h)

    // 3) draw the background underneath
    this.outputCanvasCtx!.globalCompositeOperation = 'destination-over'
    if (this.options.type === ProcessorType.BLUR) {
      const radius = this.options.blurRadius ?? DEFAULT_BLUR
      this.outputCanvasCtx!.filter = `blur(${radius}px)`
      this.outputCanvasCtx!.drawImage(this.videoElement!, 0, 0, w, h)
      this.outputCanvasCtx!.filter = 'none'
    } else if (this.virtualBackgroundImage) {
      this.outputCanvasCtx!.filter = 'none'
      this.outputCanvasCtx!.drawImage(this.virtualBackgroundImage, 0, 0, w, h)
    }
  }

  async process() {
    if (!this.videoElement || this.videoElement.videoWidth === 0) {
      console.warn('[AMP] process skipped: video not ready, readyState=', this.videoElement?.readyState)
      this.timerWorker?.postMessage({ id: SET_TIMEOUT, timeMs: 1000 / 30 })
      return
    }
    try {
      this.sizeSource()
      const rawMask = await this.segmenter!.segment(
        this.sourceImageData!,
        performance.now()
      )
      const refined = this.postPipeline!.apply(
        rawMask,
        this.processingWidth,
        this.processingHeight,
        this.sourceImageData!
      )
      this.writeMaskAlpha(refined)
      this.composite()
    } catch (e) {
      console.error('[AMP] process error', e)
    }
    this.timerWorker?.postMessage({ id: SET_TIMEOUT, timeMs: 1000 / 30 })
  }

  private _createMainCanvas() {
    let canvas = document.querySelector(
      `canvas#${BLUR_CANVAS_ID}`
    ) as HTMLCanvasElement | null
    if (!canvas) {
      canvas = this._createCanvas(
        BLUR_CANVAS_ID,
        this.sourceSettings!.width ?? 1280,
        this.sourceSettings!.height ?? 720
      )
    }
    this.outputCanvas = canvas
    this.outputCanvasCtx = canvas.getContext('2d')!
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
    this.timerWorker?.postMessage({ id: CLEAR_TIMEOUT })
    this.timerWorker?.terminate()
    this.timerWorker = undefined
    this.segmenter?.destroy()
    this.segmenter = undefined
    this.postPipeline?.reset()
  }
}
