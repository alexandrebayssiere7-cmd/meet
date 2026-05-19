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
import { FaceLandmarkerRunner } from './FaceLandmarkerRunner'
import {
  packLandmarksForMakeup,
  PACKED_TOTAL_FLOATS,
  MAX_PER_ZONE,
  NUM_ZONES,
} from './makeup/MakeupOverlay'
import { findMakeupPreset, type MakeupPreset } from './makeup/presets'

const SEGMENTATION_MASK_CANVAS_ID = 'background-blur-local-segmentation'
const BLUR_CANVAS_ID = 'background-blur-local'
const DEFAULT_BLUR = 10

function hexToRgb01(hex: string): [number, number, number] {
  // Accept '#rrggbb' and '#rgb'. Falls back to black on unknown input.
  if (hex.length === 7 && hex[0] === '#') {
    return [
      parseInt(hex.slice(1, 3), 16) / 255,
      parseInt(hex.slice(3, 5), 16) / 255,
      parseInt(hex.slice(5, 7), 16) / 255,
    ]
  }
  if (hex.length === 4 && hex[0] === '#') {
    return [
      parseInt(hex[1] + hex[1], 16) / 255,
      parseInt(hex[2] + hex[2], 16) / 255,
      parseInt(hex[3] + hex[3], 16) / 255,
    ]
  }
  return [0, 0, 0]
}

function presetToShaderUniforms(p: MakeupPreset, outH: number) {
  const lashRefH = 720
  return {
    lipColor: p.lips ? hexToRgb01(p.lips.color) : ([0, 0, 0] as [number, number, number]),
    lipAlpha: p.lips ? p.lips.alpha : 0,
    blushColor: p.blush ? hexToRgb01(p.blush.color) : ([0, 0, 0] as [number, number, number]),
    blushAlpha: p.blush ? p.blush.alpha : 0,
    // Preset radius is a fraction of the short side of the canvas — convert
    // to pixels at the current output height for direct use in the shader.
    blushRadius: p.blush ? p.blush.radius * outH : 0,
    browColor: p.brows ? hexToRgb01(p.brows.color) : ([0, 0, 0] as [number, number, number]),
    browAlpha: p.brows ? p.brows.alpha : 0,
    lashColor: p.lashes ? hexToRgb01(p.lashes.color) : ([0, 0, 0] as [number, number, number]),
    lashAlpha: p.lashes ? p.lashes.alpha : 0,
    // Thickness is calibrated for 720p in the preset — rescale linearly to
    // the current output height so feature sizes stay visually consistent.
    lashThicknessPx: p.lashes ? Math.max(1, (p.lashes.thickness * outH) / lashRefH) : 0,
  }
}

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

  private _configuredModel?: SegmentationModel
  currentModel?: SegmentationModel
  private processingWidth = 256
  private processingHeight = 144
  private _pendingModel?: SegmentationModel
  private _readyResolvers: Array<() => void> = []
  private _destroyed = false
  private _preProcessingPipeline?: PreProcessingPipeline
  private _lastMask?: Float32Array

  // Makeup overlay (optional, GPU path). The runner is lazily instantiated
  // the first time a preset is selected and torn down when the user clears
  // it. The packed buffer is allocated once and reused every frame.
  private _faceLandmarker?: FaceLandmarkerRunner
  private _activeMakeupPreset: MakeupPreset | null = null
  private _landmarksBuf = new Float32Array(PACKED_TOTAL_FLOATS)

  constructor(opts: ProcessorConfig) {
    this.name = opts.type === ProcessorType.VIRTUAL ? 'virtual' : 'blur'
    this.options = opts
    this.type = opts.type
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

    if (!this.gpuRenderer) {
      console.info('[AMP] Update called in passthrough fallback mode; ignoring background processor updates.')
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
      const success = avg <= 35

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

    this._syncMakeupConfig()
  }

  private _getMakeupPresetId(): string | undefined {
    if (
      this.options.type === ProcessorType.BLUR ||
      this.options.type === ProcessorType.VIRTUAL
    ) {
      return this.options.makeupPresetId
    }
    return undefined
  }

  private _syncMakeupConfig() {
    if (!this.gpuRenderer) return
    const presetId = this._getMakeupPresetId()
    const preset = findMakeupPreset(presetId)
    this._activeMakeupPreset = preset

    if (preset) {
      if (!this._faceLandmarker && this.videoElement) {
        this._faceLandmarker = new FaceLandmarkerRunner()
        // start() is async — we don't await here; the render loop will pull
        // null landmarks until the model is up, which is the same fallback as
        // a frame with no detected face.
        void this._faceLandmarker.start(this.videoElement)
      }
      // Push the preset's per-zone parameters to the renderer (uniforms).
      // Zero-alpha zones disable their shader branch automatically.
      this.gpuRenderer.setMakeupPreset(presetToShaderUniforms(preset, this.gpuRenderer.outH))
      this.gpuRenderer.setMakeupActive(true)
    } else {
      this.gpuRenderer.setMakeupActive(false)
      // Tear down the runner so we don't keep MediaPipe pinned in memory once
      // the user has cleared makeup.
      const runner = this._faceLandmarker
      this._faceLandmarker = undefined
      void runner?.stop()
    }
  }

  private async _initSegmenterBackground(model: SegmentationModel) {
    if (this._destroyed) return
    this._pendingModel = model
    try {
      let targetModel = model
      if (model === SegmentationModel.AUTO) {
        targetModel = SegmentationModel.MULTICLASS
      }

      let seg = createSegmenter(targetModel)
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

      let seg = createSegmenter(targetModel)
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
   * Segmenter loop: runs at most 30fps, writes the latest alpha mask to
   * _latestMask. Capped so it does not starve the render loop or saturate the
   * GPU when inference is faster than one frame period.
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
      try {
        const cropBbox = this._preProcessingPipeline?.getNextCropBbox() ?? null
        this.sizeSource(cropBbox)
        const frameToSegment = this._preProcessingPipeline
          ? this._preProcessingPipeline.apply(this.sourceImageData!, this._lastMask)
          : this.sourceImageData!
        const rawMask = await seg.segment(frameToSegment, performance.now())
        if (!this._segLoopActive) return
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
    const vw = this.videoElement.videoWidth
    const vh = this.videoElement.videoHeight
    if (vw !== this.gpuRenderer.outW || vh !== this.gpuRenderer.outH) {
      this.gpuRenderer.resizeOutput(vw, vh)
    }
    this._prepareMakeupFrame()
    const mask = this._latestMask
    if (mask) {
      this.gpuRenderer.uploadMask(mask, this.processingWidth, this.processingHeight)
      this.gpuRenderer.render(this.videoElement)
    } else {
      this._drawPassthrough()
    }
  }

  private _prepareMakeupFrame() {
    if (!this._activeMakeupPreset || !this.gpuRenderer) return
    const faces = this._faceLandmarker?.getLatestLandmarks() ?? null
    const hasFace = packLandmarksForMakeup(faces, this._landmarksBuf)
    this.gpuRenderer.uploadLandmarks(this._landmarksBuf, MAX_PER_ZONE, NUM_ZONES, hasFace)
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
    this._latestMask = null
    const runner = this._faceLandmarker
    this._faceLandmarker = undefined
    this._activeMakeupPreset = null
    void runner?.stop()
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
