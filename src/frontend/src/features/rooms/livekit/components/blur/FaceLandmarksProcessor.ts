import { ProcessorOptions, Track, TrackProcessor } from 'livekit-client'
import posthog from 'posthog-js'
import {
  FilesetResolver,
  FaceLandmarker,
  FaceLandmarkerResult,
} from '@mediapipe/tasks-vision'
import {
  CLEAR_TIMEOUT,
  SET_TIMEOUT,
  TIMEOUT_TICK,
  timerWorkerScript,
} from './TimerWorker'
import { ProcessorType } from '.'

const PROCESSING_WIDTH = 256 * 3
const PROCESSING_HEIGHT = 144 * 3

const FACE_LANDMARKS_CANVAS_ID = 'face-landmarks-local'

/**
 * Configuration options for the FaceLandmarksProcessor.
 */
export type FaceLandmarksOptions = {
  /** If true, overlay glasses on the user's eyes. */
  showGlasses: boolean
  /** If true, overlay a beret and a mustache (French theme) on the user's face. */
  showFrench: boolean
}

/**
 * TrackProcessor that overlays 2D asset decals (glasses, beret, moustache)
 * onto a local video track using MediaPipe FaceLandmaker landmarks.
 */
export class FaceLandmarksProcessor implements TrackProcessor<Track.Kind> {
  /** Active options (which assets to overlay). */
  options: FaceLandmarksOptions
  /** Unique name of the processor. */
  name: string
  /** Output video track containing the rendered effects canvas. */
  processedTrack?: MediaStreamTrack | undefined

  /** Original input video track. */
  source?: MediaStreamTrack
  /** Settings of the input video track (width, height, etc.). */
  sourceSettings?: MediaTrackSettings
  /** The source HTML5 <video> element. */
  videoElement?: HTMLVideoElement
  /** True when the video element has loaded and started playing. */
  videoElementLoaded?: boolean

  /** Offscreen Canvas containing the final processed result. */
  outputCanvas?: HTMLCanvasElement
  /** Context 2D of the output canvas. */
  outputCanvasCtx?: CanvasRenderingContext2D

  /** Instance of MediaPipe FaceLandmarker for face tracking. */
  faceLandmarker?: FaceLandmarker
  /** Last detected face landmark coordinates. */
  faceLandmarkerResult?: FaceLandmarkerResult

  /** Current frame's resized raw image data used as model input. */
  sourceImageData?: ImageData

  /** Dedicated worker for custom timing loops (avoiding background tab throttling). */
  timerWorker?: Worker

  /** Discriminated processor type. */
  type: ProcessorType

  /** Glasses decal image element. */
  glassesImage?: HTMLImageElement
  /** Mustache decal image element. */
  mustacheImage?: HTMLImageElement
  /** Beret decal image element. */
  beretImage?: HTMLImageElement

  /**
   * Creates an instance of FaceLandmarksProcessor.
   */
  constructor(opts: FaceLandmarksOptions) {
    this.name = 'face_landmarks'
    this.options = opts
    this.type = ProcessorType.FACE_LANDMARKS
    this._initEffectImages()
  }

  /**
   * Preload all effect decal images (glasses, mustache, beret) in the background
   * so they are ready when the first frame is processed.
   */
  private _initEffectImages() {
    this.glassesImage = new Image()
    this.glassesImage.src = '/assets/glasses.png'
    this.glassesImage.crossOrigin = 'anonymous'

    this.mustacheImage = new Image()
    this.mustacheImage.src = '/assets/mustache.png'
    this.mustacheImage.crossOrigin = 'anonymous'

    this.beretImage = new Image()
    this.beretImage.src = '/assets/beret.png'
    this.beretImage.crossOrigin = 'anonymous'
  }

  static get isSupported() {
    return true // Face landmarks should work in all modern browsers
  }

  /**
   * Initializes the processor with the video track options and media elements.
   * Sets up the canvas, face landmarker model, and custom timer worker.
   */
  async init(opts: ProcessorOptions<Track.Kind>) {
    if (!opts.element) {
      throw new Error('Element is required for processing')
    }

    this.source = opts.track as MediaStreamTrack
    this.sourceSettings = this.source!.getSettings()
    this.videoElement = opts.element as HTMLVideoElement

    this._createMainCanvas()

    const stream = this.outputCanvas!.captureStream()
    const tracks = stream.getVideoTracks()
    if (tracks.length == 0) {
      throw new Error('No tracks found for processing')
    }
    this.processedTrack = tracks[0]

    await this.initFaceLandmarker()
    this._initWorker()

    posthog.capture('face-landmarks-init')
  }

  /**
   * Create the `TimerWorker` Web Worker and hook it up to the processing loop.
   * The worker fires a `TIMEOUT_TICK` message every `1000/30` ms (~33 ms)
   * regardless of whether the tab is in the background, avoiding the browser's
   * throttling of `setInterval` in inactive tabs.
   * If the video element is already loaded, the timer starts immediately;
   * otherwise it waits for the `loadeddata` event.
   */
  _initWorker() {
    this.timerWorker = new Worker(timerWorkerScript, {
      name: 'FaceLandmarks',
    })
    this.timerWorker.onmessage = (data) => this.onTimerMessage(data)
    if (this.videoElementLoaded) {
      this.timerWorker!.postMessage({
        id: SET_TIMEOUT,
        timeMs: 1000 / 30,
      })
    } else {
      this.videoElement!.onloadeddata = () => {
        this.videoElementLoaded = true
        this.timerWorker!.postMessage({
          id: SET_TIMEOUT,
          timeMs: 1000 / 30,
        })
      }
    }
  }

  /**
   * Handle messages from the `TimerWorker`. Dispatches a new processing tick
   * on every `TIMEOUT_TICK` message, which fires at ~30 fps.
   *
   * @param response Worker message event containing `data.id`.
   */
  onTimerMessage(response: { data: { id: number } }) {
    if (response.data.id === TIMEOUT_TICK) {
      this.process()
    }
  }

  /**
   * Download and initialise the MediaPipe FaceLandmarker model in VIDEO mode.
   * Requests GPU delegate and enables both face blend-shapes and facial
   * transformation matrices outputs.
   */
  async initFaceLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
    )
    this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    })
  }

  /**
   * Draw the current video frame onto the output canvas at the processing
   * resolution (768×432) and capture it as `sourceImageData` for use by
   * the face landmark detector.
   */
  async sizeSource() {
    this.outputCanvasCtx?.drawImage(
      this.videoElement!,
      0,
      0,
      this.videoElement!.videoWidth,
      this.videoElement!.videoHeight,
      0,
      0,
      PROCESSING_WIDTH,
      PROCESSING_HEIGHT
    )

    this.sourceImageData = this.outputCanvasCtx?.getImageData(
      0,
      0,
      PROCESSING_WIDTH,
      PROCESSING_HEIGHT
    )
  }

  /**
   * Run face landmark detection on the current `sourceImageData` using the
   * MediaPipe FaceLandmarker in VIDEO mode. Stores the result in
   * `faceLandmarkerResult` for use by `drawFaceLandmarks()`.
   */
  async detectFaces() {
    const startTimeMs = performance.now()
    this.faceLandmarkerResult = this.faceLandmarker!.detectForVideo(
      this.sourceImageData!,
      startTimeMs
    )
  }

  /**
   * Draw a decal image centred and rotated between two facial landmark points.
   *
   * The image is scaled proportionally to the Euclidean distance between the
   * two anchor points, then rotated to match their orientation angle.
   *
   * @param leftPoint   Normalised [0,1] coordinates of the left anchor landmark.
   * @param rightPoint  Normalised [0,1] coordinates of the right anchor landmark.
   * @param image       Decal `<img>` element to draw.
   * @param widthScale  Multiplier applied to the inter-point distance to get the draw width.
   * @param heightScale Height-to-width ratio of the decal (preserves aspect ratio).
   * @param yOffset     Optional vertical offset in normalised coordinates (default 0).
   */
  private drawEffect(
    leftPoint: { x: number; y: number },
    rightPoint: { x: number; y: number },
    image: HTMLImageElement,
    widthScale: number,
    heightScale: number,
    yOffset: number = 0
  ) {
    // Calculate distance between points
    const distance = Math.sqrt(
      Math.pow(rightPoint.x - leftPoint.x, 2) +
        Math.pow(rightPoint.y - leftPoint.y, 2)
    )

    // Scale image based on distance
    const width = distance * PROCESSING_WIDTH * widthScale
    const height = width * heightScale

    // Calculate center position between points
    const centerX = (leftPoint.x + rightPoint.x) / 2
    const centerY = (leftPoint.y + rightPoint.y) / 2 + yOffset

    // Draw image
    this.outputCanvasCtx!.save()
    this.outputCanvasCtx!.translate(
      centerX * PROCESSING_WIDTH,
      centerY * PROCESSING_HEIGHT
    )

    // Calculate rotation angle based on point positions
    const angle = Math.atan2(
      rightPoint.y - leftPoint.y,
      rightPoint.x - leftPoint.x
    )
    this.outputCanvasCtx!.rotate(angle)

    // Draw image centered at the midpoint between points
    this.outputCanvasCtx!.drawImage(
      image,
      -width / 2,
      -height / 2,
      width,
      height
    )

    this.outputCanvasCtx!.restore()
  }

  /**
   * Composite one processed frame: draws the video frame at processing resolution,
   * then overlays the enabled decals (glasses and/or French set) on every detected
   * face using the stored `faceLandmarkerResult` landmark coordinates.
   */
  async drawFaceLandmarks() {
    // Draw the original video frame at the canvas size
    this.outputCanvasCtx!.drawImage(
      this.videoElement!,
      0,
      0,
      this.videoElement!.videoWidth,
      this.videoElement!.videoHeight,
      0,
      0,
      PROCESSING_WIDTH,
      PROCESSING_HEIGHT
    )

    if (!this.faceLandmarkerResult?.faceLandmarks) {
      return
    }

    // Draw face landmarks (optional, for debugging)
    this.outputCanvasCtx!.strokeStyle = '#00FF00'
    this.outputCanvasCtx!.lineWidth = 2

    for (const face of this.faceLandmarkerResult.faceLandmarks) {
      // Find eye landmarks
      const leftEye = face[468]
      const rightEye = face[473]

      // Find mouth landmarks for mustache
      const leftMoustache = face[92]
      const rightMoustache = face[322]

      // Find forehead landmarks for beret
      const leftForehead = face[103]
      const rightForehead = face[332]

      if (leftEye && rightEye && this.options.showGlasses) {
        this.drawEffect(leftEye, rightEye, this.glassesImage!, 2.5, 0.7)
      }

      if (leftMoustache && rightMoustache && this.options.showFrench) {
        this.drawEffect(
          leftMoustache,
          rightMoustache,
          this.mustacheImage!,
          1.5,
          0.5
        )
      }

      if (leftForehead && rightForehead && this.options.showFrench) {
        this.drawEffect(
          leftForehead,
          rightForehead,
          this.beretImage!,
          2.1,
          0.7,
          -0.1
        )
      }
    }
  }

  /**
   * The core processing tick. Extracts the video frame, runs face landmark
   * detection, draws overlays, and schedules the next tick via the timer worker.
   */
  async process() {
    await this.sizeSource()
    await this.detectFaces()
    await this.drawFaceLandmarks()

    this.timerWorker!.postMessage({
      id: SET_TIMEOUT,
      timeMs: 1000 / 30,
    })
  }

  /**
   * Find or create the output canvas element for face landmark compositing.
   * Re-uses an existing DOM canvas with the matching ID to avoid duplicates
   * across hot-reloads or processor restarts.
   */
  _createMainCanvas() {
    this.outputCanvas = document.querySelector(
      `#${FACE_LANDMARKS_CANVAS_ID}`
    ) as HTMLCanvasElement
    if (!this.outputCanvas) {
      this.outputCanvas = this._createCanvas(
        FACE_LANDMARKS_CANVAS_ID,
        PROCESSING_WIDTH,
        PROCESSING_HEIGHT
      )
    }
    this.outputCanvasCtx = this.outputCanvas.getContext('2d')!
  }

  /**
   * Create a new `<canvas>` element with the given ID and dimensions.
   *
   * @param id     HTML `id` attribute to assign.
   * @param width  Canvas width in pixels.
   * @param height Canvas height in pixels.
   * @returns      The created (but not yet attached) canvas element.
   */
  _createCanvas(id: string, width: number, height: number) {
    const element = document.createElement('canvas')
    element.setAttribute('id', id)
    element.setAttribute('width', '' + width)
    element.setAttribute('height', '' + height)
    return element
  }

  /**
   * Update which decals are displayed without restarting the processor.
   * Takes effect on the next processed frame.
   *
   * @param opts New decal visibility options.
   */
  update(opts: FaceLandmarksOptions): void {
    this.options = opts
  }

  /**
   * Fully restart the processor: destroys the current instance and re-initialises
   * it with the new track options. Useful when the source track changes.
   *
   * @param opts New processor options (track + element).
   */
  async restart(opts: ProcessorOptions<Track.Kind>) {
    await this.destroy()
    return this.init(opts)
  }

  /**
   * Stops the timer worker and releases the MediaPipe FaceLandmarker model resources.
   */
  async destroy() {
    this.timerWorker?.postMessage({
      id: CLEAR_TIMEOUT,
    })

    this.timerWorker?.terminate()
    this.faceLandmarker?.close()
  }
}
