// Asynchronous wrapper around MediaPipe's FaceLandmarker. Owns a single async
// loop that detects faces from a <video> at ~30 fps, stores the latest result
// in memory, and never blocks the render loop. Failures are surfaced through
// the matting error store; the runner never throws to its caller.

import {
  FaceLandmarker,
  type FaceLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import { getMediapipeFileset, probeMediapipeDelegate } from './segmenters/Segmenter'
import { pushMattingError } from './errors/MattingErrorStore'

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

const TARGET_INTERVAL_MS = 1000 / 30

export class FaceLandmarkerRunner {
  private _landmarker: FaceLandmarker | null = null
  private _videoElement: HTMLVideoElement | null = null
  private _latest: NormalizedLandmark[][] | null = null
  private _loopActive = false
  private _destroyed = false
  private _initPromise: Promise<void> | null = null
  private _lastTimestamp = -1

  async start(videoElement: HTMLVideoElement): Promise<void> {
    if (this._destroyed) return
    this._videoElement = videoElement
    if (!this._initPromise) {
      this._initPromise = this._init()
    }
    try {
      await this._initPromise
    } catch {
      // _init has already reported the error; bail out without throwing so the
      // matting pipeline keeps running without makeup.
      return
    }
    if (this._destroyed || this._loopActive) return
    this._loopActive = true
    void this._loop()
  }

  getLatestLandmarks(): NormalizedLandmark[][] | null {
    return this._latest
  }

  async stop(): Promise<void> {
    this._loopActive = false
    this._destroyed = true
    this._latest = null
    try {
      this._landmarker?.close()
    } catch {
      // Closing an already-disposed landmarker can throw on some MediaPipe
      // builds — ignore, we are tearing down.
    }
    this._landmarker = null
  }

  private async _init(): Promise<void> {
    try {
      const fileset = await getMediapipeFileset()
      const delegate = await probeMediapipeDelegate()
      this._landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate,
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      })
    } catch (e) {
      pushMattingError({
        code: 'FACELANDMARKER_INIT_FAILED',
        level: 'error',
        detail: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
      })
      throw e
    }
  }

  private async _loop(): Promise<void> {
    while (this._loopActive && !this._destroyed) {
      const t0 = performance.now()
      const video = this._videoElement
      const lm = this._landmarker
      if (!lm || !video || video.videoWidth === 0 || video.readyState < 2) {
        await new Promise<void>((r) => setTimeout(r, TARGET_INTERVAL_MS))
        continue
      }
      try {
        // detectForVideo requires strictly monotonic timestamps. performance.now
        // can return the same value twice in a tight loop, so bump it explicitly.
        let ts = performance.now()
        if (ts <= this._lastTimestamp) ts = this._lastTimestamp + 1
        this._lastTimestamp = ts
        const result: FaceLandmarkerResult = lm.detectForVideo(video, ts)
        if (!this._loopActive) return
        this._latest =
          result.faceLandmarks && result.faceLandmarks.length > 0
            ? (result.faceLandmarks as NormalizedLandmark[][])
            : null
      } catch (e) {
        if (!this._loopActive) return
        // Don't spam the error store on transient inference hiccups — log and
        // try again next tick.
        console.warn('[FaceLandmarkerRunner] detectForVideo failed', e)
      }
      const elapsed = performance.now() - t0
      await new Promise<void>((r) =>
        setTimeout(r, Math.max(0, TARGET_INTERVAL_MS - elapsed))
      )
    }
  }
}
