import { GpuRenderer } from './GpuRenderer'
import { DynamicLatencyEngine, STATIC_MODE_TABLE } from '../stats/DynamicLatencyEngine'
import { MaskMotionTracker } from '../preprocessing/MaskMotionTracker'
import { VideoFrameTracker } from '../preprocessing/VideoFrameTracker'
import { FrameMaskPair } from '../segmenters/SegmenterLoopRunner'
import { LatencyMode, MaskBlendMode } from '..'
import {
  pushGapSample,
  pushLatencySample,
  tickCameraFrame,
  tickRenderFrame,
  setEffectiveLatencyMode,
  setMotionScore,
  setMaskOffset,
  setPredictionActive,
} from '../stats/MattingStatsStore'

export class RenderLoopRunner {
  private videoElement?: HTMLVideoElement
  private outputCanvas?: HTMLCanvasElement
  private _renderLoopActive = false
  private _renderLoopHandle: number | null = null
  private _lastRenderedSeq = -1
  private _lastVideoTime = -1

  constructor(
    private getGpuRenderer: () => GpuRenderer | undefined,
    private getLatencyEngine: () => DynamicLatencyEngine,
    private getMotionTracker: () => MaskMotionTracker,
    private getFrameTracker: () => VideoFrameTracker,
    private getLatestPair: () => FrameMaskPair | null,
    private getPassthroughMask: (w: number, h: number) => Float32Array,
    private getLatencyParams: () => {
      latencyMode: LatencyMode
      latencyAuto: boolean
      maskPrediction: boolean
    },
    private getProcessingDimensions: () => { w: number; h: number }
  ) {}

  start(videoElement: HTMLVideoElement, outputCanvas: HTMLCanvasElement) {
    this.videoElement = videoElement
    this.outputCanvas = outputCanvas
    this._renderLoopActive = true
    this._lastRenderedSeq = -1
    this._lastVideoTime = -1
    this._scheduleRender()
  }

  stop() {
    this._renderLoopActive = false
    this._cancelRender()
    this.videoElement = undefined
    this.outputCanvas = undefined
  }

  private _scheduleRender(): void {
    if (!this._renderLoopActive) return
    const tracker = this.getFrameTracker()

    this._renderLoopHandle = requestAnimationFrame(() => {
      const hasRvfc = tracker.latestVideoFrameMeta !== undefined
      if (hasRvfc) {
        const seq = tracker.videoFrameSeq
        if (seq > this._lastRenderedSeq) {
          this._lastRenderedSeq = seq
          this._renderFrame()
        }
      } else {
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
    const gpuRenderer = this.getGpuRenderer()
    if (!gpuRenderer || !this.videoElement || this.videoElement.videoWidth === 0) {
      return
    }

    const pair = this.getLatestPair()
    if (!pair) {
      const vw = this.videoElement.videoWidth
      const vh = this.videoElement.videoHeight
      if (vw !== gpuRenderer.outW || vh !== gpuRenderer.outH) {
        gpuRenderer.resizeOutput(vw, vh)
      }
      gpuRenderer.setMaskOffset(0, 0)
      gpuRenderer.setBlendMix(0)
      setEffectiveLatencyMode(null)
      setMotionScore(0)
      setMaskOffset(0, 0)
      setPredictionActive(false)
      this._drawPassthrough()
      return
    }

    gpuRenderer.uploadMask(pair.mask, pair.procW, pair.procH)

    const motionTracker = this.getMotionTracker()
    const motionScore = motionTracker.isValid() ? motionTracker.getMotionScore() : 0
    setMotionScore(motionScore)

    const { latencyMode, latencyAuto, maskPrediction } = this.getLatencyParams()
    const latencyEngine = this.getLatencyEngine()

    let effectiveMode: MaskBlendMode
    let predictionGain: number
    let blendT = 0

    if (latencyAuto && motionTracker.isValid()) {
      effectiveMode = latencyEngine.resolveAutoMode(motionScore)
      if (maskPrediction && effectiveMode === 'frameLock') {
        effectiveMode = 'blend'
      }
      predictionGain =
        effectiveMode === 'live' ? 1.0 : effectiveMode === 'blend' ? 0.3 : 0
      blendT = latencyEngine.computeBlendT(
        latencyAuto,
        motionTracker.isValid(),
        motionScore,
        maskPrediction,
        effectiveMode,
        latencyMode,
        pair.captureTime
      )
    } else {
      const entry = STATIC_MODE_TABLE[latencyMode]
      effectiveMode = entry.mode
      predictionGain = entry.predictionGain
      blendT = latencyEngine.computeBlendT(
        latencyAuto,
        motionTracker.isValid(),
        motionScore,
        maskPrediction,
        effectiveMode,
        latencyMode,
        pair.captureTime
      )
    }
    latencyEngine.lastEffectiveMode = effectiveMode
    setEffectiveLatencyMode(effectiveMode)

    const velocity = motionTracker.isValid() ? motionTracker.getVelocityUv() : { vx: 0, vy: 0 }
    const { offsetU, offsetV, predictionWillRun } = latencyEngine.computePredictionOffset(
      maskPrediction,
      motionTracker.isValid(),
      effectiveMode,
      velocity,
      pair.cameraCaptureTime,
      predictionGain
    )

    gpuRenderer.setMaskOffset(offsetU, offsetV)
    setMaskOffset(offsetU, offsetV)
    setPredictionActive(predictionWillRun)
    gpuRenderer.setBlendMix(effectiveMode === 'blend' ? blendT : 0)

    if (effectiveMode === 'frameLock') {
      const sw = pair.source.width
      const sh = pair.source.height
      if (sw !== gpuRenderer.outW || sh !== gpuRenderer.outH) {
        gpuRenderer.resizeOutput(sw, sh)
      }
      gpuRenderer.render(pair.source)
    } else if (effectiveMode === 'live') {
      const vw = this.videoElement.videoWidth
      const vh = this.videoElement.videoHeight
      if (vw !== gpuRenderer.outW || vh !== gpuRenderer.outH) {
        gpuRenderer.resizeOutput(vw, vh)
      }
      gpuRenderer.render(this.videoElement)
    } else {
      const sw = pair.source.width
      const sh = pair.source.height
      if (sw !== gpuRenderer.outW || sh !== gpuRenderer.outH) {
        gpuRenderer.resizeOutput(sw, sh)
      }
      gpuRenderer.render(pair.source, this.videoElement)
    }

    const now = performance.now()
    const tracker = this.getFrameTracker()
    const liveCameraTime = tracker.latestVideoFrameMeta?.captureTime ?? now
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

  private _drawPassthrough() {
    const gpuRenderer = this.getGpuRenderer()
    if (!gpuRenderer || !this.videoElement) return
    const dims = this.getProcessingDimensions()
    const passthrough = this.getPassthroughMask(dims.w, dims.h)
    gpuRenderer.uploadMask(passthrough, dims.w, dims.h)
    gpuRenderer.render(this.videoElement)
  }
}
