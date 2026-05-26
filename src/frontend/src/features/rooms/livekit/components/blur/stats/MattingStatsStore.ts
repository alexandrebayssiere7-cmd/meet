import { proxy, useSnapshot } from 'valtio'
import { MaskBlendMode, SegmentationModel } from '..'

/**
 * Live stats published by the matting processor for diagnostic UI (HUD +
 * EffectsConfiguration panel). Ring buffers are kept outside the proxy and
 * only aggregated values (means) are pushed into valtio, so we don't trigger
 * a re-render on every render-loop tick (~50 Hz).
 */
export interface CameraSettings {
  frameRateRequested: number | null   // ce qu'on a demandé (ideal)
  frameRateActual: number | null      // ce que getSettings() rapporte
  frameRateMax: number | null         // max hardware via getCapabilities()
  width: number | null
  height: number | null
}

export interface MattingStatsState {
  active: boolean
  configuredModel: SegmentationModel | null
  currentModel: SegmentationModel | null
  captureToDisplayLatencyMs: number
  maskFrameGapMs: number
  segmenterInferenceMs: number
  renderFps: number
  segmenterFps: number
  cameraFps: number
  cameraSettings: CameraSettings | null
  samples: number
  motionScoreUvPerSec: number
  effectiveLatencyMode: MaskBlendMode | null
  maskOffsetUv: { u: number; v: number }
  predictionActive: boolean
  segmenterFrameSkip: number
}

export const mattingStatsStore = proxy<MattingStatsState>({
  active: false,
  configuredModel: null,
  currentModel: null,
  captureToDisplayLatencyMs: 0,
  maskFrameGapMs: 0,
  segmenterInferenceMs: 0,
  renderFps: 0,
  segmenterFps: 0,
  cameraFps: 0,
  cameraSettings: null,
  samples: 0,
  motionScoreUvPerSec: 0,
  effectiveLatencyMode: null,
  maskOffsetUv: { u: 0, v: 0 },
  predictionActive: false,
  segmenterFrameSkip: 2,
})

const BUF_SIZE = 50

class RingBuffer {
  private buf = new Float32Array(BUF_SIZE)
  private idx = 0
  private count = 0
  private sum = 0

  push(v: number): number {
    if (!Number.isFinite(v)) return this.mean()
    if (this.count === BUF_SIZE) {
      this.sum -= this.buf[this.idx]
    } else {
      this.count++
    }
    this.buf[this.idx] = v
    this.sum += v
    this.idx = (this.idx + 1) % BUF_SIZE
    return this.mean()
  }

  mean(): number {
    return this.count === 0 ? 0 : this.sum / this.count
  }

  reset(): void {
    this.idx = 0
    this.count = 0
    this.sum = 0
  }

  get size(): number {
    return this.count
  }
}

const latencyBuf = new RingBuffer()
const gapBuf = new RingBuffer()
const inferenceBuf = new RingBuffer()

// FPS counters: accumulate ticks over a 1s sliding window, then publish.
const FPS_WINDOW_MS = 1000
let renderTicks = 0
let segTicks = 0
let fpsWindowStart = 0

let cameraTicks = 0

function tickFps(kind: 'render' | 'segmenter' | 'camera'): void {
  const now = performance.now()
  if (fpsWindowStart === 0) fpsWindowStart = now
  if (kind === 'render') renderTicks++
  else if (kind === 'segmenter') segTicks++
  else cameraTicks++
  const elapsed = now - fpsWindowStart
  if (elapsed >= FPS_WINDOW_MS) {
    mattingStatsStore.renderFps = (renderTicks * 1000) / elapsed
    mattingStatsStore.segmenterFps = (segTicks * 1000) / elapsed
    mattingStatsStore.cameraFps = (cameraTicks * 1000) / elapsed
    renderTicks = 0
    segTicks = 0
    cameraTicks = 0
    fpsWindowStart = now
  }
}

function resetFps(): void {
  renderTicks = 0
  segTicks = 0
  cameraTicks = 0
  fpsWindowStart = 0
  mattingStatsStore.renderFps = 0
  mattingStatsStore.segmenterFps = 0
  mattingStatsStore.cameraFps = 0
}

// Throttle valtio writes: aggregate samples in the ring buffer at full
// 50 Hz cadence but only flush the mean into the proxy ~5×/s.
const FLUSH_INTERVAL_MS = 200
let lastFlush = 0
let pendingFlush = false

function maybeFlush(force = false): void {
  const now = performance.now()
  if (!force && now - lastFlush < FLUSH_INTERVAL_MS) {
    pendingFlush = true
    return
  }
  lastFlush = now
  pendingFlush = false
  mattingStatsStore.captureToDisplayLatencyMs = latencyBuf.mean()
  mattingStatsStore.maskFrameGapMs = gapBuf.mean()
  mattingStatsStore.segmenterInferenceMs = inferenceBuf.mean()
  mattingStatsStore.samples = latencyBuf.size
}

export function pushLatencySample(ms: number): void {
  latencyBuf.push(ms)
  maybeFlush()
}

export function pushGapSample(ms: number): void {
  gapBuf.push(ms)
  maybeFlush()
}

export function pushInferenceSample(ms: number): void {
  inferenceBuf.push(ms)
  maybeFlush()
}

export function tickRenderFrame(): void {
  tickFps('render')
}

export function tickSegmenterFrame(): void {
  tickFps('segmenter')
}

export function tickCameraFrame(): void {
  tickFps('camera')
}

export function setMattingStatsModel(
  configured: SegmentationModel | null,
  current: SegmentationModel | null
): void {
  mattingStatsStore.configuredModel = configured
  mattingStatsStore.currentModel = current
}

export function setMattingStatsActive(active: boolean): void {
  mattingStatsStore.active = active
  if (active) {
    // Ensure stale buffers don't bleed across activations.
    latencyBuf.reset()
    gapBuf.reset()
    inferenceBuf.reset()
    resetFps()
    lastFlush = 0
  } else if (pendingFlush) {
    maybeFlush(true)
  }
}

export function setCameraSettings(settings: CameraSettings): void {
  mattingStatsStore.cameraSettings = settings
}

export function setMotionScore(uvPerSec: number): void {
  mattingStatsStore.motionScoreUvPerSec = Number.isFinite(uvPerSec) ? uvPerSec : 0
}

export function setEffectiveLatencyMode(mode: MaskBlendMode | null): void {
  mattingStatsStore.effectiveLatencyMode = mode
}

export function setMaskOffset(u: number, v: number): void {
  mattingStatsStore.maskOffsetUv = {
    u: Number.isFinite(u) ? u : 0,
    v: Number.isFinite(v) ? v : 0,
  }
}

export function setPredictionActive(active: boolean): void {
  mattingStatsStore.predictionActive = active
}

export function setSegmenterFrameSkip(skip: number): void {
  mattingStatsStore.segmenterFrameSkip = skip
}

export function resetMattingStats(): void {
  latencyBuf.reset()
  gapBuf.reset()
  inferenceBuf.reset()
  resetFps()
  mattingStatsStore.active = false
  mattingStatsStore.configuredModel = null
  mattingStatsStore.currentModel = null
  mattingStatsStore.captureToDisplayLatencyMs = 0
  mattingStatsStore.maskFrameGapMs = 0
  mattingStatsStore.segmenterInferenceMs = 0
  mattingStatsStore.cameraSettings = null
  mattingStatsStore.samples = 0
  mattingStatsStore.motionScoreUvPerSec = 0
  mattingStatsStore.effectiveLatencyMode = null
  mattingStatsStore.maskOffsetUv = { u: 0, v: 0 }
  mattingStatsStore.predictionActive = false
  mattingStatsStore.segmenterFrameSkip = 2
  lastFlush = 0
  pendingFlush = false
}

export function useMattingStats(): MattingStatsState {
  return useSnapshot(mattingStatsStore) as MattingStatsState
}
