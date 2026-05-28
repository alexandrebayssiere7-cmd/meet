import { LatencyMode, MaskBlendMode, ProcessorConfig, ProcessorType } from '..'

// Auto-tuning thresholds for the latency/halo trade-off (uv per second).
export const AUTO_LOCK_THRESHOLD = 0.1
export const AUTO_LIVE_THRESHOLD = 0.6
export const AUTO_HYSTERESIS = 0.05
export const AUTO_PRED_BLEND_BASELINE = 0.5

// Hard caps for the mask warp prediction so a noisy velocity never produces
// visible halos. 0.08 uv ≈ 8% of the frame width.
export const MAX_PREDICTION_OFFSET_UV = 0.08
export const FRAME_MS = 1000 / 30
export const BLEND_MODE_MAX_AGE_MS = FRAME_MS * 4
export const DEFAULT_LATENCY_MODE: LatencyMode = 2

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

function clampOffset(v: number): number {
  if (v > MAX_PREDICTION_OFFSET_UV) return MAX_PREDICTION_OFFSET_UV
  if (v < -MAX_PREDICTION_OFFSET_UV) return -MAX_PREDICTION_OFFSET_UV
  return v
}

// Mapping from the user-facing LatencyMode (0..4) to the internal effective
// blend mode + prediction gain. Used when `latencyAuto` is false.
export const STATIC_MODE_TABLE: ReadonlyArray<{
  mode: MaskBlendMode
  predictionGain: number
}> = [
  { mode: 'frameLock', predictionGain: 0 }, // 0 Lock
  { mode: 'frameLock', predictionGain: 0 }, // 1 Stable (handled separately by EMA boost)
  { mode: 'blend', predictionGain: 0 }, // 2 Équilibré
  { mode: 'live', predictionGain: 0.5 }, // 3 Réactif
  { mode: 'live', predictionGain: 1.0 }, // 4 Live
]

export class DynamicLatencyEngine {
  private _lastEffectiveMode: MaskBlendMode = 'frameLock'

  get lastEffectiveMode(): MaskBlendMode {
    return this._lastEffectiveMode
  }

  set lastEffectiveMode(mode: MaskBlendMode) {
    this._lastEffectiveMode = mode
  }

  reset() {
    this._lastEffectiveMode = 'frameLock'
  }

  /**
   * Resolve the latency/halo controls from the processor config, applying the
   * "auto + prediction require ROI cropping" guard rail. When ROI cropping is
   * off the auto-tuning and prediction features have no motion signal to act on,
   * so we force-disable them at the source rather than rely on runtime checks.
   */
  static getLatencyConfig(opts: ProcessorConfig): {
    mode: LatencyMode
    auto: boolean
    prediction: boolean
  } {
    if (
      opts.type !== ProcessorType.BLUR &&
      opts.type !== ProcessorType.VIRTUAL
    ) {
      return { mode: DEFAULT_LATENCY_MODE, auto: false, prediction: false }
    }
    const rawMode = opts.latencyMode
    const mode: LatencyMode =
      rawMode === 0 ||
      rawMode === 1 ||
      rawMode === 2 ||
      rawMode === 3 ||
      rawMode === 4
        ? rawMode
        : DEFAULT_LATENCY_MODE
    const roiEnabled = opts.preProcessing?.roiCropping?.enabled === true
    return {
      mode,
      auto: roiEnabled && opts.latencyAuto !== false,
      prediction: roiEnabled && opts.maskPrediction === true,
    }
  }

  /**
   * Apply the auto-tuning thresholds with hysteresis around `_lastEffectiveMode`
   * so the resolved mode doesn't flap when the motion score sits right on a
   * boundary. The hysteresis band is asymmetric — leaving a mode requires
   * crossing a slightly stricter threshold than entering it.
   */
  resolveAutoMode(motionScore: number): MaskBlendMode {
    const lock = AUTO_LOCK_THRESHOLD
    const live = AUTO_LIVE_THRESHOLD
    const h = AUTO_HYSTERESIS
    const prev = this._lastEffectiveMode
    if (motionScore < lock - h) return 'frameLock'
    if (motionScore > live + h) return 'live'
    if (motionScore < lock + h && prev === 'frameLock') return 'frameLock'
    if (motionScore > live - h && prev === 'live') return 'live'
    return 'blend'
  }

  /**
   * Compute prediction offset (uv coords). Only applied when the user has
   * enabled it AND we are reading from a live frame (frame-locked composite
   * doesn't benefit — the mask already matches the displayed pixels).
   */
  computePredictionOffset(
    maskPredictionEnabled: boolean,
    motionTrackerValid: boolean,
    effectiveMode: MaskBlendMode,
    velocity: { vx: number; vy: number },
    pairCameraCaptureTime: number,
    predictionGain: number
  ): { offsetU: number; offsetV: number; predictionWillRun: boolean } {
    const predictionWillRun =
      maskPredictionEnabled &&
      predictionGain > 0 &&
      motionTrackerValid &&
      effectiveMode !== 'frameLock'

    let offsetU = 0
    let offsetV = 0
    if (predictionWillRun) {
      const predictionDt_s = (performance.now() - pairCameraCaptureTime) / 1000
      offsetU = clampOffset(velocity.vx * predictionDt_s * predictionGain)
      offsetV = clampOffset(velocity.vy * predictionDt_s * predictionGain)
    }

    return { offsetU, offsetV, predictionWillRun }
  }

  /**
   * Compute the blend weight (blendT) for cross-fading when in "blend" mode.
   */
  computeBlendT(
    latencyAuto: boolean,
    motionTrackerValid: boolean,
    motionScore: number,
    maskPredictionEnabled: boolean,
    effectiveMode: MaskBlendMode,
    latencyMode: LatencyMode,
    pairCaptureTime: number
  ): number {
    if (effectiveMode !== 'blend') {
      return 0
    }

    if (latencyAuto && motionTrackerValid) {
      const span = AUTO_LIVE_THRESHOLD - AUTO_LOCK_THRESHOLD
      const motionBlend = clamp01((motionScore - AUTO_LOCK_THRESHOLD) / span)
      const baseline = maskPredictionEnabled ? AUTO_PRED_BLEND_BASELINE : 0
      return clamp01(baseline + (1 - baseline) * motionBlend)
    } else {
      const ageMs = performance.now() - pairCaptureTime
      return clamp01(ageMs / BLEND_MODE_MAX_AGE_MS)
    }
  }
}
