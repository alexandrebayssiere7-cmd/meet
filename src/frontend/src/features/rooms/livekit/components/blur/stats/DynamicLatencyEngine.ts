import { LatencyMode, MaskBlendMode, ProcessorConfig } from '..'

export const FRAME_MS = 1000 / 30
export const BLEND_MODE_MAX_AGE_MS = FRAME_MS * 4

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

export const STATIC_MODE_TABLE: ReadonlyArray<{ mode: MaskBlendMode }> = [
  { mode: 'frameLock' }, // 0 Lock
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

  static getLatencyConfig(_opts: ProcessorConfig): { mode: LatencyMode } {
    return { mode: 0 }
  }

  /**
   * Compute the blend weight (blendT) for cross-fading when in "blend" mode.
   */
  computeBlendT(effectiveMode: MaskBlendMode, pairCaptureTime: number): number {
    if (effectiveMode !== 'blend') return 0
    const ageMs = performance.now() - pairCaptureTime
    return clamp01(ageMs / BLEND_MODE_MAX_AGE_MS)
  }
}
