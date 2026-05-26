/**
 * Temporal Exponential Moving Average on a Float32 mask.
 *   out = alpha * current + (1 - alpha) * previous
 * alpha closer to 1 → reactive, closer to 0 → smoother but laggy.
 * Resets automatically if the mask size changes.
 */
export class TemporalEMA {
  private prev?: Float32Array

  /**
   * @param alpha Smoothing factor in (0, 1].
   *              1.0 → no smoothing (output = current input).
   *              Values closer to 0 → stronger smoothing, more temporal lag.
   */
  constructor(private alpha: number) {}

  /**
   * Apply EMA smoothing to the current mask.
   * On the first call (or after a reset / size change), the previous frame is
   * initialised to the current mask so there is no cold-start artefact.
   *
   * @param mask Current frame mask in [0, 1] (Float32Array, length = W*H).
   * @returns    Smoothed mask as a new Float32Array.
   */
  apply(mask: Float32Array): Float32Array {
    if (!this.prev || this.prev.length !== mask.length) {
      this.prev = new Float32Array(mask)
      return new Float32Array(mask)
    }
    const a = this.alpha
    const inv = 1 - a
    const out = new Float32Array(mask.length)
    for (let i = 0; i < mask.length; i++) {
      out[i] = a * mask[i] + inv * this.prev[i]
    }
    this.prev = out
    return out
  }

  /**
   * Discard the stored previous frame so the next `apply()` call starts fresh.
   * Call this when the segmenter model or processing resolution changes.
   */
  reset(): void {
    this.prev = undefined
  }
}
