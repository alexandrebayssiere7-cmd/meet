/**
 * Temporal Exponential Moving Average on a Float32 mask.
 *   out = alpha * current + (1 - alpha) * previous
 * alpha closer to 1 → reactive, closer to 0 → smoother but laggy.
 * Resets automatically if the mask size changes.
 */
export class TemporalEMA {
  private prev?: Float32Array

  constructor(private alpha: number) {}

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

  reset(): void {
    this.prev = undefined
  }
}
