import { BBox } from './RoiCropper'

const VELOCITY_EMA_ALPHA = 0.3
const MIN_DT_S = 0.005
const MAX_DT_S = 0.2
const TELEPORT_THRESHOLD_UV = 0.25
const MAX_VELOCITY_UV_PER_SEC = 2.5

/**
 * Tracks the 2D velocity of the person's centroid (derived from the stabilised
 * RoiCropper bbox) in normalised uv coordinates per second. Used to drive both
 * the auto-tuning of the latency/halo trade-off and the mask warp prediction.
 *
 * The tracker is fed once per produced mask (segmenter loop) with the camera
 * shutter timestamp, so velocity reflects real-world motion regardless of the
 * render loop cadence.
 */
export class MaskMotionTracker {
  private prevCx = 0
  private prevCy = 0
  private prevT = 0
  private emaVx = 0
  private emaVy = 0
  private valid = false

  reset(): void {
    this.prevCx = 0
    this.prevCy = 0
    this.prevT = 0
    this.emaVx = 0
    this.emaVy = 0
    this.valid = false
  }

  update(bbox: BBox | null, cameraCaptureTime: number): void {
    if (!bbox) {
      this.valid = false
      this.emaVx = 0
      this.emaVy = 0
      return
    }

    const cx = bbox.x + bbox.width / 2
    const cy = bbox.y + bbox.height / 2

    if (!this.valid) {
      this.prevCx = cx
      this.prevCy = cy
      this.prevT = cameraCaptureTime
      this.emaVx = 0
      this.emaVy = 0
      this.valid = true
      return
    }

    const dt_s = (cameraCaptureTime - this.prevT) / 1000
    if (dt_s < MIN_DT_S || dt_s > MAX_DT_S) return

    const dx = cx - this.prevCx
    const dy = cy - this.prevCy
    if (Math.abs(dx) > TELEPORT_THRESHOLD_UV || Math.abs(dy) > TELEPORT_THRESHOLD_UV) {
      this.prevCx = cx
      this.prevCy = cy
      this.prevT = cameraCaptureTime
      this.emaVx = 0
      this.emaVy = 0
      return
    }

    let vxRaw = dx / dt_s
    let vyRaw = dy / dt_s
    if (vxRaw > MAX_VELOCITY_UV_PER_SEC) vxRaw = MAX_VELOCITY_UV_PER_SEC
    else if (vxRaw < -MAX_VELOCITY_UV_PER_SEC) vxRaw = -MAX_VELOCITY_UV_PER_SEC
    if (vyRaw > MAX_VELOCITY_UV_PER_SEC) vyRaw = MAX_VELOCITY_UV_PER_SEC
    else if (vyRaw < -MAX_VELOCITY_UV_PER_SEC) vyRaw = -MAX_VELOCITY_UV_PER_SEC

    const a = VELOCITY_EMA_ALPHA
    this.emaVx = (1 - a) * this.emaVx + a * vxRaw
    this.emaVy = (1 - a) * this.emaVy + a * vyRaw

    this.prevCx = cx
    this.prevCy = cy
    this.prevT = cameraCaptureTime
  }

  getVelocityUv(): { vx: number; vy: number } {
    return { vx: this.emaVx, vy: this.emaVy }
  }

  getMotionScore(): number {
    return Math.sqrt(this.emaVx * this.emaVx + this.emaVy * this.emaVy)
  }

  isValid(): boolean {
    return this.valid
  }
}
