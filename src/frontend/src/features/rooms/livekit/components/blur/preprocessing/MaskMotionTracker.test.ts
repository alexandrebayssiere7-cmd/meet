import { describe, it, expect, beforeEach } from 'vitest'
import { MaskMotionTracker } from './MaskMotionTracker'
import { BBox } from './RoiCropper'

/** A bbox centred at (cx, cy) with a fixed 0.2×0.2 size. */
function centredBbox(cx: number, cy: number): BBox {
  return { x: cx - 0.1, y: cy - 0.1, width: 0.2, height: 0.2 }
}

describe('MaskMotionTracker', () => {
  let tracker: MaskMotionTracker

  beforeEach(() => {
    tracker = new MaskMotionTracker()
  })

  it('starts invalid with zero motion', () => {
    expect(tracker.isValid()).toBe(false)
    expect(tracker.getMotionScore()).toBe(0)
  })

  it('becomes valid with zero velocity on the first sample', () => {
    tracker.update(centredBbox(0.5, 0.5), 0)
    expect(tracker.isValid()).toBe(true)
    expect(tracker.getVelocityUv()).toEqual({ vx: 0, vy: 0 })
  })

  it('computes EMA-smoothed velocity from displacement over time', () => {
    tracker.update(centredBbox(0.5, 0.5), 0)
    // +0.05 in x over 100 ms → raw vx = 0.5 uv/s; EMA alpha 0.3 → 0.15.
    tracker.update(centredBbox(0.55, 0.5), 100)
    const { vx, vy } = tracker.getVelocityUv()
    expect(vx).toBeCloseTo(0.15, 5)
    expect(vy).toBeCloseTo(0, 5)
    expect(tracker.getMotionScore()).toBeCloseTo(0.15, 5)
  })

  it('caps raw velocity at the max before smoothing', () => {
    tracker.update(centredBbox(0.5, 0.5), 0)
    // +0.2 over 10 ms → raw vx = 20 uv/s, clamped to 2.5; EMA → 0.75.
    tracker.update(centredBbox(0.7, 0.5), 10)
    expect(tracker.getVelocityUv().vx).toBeCloseTo(0.75, 5)
  })

  it('treats a large jump as a teleport and resets velocity', () => {
    tracker.update(centredBbox(0.5, 0.5), 0)
    tracker.update(centredBbox(0.55, 0.5), 100) // builds up some velocity
    expect(tracker.getMotionScore()).toBeGreaterThan(0)
    // Jump of 0.3 uv (> 0.25 teleport threshold) → velocity wiped.
    tracker.update(centredBbox(0.85, 0.5), 200)
    expect(tracker.getVelocityUv()).toEqual({ vx: 0, vy: 0 })
  })

  it('ignores samples whose dt is out of the valid window', () => {
    tracker.update(centredBbox(0.5, 0.5), 0)
    // dt = 1 ms < 5 ms min → sample dropped, velocity stays 0.
    tracker.update(centredBbox(0.6, 0.5), 1)
    expect(tracker.getVelocityUv()).toEqual({ vx: 0, vy: 0 })
    // dt = 500 ms > 200 ms max → also dropped.
    tracker.update(centredBbox(0.6, 0.5), 501)
    expect(tracker.getVelocityUv()).toEqual({ vx: 0, vy: 0 })
  })

  it('invalidates on a null bbox', () => {
    tracker.update(centredBbox(0.5, 0.5), 0)
    tracker.update(null, 100)
    expect(tracker.isValid()).toBe(false)
    expect(tracker.getMotionScore()).toBe(0)
  })

  it('reset() clears all state', () => {
    tracker.update(centredBbox(0.5, 0.5), 0)
    tracker.update(centredBbox(0.6, 0.5), 100)
    tracker.reset()
    expect(tracker.isValid()).toBe(false)
    expect(tracker.getMotionScore()).toBe(0)
  })
})
