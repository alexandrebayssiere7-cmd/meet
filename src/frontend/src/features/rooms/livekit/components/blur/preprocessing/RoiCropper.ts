const DEAD_ZONE_POSITION = 0.03
const DEAD_ZONE_SIZE = 0.015
const SMOOTHING = 0.5
const BBOX_PADDING = 0.05
const MASK_THRESHOLD = 0.5

export interface BBox {
  x: number      // normalised left edge [0, 1]
  y: number      // normalised top edge  [0, 1]
  width: number  // normalised width     [0, 1]
  height: number // normalised height    [0, 1]
}

const FULL_FRAME: BBox = { x: 0, y: 0, width: 1, height: 1 }

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/** Scan the mask and return the tightest bbox around pixels above threshold,
 *  expanded by BBOX_PADDING, clamped to [0, 1]. Returns null if no pixel qualifies. */
export function computePersonBbox(
  mask: Float32Array,
  maskW: number,
  maskH: number
): BBox | null {
  let minX = maskW, maxX = -1, minY = maskH, maxY = -1

  for (let y = 0; y < maskH; y++) {
    for (let x = 0; x < maskW; x++) {
      if (mask[y * maskW + x] > MASK_THRESHOLD) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }

  if (maxX < 0) return null

  const nx = minX / maskW
  const ny = minY / maskH
  const nw = (maxX - minX + 1) / maskW
  const nh = (maxY - minY + 1) / maskH

  return {
    x: clamp(nx - BBOX_PADDING, 0, 1),
    y: clamp(ny - BBOX_PADDING, 0, 1),
    width: clamp(nw + 2 * BBOX_PADDING, 0, 1 - clamp(nx - BBOX_PADDING, 0, 1)),
    height: clamp(nh + 2 * BBOX_PADDING, 0, 1 - clamp(ny - BBOX_PADDING, 0, 1)),
  }
}

/** Stabilise a raw bbox against the current stable bbox using a dead zone + EMA. */
export function stabilizeBbox(current: BBox, next: BBox): BBox {
  const cxCurr = current.x + current.width / 2
  const cyCurr = current.y + current.height / 2
  const cxNext = next.x + next.width / 2
  const cyNext = next.y + next.height / 2

  const positionMoved =
    Math.abs(cxNext - cxCurr) > DEAD_ZONE_POSITION ||
    Math.abs(cyNext - cyCurr) > DEAD_ZONE_POSITION
  const sizeMoved =
    Math.abs(next.width - current.width) > DEAD_ZONE_SIZE ||
    Math.abs(next.height - current.height) > DEAD_ZONE_SIZE

  if (!positionMoved && !sizeMoved) return current

  const s = SMOOTHING
  const inv = 1 - s
  return {
    x: inv * current.x + s * next.x,
    y: inv * current.y + s * next.y,
    width: inv * current.width + s * next.width,
    height: inv * current.height + s * next.height,
  }
}

/** Bilinear resize of a Float32 single-channel image. */
function resizeFloat32(
  src: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Float32Array {
  const dst = new Float32Array(dstW * dstH)
  const scaleX = srcW / dstW
  const scaleY = srcH / dstH

  for (let dy = 0; dy < dstH; dy++) {
    const sy = (dy + 0.5) * scaleY - 0.5
    const sy0 = Math.floor(sy)
    const sy1 = sy0 + 1
    const fy = sy - sy0
    const iy0 = sy0 < 0 ? 0 : sy0 >= srcH ? srcH - 1 : sy0
    const iy1 = sy1 < 0 ? 0 : sy1 >= srcH ? srcH - 1 : sy1

    for (let dx = 0; dx < dstW; dx++) {
      const sx = (dx + 0.5) * scaleX - 0.5
      const sx0 = Math.floor(sx)
      const sx1 = sx0 + 1
      const fx = sx - sx0
      const ix0 = sx0 < 0 ? 0 : sx0 >= srcW ? srcW - 1 : sx0
      const ix1 = sx1 < 0 ? 0 : sx1 >= srcW ? srcW - 1 : sx1

      const v = (1 - fy) * ((1 - fx) * src[iy0 * srcW + ix0] + fx * src[iy0 * srcW + ix1]) +
                      fy  * ((1 - fx) * src[iy1 * srcW + ix0] + fx * src[iy1 * srcW + ix1])
      dst[dy * dstW + dx] = v
    }
  }
  return dst
}

/**
 * ROI Cropper — maintains a stabilised bounding box of the person across frames.
 *
 * Per-frame call order in AdvancedMattingProcessor:
 *   1. bbox = roiCropper.getNextCropBbox()         (used by sizeSource to crop the video)
 *   2. model.segment(croppedFrame)                 → maskInCropSpace
 *   3. fullMask = roiCropper.remapMask(maskInCropSpace, maskW, maskH, bbox)
 *   4. roiCropper.updateWithMask(fullMask, maskW, maskH)
 *
 * Zero-allocation design: two pre-allocated Float32Arrays are alternated each
 * frame (ring-buffer / double buffering) so the hot path never triggers GC.
 */
export class RoiCropper {
  private currentBbox: BBox = { ...FULL_FRAME }
  private hasMask = false
  private frameCounter = 0

  // Pre-allocated ring-buffer for remapMask output. Sized for the maximum
  // processing resolution the pipeline supports (256×256). The caller receives
  // a reference that stays valid until the NEXT remapMask call.
  private fullMaskBuffers: [Float32Array, Float32Array]
  private fullBufIdx = 0
  private fullBufW = 0
  private fullBufH = 0

  constructor(maxW = 256, maxH = 256) {
    this.fullMaskBuffers = [
      new Float32Array(maxW * maxH),
      new Float32Array(maxW * maxH),
    ]
    this.fullBufW = maxW
    this.fullBufH = maxH
  }

  /** Returns the stabilised bbox to use when extracting the model input for this frame. */
  getNextCropBbox(): BBox {
    this.frameCounter++
    if (this.frameCounter % 45 === 0) {
      return { ...FULL_FRAME }
    }
    return this.currentBbox
  }

  /**
   * Remap a mask that lives in crop-bbox space back to the full-frame mask space.
   *
   * Fused remap + bilinear resize in a single pass: for each destination pixel
   * that falls inside the bbox region, compute its source coordinate in the
   * crop mask, bilinearly interpolate, and write directly into the pre-allocated
   * output buffer. All other pixels are zeroed. Zero allocations per call.
   */
  remapMask(
    cropMask: Float32Array,
    cropMaskW: number,
    cropMaskH: number,
    usedBbox: BBox,
    fullW: number,
    fullH: number
  ): Float32Array {
    // Ensure buffers are large enough (reallocate only on resolution change).
    if (fullW * fullH > this.fullBufW * this.fullBufH) {
      this.fullMaskBuffers = [
        new Float32Array(fullW * fullH),
        new Float32Array(fullW * fullH),
      ]
      this.fullBufW = fullW
      this.fullBufH = fullH
    }

    // Pick the current buffer and swap the index for next frame.
    const full = this.fullMaskBuffers[this.fullBufIdx]
    this.fullBufIdx ^= 1

    // Zero the entire buffer (TypedArray.fill is a fast memset).
    full.fill(0)

    const dstX = Math.round(usedBbox.x * fullW)
    const dstY = Math.round(usedBbox.y * fullH)
    const dstW = Math.round(usedBbox.width * fullW)
    const dstH = Math.round(usedBbox.height * fullH)

    if (dstW <= 0 || dstH <= 0) return full

    // Fused bilinear resize + remap: compute source coordinates directly.
    const scaleX = cropMaskW / dstW
    const scaleY = cropMaskH / dstH

    for (let dy = 0; dy < dstH; dy++) {
      const fy = dstY + dy
      if (fy < 0 || fy >= fullH) continue

      const sy = (dy + 0.5) * scaleY - 0.5
      const sy0 = Math.floor(sy)
      const fracY = sy - sy0
      const iy0 = sy0 < 0 ? 0 : sy0 >= cropMaskH ? cropMaskH - 1 : sy0
      const iy1 = sy0 + 1 < 0 ? 0 : sy0 + 1 >= cropMaskH ? cropMaskH - 1 : sy0 + 1
      const row0 = iy0 * cropMaskW
      const row1 = iy1 * cropMaskW

      for (let dx = 0; dx < dstW; dx++) {
        const fx = dstX + dx
        if (fx < 0 || fx >= fullW) continue

        const sx = (dx + 0.5) * scaleX - 0.5
        const sx0 = Math.floor(sx)
        const fracX = sx - sx0
        const ix0 = sx0 < 0 ? 0 : sx0 >= cropMaskW ? cropMaskW - 1 : sx0
        const ix1 = sx0 + 1 < 0 ? 0 : sx0 + 1 >= cropMaskW ? cropMaskW - 1 : sx0 + 1

        const v =
          (1 - fracY) * ((1 - fracX) * cropMask[row0 + ix0] + fracX * cropMask[row0 + ix1]) +
          fracY        * ((1 - fracX) * cropMask[row1 + ix0] + fracX * cropMask[row1 + ix1])

        full[fy * fullW + fx] = v
      }
    }

    return full
  }

  /** Update internal state from the full-frame mask produced this frame. */
  updateWithMask(fullMask: Float32Array, maskW: number, maskH: number): void {
    this.hasMask = true
    const raw = computePersonBbox(fullMask, maskW, maskH)
    if (!raw) {
      // No person detected — keep current bbox so the crop doesn't jump to full frame.
      return
    }
    this.currentBbox = stabilizeBbox(this.currentBbox, raw)
  }

  reset(): void {
    this.currentBbox = { ...FULL_FRAME }
    this.hasMask = false
    this.frameCounter = 0
  }

  getCurrentBbox(): BBox {
    return this.currentBbox
  }

  isInitialised(): boolean {
    return this.hasMask
  }
}

