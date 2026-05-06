/**
 * Guided Filter (He et al., 2010) — edge-preserving mask refinement.
 *
 * Uses the RGB image as a "guide" to align mask boundaries with real image
 * edges (hair, fingers, clothes). Implementation follows the O(N) box-filter
 * formulation with the gray-scale guide variant for speed.
 *
 *   I = grayscale(guide)        // luminance in [0, 1]
 *   p = mask                    // input in [0, 1]
 *   meanI  = box(I)
 *   meanP  = box(p)
 *   corrI  = box(I * I)
 *   corrIp = box(I * p)
 *   varI   = corrI  - meanI * meanI
 *   covIp  = corrIp - meanI * meanP
 *   a = covIp / (varI + eps)
 *   b = meanP - a * meanI
 *   q = box(a) * I + box(b)
 *
 * Cost ~5-10ms at 256x256 on a modern laptop.
 */

function boxFilter(
  src: Float32Array,
  width: number,
  height: number,
  radius: number
): Float32Array {
  const tmp = new Float32Array(src.length)
  const out = new Float32Array(src.length)

  // horizontal pass with running sum
  for (let y = 0; y < height; y++) {
    const off = y * width
    let sum = 0
    for (let x = 0; x <= radius && x < width; x++) sum += src[off + x]
    for (let x = 0; x < width; x++) {
      const x0 = x - radius - 1
      const x1 = x + radius
      if (x1 < width) sum += src[off + x1]
      if (x0 >= 0) sum -= src[off + x0]
      const lo = x - radius < 0 ? 0 : x - radius
      const hi = x + radius >= width ? width - 1 : x + radius
      tmp[off + x] = sum / (hi - lo + 1)
    }
  }

  // vertical pass with running sum
  for (let x = 0; x < width; x++) {
    let sum = 0
    for (let y = 0; y <= radius && y < height; y++) sum += tmp[y * width + x]
    for (let y = 0; y < height; y++) {
      const y0 = y - radius - 1
      const y1 = y + radius
      if (y1 < height) sum += tmp[y1 * width + x]
      if (y0 >= 0) sum -= tmp[y0 * width + x]
      const lo = y - radius < 0 ? 0 : y - radius
      const hi = y + radius >= height ? height - 1 : y + radius
      out[y * width + x] = sum / (hi - lo + 1)
    }
  }
  return out
}

function rgbToGrayFloat(rgba: Uint8ClampedArray): Float32Array {
  const n = rgba.length >> 2
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const j = i << 2
    // Rec. 601 luma, normalized to [0, 1]
    out[i] = (0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]) / 255
  }
  return out
}

export function applyGuidedFilter(
  mask: Float32Array,
  guideRgb: ImageData,
  radius: number,
  eps: number
): Float32Array {
  const width = guideRgb.width
  const height = guideRgb.height
  const I = rgbToGrayFloat(guideRgb.data)
  const p = mask

  const meanI = boxFilter(I, width, height, radius)
  const meanP = boxFilter(p, width, height, radius)

  const II = new Float32Array(I.length)
  const IP = new Float32Array(I.length)
  for (let i = 0; i < I.length; i++) {
    II[i] = I[i] * I[i]
    IP[i] = I[i] * p[i]
  }
  const corrI = boxFilter(II, width, height, radius)
  const corrIp = boxFilter(IP, width, height, radius)

  const a = new Float32Array(I.length)
  const b = new Float32Array(I.length)
  for (let i = 0; i < I.length; i++) {
    const varI = corrI[i] - meanI[i] * meanI[i]
    const covIp = corrIp[i] - meanI[i] * meanP[i]
    const ai = covIp / (varI + eps)
    a[i] = ai
    b[i] = meanP[i] - ai * meanI[i]
  }
  const meanA = boxFilter(a, width, height, radius)
  const meanB = boxFilter(b, width, height, radius)

  const out = new Float32Array(I.length)
  for (let i = 0; i < I.length; i++) {
    const v = meanA[i] * I[i] + meanB[i]
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v
  }
  return out
}
