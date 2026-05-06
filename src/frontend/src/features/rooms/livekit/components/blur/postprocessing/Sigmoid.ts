/**
 * Pixel-wise sigmoid: 1 / (1 + exp(-steepness * (x - threshold))).
 * Sharpens the mask transition around the threshold while keeping it smooth.
 */
export function applySigmoid(
  mask: Float32Array,
  steepness: number,
  threshold: number
): Float32Array {
  const out = new Float32Array(mask.length)
  for (let i = 0; i < mask.length; i++) {
    out[i] = 1 / (1 + Math.exp(-steepness * (mask[i] - threshold)))
  }
  return out
}
