import { MorphologyOp } from '..'

/**
 * 2D grayscale morphological operations on a Float32 mask in [0, 1].
 * - erosion : pixel = min over a square neighborhood (shrinks mask, reduces halo)
 * - dilation: pixel = max over a square neighborhood (expands mask, fills holes)
 * - opening : erosion then dilation (removes small isolated specks)
 * - closing : dilation then erosion (fills small holes inside the mask)
 *
 * Implementation: separable min/max via two 1-D passes (row + column),
 * O(N * kernelSize) per pass. Sufficient for 256x256.
 */

type Reducer = (a: number, b: number) => number
const minReducer: Reducer = (a, b) => (a < b ? a : b)
const maxReducer: Reducer = (a, b) => (a > b ? a : b)

function pass1D(
  src: Float32Array,
  dst: Float32Array,
  width: number,
  height: number,
  radius: number,
  reducer: Reducer,
  horizontal: boolean
): void {
  if (horizontal) {
    for (let y = 0; y < height; y++) {
      const rowOff = y * width
      for (let x = 0; x < width; x++) {
        let v = src[rowOff + x]
        const x0 = x - radius < 0 ? 0 : x - radius
        const x1 = x + radius >= width ? width - 1 : x + radius
        for (let i = x0; i <= x1; i++) {
          v = reducer(v, src[rowOff + i])
        }
        dst[rowOff + x] = v
      }
    }
  } else {
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        let v = src[y * width + x]
        const y0 = y - radius < 0 ? 0 : y - radius
        const y1 = y + radius >= height ? height - 1 : y + radius
        for (let i = y0; i <= y1; i++) {
          v = reducer(v, src[i * width + x])
        }
        dst[y * width + x] = v
      }
    }
  }
}

function applyMinOrMax(
  mask: Float32Array,
  width: number,
  height: number,
  radius: number,
  reducer: Reducer
): Float32Array {
  const tmp = new Float32Array(mask.length)
  const out = new Float32Array(mask.length)
  pass1D(mask, tmp, width, height, radius, reducer, true)
  pass1D(tmp, out, width, height, radius, reducer, false)
  return out
}

export function applyMorphology(
  mask: Float32Array,
  width: number,
  height: number,
  op: MorphologyOp,
  kernelSize: 3 | 5 | 7
): Float32Array {
  const radius = (kernelSize - 1) >> 1
  switch (op) {
    case 'erosion':
      return applyMinOrMax(mask, width, height, radius, minReducer)
    case 'dilation':
      return applyMinOrMax(mask, width, height, radius, maxReducer)
    case 'opening': {
      const eroded = applyMinOrMax(mask, width, height, radius, minReducer)
      return applyMinOrMax(eroded, width, height, radius, maxReducer)
    }
    case 'closing': {
      const dilated = applyMinOrMax(mask, width, height, radius, maxReducer)
      return applyMinOrMax(dilated, width, height, radius, minReducer)
    }
    default:
      return mask
  }
}
