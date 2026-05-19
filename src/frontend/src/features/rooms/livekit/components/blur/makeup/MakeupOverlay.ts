// Makeup overlay — GPU path.
//
// This module is purely CPU-side data preparation: it knows which MediaPipe
// FaceMesh landmark indices form each makeup zone (lips, brows, lashes,
// blush) and packs them into a flat Float32Array laid out as rows of a small
// 2D texture. That texture is then sampled by the makeup fragment shader,
// which evaluates point-in-polygon / distance-to-polyline tests per pixel.
// There is no Canvas 2D rendering anymore.

import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

// Zone row indices in the landmarks texture. The shader uses the same
// constants — keep them in sync.
export const ZONE_LIP_OUTER = 0
export const ZONE_LIP_INNER = 1
export const ZONE_BROW_LEFT = 2
export const ZONE_BROW_RIGHT = 3
export const ZONE_LASH_LEFT = 4
export const ZONE_LASH_RIGHT = 5
export const ZONE_BLUSH = 6
export const NUM_ZONES = 7

// Outer/inner lip contours. The shader fills (outer XOR inner) so the
// pigment stops at the mouth opening.
const LIP_OUTER = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]
const LIP_INNER = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

// Eyebrows — single closed polygon per brow.
const BROW_LEFT = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
const BROW_RIGHT = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

// Upper-eyelid polylines for lashes. Order: outer → inner corner.
const LASH_LEFT = [33, 246, 161, 160, 159, 158, 157, 173]
const LASH_RIGHT = [263, 466, 388, 387, 386, 385, 384, 398]

// Blush — single landmark per cheek (apple of the cheek).
const BLUSH_LEFT = 50
const BLUSH_RIGHT = 280

// One row per zone, MAX_PER_ZONE columns. Rows shorter than MAX_PER_ZONE
// are padded by repeating the last vertex — distance calculations against
// a degenerate segment are stable (zero-length segments contribute the
// vertex itself as the closest point).
export const MAX_PER_ZONE = 20

// Number of valid vertices per zone (matches the index arrays above).
export const ZONE_VERTEX_COUNTS = {
  lipOuter: LIP_OUTER.length,
  lipInner: LIP_INNER.length,
  browLeft: BROW_LEFT.length,
  browRight: BROW_RIGHT.length,
  lashLeft: LASH_LEFT.length,
  lashRight: LASH_RIGHT.length,
  blush: 2,
} as const

// Size of the packed buffer: NUM_ZONES rows × MAX_PER_ZONE cols × 2 floats (xy).
export const PACKED_FLOATS_PER_ROW = MAX_PER_ZONE * 2
export const PACKED_TOTAL_FLOATS = NUM_ZONES * PACKED_FLOATS_PER_ROW

function writeZone(
  out: Float32Array,
  rowIdx: number,
  face: NormalizedLandmark[],
  indices: readonly number[]
) {
  const base = rowIdx * PACKED_FLOATS_PER_ROW
  const n = indices.length
  for (let i = 0; i < MAX_PER_ZONE; i++) {
    // Pad rows shorter than MAX_PER_ZONE by repeating the last valid index.
    const srcIdx = i < n ? indices[i] : indices[n - 1]
    const lm = face[srcIdx]
    out[base + i * 2] = lm.x
    out[base + i * 2 + 1] = lm.y
  }
}

/**
 * Pack the landmarks of the first detected face into `out`, laid out as
 * NUM_ZONES rows of MAX_PER_ZONE (x, y) pairs. `out` must be exactly
 * PACKED_TOTAL_FLOATS long. Returns `true` if a usable face was packed.
 */
export function packLandmarksForMakeup(
  faces: NormalizedLandmark[][] | null,
  out: Float32Array
): boolean {
  if (!faces || faces.length === 0) return false
  const face = faces[0]
  if (!face || face.length < 400) return false

  writeZone(out, ZONE_LIP_OUTER, face, LIP_OUTER)
  writeZone(out, ZONE_LIP_INNER, face, LIP_INNER)
  writeZone(out, ZONE_BROW_LEFT, face, BROW_LEFT)
  writeZone(out, ZONE_BROW_RIGHT, face, BROW_RIGHT)
  writeZone(out, ZONE_LASH_LEFT, face, LASH_LEFT)
  writeZone(out, ZONE_LASH_RIGHT, face, LASH_RIGHT)

  // Blush row: only 2 valid entries (left center, right center). Rest is
  // padding (last valid vertex repeated).
  const blushBase = ZONE_BLUSH * PACKED_FLOATS_PER_ROW
  const bl = face[BLUSH_LEFT]
  const br = face[BLUSH_RIGHT]
  out[blushBase + 0] = bl.x
  out[blushBase + 1] = bl.y
  out[blushBase + 2] = br.x
  out[blushBase + 3] = br.y
  for (let i = 2; i < MAX_PER_ZONE; i++) {
    out[blushBase + i * 2] = br.x
    out[blushBase + i * 2 + 1] = br.y
  }

  return true
}
