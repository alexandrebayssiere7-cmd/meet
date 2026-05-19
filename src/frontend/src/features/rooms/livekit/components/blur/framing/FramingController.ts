import { BBox } from '../preprocessing/RoiCropper'

/** Normalised viewport over the composite texture. (0,0,1,1) means "no framing". */
export interface Viewport {
  x: number
  y: number
  width: number
  height: number
}

export interface FramingConfig {
  /** Maximum zoom-in factor. 2.0 ⇒ viewport min side = 1/2.0 = 0.5 in UV. */
  maxZoom: number
  /** Target apparent width of the person as a fraction of the canvas. side =
   *  wPad / targetPersonWidth, clamped to [1/maxZoom, 1]. */
  targetPersonWidth: number
  /** Margin added around the bbox width before sizing the viewport. */
  paddingRatio: number
  /** Vertical anchor as a fraction of the bbox height (0 = top of bbox,
   *  1 = bottom). The anchor stays at the same canvas y regardless of zoom,
   *  so the subject doesn't drift vertically when distance changes.
   *  1.0 ≈ "bottom of bbox / torso bottom" — keeps the torso planted while
   *  zoom pushes the area ABOVE the person (sky / ceiling, eventually the
   *  forehead) out of the top of the canvas. Matches the natural intuition
   *  of "zoom in on a fixed subject without losing the body". */
  anchorYFraction: number
  /** Duration of the cubic ease-in-out between two stable viewports, ms. */
  easingMs: number
  /** When false, target snaps to identity and the current viewport eases back to full frame. */
  enabled: boolean
}

export const DEFAULT_FRAMING_CONFIG: FramingConfig = {
  maxZoom: 2.0,
  targetPersonWidth: 0.7,
  paddingRatio: 0.15,
  anchorYFraction: 1.0,
  easingMs: 500,
  enabled: false,
}

const IDENTITY: Viewport = { x: 0, y: 0, width: 1, height: 1 }

const TARGET_DEAD_ZONE = 0.015

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

function easeInOutCubic(t: number): number {
  if (t <= 0) return 0
  if (t >= 1) return 1
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function viewportsEqual(a: Viewport, b: Viewport, eps: number): boolean {
  return (
    Math.abs(a.x - b.x) < eps &&
    Math.abs(a.y - b.y) < eps &&
    Math.abs(a.width - b.width) < eps &&
    Math.abs(a.height - b.height) < eps
  )
}

/**
 * Convert a normalised person bbox (in image coords where y=0 is the visual TOP
 * of the frame) into a viewport in FBO UV coords (where y=0 is the visual
 * BOTTOM of the frame) that preserves the output aspect ratio.
 *
 * Key constraints:
 *  - The viewport is expressed in UV [0, 1] of the intermediate composite
 *    texture (outW × outH) and rendered onto an outW × outH canvas. The pixel
 *    aspect ratio of the sampled region must equal outW / outH, which in UV
 *    space means vp.width == vp.height (the viewport must be SQUARE in UV).
 *  - The bbox y-axis is inverted relative to the FBO's UV y-axis: bbox.y=0
 *    means the top of the image visually, but viewport.y=0 samples the bottom
 *    of the FBO. We compensate at the end.
 *  - The segmenter mask is reliable on head/torso but weak on lower body;
 *    `verticalBias` shifts the centre downward (in image coords) so the
 *    framing keeps the body in shot instead of cropping above the chin.
 *
 * Steps:
 *   1. Pad bbox on every side.
 *   2. Square it (max(w, h)) so the rendered region matches outAspect.
 *   3. Clamp side ∈ [1/maxZoom, 1].
 *   4. Apply verticalBias to the bbox centre (image-coords).
 *   5. Translate cx, cy to keep the viewport inside [0, 1].
 *   6. Convert y to FBO UV (1 − cy_image − side/2).
 *
 * outAspect is kept in the signature for future variants (letterboxing, etc.).
 */
/**
 * Compute the viewport for zoom + horizontal recentre framing.
 *
 * Vertical anchor (anchorYFraction=1.0 by default): the BOTTOM of the bbox
 * stays at its natural canvas y across all zoom levels. When zoom kicks in
 * (subject far → side < 1), the source is stretched ABOVE the anchor —
 * the head rises in canvas space, and the empty area above it (sky /
 * ceiling) is the first thing to be pushed off the top of the canvas.
 * The torso remains visible.
 *
 * Formula derivation. Without framing the image content at image y = a maps
 * to canvas y = 1 − a (the FBO is y-flipped relative to image coords). To
 * keep image y = a at canvas y = 1 − a even with viewport (vy, side):
 *     canvas_y(a) = (FBO_y(a) − vy) / side = (1 − a − vy) / side
 *     setting canvas_y(a) = 1 − a ⇒ vy = (1 − a)(1 − side)
 * We use a = bbox.y + bbox.height * anchorYFraction as the anchor.
 *
 * Horizontal works the same way around cxBbox, but because we WANT the
 * person to be recentred horizontally we map cxBbox → canvas centre (0.5)
 * instead of to its original canvas position. That's the "translation"
 * component the user explicitly asked for.
 *
 * Out-of-bounds samples (from translation pushing the viewport past
 * [0, 1]) are masked to 0 by the viewport-remap shader, so the background
 * shows at the edges with no phantom person.
 */
function bboxToViewport(
  bbox: BBox,
  maxZoom: number,
  targetPersonWidth: number,
  paddingRatio: number,
  anchorYFraction: number
): Viewport {
  const cxBbox = bbox.x + bbox.width / 2
  const wPad = bbox.width * (1 + 2 * paddingRatio)

  // Side: aim for the person to occupy `targetPersonWidth` of the canvas
  // horizontally. Clamped to [1/maxZoom, 1] — never zoom OUT (the subject
  // should never appear smaller than naturally).
  const minSide = 1 / maxZoom
  let side = wPad / targetPersonWidth
  if (side < minSide) side = minSide
  if (side > 1.0) side = 1.0

  // Horizontal: centre the bbox cx on the canvas centre.
  const x = cxBbox - 0.5 * side

  // Vertical: anchor on top-of-bbox so the head stays fixed across zoom
  // levels. anchorY in image coords.
  const anchorY = bbox.y + bbox.height * anchorYFraction
  const y = (1 - anchorY) * (1 - side)

  return { x, y, width: side, height: side }
}

/**
 * Owns the animated viewport applied as a final framing pass on top of the
 * composited matting output. Sits on top of RoiCropper's already-smoothed bbox
 * and adds a cinematographic easing of the *presented* viewport.
 */
export class FramingController {
  private targetViewport: Viewport = { ...IDENTITY }
  private currentViewport: Viewport = { ...IDENTITY }
  private startViewport: Viewport = { ...IDENTITY }
  private animStartMs = 0
  private animDurationMs = 0
  private initialised = false

  update(
    personBbox: BBox | null,
    _outAspect: number,
    nowMs: number,
    cfg: FramingConfig
  ): void {
    void _outAspect
    const nextTarget: Viewport =
      !cfg.enabled || !personBbox
        ? { ...IDENTITY }
        : bboxToViewport(
            personBbox,
            cfg.maxZoom,
            cfg.targetPersonWidth,
            cfg.paddingRatio,
            cfg.anchorYFraction
          )

    if (!this.initialised) {
      this.targetViewport = nextTarget
      this.currentViewport = { ...nextTarget }
      this.startViewport = { ...nextTarget }
      this.animStartMs = nowMs
      this.animDurationMs = 0
      this.initialised = true
      return
    }

    // Second-level dead zone: avoid restarting the easing on every micro-jitter.
    if (!viewportsEqual(nextTarget, this.targetViewport, TARGET_DEAD_ZONE)) {
      this.startViewport = { ...this.currentViewport }
      this.targetViewport = nextTarget
      this.animStartMs = nowMs
      this.animDurationMs = cfg.easingMs
    }

    if (this.animDurationMs <= 0) {
      this.currentViewport = { ...this.targetViewport }
      return
    }

    const raw = (nowMs - this.animStartMs) / this.animDurationMs
    const t = easeInOutCubic(raw)
    this.currentViewport = {
      x: lerp(this.startViewport.x, this.targetViewport.x, t),
      y: lerp(this.startViewport.y, this.targetViewport.y, t),
      width: lerp(this.startViewport.width, this.targetViewport.width, t),
      height: lerp(this.startViewport.height, this.targetViewport.height, t),
    }

    if (raw >= 1) {
      this.currentViewport = { ...this.targetViewport }
      this.animDurationMs = 0
    }
  }

  getViewport(): Viewport {
    return this.currentViewport
  }

  reset(): void {
    this.targetViewport = { ...IDENTITY }
    this.currentViewport = { ...IDENTITY }
    this.startViewport = { ...IDENTITY }
    this.animStartMs = 0
    this.animDurationMs = 0
    this.initialised = false
  }
}
