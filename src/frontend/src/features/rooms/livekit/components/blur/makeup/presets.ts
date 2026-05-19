// Makeup presets used by the matting pipeline.
//
// Each preset is a fixed combination of optional zones (lips, blush, brows,
// lashes). A preset omits a zone by leaving its field undefined. Intensities
// are encoded directly into the alpha/strength fields — there is no
// user-adjustable slider.

export type MakeupLips = {
  color: string
  alpha: number
  // CanvasRenderingContext2D.globalCompositeOperation value used when filling
  // the lip polygon. 'multiply' gives a natural pigmented look; 'source-over'
  // gives a flat opaque fill.
  blend: GlobalCompositeOperation
}

export type MakeupBlush = {
  color: string
  alpha: number
  // Radius of the radial-gradient blush, as a fraction of the canvas
  // shorter side. 0.10 ≈ a coin-sized soft circle on a 720p canvas.
  radius: number
}

export type MakeupBrows = {
  color: string
  alpha: number
}

export type MakeupLashes = {
  color: string
  alpha: number
  // Stroke width in pixels at the reference output canvas resolution
  // (1280×720). Scaled at draw time by the actual canvas height.
  thickness: number
}

export type MakeupPreset = {
  id: string
  // i18n key suffix under `effects.makeup.presets.<id>.label`.
  labelKey: string
  // A 1–2 char glyph used in the UI button as a visual hint. Kept as text
  // so the bundle stays asset-free.
  swatchColor: string
  lips?: MakeupLips
  blush?: MakeupBlush
  brows?: MakeupBrows
  lashes?: MakeupLashes
}

export const MAKEUP_PRESETS: readonly MakeupPreset[] = [
  {
    id: 'natural',
    labelKey: 'natural',
    swatchColor: '#d68a8a',
    lips: { color: '#c97a72', alpha: 0.38, blend: 'multiply' },
    blush: { color: '#e6a0a0', alpha: 0.22, radius: 0.10 },
  },
  {
    id: 'glamour',
    labelKey: 'glamour',
    swatchColor: '#b03a3a',
    lips: { color: '#a3201f', alpha: 0.55, blend: 'multiply' },
    blush: { color: '#d96a6a', alpha: 0.30, radius: 0.11 },
    brows: { color: '#3b2a22', alpha: 0.30 },
    lashes: { color: '#1f1610', alpha: 0.55, thickness: 2.6 },
  },
  {
    id: 'boldLips',
    labelKey: 'boldLips',
    swatchColor: '#8e1212',
    lips: { color: '#7a0d12', alpha: 0.72, blend: 'multiply' },
  },
  {
    id: 'smokyEyes',
    labelKey: 'smokyEyes',
    swatchColor: '#2a1f1c',
    brows: { color: '#1a1209', alpha: 0.45 },
    lashes: { color: '#0d0a08', alpha: 0.70, thickness: 3.2 },
    lips: { color: '#b88475', alpha: 0.28, blend: 'multiply' },
  },
  {
    id: 'softGlow',
    labelKey: 'softGlow',
    swatchColor: '#f0b6b6',
    blush: { color: '#f1a8a8', alpha: 0.32, radius: 0.13 },
  },
] as const

export function findMakeupPreset(id: string | undefined): MakeupPreset | null {
  if (!id) return null
  return MAKEUP_PRESETS.find((p) => p.id === id) ?? null
}

export function isKnownMakeupPresetId(id: unknown): id is string {
  return typeof id === 'string' && MAKEUP_PRESETS.some((p) => p.id === id)
}
