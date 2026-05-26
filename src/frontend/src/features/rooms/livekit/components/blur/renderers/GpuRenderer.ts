import { PostProcessingConfig, UpsamplingConfig } from '..'

/**
 * Source accepted by GpuRenderer.render(). HTMLVideoElement is the live
 * camera path (passthrough or maxFrameOffset > 0 with fresh mask). ImageBitmap
 * is used for frame-locked composite, where the source is the exact frame
 * that produced the current mask.
 */
export type RenderSource = HTMLVideoElement | ImageBitmap

export interface GpuRendererInitOpts {
  processingW: number
  processingH: number
  outW: number
  outH: number
  postProcessing: PostProcessingConfig
  upsampling: UpsamplingConfig
}

/**
 * Backend-agnostic GPU renderer. The orchestrator (AdvancedMattingProcessor)
 * uses this to composite a final frame from the camera + mask + background.
 *
 * Implementations must NEVER use Canvas2D `ctx.filter` (unreliable on Safari).
 * All blur passes are GPU shaders.
 */
export interface GpuRenderer {
  readonly backend: 'webgpu' | 'webgl2' | 'canvas2d'
  outW: number
  outH: number
  init(canvas: HTMLCanvasElement, opts: GpuRendererInitOpts): Promise<void>
  resizeProcessing(w: number, h: number): void
  resizeOutput(w: number, h: number): void
  uploadMask(mask: Float32Array, w: number, h: number): void
  setVirtualBackground(img: HTMLImageElement | null): void
  setBlurRadius(px: number): void
  setMode(mode: 'blur' | 'virtual'): void
  setPostProcessing(cfg: PostProcessingConfig): void
  setUpsampling(cfg: UpsamplingConfig): void
  /** UV-space mask offset applied at composite time (prediction warp). */
  setMaskOffset(u: number, v: number): void
  /** Blend ratio between frame-locked source and live source (0..1). */
  setBlendMix(t: number): void
  render(source: RenderSource, liveSource?: RenderSource): void
  /** Read a small RGBA8 patch — used by preflight diagnostics. */
  readPixels(x: number, y: number, w: number, h: number): Uint8Array
  destroy(): void
}
