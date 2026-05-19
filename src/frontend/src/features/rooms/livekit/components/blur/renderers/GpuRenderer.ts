import { PostProcessingConfig, UpsamplingConfig } from '..'
import { Viewport } from '../framing/FramingController'

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
  readonly backend: 'webgpu' | 'webgl2'
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
  /** Normalised viewport applied as a final framing pass over the composite.
   *  Identity = {x:0, y:0, width:1, height:1}. */
  setViewport(vp: Viewport): void
  render(videoElement: HTMLVideoElement): void
  /** Read a small RGBA8 patch — used by preflight diagnostics. */
  readPixels(x: number, y: number, w: number, h: number): Uint8Array
  destroy(): void
}
