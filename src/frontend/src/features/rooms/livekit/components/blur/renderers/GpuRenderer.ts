import { PostProcessingConfig, UpsamplingConfig } from '..'

/**
 * Source accepted by GpuRenderer.render(). HTMLVideoElement is the live
 * camera path (passthrough or maxFrameOffset > 0 with fresh mask). ImageBitmap
 * is used for frame-locked composite, where the source is the exact frame
 * that produced the current mask.
 */
export type RenderSource = HTMLVideoElement | ImageBitmap

/**
 * Initialisation options passed to `GpuRenderer.init()`.
 */
export interface GpuRendererInitOpts {
  /** Width of the segmentation mask (model input resolution). */
  processingW: number
  /** Height of the segmentation mask (model input resolution). */
  processingH: number
  /** Width of the final output canvas (= camera resolution). */
  outW: number
  /** Height of the final output canvas (= camera resolution). */
  outH: number
  /** Initial post-processing pipeline configuration. */
  postProcessing: PostProcessingConfig
  /** Initial upsampling configuration. */
  upsampling: UpsamplingConfig
}

/**
 * Backend-agnostic GPU renderer. The orchestrator (AdvancedMattingProcessor)
 * uses this to composite a final frame from the camera + mask + background.
 *
 * Implementations must NEVER use Canvas2D `ctx.filter` (unreliable on Safari).
 * All blur passes are GPU shaders.
 */
/**
 * Backend-agnostic GPU renderer. The orchestrator (AdvancedMattingProcessor)
 * uses this to composite a final frame from the camera + mask + background.
 *
 * Implementations must NEVER use Canvas2D `ctx.filter` (unreliable on Safari).
 * All blur passes are GPU shaders.
 */
export interface GpuRenderer {
  /** Identifies the underlying graphics API used by this implementation. */
  readonly backend: 'webgpu' | 'webgl2'
  /** Current output canvas width in pixels. */
  outW: number
  /** Current output canvas height in pixels. */
  outH: number

  /**
   * Initialise the renderer on the provided canvas with the given options.
   * Must be called once before any other method.
   */
  init(canvas: HTMLCanvasElement, opts: GpuRendererInitOpts): Promise<void>

  /**
   * Resize internal processing-resolution resources (mask textures, FBOs)
   * when the segmenter model changes dimensions.
   */
  resizeProcessing(w: number, h: number): void

  /**
   * Resize output-resolution resources (video texture, blur buffers, canvas)
   * when the camera resolution changes.
   */
  resizeOutput(w: number, h: number): void

  /**
   * Upload a new segmentation mask produced by the segmenter.
   * @param mask Float32Array in [0, 1], size w×h (1 = person, 0 = background).
   * @param w    Mask width (= processingW).
   * @param h    Mask height (= processingH).
   */
  uploadMask(mask: Float32Array, w: number, h: number): void

  /**
   * Set (or clear) the virtual background image.
   * Pass `null` to revert to blurred-camera mode.
   */
  setVirtualBackground(img: HTMLImageElement | null): void

  /**
   * Set the Gaussian blur radius in output pixels used for the background blur pass.
   * Only effective when mode is 'blur'.
   */
  setBlurRadius(px: number): void

  /**
   * Switch between 'blur' (Gaussian camera blur) and 'virtual' (image replacement) modes.
   */
  setMode(mode: 'blur' | 'virtual'): void

  /**
   * Update the post-processing pipeline configuration applied to the mask.
   * Resets temporal EMA state.
   */
  setPostProcessing(cfg: PostProcessingConfig): void

  /**
   * Update mask upsampling method and parameters.
   * Destroys the guided filter instance if switching away from guided mode.
   */
  setUpsampling(cfg: UpsamplingConfig): void

  /**
   * Composite a new output frame using the latest uploaded mask.
   * @param source Live camera `<video>` element or a pre-captured `ImageBitmap`.
   */
  render(source: RenderSource): void

  /**
   * Read a small RGBA8 patch from the output canvas.
   * Used by preflight diagnostics to verify the renderer is producing output.
   */
  readPixels(x: number, y: number, w: number, h: number): Uint8Array

  /** Release all GPU resources (textures, FBOs, programs, VAOs). */
  destroy(): void
}
