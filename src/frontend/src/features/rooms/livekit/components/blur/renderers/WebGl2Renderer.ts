import { PostProcessingConfig, UpsamplingConfig } from '..'
import { pushMattingError } from '../errors/MattingErrorStore'
import { GpuRenderer, GpuRendererInitOpts } from './GpuRenderer'
import { GpuGuidedFilter } from './GpuGuidedFilter'

/**
 * WebGL2 implementation of the matting compositor.
 *
 * Pipeline per frame (`render(videoElement)`):
 *   videoTex ← upload from <video>
 *   maskTex  ← uploaded once per new mask (uploadMask)
 *   maskRefined ← post-processing chain (morpho → ema)
 *   bgBlur ← (mode === 'blur') maskedDownsample(videoTex, mask)
 *                              → maskWeightedGaussH → maskWeightedGaussV  (half-res)
 *           (mode === 'virtual') virtualBgTex
 *   canvas  ← composite(videoTex, bgBlur, maskRefined)
 *
 * SAFARI: never uses ctx.filter. Every blur is a shader.
 * NOTE: Guided filter is NOT implemented in shaders here yet — the orchestrator
 *       falls back to the CPU implementation when guided filter is enabled.
 */
export class WebGl2Renderer implements GpuRenderer {
  readonly backend = 'webgl2'

  private gl!: WebGL2RenderingContext
  outW = 0
  outH = 0
  private procW = 0
  private procH = 0
  private postCfg: PostProcessingConfig = {}
  private upsamplingCfg: UpsamplingConfig = {}
  private gf: GpuGuidedFilter | null = null
  private mode: 'blur' | 'virtual' = 'blur'
  private blurRadius = 10

  private vao!: WebGLVertexArrayObject
  private quadBuffer!: WebGLBuffer

  // programs
  private pUploadMask!: WebGLProgram
  private pEma!: WebGLProgram
  private pCopyR!: WebGLProgram
  private pMaskedDownsample!: WebGLProgram
  private pMaskWeightedBlur!: WebGLProgram
  private pMorphology!: WebGLProgram
  private pComposite!: WebGLProgram
  // Segmo-style virtual-background compositor (foreground recovery + edge-adaptive
  // sharpening + closed-form alpha matting). Used ONLY when mode === 'virtual' and
  // a virtual background image is uploaded. Never runs in the blur path.
  private pCompositeSegmo!: WebGLProgram
  // Edge-only feather pass: gaussian-blurs the mask near silhouette edges, leaves
  // interior/exterior pixels untouched. Widens the transition band so segmo's
  // closed-form matting has more pixels to operate on. Virtual path only.
  private pSegmoEdgeFeather!: WebGLProgram
  private segmoFeatheredMaskTex: WebGLTexture | null = null
  private fboSegmoFeatheredMask: WebGLFramebuffer | null = null
  private segmoFeatherRadius = 3.0
  // Light wrap pass: mixes a small amount of background color into the foreground
  // edge band so the subject looks lit by the new scene. Virtual path only,
  // skipped entirely when strength <= 0.
  private pLightWrap!: WebGLProgram
  private segmoCompositeTex: WebGLTexture | null = null
  private fboSegmoComposite: WebGLFramebuffer | null = null
  private segmoLightWrapStrength = 0.08
  // Foreground color cast: tints the camera frame toward the background's mean
  // color so the subject reads as lit by the virtual scene. Pure GPU — mean
  // colors live in the top mip of mipmapped textures and are sampled via
  // textureLod (no readback). Virtual path only, skipped when strength <= 0.
  private pMaskedFg!: WebGLProgram
  private pFgColorCast!: WebGLProgram
  private maskedFgTex: WebGLTexture | null = null
  private fboMaskedFg: WebGLFramebuffer | null = null
  private tintedVideoTex: WebGLTexture | null = null
  private fboTintedVideo: WebGLFramebuffer | null = null
  private segmoForegroundTintStrength = 0.15
  private _segmoBgMipmapsValid = false

  // textures
  private videoTex!: WebGLTexture
  private rawMaskTex!: WebGLTexture // R8 at proc res — uploaded from segmenter
  private maskA!: WebGLTexture // R8 ping
  private maskB!: WebGLTexture // R8 pong
  private emaTex!: WebGLTexture // R8, persistent across frames
  private bgDownTex!: WebGLTexture // RGBA half-res masked downsample
  private bgBlurPingTex!: WebGLTexture // RGBA half-res after H blur
  private bgBlurPongTex!: WebGLTexture // RGBA half-res after V blur
  private virtualBgTex: WebGLTexture | null = null

  // FBOs
  private fboMaskA!: WebGLFramebuffer
  private fboMaskB!: WebGLFramebuffer
  private fboEma!: WebGLFramebuffer
  private fboBgDown!: WebGLFramebuffer
  private fboBgBlurPing!: WebGLFramebuffer
  private fboBgBlurPong!: WebGLFramebuffer

  // Makeup overlay (GPU path).
  //
  // When a preset is active the composite shader writes to `compositeTex`
  // instead of the canvas, and a final makeup pass reads (composite +
  // landmarks + preset params) and writes the final image to the canvas.
  // When no preset is active the composite renders to the canvas as before
  // (zero overhead path).
  private pMakeupGl: WebGLProgram | null = null
  private compositeTex: WebGLTexture | null = null
  private fboComposite: WebGLFramebuffer | null = null
  private landmarksTex: WebGLTexture | null = null
  private _makeupActive = false
  private _makeupHasFace = false
  private _makeupPreset: {
    lipColor: [number, number, number]; lipAlpha: number
    blushColor: [number, number, number]; blushAlpha: number; blushRadius: number
    browColor: [number, number, number]; browAlpha: number
    lashColor: [number, number, number]; lashAlpha: number; lashThicknessPx: number
  } = {
    lipColor: [0, 0, 0], lipAlpha: 0,
    blushColor: [0, 0, 0], blushAlpha: 0, blushRadius: 0,
    browColor: [0, 0, 0], browAlpha: 0,
    lashColor: [0, 0, 0], lashAlpha: 0, lashThicknessPx: 0,
  }

  private halfW = 0
  private halfH = 0
  private hasEmaState = false
  private virtualImgPending: HTMLImageElement | null = null
  private virtualImgUploaded = false

  async init(canvas: HTMLCanvasElement, opts: GpuRendererInitOpts) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      premultipliedAlpha: true,
      // Must be true for captureStream() to read WebGL output correctly on Safari.
      // With false, the browser clears the buffer before captureStream can grab it.
      preserveDrawingBuffer: true,
      // Important for Safari: ensure the GPU is allowed.
      powerPreference: 'high-performance',
    })
    if (!gl) {
      pushMattingError({
        code: 'WEBGL2_INIT_FAILED',
        level: 'error',
        detail: 'getContext("webgl2") returned null',
      })
      throw new Error('WebGL2 unavailable')
    }
    this.gl = gl
    this.outW = opts.outW
    this.outH = opts.outH
    this.procW = opts.processingW
    this.procH = opts.processingH
    this.postCfg = opts.postProcessing
    this.upsamplingCfg = opts.upsampling
    this.halfW = Math.max(2, Math.floor(this.outW / 2))
    this.halfH = Math.max(2, Math.floor(this.outH / 2))

    canvas.width = this.outW
    canvas.height = this.outH
    gl.viewport(0, 0, this.outW, this.outH)
    // HTML element uploads (video, virtual bg image) get Y-flipped on upload so
    // that texture coord (0,0) corresponds to the BOTTOM-LEFT pixel of the source
    // image — matching WebGL's bottom-up coord system. Mask typed-array uploads
    // are not affected (UNPACK_FLIP_Y_WEBGL does not apply); we compensate by
    // sampling the mask with `(x, 1-y)` in the composite shader.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)

    try {
      this._buildPrograms()
      this._buildQuad()
      this._buildTexturesAndFBOs()
    } catch (e) {
      pushMattingError({
        code: 'POSTPROCESS_SHADER_COMPILE_FAILED',
        level: 'error',
        detail: e instanceof Error ? e.message : String(e),
      })
      throw e
    }
  }

  resizeProcessing(w: number, h: number) {
    if (w === this.procW && h === this.procH) return
    this.procW = w
    this.procH = h
    const gl = this.gl
    // Reallocate proc-sized textures (rawMask, maskA, maskB, ema)
    for (const tex of [
      this.rawMaskTex,
      this.maskA,
      this.maskB,
      this.emaTex,
    ]) {
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R8,
        w,
        h,
        0,
        gl.RED,
        gl.UNSIGNED_BYTE,
        null
      )
    }
    this.hasEmaState = false
  }

  resizeOutput(w: number, h: number) {
    if (w === this.outW && h === this.outH) return
    this.outW = w
    this.outH = h
    this.halfW = Math.max(2, Math.floor(w / 2))
    this.halfH = Math.max(2, Math.floor(h / 2))

    const gl = this.gl
    if (!gl) return

    // 1. Reallocate videoTex at new size
    if (this.videoTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    }

    // 2. Reallocate half-res textures
    if (this.bgDownTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.bgDownTex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.halfW, this.halfH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    }
    if (this.bgBlurPingTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.bgBlurPingTex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.halfW, this.halfH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    }
    if (this.bgBlurPongTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.bgBlurPongTex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.halfW, this.halfH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    }

    // 3. Clear lazily-allocated Segmo textures so they get recreated at the new size
    if (this.segmoFeatheredMaskTex) {
      gl.deleteTexture(this.segmoFeatheredMaskTex)
      this.segmoFeatheredMaskTex = null
    }
    if (this.fboSegmoFeatheredMask) {
      gl.deleteFramebuffer(this.fboSegmoFeatheredMask)
      this.fboSegmoFeatheredMask = null
    }
    if (this.segmoCompositeTex) {
      gl.deleteTexture(this.segmoCompositeTex)
      this.segmoCompositeTex = null
    }
    if (this.fboSegmoComposite) {
      gl.deleteFramebuffer(this.fboSegmoComposite)
      this.fboSegmoComposite = null
    }
    if (this.maskedFgTex) {
      gl.deleteTexture(this.maskedFgTex)
      this.maskedFgTex = null
    }
    if (this.fboMaskedFg) {
      gl.deleteFramebuffer(this.fboMaskedFg)
      this.fboMaskedFg = null
    }
    if (this.tintedVideoTex) {
      gl.deleteTexture(this.tintedVideoTex)
      this.tintedVideoTex = null
    }
    if (this.fboTintedVideo) {
      gl.deleteFramebuffer(this.fboTintedVideo)
      this.fboTintedVideo = null
    }

    // 4. Clear the makeup composite target so it gets recreated at the new
    //    size on the next frame that needs it.
    if (this.compositeTex) {
      gl.deleteTexture(this.compositeTex)
      this.compositeTex = null
    }
    if (this.fboComposite) {
      gl.deleteFramebuffer(this.fboComposite)
      this.fboComposite = null
    }

    // 5. Destroy Guided Filter so it gets recreated at the new size
    if (this.gf) {
      this.gf.destroy()
      this.gf = null
    }

    // 6. Update canvas dimensions
    const canvas = gl.canvas as HTMLCanvasElement
    if (canvas) {
      canvas.width = w
      canvas.height = h
    }
  }

  uploadMask(mask: Float32Array, w: number, h: number) {
    if (w !== this.procW || h !== this.procH) {
      this.resizeProcessing(w, h)
    }
    // Convert Float32 [0,1] → Uint8.
    const u8 = new Uint8Array(mask.length)
    for (let i = 0; i < mask.length; i++) {
      const v = mask[i]
      u8[i] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255)
    }
    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, this.rawMaskTex)
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      w,
      h,
      gl.RED,
      gl.UNSIGNED_BYTE,
      u8
    )
  }

  setVirtualBackground(img: HTMLImageElement | null) {
    if (img === null) {
      this.virtualImgPending = null
      this.virtualImgUploaded = false
      return
    }
    this.virtualImgPending = img
    this.virtualImgUploaded = false
  }

  setBlurRadius(px: number) {
    this.blurRadius = px
  }

  setMode(mode: 'blur' | 'virtual') {
    this.mode = mode
  }

  setMakeupActive(active: boolean) {
    this._makeupActive = active
  }

  /**
   * Configure the per-zone colors/alphas for the makeup shader. Called when
   * the user picks a preset — zero-alpha zones disable their respective
   * shader branches.
   */
  setMakeupPreset(p: {
    lipColor: [number, number, number]; lipAlpha: number
    blushColor: [number, number, number]; blushAlpha: number; blushRadius: number
    browColor: [number, number, number]; browAlpha: number
    lashColor: [number, number, number]; lashAlpha: number; lashThicknessPx: number
  }) {
    this._makeupPreset = p
  }

  /**
   * Upload the packed landmark buffer (NUM_ZONES rows of MAX_PER_ZONE (x,y)
   * pairs) into a small RG32F sampler texture. `hasFace` is false when no
   * face was detected this frame — the shader then leaves the image
   * untouched without us having to clear anything.
   */
  uploadLandmarks(packed: Float32Array, width: number, height: number, hasFace: boolean) {
    const gl = this.gl
    if (!gl) return
    this._makeupHasFace = hasFace
    if (!hasFace) return
    if (!this.landmarksTex) {
      this.landmarksTex = gl.createTexture()!
      gl.bindTexture(gl.TEXTURE_2D, this.landmarksTex)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      // FLIP_Y is global and on by default — for a raw float texture the
      // ordering must match what we wrote, so disable it for this upload.
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RG32F,
        width,
        height,
        0,
        gl.RG,
        gl.FLOAT,
        packed
      )
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
      return
    }
    gl.bindTexture(gl.TEXTURE_2D, this.landmarksTex)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      width,
      height,
      gl.RG,
      gl.FLOAT,
      packed
    )
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
  }

  setPostProcessing(cfg: PostProcessingConfig) {
    this.postCfg = cfg
    this.hasEmaState = false
  }

  setUpsampling(cfg: UpsamplingConfig) {
    this.upsamplingCfg = cfg
    if (cfg.method !== 'guided') {
      this.gf?.destroy()
      this.gf = null
    }
  }

  render(videoElement: HTMLVideoElement) {
    if (!videoElement || videoElement.videoWidth === 0) return
    const gl = this.gl

    // 1. Upload current video frame to videoTex (full output size).
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
    try {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        videoElement
      )
    } catch (e) {
      // Some browsers throw if the video frame isn't ready yet — skip this tick.
      void e
      return
    }

    // 2. Run post-processing chain on the mask at processing resolution.
    const procMaskTex = this._runPostProcessing()

    // 3. Upsample mask to full output resolution (bilinear or guided filter).
    const finalMaskTex = this._upsampleMask(procMaskTex)

    // 4. Build background (blurred camera or virtual image).
    const bgTex = this._buildBackground(finalMaskTex)

    // 5. Composite — segmo-style path is taken ONLY for virtual mode with an
    //    uploaded virtual background. The blur path (and the virtual-no-image
    //    fallback, which currently returns the blurred camera) falls through to
    //    the original composite below, unchanged.
    if (
      this.mode === 'virtual' &&
      this.virtualImgUploaded &&
      this.virtualBgTex !== null &&
      bgTex === this.virtualBgTex
    ) {
      this._compositeVirtualSegmo(bgTex, finalMaskTex)
      if (this._makeupActive && this.compositeTex) this._drawMakeupGl()
      gl.flush()
      return
    }

    // 5. Composite — straight to the canvas, unless makeup is active in which
    //    case we render into fboComposite first so the makeup shader can
    //    sample the result.
    const routeToFbo = this._routeCompositeToFbo()
    if (routeToFbo) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboComposite)
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    }
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pComposite)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
    gl.uniform1i(gl.getUniformLocation(this.pComposite, 'uVideo'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, bgTex)
    gl.uniform1i(gl.getUniformLocation(this.pComposite, 'uBg'), 1)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, finalMaskTex)
    gl.uniform1i(gl.getUniformLocation(this.pComposite, 'uMask'), 2)
    gl.uniform1f(
      gl.getUniformLocation(this.pComposite, 'uErosionRadius'),
      this.postCfg.erosion?.pixels ?? 0
    )
    gl.uniform2f(
      gl.getUniformLocation(this.pComposite, 'uOutTexel'),
      1 / this.outW,
      1 / this.outH
    )
    this._drawQuad()

    if (routeToFbo) this._drawMakeupGl()

    gl.flush()
  }

  // Allocate (or re-allocate after a resize) the composite FBO we redirect
  // the standard compositor to when makeup is active. Lazy: zero cost when
  // makeup is never used.
  private _ensureCompositeTarget(): boolean {
    const gl = this.gl
    if (this.compositeTex && this.fboComposite) return true
    const t = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, t)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.outW, this.outH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const f = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, f)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0)
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      gl.deleteTexture(t)
      gl.deleteFramebuffer(f)
      return false
    }
    this.compositeTex = t
    this.fboComposite = f
    return true
  }

  // True iff the next composite should be redirected to fboComposite (so the
  // makeup pass can sample it). False ⇒ render straight to canvas (zero-cost
  // path when makeup is off, or off because the GPU resources couldn't be
  // allocated).
  private _routeCompositeToFbo(): boolean {
    return this._makeupActive && this._ensureCompositeTarget()
  }

  private _drawMakeupGl() {
    if (!this._makeupActive || !this.pMakeupGl || !this.compositeTex || !this.landmarksTex) return
    const gl = this.gl
    const p = this.pMakeupGl
    const mp = this._makeupPreset

    // Reference values from MakeupOverlay.ts. Kept in JS-side constants only
    // to set the per-zone vertex-count uniforms.
    const N_LIP_OUTER = 20
    const N_LIP_INNER = 20
    const N_BROW_LEFT = 10
    const N_BROW_RIGHT = 10
    const N_LASH_LEFT = 8
    const N_LASH_RIGHT = 8

    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(p)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.compositeTex)
    gl.uniform1i(gl.getUniformLocation(p, 'uComposite'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.landmarksTex)
    gl.uniform1i(gl.getUniformLocation(p, 'uLandmarks'), 1)
    gl.uniform1i(gl.getUniformLocation(p, 'uHasFace'), this._makeupHasFace ? 1 : 0)
    gl.uniform2f(gl.getUniformLocation(p, 'uOutSize'), this.outW, this.outH)
    gl.uniform1i(gl.getUniformLocation(p, 'uNumLipOuter'),  N_LIP_OUTER)
    gl.uniform1i(gl.getUniformLocation(p, 'uNumLipInner'),  N_LIP_INNER)
    gl.uniform1i(gl.getUniformLocation(p, 'uNumBrowLeft'),  N_BROW_LEFT)
    gl.uniform1i(gl.getUniformLocation(p, 'uNumBrowRight'), N_BROW_RIGHT)
    gl.uniform1i(gl.getUniformLocation(p, 'uNumLashLeft'),  N_LASH_LEFT)
    gl.uniform1i(gl.getUniformLocation(p, 'uNumLashRight'), N_LASH_RIGHT)
    gl.uniform3fv(gl.getUniformLocation(p, 'uLipColor'),   mp.lipColor)
    gl.uniform1f (gl.getUniformLocation(p, 'uLipAlpha'),   mp.lipAlpha)
    gl.uniform3fv(gl.getUniformLocation(p, 'uBlushColor'), mp.blushColor)
    gl.uniform1f (gl.getUniformLocation(p, 'uBlushAlpha'), mp.blushAlpha)
    gl.uniform1f (gl.getUniformLocation(p, 'uBlushRadius'), mp.blushRadius)
    gl.uniform3fv(gl.getUniformLocation(p, 'uBrowColor'),  mp.browColor)
    gl.uniform1f (gl.getUniformLocation(p, 'uBrowAlpha'),  mp.browAlpha)
    gl.uniform3fv(gl.getUniformLocation(p, 'uLashColor'),  mp.lashColor)
    gl.uniform1f (gl.getUniformLocation(p, 'uLashAlpha'),  mp.lashAlpha)
    gl.uniform1f (gl.getUniformLocation(p, 'uLashThicknessPx'), mp.lashThicknessPx)
    this._drawQuad()
  }

  readPixels(x: number, y: number, w: number, h: number): Uint8Array {
    const out = new Uint8Array(w * h * 4)
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out)
    return out
  }

  private _upsampleMask(procMaskTex: WebGLTexture): WebGLTexture {
    if (this.upsamplingCfg.method !== 'guided') {
      return procMaskTex
    }
    if (!this.gf) {
      try {
        this.gf = new GpuGuidedFilter(this.gl, this.outW, this.outH)
      } catch (e) {
        pushMattingError({
          code: 'POSTPROCESS_SHADER_COMPILE_FAILED',
          level: 'warn',
          detail: e instanceof Error ? e.message : String(e),
        })
        this.upsamplingCfg = {}
        return procMaskTex
      }
    }
    const radius = this.upsamplingCfg.radius ?? 8
    const eps    = this.upsamplingCfg.eps    ?? 0.01
    return this.gf.run(this.videoTex, procMaskTex, radius, eps, this.vao)
  }

  destroy() {
    if (!this.gl) return
    this.gf?.destroy()
    this.gf = null
    const gl = this.gl
    const tex = [
      this.videoTex,
      this.rawMaskTex,
      this.maskA,
      this.maskB,
      this.emaTex,
      this.bgDownTex,
      this.bgBlurPingTex,
      this.bgBlurPongTex,
      this.virtualBgTex,
      this.segmoFeatheredMaskTex,
      this.segmoCompositeTex,
      this.maskedFgTex,
      this.tintedVideoTex,
      this.compositeTex,
      this.landmarksTex,
    ]
    for (const t of tex) if (t) gl.deleteTexture(t)
    const fbo = [
      this.fboMaskA,
      this.fboMaskB,
      this.fboEma,
      this.fboBgDown,
      this.fboBgBlurPing,
      this.fboBgBlurPong,
      this.fboSegmoFeatheredMask,
      this.fboSegmoComposite,
      this.fboMaskedFg,
      this.fboTintedVideo,
      this.fboComposite,
    ]
    for (const f of fbo) if (f) gl.deleteFramebuffer(f)
    if (this.quadBuffer) gl.deleteBuffer(this.quadBuffer)
    if (this.vao) gl.deleteVertexArray(this.vao)
    const programs = [
      this.pUploadMask,
      this.pEma,
      this.pCopyR,
      this.pMaskedDownsample,
      this.pMaskWeightedBlur,
      this.pMorphology,
      this.pComposite,
      this.pCompositeSegmo,
      this.pSegmoEdgeFeather,
      this.pLightWrap,
      this.pMaskedFg,
      this.pFgColorCast,
      this.pMakeupGl,
    ]
    for (const p of programs) if (p) gl.deleteProgram(p)
  }

  // ─────────────────────────────── internals ───────────────────────────────

  private _runPostProcessing(): WebGLTexture {
    const gl = this.gl
    gl.viewport(0, 0, this.procW, this.procH)
    let src = this.rawMaskTex
    let dstTex = this.maskA
    let dstFbo = this.fboMaskA
    const swap = () => {
      // swap A/B
      if (dstTex === this.maskA) {
        dstTex = this.maskB
        dstFbo = this.fboMaskB
      } else {
        dstTex = this.maskA
        dstFbo = this.fboMaskA
      }
    }
    const advance = () => {
      src = dstTex === this.maskA ? this.maskA : this.maskB
      swap()
    }

    // Closing (Dilation then Erosion to fill holes)
    if (this.postCfg.closing && this.postCfg.closing.radius > 0) {
      const r = this.postCfg.closing.radius
      this._applyMorphology(dstFbo, src, r) // Dilation
      advance()
      this._applyMorphology(dstFbo, src, -r) // Erosion
      advance()
    }

    // EMA
    if (this.postCfg.ema) {
      const alpha = this.postCfg.ema.alpha
      // out = alpha * src + (1 - alpha) * prev
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFbo)
      gl.useProgram(this.pEma)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, src)
      gl.uniform1i(gl.getUniformLocation(this.pEma, 'uTex'), 0)
      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, this.emaTex)
      gl.uniform1i(gl.getUniformLocation(this.pEma, 'uPrev'), 1)
      gl.uniform1f(
        gl.getUniformLocation(this.pEma, 'uAlpha'),
        this.hasEmaState ? alpha : 1.0
      )
      this._drawQuad()
      advance()
      // Copy current result into emaTex for next frame
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboEma)
      gl.useProgram(this.pCopyR)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, src)
      gl.uniform1i(gl.getUniformLocation(this.pCopyR, 'uTex'), 0)
      this._drawQuad()
      this.hasEmaState = true
    }

    return src
  }

  private _applyMorphology(fbo: WebGLFramebuffer, src: WebGLTexture, radius: number) {
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.useProgram(this.pMorphology)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, src)
    gl.uniform1i(gl.getUniformLocation(this.pMorphology, 'uTex'), 0)
    gl.uniform1f(gl.getUniformLocation(this.pMorphology, 'uRadius'), radius)
    gl.uniform2f(
      gl.getUniformLocation(this.pMorphology, 'uTexel'),
      1 / this.procW,
      1 / this.procH
    )
    this._drawQuad()
  }

  private _buildBackground(maskTex: WebGLTexture): WebGLTexture {
    const gl = this.gl

    if (this.mode === 'virtual') {
      // Lazy upload virtual bg image when ready.
      if (this.virtualImgPending && !this.virtualImgUploaded) {
        const img = this.virtualImgPending
        if (img.complete && img.naturalWidth > 0) {
          if (!this.virtualBgTex) {
            this.virtualBgTex = gl.createTexture()!
          }
          gl.bindTexture(gl.TEXTURE_2D, this.virtualBgTex)
          gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            img
          )
          gl.texParameteri(
            gl.TEXTURE_2D,
            gl.TEXTURE_MIN_FILTER,
            gl.LINEAR
          )
          gl.texParameteri(
            gl.TEXTURE_2D,
            gl.TEXTURE_MAG_FILTER,
            gl.LINEAR
          )
          gl.texParameteri(
            gl.TEXTURE_2D,
            gl.TEXTURE_WRAP_S,
            gl.CLAMP_TO_EDGE
          )
          gl.texParameteri(
            gl.TEXTURE_2D,
            gl.TEXTURE_WRAP_T,
            gl.CLAMP_TO_EDGE
          )
          this.virtualImgUploaded = true
        }
      }
      if (this.virtualBgTex && this.virtualImgUploaded) {
        return this.virtualBgTex
      }
      // Fallback to blur if image not ready.
    }

    // Blur path: masked downsample → mask-weighted gaussian H → V on half-res buffers.
    const radius = Math.max(1, this.blurRadius / 2)
    gl.viewport(0, 0, this.halfW, this.halfH)

    // Stage 1: masked downsample — 3x3 weighted average sampling the FULL-res
    // source, normalised by accumulated bgWeight so transition-zone pixels
    // don't darken the result (this is what causes the halo if omitted).
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBgDown)
    gl.useProgram(this.pMaskedDownsample)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
    gl.uniform1i(gl.getUniformLocation(this.pMaskedDownsample, 'uFrame'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(gl.getUniformLocation(this.pMaskedDownsample, 'uMask'), 1)
    gl.uniform2f(
      gl.getUniformLocation(this.pMaskedDownsample, 'uSourceTexelSize'),
      1.0 / this.outW,
      1.0 / this.outH
    )
    this._drawQuad()

    // Stage 2: horizontal mask-weighted gaussian.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBgBlurPing)
    gl.useProgram(this.pMaskWeightedBlur)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.bgDownTex)
    gl.uniform1i(gl.getUniformLocation(this.pMaskWeightedBlur, 'uImage'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(gl.getUniformLocation(this.pMaskWeightedBlur, 'uMask'), 1)
    gl.uniform2f(gl.getUniformLocation(this.pMaskWeightedBlur, 'uDirection'), 1.0, 0.0)
    gl.uniform2f(gl.getUniformLocation(this.pMaskWeightedBlur, 'uTexelSize'), 1.0 / this.halfW, 1.0 / this.halfH)
    gl.uniform1f(gl.getUniformLocation(this.pMaskWeightedBlur, 'uRadius'), radius)
    this._drawQuad()

    // Stage 3: vertical mask-weighted gaussian.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBgBlurPong)
    gl.useProgram(this.pMaskWeightedBlur)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.bgBlurPingTex)
    gl.uniform1i(gl.getUniformLocation(this.pMaskWeightedBlur, 'uImage'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(gl.getUniformLocation(this.pMaskWeightedBlur, 'uMask'), 1)
    gl.uniform2f(gl.getUniformLocation(this.pMaskWeightedBlur, 'uDirection'), 0.0, 1.0)
    gl.uniform2f(gl.getUniformLocation(this.pMaskWeightedBlur, 'uTexelSize'), 1.0 / this.halfW, 1.0 / this.halfH)
    gl.uniform1f(gl.getUniformLocation(this.pMaskWeightedBlur, 'uRadius'), radius)
    this._drawQuad()

    return this.bgBlurPongTex
  }

  /**
   * Segmo-style compositor for virtual backgrounds.
   *
   * Runs the foreground-recovery composite shader: edge-adaptive sharpening from
   * the camera gradient, closed-form alpha matting on a 13-tap cross pattern in
   * the transition zone, chroma-aware color-separation gate, and the VFX
   * decontamination equation `output = I + (B_new − B_old) * (1 − α)` to remove
   * the old background's color contribution from contaminated edge pixels.
   *
   * Intentionally does NOT include the erosion step from the standard compositor:
   * segmo's transition-zone matting subsumes that need. The user-selectable
   * postprocess chain (morphology/EMA/guided-upsample) ran upstream and
   * is unaffected.
   */
  private _segmoLogged = false
  private _ensureSegmoFeatherTarget() {
    if (this.segmoFeatheredMaskTex && this.fboSegmoFeatheredMask) return
    const gl = this.gl
    const t = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, t)
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.R8, this.outW, this.outH, 0,
      gl.RED, gl.UNSIGNED_BYTE, null
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const f = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, f)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0
    )
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`segmo feather FBO incomplete: 0x${status.toString(16)}`)
    }
    this.segmoFeatheredMaskTex = t
    this.fboSegmoFeatheredMask = f
  }

  private _ensureSegmoCompositeTarget() {
    if (this.segmoCompositeTex && this.fboSegmoComposite) return
    const gl = this.gl
    const t = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, t)
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, this.outW, this.outH, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const f = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, f)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0
    )
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`segmo composite FBO incomplete: 0x${status.toString(16)}`)
    }
    this.segmoCompositeTex = t
    this.fboSegmoComposite = f
  }

  private _ensureSegmoTintTargets() {
    if (this.maskedFgTex && this.fboMaskedFg && this.tintedVideoTex && this.fboTintedVideo) return
    const gl = this.gl
    // Masked foreground: mipmapped, RGBA8. rgb = video * weight, a = weight.
    // Top mip yields (sum(video*weight), sum(weight)) so the cast shader can
    // recover the foreground mean color as rgb/a.
    const mft = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, mft)
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, this.outW, this.outH, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const mff = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, mff)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mft, 0
    )
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('segmo masked-fg FBO incomplete')
    }

    // Tinted video target: plain RGBA8, no mipmaps needed (it's consumed by
    // a full-resolution sampler at mip 0).
    const tvt = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tvt)
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, this.outW, this.outH, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const tvf = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, tvf)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tvt, 0
    )
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('segmo tinted-video FBO incomplete')
    }

    this.maskedFgTex = mft
    this.fboMaskedFg = mff
    this.tintedVideoTex = tvt
    this.fboTintedVideo = tvf
  }

  private _compositeVirtualSegmo(bgTex: WebGLTexture, maskTex: WebGLTexture) {
    const gl = this.gl
    if (!this._segmoLogged) {
      this._segmoLogged = true
      // One-shot confirmation that the segmo composite path is active.
      console.log('[WebGl2Renderer] segmo virtual-bg composite active')
    }

    // Pass A: edge-only feather (widens the transition band near silhouettes,
    // leaves interior/exterior alone). Output is R8 at output resolution.
    this._ensureSegmoFeatherTarget()
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboSegmoFeatheredMask)
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pSegmoEdgeFeather)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(gl.getUniformLocation(this.pSegmoEdgeFeather, 'uMask'), 0)
    gl.uniform2f(
      gl.getUniformLocation(this.pSegmoEdgeFeather, 'uTexel'),
      1 / this.outW,
      1 / this.outH
    )
    gl.uniform1f(
      gl.getUniformLocation(this.pSegmoEdgeFeather, 'uRadius'),
      this.segmoFeatherRadius
    )
    this._drawQuad()

    // Passes T1+T2 (foreground color cast — pure GPU via mipmaps). Skipped when
    // strength <= 0; otherwise produces tintedVideoTex which feeds the composite
    // in place of videoTex. Bg mipmaps are regenerated lazily after each upload.
    const useTint = this.segmoForegroundTintStrength > 0.0
    let videoSrc: WebGLTexture = this.videoTex
    if (useTint) {
      // Lazy: ensure the bg texture has a mipmap chain after upload. Detects
      // upload-completed transitions via virtualImgUploaded (reset to false in
      // setVirtualBackground on every new image).
      if (this.virtualImgUploaded && !this._segmoBgMipmapsValid && this.virtualBgTex) {
        gl.bindTexture(gl.TEXTURE_2D, this.virtualBgTex)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR)
        gl.generateMipmap(gl.TEXTURE_2D)
        this._segmoBgMipmapsValid = true
      } else if (!this.virtualImgUploaded) {
        this._segmoBgMipmapsValid = false
      }

      if (this._segmoBgMipmapsValid) {
        this._ensureSegmoTintTargets()
        // T1: render video × foreground weight to maskedFgTex. Weight in alpha
        // lets the cast shader recover the weighted mean as rgb/a from the top
        // mip level.
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboMaskedFg)
        gl.viewport(0, 0, this.outW, this.outH)
        gl.useProgram(this.pMaskedFg)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
        gl.uniform1i(gl.getUniformLocation(this.pMaskedFg, 'uVideo'), 0)
        gl.activeTexture(gl.TEXTURE1)
        gl.bindTexture(gl.TEXTURE_2D, this.segmoFeatheredMaskTex!)
        gl.uniform1i(gl.getUniformLocation(this.pMaskedFg, 'uMask'), 1)
        this._drawQuad()
        // Build the mip pyramid so textureLod can fetch the global mean.
        gl.bindTexture(gl.TEXTURE_2D, this.maskedFgTex!)
        gl.generateMipmap(gl.TEXTURE_2D)

        // T2: tint video toward bg's tint. Both means read from top mip via
        // textureLod inside the shader.
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboTintedVideo)
        gl.viewport(0, 0, this.outW, this.outH)
        gl.useProgram(this.pFgColorCast)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
        gl.uniform1i(gl.getUniformLocation(this.pFgColorCast, 'uVideo'), 0)
        gl.activeTexture(gl.TEXTURE1)
        gl.bindTexture(gl.TEXTURE_2D, this.maskedFgTex!)
        gl.uniform1i(gl.getUniformLocation(this.pFgColorCast, 'uFgMasked'), 1)
        gl.activeTexture(gl.TEXTURE2)
        gl.bindTexture(gl.TEXTURE_2D, bgTex)
        gl.uniform1i(gl.getUniformLocation(this.pFgColorCast, 'uBg'), 2)
        gl.uniform1f(
          gl.getUniformLocation(this.pFgColorCast, 'uStrength'),
          this.segmoForegroundTintStrength
        )
        this._drawQuad()
        videoSrc = this.tintedVideoTex!
      }
    }

    // Pass B: segmo composite, fed by the feathered mask. When light wrap is
    // enabled we render to an intermediate texture; otherwise straight to the
    // final target (canvas or fboComposite if makeup is on).
    const useLightWrap = this.segmoLightWrapStrength > 0.0
    const finalRouteToFbo = this._routeCompositeToFbo()
    if (useLightWrap) {
      this._ensureSegmoCompositeTarget()
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboSegmoComposite)
    } else if (finalRouteToFbo) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboComposite)
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    }
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pCompositeSegmo)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, videoSrc)
    gl.uniform1i(gl.getUniformLocation(this.pCompositeSegmo, 'uVideo'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, bgTex)
    gl.uniform1i(gl.getUniformLocation(this.pCompositeSegmo, 'uBg'), 1)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, this.segmoFeatheredMaskTex!)
    gl.uniform1i(gl.getUniformLocation(this.pCompositeSegmo, 'uMask'), 2)
    gl.uniform2f(
      gl.getUniformLocation(this.pCompositeSegmo, 'uOutTexel'),
      1 / this.outW,
      1 / this.outH
    )
    gl.uniform1f(
      gl.getUniformLocation(this.pCompositeSegmo, 'uErosionRadius'),
      this.postCfg.erosion?.pixels ?? 0
    )
    this._drawQuad()

    if (!useLightWrap) return

    // Pass C: light wrap — mix a small amount of the background color into the
    // narrow edge band so the subject looks lit by the virtual scene. Target
    // is fboComposite when makeup is on, otherwise the canvas directly.
    if (finalRouteToFbo) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboComposite)
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    }
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pLightWrap)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.segmoCompositeTex!)
    gl.uniform1i(gl.getUniformLocation(this.pLightWrap, 'uComposite'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, bgTex)
    gl.uniform1i(gl.getUniformLocation(this.pLightWrap, 'uBg'), 1)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, this.segmoFeatheredMaskTex!)
    gl.uniform1i(gl.getUniformLocation(this.pLightWrap, 'uMask'), 2)
    gl.uniform1f(
      gl.getUniformLocation(this.pLightWrap, 'uStrength'),
      this.segmoLightWrapStrength
    )
    this._drawQuad()
  }

  private _drawQuad() {
    const gl = this.gl
    gl.bindVertexArray(this.vao)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }

  private _buildQuad() {
    const gl = this.gl
    this.vao = gl.createVertexArray()!
    gl.bindVertexArray(this.vao)
    this.quadBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer)
    // Full-screen triangle covering [-1,1]² with UVs in [0,1]² (Y flipped to
    // sample non-mirrored video).
    // Vertex shader expects only position; it computes UV from gl_Position.
    const verts = new Float32Array([-1, -1, 3, -1, -1, 3])
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.bindVertexArray(null)
  }

  private _buildTexturesAndFBOs() {
    const gl = this.gl

    const makeTex = (
      w: number,
      h: number,
      internal: number,
      format: number,
      type: number,
      filter: number = gl.LINEAR
    ) => {
      const t = gl.createTexture()!
      gl.bindTexture(gl.TEXTURE_2D, t)
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      return t
    }
    const makeFbo = (tex: WebGLTexture) => {
      const f = gl.createFramebuffer()!
      gl.bindFramebuffer(gl.FRAMEBUFFER, f)
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        tex,
        0
      )
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`FBO incomplete: 0x${status.toString(16)}`)
      }
      return f
    }

    this.videoTex = makeTex(
      this.outW,
      this.outH,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE
    )
    // Mask textures at processing res
    this.rawMaskTex = makeTex(
      this.procW,
      this.procH,
      gl.R8,
      gl.RED,
      gl.UNSIGNED_BYTE
    )
    this.maskA = makeTex(
      this.procW,
      this.procH,
      gl.R8,
      gl.RED,
      gl.UNSIGNED_BYTE
    )
    this.maskB = makeTex(
      this.procW,
      this.procH,
      gl.R8,
      gl.RED,
      gl.UNSIGNED_BYTE
    )
    this.emaTex = makeTex(
      this.procW,
      this.procH,
      gl.R8,
      gl.RED,
      gl.UNSIGNED_BYTE
    )
    this.fboMaskA = makeFbo(this.maskA)
    this.fboMaskB = makeFbo(this.maskB)
    this.fboEma = makeFbo(this.emaTex)

    // BG half-res buffers (RGBA8)
    this.bgDownTex = makeTex(this.halfW, this.halfH, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE)
    this.bgBlurPingTex = makeTex(this.halfW, this.halfH, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE)
    this.bgBlurPongTex = makeTex(this.halfW, this.halfH, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE)
    this.fboBgDown = makeFbo(this.bgDownTex)
    this.fboBgBlurPing = makeFbo(this.bgBlurPingTex)
    this.fboBgBlurPong = makeFbo(this.bgBlurPongTex)

    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  }

  private _compile(stage: number, src: string): WebGLShader {
    const gl = this.gl
    const sh = gl.createShader(stage)!
    gl.shaderSource(sh, src)
    gl.compileShader(sh)
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) ?? '<no log>'
      gl.deleteShader(sh)
      throw new Error(`Shader compile failed: ${log}\nSrc:\n${src}`)
    }
    return sh
  }

  private _link(vsSrc: string, fsSrc: string): WebGLProgram {
    const gl = this.gl
    const vs = this._compile(gl.VERTEX_SHADER, vsSrc)
    const fs = this._compile(gl.FRAGMENT_SHADER, fsSrc)
    const p = gl.createProgram()!
    gl.attachShader(p, vs)
    gl.attachShader(p, fs)
    gl.bindAttribLocation(p, 0, 'aPos')
    gl.linkProgram(p)
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p) ?? '<no log>'
      gl.deleteProgram(p)
      throw new Error(`Program link failed: ${log}`)
    }
    gl.deleteShader(vs)
    gl.deleteShader(fs)
    return p
  }

  private _buildPrograms() {
    const VS = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  // Map clip-space [-1,1] to UV [0,1].
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`

    const FS_COPY_R = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
  fragColor = texture(uTex, vUv);
}`

    const FS_EMA = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTex;
uniform sampler2D uPrev;
uniform float uAlpha; // 1.0 means "no smoothing — use current"
out vec4 fragColor;
void main() {
  float cur = texture(uTex, vUv).r;
  float prev = texture(uPrev, vUv).r;
  fragColor = vec4(uAlpha * cur + (1.0 - uAlpha) * prev, 0.0, 0.0, 1.0);
}`

    const FS_MASKED_DOWNSAMPLE = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uFrame;
uniform sampler2D uMask;
uniform vec2 uSourceTexelSize; // 1/srcW, 1/srcH (full-res input)
out vec4 fragColor;
void main() {
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 sampleCoord = vUv + vec2(float(dx), float(dy)) * uSourceTexelSize;
      float fg = texture(uMask, sampleCoord).r;
      float bgWeight = 1.0 - smoothstep(0.12, 0.55, fg);
      acc += texture(uFrame, sampleCoord).rgb * bgWeight;
      wsum += bgWeight;
    }
  }
  // Fallback: if the whole 3x3 is foreground, use the center sample unweighted
  // (this region will be hidden by the foreground in the composite anyway).
  if (wsum < 0.001) {
    acc = texture(uFrame, vUv).rgb;
    wsum = 1.0;
  }
  fragColor = vec4(acc / wsum, 1.0);
}`

    const FS_MASK_WEIGHTED_BLUR = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uImage;
uniform sampler2D uMask;
uniform vec2 uDirection;
uniform vec2 uTexelSize;
uniform float uRadius;
out vec4 fragColor;
void main() {
  float sigma = uRadius;
  float twoSigmaSq = 2.0 * sigma * sigma;
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  const int MAX_SAMPLES = 16;
  int radius = int(min(float(MAX_SAMPLES), ceil(uRadius)));
  for (int i = -MAX_SAMPLES; i <= MAX_SAMPLES; ++i) {
    float offset = float(i);
    if (abs(offset) > float(radius)) continue;
    float gaussW = exp(-(offset * offset) / twoSigmaSq);
    vec2 sampleCoord = vUv + uDirection * uTexelSize * offset;
    float maskVal = texture(uMask, sampleCoord).r;
    // Floor at 0.001 to avoid div-by-zero; small enough to hide foreground ghosts.
    float maskW = max(1.0 - maskVal, 0.001);
    float w = gaussW * maskW;
    acc += texture(uImage, sampleCoord).rgb * w;
    wsum += w;
  }
  fragColor = vec4(acc / max(wsum, 0.001), 1.0);
}`

    const FS_COMPOSITE = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uVideo;
uniform sampler2D uBg;
uniform sampler2D uMask;
uniform float uErosionRadius; // pixels at output resolution, 0 = disabled
uniform vec2 uOutTexel;       // vec2(1/outW, 1/outH)
out vec4 fragColor;
void main() {
  vec3 fg = texture(uVideo, vUv).rgb;
  vec3 bg = texture(uBg, vUv).rgb;
  // Erosion applied here at output resolution so that uErosionRadius is measured
  // in actual output pixels — not in the coarse processing-resolution pixels that
  // would produce large blocky artefacts after upsampling.
  // Diamond kernel (H + V in one pass): accurate enough for edge trimming.
  float m = texture(uMask, vUv).r;
  if (uErosionRadius > 0.0) {
    for (int i = 1; i <= 16; i++) {
      if (float(i) > uErosionRadius) break;
      float fi = float(i);
      m = min(m, texture(uMask, vUv + vec2(uOutTexel.x * fi, 0.0)).r);
      m = min(m, texture(uMask, vUv - vec2(uOutTexel.x * fi, 0.0)).r);
      m = min(m, texture(uMask, vUv + vec2(0.0, uOutTexel.y * fi)).r);
      m = min(m, texture(uMask, vUv - vec2(0.0, uOutTexel.y * fi)).r);
    }
  }
  // +0.035 foreground bias preserves edges that conservative segmentation models clip.
  float t = smoothstep(0.26, 0.72, clamp(m + 0.035, 0.0, 1.0));
  fragColor = vec4(mix(bg, fg, t), 1.0);
}`

    this.pUploadMask = this._link(VS, FS_COPY_R)
    this.pEma = this._link(VS, FS_EMA)
    this.pCopyR = this._link(VS, FS_COPY_R)
    this.pMaskedDownsample = this._link(VS, FS_MASKED_DOWNSAMPLE)
    this.pMaskWeightedBlur = this._link(VS, FS_MASK_WEIGHTED_BLUR)
    
    const FS_MORPHOLOGY = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTex;
uniform float uRadius; // Positive = Dilation, Negative = Erosion
uniform vec2 uTexel;
out vec4 fragColor;
void main() {
  float r = abs(uRadius);
  float val = texture(uTex, vUv).r;
  for (float i = 1.0; i <= 8.0; i++) {
    if (i > r) break;
    vec2 off = uTexel * i;
    float v1 = texture(uTex, vUv + vec2(off.x, 0.0)).r;
    float v2 = texture(uTex, vUv - vec2(off.x, 0.0)).r;
    float v3 = texture(uTex, vUv + vec2(0.0, off.y)).r;
    float v4 = texture(uTex, vUv - vec2(0.0, off.y)).r;
    if (uRadius > 0.0) {
      val = max(val, max(max(v1, v2), max(v3, v4)));
    } else {
      val = min(val, min(min(v1, v2), min(v3, v4)));
    }
  }
  fragColor = vec4(val, 0.0, 0.0, 1.0);
}`
    this.pMorphology = this._link(VS, FS_MORPHOLOGY)
    this.pComposite = this._link(VS, FS_COMPOSITE)

    // Makeup pass: reads the already-composed image from `uComposite`,
    // evaluates per-pixel polygon / polyline tests against the packed
    // landmark texture, and writes the final image to the canvas.
    //
    // Zone layout matches MakeupOverlay.ts. Per-pixel cost is bounded by
    // MAX_PER_ZONE iterations per zone — about ~120 ops per pixel in the
    // worst-case preset. Branches on per-zone alpha are uniform-controlled
    // (same across the whole frame) so they cost nothing on a GPU warp.
    const FS_MAKEUP_GL = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uComposite;
uniform sampler2D uLandmarks; // RG32F, MAX_PER_ZONE × NUM_ZONES
uniform bool uHasFace;
uniform vec2 uOutSize;        // (outW, outH) in pixels

uniform int uNumLipOuter;
uniform int uNumLipInner;
uniform int uNumBrowLeft;
uniform int uNumBrowRight;
uniform int uNumLashLeft;
uniform int uNumLashRight;

uniform vec3 uLipColor;     uniform float uLipAlpha;
uniform vec3 uBlushColor;   uniform float uBlushAlpha;  uniform float uBlushRadius;
uniform vec3 uBrowColor;    uniform float uBrowAlpha;
uniform vec3 uLashColor;    uniform float uLashAlpha;   uniform float uLashThicknessPx;

const int ZONE_LIP_OUTER  = 0;
const int ZONE_LIP_INNER  = 1;
const int ZONE_BROW_LEFT  = 2;
const int ZONE_BROW_RIGHT = 3;
const int ZONE_LASH_LEFT  = 4;
const int ZONE_LASH_RIGHT = 5;
const int ZONE_BLUSH      = 6;
const int MAX_PER_ZONE    = 20;

out vec4 fragColor;

vec2 fetchLmPx(int zone, int idx) {
  // Landmarks are normalized in image space (top-left origin). Convert to
  // pixel coordinates so distances are isotropic regardless of aspect ratio.
  vec2 lm = texelFetch(uLandmarks, ivec2(idx, zone), 0).xy;
  return lm * uOutSize;
}

float distToSegmentSq(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float denom = max(dot(ba, ba), 1e-6);
  float h = clamp(dot(pa, ba) / denom, 0.0, 1.0);
  vec2 d = pa - ba * h;
  return dot(d, d);
}

// Even-odd point-in-polygon with anti-aliased edge. Returns a soft mask in
// [0,1]. The polygon vertices live in row \`zone\`, indices 0..n-1.
float polygonMask(vec2 p, int zone, int n) {
  bool inside = false;
  float minDistSq = 1e18;
  vec2 prev = fetchLmPx(zone, n - 1);
  for (int i = 0; i < MAX_PER_ZONE; i++) {
    if (i >= n) break;
    vec2 cur = fetchLmPx(zone, i);
    // Even-odd ray casting (horizontal ray to +x). When the edge straddles
    // p.y, prev.y - cur.y is guaranteed non-zero — direct division is safe
    // and signed correctly (no max() needed, no sign-flipping hack).
    if ((cur.y > p.y) != (prev.y > p.y)) {
      float t = (p.y - cur.y) / (prev.y - cur.y);
      float xCross = cur.x + t * (prev.x - cur.x);
      if (p.x < xCross) inside = !inside;
    }
    minDistSq = min(minDistSq, distToSegmentSq(p, prev, cur));
    prev = cur;
  }
  float dist = sqrt(minDistSq);
  float signedDist = inside ? dist : -dist;
  return smoothstep(-1.0, 1.0, signedDist);
}

// Distance (in pixels) from p to the polyline defined by row \`zone\` of
// length n. Polyline is open (not closed back to start).
float polylineDistSq(vec2 p, int zone, int n) {
  float minD = 1e18;
  for (int i = 0; i < MAX_PER_ZONE - 1; i++) {
    if (i >= n - 1) break;
    vec2 a = fetchLmPx(zone, i);
    vec2 b = fetchLmPx(zone, i + 1);
    minD = min(minD, distToSegmentSq(p, a, b));
  }
  return minD;
}

void main() {
  vec3 base = texture(uComposite, vUv).rgb;
  if (!uHasFace) { fragColor = vec4(base, 1.0); return; }

  // Composite UV: bottom-left origin (WebGL). Landmark UV: top-left origin.
  // Flip Y so the two coordinate systems agree.
  vec2 p = vec2(vUv.x, 1.0 - vUv.y) * uOutSize;

  vec3 result = base;

  if (uBlushAlpha > 0.0) {
    vec2 bl = fetchLmPx(ZONE_BLUSH, 0);
    vec2 br = fetchLmPx(ZONE_BLUSH, 1);
    float dl = length(p - bl);
    float dr = length(p - br);
    float r  = uBlushRadius;
    float blushMask = 1.0 - smoothstep(0.0, r, min(dl, dr));
    result = mix(result, result * uBlushColor, blushMask * uBlushAlpha);
  }

  if (uBrowAlpha > 0.0) {
    float mL = polygonMask(p, ZONE_BROW_LEFT,  uNumBrowLeft);
    float mR = polygonMask(p, ZONE_BROW_RIGHT, uNumBrowRight);
    float browMask = max(mL, mR);
    result = mix(result, result * uBrowColor, browMask * uBrowAlpha);
  }

  if (uLipAlpha > 0.0) {
    float outer = polygonMask(p, ZONE_LIP_OUTER, uNumLipOuter);
    float inner = polygonMask(p, ZONE_LIP_INNER, uNumLipInner);
    float lipMask = clamp(outer - inner, 0.0, 1.0);
    result = mix(result, result * uLipColor, lipMask * uLipAlpha);
  }

  if (uLashAlpha > 0.0) {
    float dL = sqrt(polylineDistSq(p, ZONE_LASH_LEFT,  uNumLashLeft));
    float dR = sqrt(polylineDistSq(p, ZONE_LASH_RIGHT, uNumLashRight));
    float d  = min(dL, dR);
    float halfThick = uLashThicknessPx * 0.5;
    float lashMask = 1.0 - smoothstep(halfThick - 1.0, halfThick, d);
    result = mix(result, uLashColor, lashMask * uLashAlpha);
  }

  fragColor = vec4(result, 1.0);
}`
    this.pMakeupGl = this._link(VS, FS_MAKEUP_GL)

    // Segmo-style compositor for virtual backgrounds. Ported from
    // eyalfishler/segmo (src/shaders.ts COMPOSITE_SHADER, MIT).
    // Implements:
    //   - Edge-adaptive sharpening using camera RGB gradient
    //   - Closed-form alpha matting on a 13-tap cross pattern
    //   - Chroma-aware color-separation gate (disables matting when F≈B)
    //   - Foreground recovery: output = I + (B_new − B_old) * (1 − α)
    // Not used by the blur path.
    const FS_COMPOSITE_SEGMO = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVideo;     // Full-res camera frame
uniform sampler2D uBg;        // Virtual background (full-res)
uniform sampler2D uMask;      // Final processed mask
uniform vec2 uOutTexel;       // (1/outW, 1/outH)
uniform float uErosionRadius;
out vec4 fragColor;

// Cross-shaped sample pattern: wider reach for fg/bg color estimation (13 samples)
const vec2 mOff[13] = vec2[13](
  vec2(0.0, 0.0),
  vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(0.0, -1.0), vec2(0.0, 1.0),
  vec2(-2.0, 0.0), vec2(2.0, 0.0), vec2(0.0, -2.0), vec2(0.0, 2.0),
  vec2(-3.0, 0.0), vec2(3.0, 0.0), vec2(0.0, -3.0), vec2(0.0, 3.0)
);

void main() {
  float rawMask = texture(uMask, vUv).r;
  if (uErosionRadius > 0.0) {
    for (int i = 1; i <= 16; i++) {
      if (float(i) > uErosionRadius) break;
      float fi = float(i);
      rawMask = min(rawMask, texture(uMask, vUv + vec2(uOutTexel.x * fi, 0.0)).r);
      rawMask = min(rawMask, texture(uMask, vUv - vec2(uOutTexel.x * fi, 0.0)).r);
      rawMask = min(rawMask, texture(uMask, vUv + vec2(0.0, uOutTexel.y * fi)).r);
      rawMask = min(rawMask, texture(uMask, vUv - vec2(0.0, uOutTexel.y * fi)).r);
    }
  }
  vec3 I = texture(uVideo, vUv).rgb;

  // Edge-adaptive sharpening: narrow the mask transition at strong camera edges
  // (shoulders), widen it at weak edges (hair).
  vec3 dx = I - texture(uVideo, vUv + vec2(uOutTexel.x, 0.0)).rgb;
  vec3 dy = I - texture(uVideo, vUv + vec2(0.0, uOutTexel.y)).rgb;
  float edgeStrength = dot(dx, dx) + dot(dy, dy);
  float sharpness = smoothstep(0.001, 0.02, edgeStrength);
  float lo = mix(0.15, 0.35, sharpness);
  float hi = mix(0.85, 0.65, sharpness);
  float mask = smoothstep(lo, hi, rawMask);

  vec3 newBg = texture(uBg, vUv).rgb;

  // Default output: standard alpha composite (used outside the transition zone).
  vec3 result = mix(newBg, I, mask);

  // Foreground recovery in transition zone [0.02, 0.98].
  // Camera pixel is contaminated: I = F_true * α + B_old * (1 − α).
  // We want: output = F_true * α + B_new * (1 − α).
  // Therefore: output = I + (B_new − B_old) * (1 − α).
  float inTransition = step(0.02, mask) * step(mask, 0.98);
  if (inTransition > 0.5) {
    vec3 fgColor = vec3(0.0);
    vec3 bgColor = vec3(0.0);
    float fgWeight = 0.0;
    float bgWeight = 0.0;
    vec2 sampleStep = uOutTexel * 4.0;

    for (int i = 0; i < 13; i++) {
      vec2 sc = vUv + mOff[i] * sampleStep;
      float m = texture(uMask, sc).r;
      vec3 col = texture(uVideo, sc).rgb;
      float dist = length(mOff[i]);
      float proximity = 1.0 / (1.0 + dist);
      float fw = smoothstep(0.6, 0.9, m) * proximity;
      float bw = smoothstep(0.4, 0.1, m) * proximity;
      fgColor += col * fw;
      fgWeight += fw;
      bgColor += col * bw;
      bgWeight += bw;
    }

    float hasBoth = step(0.01, fgWeight) * step(0.01, bgWeight);
    if (hasBoth > 0.5) {
      vec3 F = fgColor / fgWeight;
      vec3 B = bgColor / bgWeight;
      vec3 FB = F - B;
      float denom = dot(FB, FB);

      // Chroma-aware separation gate: disable matting when foreground and
      // background colors are too similar (otherwise α is numerically unstable).
      const vec3 lumW = vec3(0.299, 0.587, 0.114);
      float fbLumDiff = dot(FB, lumW);
      vec3 fbChromaDiff = FB - fbLumDiff;
      float perceptualDenom = fbLumDiff * fbLumDiff + dot(fbChromaDiff, fbChromaDiff) * 3.0;
      float colorSeparation = smoothstep(0.02, 0.08, perceptualDenom);

      float mattedAlpha = clamp(dot(I - B, FB) / max(denom, 0.01), 0.0, 1.0);

      float blendFactor = smoothstep(0.02, 0.15, rawMask)
                        * (1.0 - smoothstep(0.9, 1.0, rawMask))
                        * colorSeparation;
      float alpha = mix(mask, mattedAlpha, blendFactor * 0.8);

      vec3 recovered = I + (newBg - B) * (1.0 - alpha);
      result = mix(result, clamp(recovered, 0.0, 1.0), blendFactor);
    }
  }

  fragColor = vec4(result, 1.0);
}`
    this.pCompositeSegmo = this._link(VS, FS_COMPOSITE_SEGMO)

    // Edge-only feather. Ported from eyalfishler/segmo (EDGE_FEATHER_SHADER, MIT).
    // Detects edges (max neighbor mask diff), then blends a 5×5 gaussian-blurred
    // mask value over the original only where an edge is present.
    const FS_SEGMO_EDGE_FEATHER = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uMask;
uniform vec2 uTexel;       // 1 / target dimensions (output resolution)
uniform float uRadius;     // feather radius in texels (typical 2-5)
out vec4 fragColor;

void main() {
  float center = texture(uMask, vUv).r;

  vec2 edgeStep = uTexel * 2.0;
  float maxDiff = 0.0;
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(-edgeStep.x, -edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(0.0, -edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(edgeStep.x, -edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(-edgeStep.x, 0.0)).r));
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(edgeStep.x, 0.0)).r));
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(-edgeStep.x, edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(0.0, edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(uMask, vUv + vec2(edgeStep.x, edgeStep.y)).r));

  float edgeness = smoothstep(0.02, 0.15, maxDiff);

  if (edgeness < 0.01) {
    fragColor = vec4(center, 0.0, 0.0, 1.0);
    return;
  }

  // 5×5 gaussian: bDist[i] is the squared distance from center.
  const float bDist[25] = float[25](
    8.0, 5.0, 4.0, 5.0, 8.0,
    5.0, 2.0, 1.0, 2.0, 5.0,
    4.0, 1.0, 0.0, 1.0, 4.0,
    5.0, 2.0, 1.0, 2.0, 5.0,
    8.0, 5.0, 4.0, 5.0, 8.0
  );
  const vec2 bOff[25] = vec2[25](
    vec2(-2.0, -2.0), vec2(-1.0, -2.0), vec2(0.0, -2.0), vec2(1.0, -2.0), vec2(2.0, -2.0),
    vec2(-2.0, -1.0), vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(2.0, -1.0),
    vec2(-2.0,  0.0), vec2(-1.0,  0.0), vec2(0.0,  0.0), vec2(1.0,  0.0), vec2(2.0,  0.0),
    vec2(-2.0,  1.0), vec2(-1.0,  1.0), vec2(0.0,  1.0), vec2(1.0,  1.0), vec2(2.0,  1.0),
    vec2(-2.0,  2.0), vec2(-1.0,  2.0), vec2(0.0,  2.0), vec2(1.0,  2.0), vec2(2.0,  2.0)
  );

  vec2 blurStep = uTexel * uRadius;
  float blurred = 0.0;
  float totalWeight = 0.0;
  for (int i = 0; i < 25; i++) {
    // gaussian with sigma = 1 in cell units (segmo's formula collapses to this).
    float weight = exp(-bDist[i] * 0.5);
    blurred += texture(uMask, vUv + bOff[i] * blurStep).r * weight;
    totalWeight += weight;
  }
  blurred /= totalWeight;

  float result = mix(center, blurred, edgeness);
  fragColor = vec4(result, 0.0, 0.0, 1.0);
}`
    this.pSegmoEdgeFeather = this._link(VS, FS_SEGMO_EDGE_FEATHER)

    // Light wrap. Ported from eyalfishler/segmo (LIGHT_WRAP_SHADER, MIT).
    // Adds subtle background spill onto foreground edge pixels so the subject
    // looks lit by the virtual scene instead of pasted on top of it.
    const FS_LIGHT_WRAP = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uComposite;   // segmo composite output
uniform sampler2D uBg;          // virtual background
uniform sampler2D uMask;        // feathered mask (same as composite consumed)
uniform float uStrength;        // 0.05-0.15 typical
out vec4 fragColor;

void main() {
  vec4 comp = texture(uComposite, vUv);
  vec4 bg = texture(uBg, vUv);
  float mask = texture(uMask, vUv).r;

  // Narrow band right inside the silhouette (mask ≈ 0.5).
  float edgeMask = smoothstep(0.25, 0.45, mask) * (1.0 - smoothstep(0.55, 0.75, mask));

  fragColor = mix(comp, bg, edgeMask * uStrength);
}`
    this.pLightWrap = this._link(VS, FS_LIGHT_WRAP)

    // Masked-foreground pre-pass for color cast. Writes rgb = video * weight,
    // a = weight, where weight = smoothstep(0.3, 0.7, mask). Generating mipmaps
    // on this target produces a top mip where rgb is the weighted sum and a is
    // the weight sum; their ratio is the foreground mean color.
    const FS_MASKED_FG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVideo;
uniform sampler2D uMask;
out vec4 fragColor;
void main() {
  float w = smoothstep(0.3, 0.7, texture(uMask, vUv).r);
  vec3 v = texture(uVideo, vUv).rgb;
  fragColor = vec4(v * w, w);
}`
    this.pMaskedFg = this._link(VS, FS_MASKED_FG)

    // Foreground color cast. Reads global means from top mips via textureLod
    // (the GPU clamps the LOD argument to the deepest available level — a 1×1
    // or near-1 texel that holds the average of the texture). Computes a
    // per-channel correction that shifts the foreground toward the background's
    // tint, clamped to a safe range to protect skin tones, and applied at the
    // requested strength.
    const FS_FG_COLOR_CAST = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVideo;
uniform sampler2D uFgMasked;   // mipmapped: rgb = sum(video*w), a = sum(w)
uniform sampler2D uBg;         // mipmapped
uniform float uStrength;
out vec4 fragColor;
void main() {
  vec3 video = texture(uVideo, vUv).rgb;
  // 32.0 is well above the deepest mip level for any practical resolution;
  // the sampler clamps to the top of the pyramid.
  vec4 fgSum = textureLod(uFgMasked, vec2(0.5), 32.0);
  vec3 fgMean = fgSum.rgb / max(fgSum.a, 0.001);
  vec3 bgMean = textureLod(uBg, vec2(0.5), 32.0).rgb;

  vec3 correction = bgMean / max(fgMean, vec3(0.01));
  correction = clamp(correction, vec3(0.7), vec3(1.4));

  vec3 tinted = mix(video, video * correction, uStrength);
  fragColor = vec4(tinted, 1.0);
}`
    this.pFgColorCast = this._link(VS, FS_FG_COLOR_CAST)
  }
}
