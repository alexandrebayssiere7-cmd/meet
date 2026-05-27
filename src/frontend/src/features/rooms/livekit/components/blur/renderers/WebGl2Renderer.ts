import { PostProcessingConfig, UpsamplingConfig } from '..'
import { pushMattingError } from '../errors/MattingErrorStore'
import { GpuRenderer, GpuRendererInitOpts, RenderSource } from './GpuRenderer'
import { GpuGuidedFilter } from './GpuGuidedFilter'

/**
 * WebGL2 implementation of the matting compositor.
 *
 * Pipeline per frame (`render(videoElement)`):
 *   videoTex ← upload from <video>
 *   maskTex  ← uploaded once per new mask (uploadMask)
 *   maskRefined ← post-processing chain (sigmoid → morpho → ema)
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
  private pSigmoid!: WebGLProgram
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
  // Live-camera frame, uploaded only when the orchestrator requests a blend
  // between the frame-locked source (`videoTex`) and the live source. Allocated
  // lazily on first use to avoid the memory cost when blend mode is never used.
  private liveVideoTex: WebGLTexture | null = null
  // Mask warp offset in uv space (uMaskOffset uniform). Applied at the mask
  // sample-time in the composite shader to align the (possibly stale) mask
  // with the live frame using a velocity prediction.
  private maskOffsetU = 0
  private maskOffsetV = 0
  // Cross-fade weight between `videoTex` (frame-locked, 0.0) and `liveVideoTex`
  // (live, 1.0). 0.0 disables the blend pass entirely.
  private blendMix = 0
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

  private halfW = 0
  private halfH = 0
  private hasEmaState = false
  private virtualImgPending: HTMLImageElement | null = null
  private virtualImgUploaded = false

  // Reusable CPU buffer for Float32→Uint8 mask conversion (avoids per-frame allocation).
  private u8MaskBuffer?: Uint8Array

  // Cached uniform locations — resolved once in _buildPrograms(), reused every frame.
  // Eliminates ~43 string-lookup driver calls per frame.
  private uLoc!: {
    sigmoid: {
      uTex: WebGLUniformLocation | null
      uSteepness: WebGLUniformLocation | null
      uThreshold: WebGLUniformLocation | null
    }
    ema: {
      uTex: WebGLUniformLocation | null
      uPrev: WebGLUniformLocation | null
      uAlpha: WebGLUniformLocation | null
    }
    copyR: { uTex: WebGLUniformLocation | null }
    morphology: {
      uTex: WebGLUniformLocation | null
      uRadius: WebGLUniformLocation | null
      uTexel: WebGLUniformLocation | null
    }
    maskedDown: {
      uFrame: WebGLUniformLocation | null
      uMask: WebGLUniformLocation | null
      uSourceTexelSize: WebGLUniformLocation | null
    }
    blur: {
      uImage: WebGLUniformLocation | null
      uMask: WebGLUniformLocation | null
      uDirection: WebGLUniformLocation | null
      uTexelSize: WebGLUniformLocation | null
      uRadius: WebGLUniformLocation | null
    }
    composite: {
      uVideo: WebGLUniformLocation | null
      uBg: WebGLUniformLocation | null
      uMask: WebGLUniformLocation | null
      uErosionRadius: WebGLUniformLocation | null
      uOutTexel: WebGLUniformLocation | null
    }
    segmo: {
      uVideo: WebGLUniformLocation | null
      uBg: WebGLUniformLocation | null
      uMask: WebGLUniformLocation | null
      uOutTexel: WebGLUniformLocation | null
    }
    feather: {
      uMask: WebGLUniformLocation | null
      uTexel: WebGLUniformLocation | null
      uRadius: WebGLUniformLocation | null
    }
    lightWrap: {
      uComposite: WebGLUniformLocation | null
      uBg: WebGLUniformLocation | null
      uMask: WebGLUniformLocation | null
      uStrength: WebGLUniformLocation | null
    }
    maskedFg: {
      uVideo: WebGLUniformLocation | null
      uMask: WebGLUniformLocation | null
    }
    fgCast: {
      uVideo: WebGLUniformLocation | null
      uFgMasked: WebGLUniformLocation | null
      uBg: WebGLUniformLocation | null
      uStrength: WebGLUniformLocation | null
    }
  }

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
    for (const tex of [this.rawMaskTex, this.maskA, this.maskB, this.emaTex]) {
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

    // 1. Reallocate videoTex (and the optional liveVideoTex) at new size
    if (this.videoTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        w,
        h,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null
      )
    }
    if (this.liveVideoTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.liveVideoTex)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        w,
        h,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null
      )
    }

    // 2. Reallocate half-res textures
    if (this.bgDownTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.bgDownTex)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        this.halfW,
        this.halfH,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null
      )
    }
    if (this.bgBlurPingTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.bgBlurPingTex)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        this.halfW,
        this.halfH,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null
      )
    }
    if (this.bgBlurPongTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.bgBlurPongTex)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        this.halfW,
        this.halfH,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null
      )
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

    // 4. Destroy Guided Filter so it gets recreated at the new size
    if (this.gf) {
      this.gf.destroy()
      this.gf = null
    }

    // 5. Update canvas dimensions
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
    // Convert Float32 [0,1] → Uint8, reusing a pre-allocated buffer.
    const len = mask.length
    if (!this.u8MaskBuffer || this.u8MaskBuffer.length !== len) {
      this.u8MaskBuffer = new Uint8Array(len)
    }
    const u8 = this.u8MaskBuffer
    for (let i = 0; i < len; i++) {
      const v = mask[i]
      u8[i] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0
    }
    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, this.rawMaskTex)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RED, gl.UNSIGNED_BYTE, u8)
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

  setMaskOffset(u: number, v: number) {
    this.maskOffsetU = Number.isFinite(u) ? u : 0
    this.maskOffsetV = Number.isFinite(v) ? v : 0
  }

  setBlendMix(t: number) {
    if (!Number.isFinite(t)) {
      this.blendMix = 0
      return
    }
    this.blendMix = t < 0 ? 0 : t > 1 ? 1 : t
  }

  render(source: RenderSource, liveSource?: RenderSource) {
    if (!source) return
    const isVideo = (source as HTMLVideoElement).videoWidth !== undefined
    const sw = isVideo
      ? (source as HTMLVideoElement).videoWidth
      : (source as ImageBitmap).width
    if (!sw) return
    const gl = this.gl

    // 1. Upload current source frame to videoTex (full output size).
    // The shader assumes texture origin = bottom-left. For HTMLVideoElement
    // we let the global UNPACK_FLIP_Y_WEBGL=true do the flip. For
    // ImageBitmap, the bitmap is pre-flipped at creation (imageOrientation
    // 'flipY') because UNPACK_FLIP_Y_WEBGL is unreliable for bitmaps across
    // browsers — so we explicitly disable the GL flip for this upload, then
    // restore it afterwards to preserve global state used by other uploads.
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
    if (!isVideo) gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
    try {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        source
      )
    } catch (e) {
      if (!isVideo) gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
      // Some browsers throw if the video frame isn't ready yet — skip this tick.
      void e
      return
    }
    if (!isVideo) gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)

    // 1b. Upload optional live source (used by the standard composite shader
    // when uBlendT > 0). Allocated lazily on first use.
    const wantsBlend = this.blendMix > 0 && liveSource !== undefined
    if (wantsBlend) {
      const liveIsVideo =
        (liveSource as HTMLVideoElement).videoWidth !== undefined
      if (!this.liveVideoTex) {
        this.liveVideoTex = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, this.liveVideoTex)
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          this.outW,
          this.outH,
          0,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          null
        )
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      }
      gl.activeTexture(gl.TEXTURE3)
      gl.bindTexture(gl.TEXTURE_2D, this.liveVideoTex)
      if (!liveIsVideo) gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
      try {
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          liveSource!
        )
      } catch (e) {
        void e
      }
      if (!liveIsVideo) gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
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
      gl.flush()
      return
    }

    // 5. Composite to the canvas.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pComposite)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
    gl.uniform1i(this.uLoc.composite.uVideo, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, bgTex)
    gl.uniform1i(this.uLoc.composite.uBg, 1)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, finalMaskTex)
    gl.uniform1i(this.uLoc.composite.uMask, 2)
    // uLiveVideo defaults to the frame-locked source when blend is off, so the
    // shader's mix() degenerates to a no-op. When blend is on, bind the live
    // source we uploaded above.
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(
      gl.TEXTURE_2D,
      wantsBlend && this.liveVideoTex ? this.liveVideoTex : this.videoTex
    )
    gl.uniform1i(gl.getUniformLocation(this.pComposite, 'uLiveVideo'), 3)
    gl.uniform1f(
      gl.getUniformLocation(this.pComposite, 'uBlendT'),
      wantsBlend ? this.blendMix : 0
    )
    gl.uniform2f(
      gl.getUniformLocation(this.pComposite, 'uMaskOffset'),
      this.maskOffsetU,
      this.maskOffsetV
    )
    gl.uniform1f(
      this.uLoc.composite.uErosionRadius,
      this.postCfg.erosion?.pixels ?? 0
    )
    gl.uniform2f(this.uLoc.composite.uOutTexel, 1 / this.outW, 1 / this.outH)
    this._drawQuad()

    gl.flush()
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
    const eps = this.upsamplingCfg.eps ?? 0.01
    return this.gf.run(this.videoTex, procMaskTex, radius, eps, this.vao)
  }

  destroy() {
    if (!this.gl) return
    this.gf?.destroy()
    this.gf = null
    const gl = this.gl
    const tex = [
      this.videoTex,
      this.liveVideoTex,
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
    ]
    for (const f of fbo) if (f) gl.deleteFramebuffer(f)
    if (this.quadBuffer) gl.deleteBuffer(this.quadBuffer)
    if (this.vao) gl.deleteVertexArray(this.vao)
    const programs = [
      this.pUploadMask,
      this.pSigmoid,
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

    // Sigmoid
    if (this.postCfg.sigmoid) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFbo)
      gl.useProgram(this.pSigmoid)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, src)
      gl.uniform1i(this.uLoc.sigmoid.uTex, 0)
      gl.uniform1f(this.uLoc.sigmoid.uSteepness, this.postCfg.sigmoid.steepness)
      gl.uniform1f(this.uLoc.sigmoid.uThreshold, this.postCfg.sigmoid.threshold)
      this._drawQuad()
      advance()
    }

    // Opening (Erosion then Dilation — removes small isolated specks at mask edges)
    if (this.postCfg.opening && this.postCfg.opening.radius > 0) {
      const r = this.postCfg.opening.radius
      this._applyMorphology(dstFbo, src, -r) // Erosion
      advance()
      this._applyMorphology(dstFbo, src, r) // Dilation
      advance()
    }

    // Closing (Dilation then Erosion — fills small holes inside the mask)
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
      gl.uniform1i(this.uLoc.ema.uTex, 0)
      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, this.emaTex)
      gl.uniform1i(this.uLoc.ema.uPrev, 1)
      gl.uniform1f(this.uLoc.ema.uAlpha, this.hasEmaState ? alpha : 1.0)
      this._drawQuad()
      advance()
      // Copy current result into emaTex for next frame
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboEma)
      gl.useProgram(this.pCopyR)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, src)
      gl.uniform1i(this.uLoc.copyR.uTex, 0)
      this._drawQuad()
      this.hasEmaState = true
    }

    return src
  }

  private _applyMorphology(
    fbo: WebGLFramebuffer,
    src: WebGLTexture,
    radius: number
  ) {
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.useProgram(this.pMorphology)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, src)
    gl.uniform1i(this.uLoc.morphology.uTex, 0)
    gl.uniform1f(this.uLoc.morphology.uRadius, radius)
    gl.uniform2f(this.uLoc.morphology.uTexel, 1 / this.procW, 1 / this.procH)
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
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
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
    gl.uniform1i(this.uLoc.maskedDown.uFrame, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(this.uLoc.maskedDown.uMask, 1)
    gl.uniform2f(
      this.uLoc.maskedDown.uSourceTexelSize,
      1.0 / this.outW,
      1.0 / this.outH
    )
    this._drawQuad()

    // Stage 2: horizontal mask-weighted gaussian.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBgBlurPing)
    gl.useProgram(this.pMaskWeightedBlur)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.bgDownTex)
    gl.uniform1i(this.uLoc.blur.uImage, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(this.uLoc.blur.uMask, 1)
    gl.uniform2f(this.uLoc.blur.uDirection, 1.0, 0.0)
    gl.uniform2f(this.uLoc.blur.uTexelSize, 1.0 / this.halfW, 1.0 / this.halfH)
    gl.uniform1f(this.uLoc.blur.uRadius, radius)
    this._drawQuad()

    // Stage 3: vertical mask-weighted gaussian.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBgBlurPong)
    gl.useProgram(this.pMaskWeightedBlur)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.bgBlurPingTex)
    gl.uniform1i(this.uLoc.blur.uImage, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(this.uLoc.blur.uMask, 1)
    gl.uniform2f(this.uLoc.blur.uDirection, 0.0, 1.0)
    gl.uniform2f(this.uLoc.blur.uTexelSize, 1.0 / this.halfW, 1.0 / this.halfH)
    gl.uniform1f(this.uLoc.blur.uRadius, radius)
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
   * postprocess chain (sigmoid/morphology/EMA/guided-upsample) ran upstream and
   * is unaffected.
   */
  private _ensureSegmoFeatherTarget() {
    if (this.segmoFeatheredMaskTex && this.fboSegmoFeatheredMask) return
    const gl = this.gl
    const t = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, t)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R8,
      this.outW,
      this.outH,
      0,
      gl.RED,
      gl.UNSIGNED_BYTE,
      null
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const f = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, f)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      t,
      0
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
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      this.outW,
      this.outH,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const f = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, f)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      t,
      0
    )
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(
        `segmo composite FBO incomplete: 0x${status.toString(16)}`
      )
    }
    this.segmoCompositeTex = t
    this.fboSegmoComposite = f
  }

  private _ensureSegmoTintTargets() {
    if (
      this.maskedFgTex &&
      this.fboMaskedFg &&
      this.tintedVideoTex &&
      this.fboTintedVideo
    )
      return
    const gl = this.gl
    // Masked foreground: mipmapped, RGBA8. rgb = video * weight, a = weight.
    // Top mip yields (sum(video*weight), sum(weight)) so the cast shader can
    // recover the foreground mean color as rgb/a.
    const mft = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, mft)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      this.outW,
      this.outH,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null
    )
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MIN_FILTER,
      gl.LINEAR_MIPMAP_LINEAR
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const mff = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, mff)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      mft,
      0
    )
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('segmo masked-fg FBO incomplete')
    }

    // Tinted video target: plain RGBA8, no mipmaps needed (it's consumed by
    // a full-resolution sampler at mip 0).
    const tvt = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tvt)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      this.outW,
      this.outH,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null
    )
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const tvf = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, tvf)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      tvt,
      0
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

    // Pass A: edge-only feather (widens the transition band near silhouettes,
    // leaves interior/exterior alone). Output is R8 at output resolution.
    this._ensureSegmoFeatherTarget()
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboSegmoFeatheredMask)
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pSegmoEdgeFeather)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, maskTex)
    gl.uniform1i(this.uLoc.feather.uMask, 0)
    gl.uniform2f(this.uLoc.feather.uTexel, 1 / this.outW, 1 / this.outH)
    gl.uniform1f(this.uLoc.feather.uRadius, this.segmoFeatherRadius)
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
      if (
        this.virtualImgUploaded &&
        !this._segmoBgMipmapsValid &&
        this.virtualBgTex
      ) {
        gl.bindTexture(gl.TEXTURE_2D, this.virtualBgTex)
        gl.texParameteri(
          gl.TEXTURE_2D,
          gl.TEXTURE_MIN_FILTER,
          gl.LINEAR_MIPMAP_LINEAR
        )
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
        gl.uniform1i(this.uLoc.maskedFg.uVideo, 0)
        gl.activeTexture(gl.TEXTURE1)
        gl.bindTexture(gl.TEXTURE_2D, this.segmoFeatheredMaskTex!)
        gl.uniform1i(this.uLoc.maskedFg.uMask, 1)
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
        gl.uniform1i(this.uLoc.fgCast.uVideo, 0)
        gl.activeTexture(gl.TEXTURE1)
        gl.bindTexture(gl.TEXTURE_2D, this.maskedFgTex!)
        gl.uniform1i(this.uLoc.fgCast.uFgMasked, 1)
        gl.activeTexture(gl.TEXTURE2)
        gl.bindTexture(gl.TEXTURE_2D, bgTex)
        gl.uniform1i(this.uLoc.fgCast.uBg, 2)
        gl.uniform1f(
          this.uLoc.fgCast.uStrength,
          this.segmoForegroundTintStrength
        )
        this._drawQuad()
        videoSrc = this.tintedVideoTex!
      }
    }

    // Pass B: segmo composite, fed by the feathered mask. When light wrap is
    // enabled we render to an intermediate texture; otherwise straight to canvas.
    const useLightWrap = this.segmoLightWrapStrength > 0.0
    if (useLightWrap) {
      this._ensureSegmoCompositeTarget()
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboSegmoComposite)
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    }
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pCompositeSegmo)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, videoSrc)
    gl.uniform1i(this.uLoc.segmo.uVideo, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, bgTex)
    gl.uniform1i(this.uLoc.segmo.uBg, 1)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, this.segmoFeatheredMaskTex!)
    gl.uniform1i(this.uLoc.segmo.uMask, 2)
    gl.uniform2f(this.uLoc.segmo.uOutTexel, 1 / this.outW, 1 / this.outH)
    this._drawQuad()

    if (!useLightWrap) return

    // Pass C: light wrap — mix a small amount of the background color into the
    // narrow edge band so the subject looks lit by the virtual scene.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, this.outW, this.outH)
    gl.useProgram(this.pLightWrap)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.segmoCompositeTex!)
    gl.uniform1i(this.uLoc.lightWrap.uComposite, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, bgTex)
    gl.uniform1i(this.uLoc.lightWrap.uBg, 1)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, this.segmoFeatheredMaskTex!)
    gl.uniform1i(this.uLoc.lightWrap.uMask, 2)
    gl.uniform1f(this.uLoc.lightWrap.uStrength, this.segmoLightWrapStrength)
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
    this.bgDownTex = makeTex(
      this.halfW,
      this.halfH,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE
    )
    this.bgBlurPingTex = makeTex(
      this.halfW,
      this.halfH,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE
    )
    this.bgBlurPongTex = makeTex(
      this.halfW,
      this.halfH,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE
    )
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

    const FS_SIGMOID = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTex;
uniform float uSteepness;
uniform float uThreshold;
out vec4 fragColor;
void main() {
  float v = texture(uTex, vUv).r;
  float y = 1.0 / (1.0 + exp(-uSteepness * (v - uThreshold)));
  fragColor = vec4(y, 0.0, 0.0, 1.0);
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
uniform sampler2D uVideo;     // frame-locked source (matches uMask)
uniform sampler2D uBg;
uniform sampler2D uMask;
uniform sampler2D uLiveVideo; // live <video> source, used when uBlendT > 0
uniform float uErosionRadius; // pixels at output resolution, 0 = disabled
uniform vec2 uOutTexel;       // vec2(1/outW, 1/outH)
uniform vec2 uMaskOffset;     // uv offset applied to every mask sample (prediction)
uniform float uBlendT;        // 0 = pure uVideo, 1 = pure uLiveVideo, in between = cross-fade
out vec4 fragColor;
void main() {
  vec3 fgLocked = texture(uVideo, vUv).rgb;
  vec3 fg = uBlendT > 0.0
    ? mix(fgLocked, texture(uLiveVideo, vUv).rgb, uBlendT)
    : fgLocked;
  vec3 bg = texture(uBg, vUv).rgb;
  // Erosion applied here at output resolution so that uErosionRadius is measured
  // in actual output pixels — not in the coarse processing-resolution pixels that
  // would produce large blocky artefacts after upsampling.
  // Diamond kernel (H + V in one pass): accurate enough for edge trimming.
  vec2 mUv = vUv - uMaskOffset;
  float m = texture(uMask, mUv).r;
  if (uErosionRadius > 0.0) {
    for (int i = 1; i <= 16; i++) {
      if (float(i) > uErosionRadius) break;
      float fi = float(i);
      m = min(m, texture(uMask, mUv + vec2(uOutTexel.x * fi, 0.0)).r);
      m = min(m, texture(uMask, mUv - vec2(uOutTexel.x * fi, 0.0)).r);
      m = min(m, texture(uMask, mUv + vec2(0.0, uOutTexel.y * fi)).r);
      m = min(m, texture(uMask, mUv - vec2(0.0, uOutTexel.y * fi)).r);
    }
  }
  // +0.035 foreground bias preserves edges that conservative segmentation models clip.
  float t = smoothstep(0.26, 0.72, clamp(m + 0.035, 0.0, 1.0));
  fragColor = vec4(mix(bg, fg, t), 1.0);
}`

    this.pUploadMask = this._link(VS, FS_COPY_R)
    this.pSigmoid = this._link(VS, FS_SIGMOID)
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

    // Cache all uniform locations once — avoids per-frame string lookups
    // through the GL driver which can stall the CPU-GPU pipeline.
    const loc = (p: WebGLProgram, n: string) => this.gl.getUniformLocation(p, n)
    this.uLoc = {
      sigmoid: {
        uTex: loc(this.pSigmoid, 'uTex'),
        uSteepness: loc(this.pSigmoid, 'uSteepness'),
        uThreshold: loc(this.pSigmoid, 'uThreshold'),
      },
      ema: {
        uTex: loc(this.pEma, 'uTex'),
        uPrev: loc(this.pEma, 'uPrev'),
        uAlpha: loc(this.pEma, 'uAlpha'),
      },
      copyR: { uTex: loc(this.pCopyR, 'uTex') },
      morphology: {
        uTex: loc(this.pMorphology, 'uTex'),
        uRadius: loc(this.pMorphology, 'uRadius'),
        uTexel: loc(this.pMorphology, 'uTexel'),
      },
      maskedDown: {
        uFrame: loc(this.pMaskedDownsample, 'uFrame'),
        uMask: loc(this.pMaskedDownsample, 'uMask'),
        uSourceTexelSize: loc(this.pMaskedDownsample, 'uSourceTexelSize'),
      },
      blur: {
        uImage: loc(this.pMaskWeightedBlur, 'uImage'),
        uMask: loc(this.pMaskWeightedBlur, 'uMask'),
        uDirection: loc(this.pMaskWeightedBlur, 'uDirection'),
        uTexelSize: loc(this.pMaskWeightedBlur, 'uTexelSize'),
        uRadius: loc(this.pMaskWeightedBlur, 'uRadius'),
      },
      composite: {
        uVideo: loc(this.pComposite, 'uVideo'),
        uBg: loc(this.pComposite, 'uBg'),
        uMask: loc(this.pComposite, 'uMask'),
        uErosionRadius: loc(this.pComposite, 'uErosionRadius'),
        uOutTexel: loc(this.pComposite, 'uOutTexel'),
      },
      segmo: {
        uVideo: loc(this.pCompositeSegmo, 'uVideo'),
        uBg: loc(this.pCompositeSegmo, 'uBg'),
        uMask: loc(this.pCompositeSegmo, 'uMask'),
        uOutTexel: loc(this.pCompositeSegmo, 'uOutTexel'),
      },
      feather: {
        uMask: loc(this.pSegmoEdgeFeather, 'uMask'),
        uTexel: loc(this.pSegmoEdgeFeather, 'uTexel'),
        uRadius: loc(this.pSegmoEdgeFeather, 'uRadius'),
      },
      lightWrap: {
        uComposite: loc(this.pLightWrap, 'uComposite'),
        uBg: loc(this.pLightWrap, 'uBg'),
        uMask: loc(this.pLightWrap, 'uMask'),
        uStrength: loc(this.pLightWrap, 'uStrength'),
      },
      maskedFg: {
        uVideo: loc(this.pMaskedFg, 'uVideo'),
        uMask: loc(this.pMaskedFg, 'uMask'),
      },
      fgCast: {
        uVideo: loc(this.pFgColorCast, 'uVideo'),
        uFgMasked: loc(this.pFgColorCast, 'uFgMasked'),
        uBg: loc(this.pFgColorCast, 'uBg'),
        uStrength: loc(this.pFgColorCast, 'uStrength'),
      },
    }
  }
}
