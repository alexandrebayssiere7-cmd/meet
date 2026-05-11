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
  private outW = 0
  private outH = 0
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

    // 5. Composite to the canvas.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
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
    ]
    for (const t of tex) if (t) gl.deleteTexture(t)
    const fbo = [
      this.fboMaskA,
      this.fboMaskB,
      this.fboEma,
      this.fboBgDown,
      this.fboBgBlurPing,
      this.fboBgBlurPong,
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
      gl.uniform1i(gl.getUniformLocation(this.pSigmoid, 'uTex'), 0)
      gl.uniform1f(
        gl.getUniformLocation(this.pSigmoid, 'uSteepness'),
        this.postCfg.sigmoid.steepness
      )
      gl.uniform1f(
        gl.getUniformLocation(this.pSigmoid, 'uThreshold'),
        this.postCfg.sigmoid.threshold
      )
      this._drawQuad()
      advance()
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
  }
}
