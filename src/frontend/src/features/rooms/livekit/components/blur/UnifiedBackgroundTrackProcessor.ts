import { ProcessorOptions, Track } from 'livekit-client'
import {
  ProcessorWrapper,
  VirtualBackground,
} from '@livekit/track-processors'
import { createLiveKitBlurProcessor, LiveKitBlurProcessor } from 'gregblur/livekit'
import { isMobileBrowser } from '@livekit/components-core'
import { ProcessorConfig, BackgroundProcessorInterface, ProcessorType } from '.'

const LANDSCAPE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite'

/**
 * Segmentation provider for the Landscape model that INVERTS the mask.
 *
 * The landscape model outputs person confidence (1.0 = person, 0.0 = background).
 * Gregblur expects background confidence (1.0 = background, 0.0 = person).
 * This provider inverts the mask so the BACKGROUND gets blurred, not the person.
 *
 * The mask is tiny (256×144 = 36K pixels), so CPU inversion takes < 1ms.
 */
function createInvertedLandscapeProvider() {
  let segmenter: any = null
  let lastTimestampMs = -1
  let gl: WebGL2RenderingContext | null = null
  let invertedTexture: WebGLTexture | null = null
  let byteBuffer: Uint8Array | null = null

  return {
    async init(canvas: HTMLCanvasElement | OffscreenCanvas) {
      // Reuse the GL context that gregblur already created on this canvas
      gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null

      if (gl) {
        invertedTexture = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, invertedTexture)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      }

      // Load MediaPipe via CDN (same approach as gregblur internally)
      const visionUrl = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
      const wasmPath = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'

      const vision = await import(/* @vite-ignore */ visionUrl)
      const wasmFileset = await vision.FilesetResolver.forVisionTasks(wasmPath)

      segmenter = await vision.ImageSegmenter.createFromOptions(wasmFileset, {
        baseOptions: { modelAssetPath: LANDSCAPE_MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
        canvas,
      })

      lastTimestampMs = -1
      console.log('[InvertedLandscape] ✅ Provider initialized')
    },

    segment(source: TexImageSource, timestampMs: number) {
      if (!segmenter || !gl || !invertedTexture) return null

      const ts = Math.max(lastTimestampMs + 1, Math.floor(timestampMs))
      lastTimestampMs = ts

      const result = segmenter.segmentForVideo(source, ts)
      if (!result?.confidenceMasks?.[0]) {
        result?.close?.()
        return null
      }

      const mask = result.confidenceMasks[0]
      const floatData = mask.getAsFloat32Array()
      const w = mask.width
      const h = mask.height

      // Reuse byte buffer to avoid allocations
      if (!byteBuffer || byteBuffer.length !== floatData.length) {
        byteBuffer = new Uint8Array(floatData.length)
      }

      // INVERT the mask: person confidence → background confidence
      // This is the key fix: gregblur expects 1.0 = background
      for (let i = 0; i < floatData.length; i++) {
        byteBuffer[i] = ((1.0 - floatData[i]) * 255) | 0
      }

      // Upload inverted mask to GPU (R8 format, universally supported)
      gl.bindTexture(gl.TEXTURE_2D, invertedTexture)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, byteBuffer)

      const closeFn = () => { try { result.close?.() } catch {} }

      return { confidenceTexture: invertedTexture, close: closeFn }
    },

    destroy() {
      if (segmenter) {
        try { segmenter.close() } catch {}
        segmenter = null
      }
      if (gl && invertedTexture) {
        gl.deleteTexture(invertedTexture)
        invertedTexture = null
      }
      gl = null
      byteBuffer = null
      lastTimestampMs = -1
    },
  }
}

export class UnifiedBackgroundTrackProcessor implements BackgroundProcessorInterface {
  processor: ProcessorWrapper<{ imagePath?: string; blurRadius?: number }> | LiveKitBlurProcessor
  opts: ProcessorConfig
  processorType: ProcessorType

  constructor(opts: ProcessorConfig) {
    this.opts = opts
    this.processorType = opts.type

    if (opts.type === 'virtual') {
      this.processor = VirtualBackground(opts.imagePath)
    } else if (opts.type === 'blur') {
      const isMobile = isMobileBrowser()
      const isLowEnd = (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4)
      const usePerformanceMode = isMobile || isLowEnd

      console.log(`🚀 Gregblur ${usePerformanceMode ? '⚡ Performance (Landscape)' : '✨ Quality (Multiclass)'}`)

      this.processor = createLiveKitBlurProcessor({
        blurRadius: opts.blurRadius,
        initialEnabled: true,

        // Mobile: landscape + inverted mask | Desktop: multiclass (native)
        segmentationProvider: usePerformanceMode
          ? createInvertedLandscapeProvider()
          : undefined,
        segmentationModel: usePerformanceMode ? undefined : 'selfie-multiclass-256',

        // Perf tuning
        downsampleFactor: usePerformanceMode ? 4 : 2,
        bilateralSigmaSpace: usePerformanceMode ? 0.1 : 4.0,
        bilateralSigmaColor: usePerformanceMode ? 0.01 : 0.1,
        temporalBlendFactor: usePerformanceMode ? 0.15 : 0.24,
      })
    } else {
      throw new Error(
        'Must provide either imagePath for virtual background or blurRadius for blur'
      )
    }
  }

  async init(opts: ProcessorOptions<Track.Kind>) {
    const isMobile = isMobileBrowser()
    const isLowEnd = (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4)

    if (isMobile || isLowEnd) {
      console.log('[UnifiedProcessor] 📉 Mobile constraints: 480×270 @ 15fps')
      try {
        await opts.track.applyConstraints({
          width: { ideal: 480 },
          height: { ideal: 270 },
          frameRate: { ideal: 15 },
        })
      } catch (e) {
        console.warn('[UnifiedProcessor] Could not apply constraints', e)
      }
    }

    try {
      return await this.processor.init(opts)
    } catch (e) {
      console.error('[UnifiedProcessor] Blur init failed, camera will start without blur', e)
    }
  }

  async restart(opts: ProcessorOptions<Track.Kind>) {
    return this.processor.restart(opts)
  }

  async destroy() {
    return this.processor.destroy()
  }

  async update(opts: ProcessorConfig): Promise<void> {
    this.opts = opts

    if (this.processorType === ProcessorType.VIRTUAL && opts.type === 'virtual') {
      const wrapper = this.processor as ProcessorWrapper<{ imagePath?: string; blurRadius?: number }>
      await wrapper.updateTransformerOptions({ imagePath: opts.imagePath })
    }
  }

  get name() {
    return this.processor.name
  }

  get processedTrack() {
    return this.processor.processedTrack
  }

  get options() {
    return this.opts
  }
}
