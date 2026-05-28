import * as ort from 'onnxruntime-web'
import { Segmenter } from './Segmenter'

/**
 * Tier 1 segmenter — Depth Anything V2 via ONNX Runtime Web + WebGPU.
 *
 * Expects the model at /models/depth-anything-v2/model.onnx (public folder).
 * Returns a normalized depth map (1 = near/person, 0 = far/background).
 * The renderer interprets this as a True Bokeh mask (FS_DEPTH_BOKEH shader).
 *
 * If WebGPU is unavailable or model loading fails, init() throws and
 * AdvancedMattingProcessor falls back to Tier 2 (Multiclass MediaPipe).
 */
const MODEL_URL = '/models/depth-anything-v2/model.onnx'

// ORT WASM files (jsep variant for WebGPU) — served from CDN to avoid
// Vite WASM asset configuration. Change to a local path if offline.
const ORT_WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/'

// ImageNet normalization applied to RGB input before inference.
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

let _ortWasmConfigured = false

function configureOrtWasm() {
  if (_ortWasmConfigured) return
  ort.env.wasm.wasmPaths = ORT_WASM_CDN
  ort.env.wasm.numThreads = 1
  _ortWasmConfigured = true
}

export class DepthAnythingSegmenter implements Segmenter {
  readonly inputSize = { width: 518, height: 518 }
  readonly maskType = 'depth' as const

  private _session: ort.InferenceSession | null = null
  private _inputName = 'image'
  private _outputName = 'depth'
  private _inferenceCount = 0

  async init(): Promise<void> {
    console.log('[Depth Anything] init: checking WebGPU...')

    if (!('gpu' in navigator)) {
      console.warn('[Depth Anything] init FAILED: WebGPU not supported in this browser')
      throw new Error('DepthAnythingSegmenter: WebGPU not available in this browser')
    }
    const adapter = await (navigator as unknown as { gpu: GPU }).gpu.requestAdapter()
    if (!adapter) {
      console.warn('[Depth Anything] init FAILED: navigator.gpu.requestAdapter() returned null')
      throw new Error('DepthAnythingSegmenter: no WebGPU adapter found')
    }

    console.log('[Depth Anything] WebGPU adapter found:', adapter.info?.description ?? '(no description)')
    console.log('[Depth Anything] loading model from:', MODEL_URL)

    configureOrtWasm()

    this._session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
    })

    // Discover actual input/output names so we're not hardcoded to a specific export.
    this._inputName = this._session.inputNames[0]
    this._outputName = this._session.outputNames[0]

    console.log(
      `[Depth Anything] model loaded OK — input: "${this._inputName}" | output: "${this._outputName}"`,
      `| providers: ${JSON.stringify(this._session.handler?.backend ?? 'unknown')}`
    )
  }

  async segment(imageData: ImageData, _timestampMs: number): Promise<Float32Array> {
    if (!this._session) throw new Error('DepthAnythingSegmenter: model not loaded')

    this._inferenceCount++
    if (this._inferenceCount === 1) {
      console.log('[Depth Anything] first inference running (518x518, WebGPU)...')
    }

    const { width, height } = imageData
    const rgba = imageData.data

    // Convert RGBA uint8 → NCHW float32, ImageNet-normalized.
    // Depth Anything V2 standard export: [1, 3, H, W].
    const nchw = new Float32Array(3 * height * width)
    const hw = height * width
    for (let i = 0; i < hw; i++) {
      nchw[i]          = (rgba[i * 4]     / 255 - MEAN[0]) / STD[0] // R
      nchw[hw + i]     = (rgba[i * 4 + 1] / 255 - MEAN[1]) / STD[1] // G
      nchw[2 * hw + i] = (rgba[i * 4 + 2] / 255 - MEAN[2]) / STD[2] // B
    }

    const t0 = performance.now()
    const input = new ort.Tensor('float32', nchw, [1, 3, height, width])
    const results = await this._session.run({ [this._inputName]: input })
    const raw = results[this._outputName].data as Float32Array
    const inferMs = performance.now() - t0

    if (this._inferenceCount === 1) {
      console.log(`[Depth Anything] first inference done in ${inferMs.toFixed(1)}ms — depth map shape: [${results[this._outputName].dims.join(', ')}]`)
    }

    return this._normalize(raw)
  }

  destroy(): void {
    console.log(`[Depth Anything] destroy (ran ${this._inferenceCount} inferences)`)
    this._session?.release().catch(() => { /* best-effort */ })
    this._session = null
    this._inferenceCount = 0
  }

  private _normalize(data: Float32Array): Float32Array {
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i]
      if (data[i] > max) max = data[i]
    }
    const range = max - min || 1
    const out = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) {
      out[i] = (data[i] - min) / range
    }
    return out
  }
}
