import * as tf from '@tensorflow/tfjs-core'
import { GraphModel, loadGraphModel } from '@tensorflow/tfjs-converter'
import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-webgpu'

import { Segmenter } from './Segmenter'
import { pushMattingError } from '../errors/MattingErrorStore'

/**
 * Pick the fastest available TFJS backend.
 *
 * Tries WebGPU first (typically 2–3× faster than WebGL on modern GPUs), falls
 * back to WebGL when WebGPU is unavailable (Safari < 18, no `navigator.gpu`,
 * adapter request fails, or backend init throws). Emits a `WEBGPU_FALLBACK`
 * info error so the UI can surface the choice.
 *
 * Memoised: setBackend is global to the tf engine so only the first call matters.
 */
let _backendPick: Promise<'webgpu' | 'webgl'> | null = null
async function pickTfjsBackend(): Promise<'webgpu' | 'webgl'> {
  if (_backendPick) return _backendPick
  _backendPick = (async () => {
    const hasNavigatorGpu =
      typeof navigator !== 'undefined' && 'gpu' in navigator
    if (hasNavigatorGpu) {
      try {
        const ok = await tf.setBackend('webgpu')
        if (ok) {
          await tf.ready()
          // Force a real op to surface adapter init failures synchronously.
          tf.tidy(() => tf.add(tf.scalar(1), tf.scalar(1)).dataSync())
          return 'webgpu'
        }
      } catch (e) {
        pushMattingError({
          code: 'WEBGPU_FALLBACK',
          level: 'info',
          detail: `WebGPU init failed, falling back to WebGL: ${e instanceof Error ? e.message : String(e)}`,
        })
      }
    } else {
      pushMattingError({
        code: 'WEBGPU_FALLBACK',
        level: 'info',
        detail: 'navigator.gpu unavailable, using WebGL',
      })
    }
    const ok = await tf.setBackend('webgl')
    if (!ok) {
      throw new Error(
        `Neither webgpu nor webgl backends could be initialised (registered: ${tf.engine().backendNames().join(',')})`
      )
    }
    await tf.ready()
    return 'webgl'
  })()
  return _backendPick
}

const RVM_MODEL_URL = '/models/rvm/model.json'

// Processing resolution fed to RVM. RVM internally further downsamples with
// `downsample_ratio` for the recurrent branch — this is the resolution at which
// the alpha mask is produced and the renderer composites.
const DEFAULT_INPUT_W = 640
const DEFAULT_INPUT_H = 360

export interface RVMSegmenterOptions {
  downsampleRatio?: number
  inputSize?: { width: number; height: number }
}

/**
 * Robust Video Matting (PeterL1n/RobustVideoMatting) segmenter.
 *
 * Recurrent network — temporal state is kept internally between frames, so the
 * outward contract stays identical to the stateless MediaPipe segmenters:
 * `segment(ImageData) → Float32Array` with values in [0, 1].
 *
 * The model is converted from TF.js (NHWC). Initial recurrent states are
 * `tf.tensor1d([0])` per the official starter code convention — the model
 * recognises these as "no prior state" and emits properly-shaped output states.
 */
export class RVMSegmenter implements Segmenter {
  readonly inputSize: { width: number; height: number }

  private model?: GraphModel
  private recurrent: [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor] | null = null
  private downsampleTensor?: tf.Tensor
  private downsampleRatio: number
  private destroyed = false

  constructor(opts?: RVMSegmenterOptions) {
    this.downsampleRatio = opts?.downsampleRatio ?? 0.25
    this.inputSize = opts?.inputSize ?? {
      width: DEFAULT_INPUT_W,
      height: DEFAULT_INPUT_H,
    }
  }

  async init(): Promise<void> {
    try {
      const backend = await pickTfjsBackend()
      console.info('[RVM] tfjs ready, backend=', backend)
      this.model = await loadGraphModel(RVM_MODEL_URL)
      console.info('[RVM] model loaded, inputs=', this.model.inputs.map(i => i.name), 'outputs=', this.model.outputs.map(o => o.name))
      this.downsampleTensor = tf.scalar(this.downsampleRatio)
      this.recurrent = [
        tf.tensor1d([0]),
        tf.tensor1d([0]),
        tf.tensor1d([0]),
        tf.tensor1d([0]),
      ]
    } catch (e) {
      pushMattingError({
        code: 'RVM_INIT_FAILED',
        level: 'error',
        detail: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
      })
      throw e
    }
  }

  setDownsampleRatio(r: number): void {
    if (!this.downsampleTensor || r === this.downsampleRatio) return
    this.downsampleRatio = r
    this.downsampleTensor.dispose()
    this.downsampleTensor = tf.scalar(r)
  }

  async segment(imageData: ImageData): Promise<Float32Array> {
    if (!this.model || !this.recurrent || !this.downsampleTensor) {
      throw new Error('RVMSegmenter not initialised')
    }

    // Build src tensor inside tidy so the intermediate FromPixels/toFloat/div
    // tensors are freed automatically. Recurrent state tensors are kept alive
    // outside (returned from tidy via .keep()-style reference passing below).
    const src = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(imageData, 3)
      const float = tf.cast(pixels, 'float32')
      const normalized = tf.div(float, 255)
      return tf.expandDims(normalized, 0)
    })

    let outputs: tf.Tensor[]
    try {
      outputs = (await this.model.executeAsync(
        {
          'src:0': src,
          'r1i:0': this.recurrent[0],
          'r2i:0': this.recurrent[1],
          'r3i:0': this.recurrent[2],
          'r4i:0': this.recurrent[3],
          'downsample_ratio:0': this.downsampleTensor,
        },
        ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']
      )) as tf.Tensor[]
    } catch (e) {
      src.dispose()
      pushMattingError({
        code: 'RVM_INFERENCE_FAILED',
        level: 'error',
        detail: e instanceof Error ? e.message : String(e),
      })
      throw e
    }
    src.dispose()

    if (!Array.isArray(outputs) || outputs.length !== 6) {
      const detail = `executeAsync returned ${Array.isArray(outputs) ? `array of length ${outputs.length}` : typeof outputs}`
      pushMattingError({ code: 'RVM_INFERENCE_FAILED', level: 'error', detail })
      throw new Error(`[RVM] ${detail}`)
    }

    const [fgr, pha, r1o, r2o, r3o, r4o] = outputs
    // We only need pha; foreground decontamination is unused (the renderer
    // composites the original video frame using the alpha mask).
    fgr.dispose()

    // Replace recurrent state (dispose previous, keep new).
    this.recurrent.forEach((t) => t.dispose())
    this.recurrent = [r1o, r2o, r3o, r4o]

    // pha shape: [1, H, W, 1] → Float32Array length H*W.
    const data = (await pha.data()) as Float32Array
    pha.dispose()
    return data
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.recurrent?.forEach((t) => t.dispose())
    this.recurrent = null
    this.downsampleTensor?.dispose()
    this.downsampleTensor = undefined
    this.model?.dispose()
    this.model = undefined
  }
}
