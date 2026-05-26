import {
  ImageSegmenter,
  ImageSegmenterResult,
} from '@mediapipe/tasks-vision'
import {
  Segmenter,
  getMediapipeFileset,
  probeMediapipeDelegate,
} from './Segmenter'
import { pushMattingError } from '../errors/MattingErrorStore'

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite'

/**
 * MediaPipe Selfie Segmenter (landscape / binary) — 256×144.
 *
 * Uses the `selfie_segmenter_landscape.tflite` model which outputs a single
 * confidence mask where high values represent the person (foreground).
 * This is the fastest available segmenter and is the automatic fallback when
 * the multiclass model fails the performance benchmark.
 */
export class LandscapeSegmenter implements Segmenter {
  readonly inputSize = { width: 256, height: 144 }
  private imageSegmenter?: ImageSegmenter
  // Reusable output buffer — avoids allocating a new Float32Array every frame.
  // MediaPipe recycles its internal buffer, so a copy is mandatory, but we
  // can reuse the same destination across frames.
  private _maskBuffer?: Float32Array

  /**
   * Download the model, probe GPU delegate support, and initialise the
   * MediaPipe ImageSegmenter in VIDEO mode.
   * Pushes `MEDIAPIPE_INIT_FAILED` and re-throws on failure.
   */
  async init() {
    try {
      const [fileset, delegate] = await Promise.all([
        getMediapipeFileset(),
        probeMediapipeDelegate(),
      ])
      this.imageSegmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate,
        },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      })
    } catch (e) {
      pushMattingError({
        code: 'MEDIAPIPE_INIT_FAILED',
        level: 'error',
        detail: `Landscape model: ${e instanceof Error ? e.message : String(e)}`,
      })
      throw e
    }
  }

  /**
   * Run segmentation on one video frame.
   * The confidence mask (`confidenceMasks[0]`) is copied into a reusable buffer
   * before the promise resolves, since MediaPipe recycles its internal storage.
   * The call races against a 2-second timeout to prevent queue stalls.
   *
   * @param imageData  RGBA frame at the model's input resolution (256×144).
   * @param timestampMs Frame capture time in milliseconds (used for VIDEO mode).
   * @returns Float32Array mask [0, 1], length = 256*144. Values close to 1 = person.
   */
  async segment(
    imageData: ImageData,
    timestampMs: number
  ): Promise<Float32Array> {
    const segPromise = new Promise<Float32Array>((resolve) => {
      this.imageSegmenter!.segmentForVideo(
        imageData,
        timestampMs,
        (result: ImageSegmenterResult) => {
          // For this binary model, confidenceMasks[0] is the foreground (person) probability
          // directly — NOT background. Unlike multiclass (where class 0 = background),
          // the binary landscape model emits a single mask where high value = person.
          const fg = result.confidenceMasks![0].getAsFloat32Array()
          // Copy into reusable buffer: getAsFloat32Array() returns a view into
          // a MediaPipe-managed buffer that gets recycled on the next call.
          if (!this._maskBuffer || this._maskBuffer.length !== fg.length) {
            this._maskBuffer = new Float32Array(fg.length)
          }
          this._maskBuffer.set(fg)
          resolve(this._maskBuffer)
        }
      )
    })
    const timeout = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('segment() timeout after 2s')), 2000)
    )
    return Promise.race([segPromise, timeout])
  }

  /**
   * Close the MediaPipe ImageSegmenter and free its GPU/WASM resources.
   * The instance must not be used after this call.
   */
  destroy() {
    this.imageSegmenter?.close()
    this.imageSegmenter = undefined
  }
}
