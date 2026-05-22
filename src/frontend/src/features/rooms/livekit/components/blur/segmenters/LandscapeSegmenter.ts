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

export class LandscapeSegmenter implements Segmenter {
  readonly inputSize = { width: 256, height: 144 }
  private imageSegmenter?: ImageSegmenter
  // Ring-buffer: two pre-allocated buffers alternated each frame to avoid
  // allocating Float32Arrays in the hot segmentation loop (zero GC pressure).
  private buffers = [
    new Float32Array(256 * 144),
    new Float32Array(256 * 144),
  ]
  private bufIdx = 0

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
          // Copy into the current ring-buffer slot (zero allocation).
          const out = this.buffers[this.bufIdx]
          this.bufIdx ^= 1
          out.set(fg)
          result.close()
          resolve(out)
        }
      )
    })
    const timeout = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('segment() timeout after 2s')), 2000)
    )
    return Promise.race([segPromise, timeout])
  }

  destroy() {
    this.imageSegmenter?.close()
    this.imageSegmenter = undefined
  }
}
