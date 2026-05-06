import {
  FilesetResolver,
  ImageSegmenter,
  ImageSegmenterResult,
} from '@mediapipe/tasks-vision'
import { Segmenter, detectMediapipeDelegate } from './Segmenter'

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite'

export class LandscapeSegmenter implements Segmenter {
  readonly inputSize = { width: 256, height: 144 }
  private imageSegmenter?: ImageSegmenter

  async init() {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
    )
    this.imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: detectMediapipeDelegate(),
      },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    })
  }

  async segment(
    imageData: ImageData,
    timestampMs: number
  ): Promise<Float32Array> {
    return new Promise<Float32Array>((resolve) => {
      this.imageSegmenter!.segmentForVideo(
        imageData,
        timestampMs,
        (result: ImageSegmenterResult) => {
          const uint8 = result.categoryMask!.getAsUint8Array()
          // Selfie segmenter outputs 0 = person, 255 = background.
          // Convert to a person-prob Float32 mask (1 = person).
          const out = new Float32Array(uint8.length)
          for (let i = 0; i < uint8.length; i++) {
            out[i] = 1 - uint8[i] / 255
          }
          resolve(out)
        }
      )
    })
  }

  destroy() {
    this.imageSegmenter?.close()
    this.imageSegmenter = undefined
  }
}
