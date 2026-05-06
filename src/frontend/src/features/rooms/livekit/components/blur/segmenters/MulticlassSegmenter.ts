import {
  FilesetResolver,
  ImageSegmenter,
  ImageSegmenterResult,
} from '@mediapipe/tasks-vision'
import { Segmenter, detectMediapipeDelegate } from './Segmenter'

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'

/**
 * MediaPipe Selfie Multiclass segmenter.
 * Outputs 6 classes: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others.
 * The "person" probability is built as 1 - background_prob.
 */
export class MulticlassSegmenter implements Segmenter {
  readonly inputSize = { width: 256, height: 256 }
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
      outputCategoryMask: false,
      outputConfidenceMasks: true,
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
          const masks = result.confidenceMasks!
          // confidenceMasks[0] is the background probability.
          const bg = masks[0].getAsFloat32Array()
          const out = new Float32Array(bg.length)
          for (let i = 0; i < bg.length; i++) {
            const v = 1 - bg[i]
            out[i] = v < 0 ? 0 : v > 1 ? 1 : v
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
