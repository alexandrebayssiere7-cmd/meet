import {
  FilesetResolver,
  ImageSegmenter,
} from '@mediapipe/tasks-vision'
import {
  pushMattingError,
} from '../errors/MattingErrorStore'

/**
 * Segmenter: abstracts the segmentation model behind a uniform interface.
 * Each implementation must return a Float32Array mask with values in [0, 1],
 * where 1 = person, 0 = background.
 */
export interface Segmenter {
  init(): Promise<void>
  segment(imageData: ImageData, timestampMs: number): Promise<Float32Array>
  destroy(): void
  readonly inputSize: { width: number; height: number }
}

const MEDIAPIPE_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'

let _filesetPromise: Promise<FilesetResolver> | null =
  null

/** Cache the FilesetResolver across segmenter instances — it loads ~1MB of WASM. */
export function getMediapipeFileset(): Promise<FilesetResolver> {
  if (!_filesetPromise) {
    _filesetPromise = FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL).catch(
      (e) => {
        _filesetPromise = null
        throw e
      }
    )
  }
  return _filesetPromise
}

let _delegateProbe: Promise<'GPU' | 'CPU'> | null = null

/**
 * Probe MediaPipe's GPU delegate by really trying to spin up an ImageSegmenter.
 * Falls back to CPU on failure. Memoised for the session.
 *
 * Replaces the previous UA-sniff that blanket-disabled GPU on Safari — Safari ≥ 17
 * supports the GPU delegate, and the user-facing impact of forcing CPU is severe
 * (80–150 ms per frame at 256² → queue saturation → frames look like passthrough).
 */
export function probeMediapipeDelegate(): Promise<'GPU' | 'CPU'> {
  if (_delegateProbe) return _delegateProbe
  _delegateProbe = (async () => {
    // Quick WebGL2 check first — without it the GPU delegate has nowhere to run.
    let webgl2Available = false
    try {
      const c = document.createElement('canvas')
      webgl2Available = !!c.getContext('webgl2')
    } catch {
      webgl2Available = false
    }
    if (!webgl2Available) return 'CPU'

    try {
      const fileset = await getMediapipeFileset()
      // Use a small/cheap model just to test the delegate. The Landscape model
      // is the smallest of the two we ship.
      const probe = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      })
      probe.close()
      return 'GPU'
    } catch (e) {
      pushMattingError({
        code: 'MEDIAPIPE_GPU_FALLBACK_TO_CPU',
        level: 'info',
        detail: e instanceof Error ? e.message : String(e),
      })
      return 'CPU'
    }
  })()
  return _delegateProbe
}
