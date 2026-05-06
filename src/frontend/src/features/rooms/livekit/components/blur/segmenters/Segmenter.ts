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

export function detectMediapipeDelegate(): 'GPU' | 'CPU' {
  // Safari's WebGL2 exists but MediaPipe's GPU delegate crashes on it — force CPU.
  const isSafari =
    /^((?!chrome|android).)*safari/i.test(navigator.userAgent) &&
    !/CriOS|FxiOS/i.test(navigator.userAgent)
  if (isSafari) return 'CPU'
  try {
    const canvas = document.createElement('canvas')
    return canvas.getContext('webgl2') ? 'GPU' : 'CPU'
  } catch {
    return 'CPU'
  }
}
