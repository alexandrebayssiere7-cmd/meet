import { SegmentationModel } from '..'
import { LandscapeSegmenter } from './LandscapeSegmenter'
import { MulticlassSegmenter } from './MulticlassSegmenter'
import { Segmenter } from './Segmenter'

export type { Segmenter } from './Segmenter'
export { probeMediapipeDelegate } from './Segmenter'

/**
 * Options forwarded to the segmenter factory for model-specific configuration.
 */
export interface CreateSegmenterOptions {
  /**
   * Downsample ratio for the RVM model (0 < ratio ≤ 1).
   * Lower values = smaller input = faster inference at the cost of mask detail.
   * Ignored for MediaPipe-based segmenters (Landscape / Multiclass).
   */
  rvmDownsampleRatio?: number
}

/**
 * Factory that instantiates the appropriate `Segmenter` for the given model enum.
 *
 * - `RVM`        → `RVMSegmenter` (Robust Video Matting, ONNX inference).
 * - `MULTICLASS` → `MulticlassSegmenter` (MediaPipe 256×256, 6 classes).
 * - `LANDSCAPE` or default → `LandscapeSegmenter` (MediaPipe 256×144, binary, fastest).
 *
 * `AUTO` is resolved by `AdvancedMattingProcessor._initSegmenterBackground` which
 * first tries Multiclass and falls back to Landscape based on a benchmark.
 *
 * @param model Requested segmentation model (defaults to Landscape if omitted).
 * @param opts  Additional options (e.g. RVM downsample ratio).
 * @returns     A new, uninitialised `Segmenter` instance.
 */
export function createSegmenter(
  model?: SegmentationModel,
  opts?: CreateSegmenterOptions
): Segmenter {
  if (model === SegmentationModel.RVM) {
    return new RVMSegmenter({ downsampleRatio: opts?.rvmDownsampleRatio })
  }
  if (model === SegmentationModel.MULTICLASS) {
    return new MulticlassSegmenter()
  }
  return new LandscapeSegmenter()
}
