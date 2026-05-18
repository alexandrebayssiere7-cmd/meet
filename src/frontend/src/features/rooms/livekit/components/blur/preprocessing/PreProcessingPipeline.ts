import { PreProcessingConfig } from '..'
import { BBox, RoiCropper } from './RoiCropper'

/**
 * Orchestrates pre-processing filters applied to the raw video frame
 * before (and surrounding) the segmentation model.
 *
 * Per-frame call order in AdvancedMattingProcessor:
 *
 *   bbox = pipeline.getNextCropBbox()       // used by sizeSource() to crop the video
 *   sizeSource(bbox)                        // extracts crop from full-res video → ImageData
 *   frame = pipeline.apply(frame, prevMask) // frame-level transforms (future techniques)
 *   rawMask = segmenter.segment(frame)      // inference in crop space
 *   guidedFilter(rawMask, frame)            // optional, still in crop space
 *   fullMask = pipeline.applyAfterInference(refinedMask, maskW, maskH, bbox)
 *                                           // remap crop-space mask → full-frame space
 *                                           // + update RoiCropper internal state
 */
export class PreProcessingPipeline {
  private roiCropper?: RoiCropper

  constructor(cfg: PreProcessingConfig) {
    if (cfg.roiCropping?.enabled) this.roiCropper = new RoiCropper()
  }

  /**
   * Returns the bbox to use when extracting the model input from the full-resolution
   * video frame. Must be called before sizeSource() each frame.
   * Returns null when no spatial crop is needed (full frame).
   */
  getNextCropBbox(): BBox | null {
    return this.roiCropper?.getNextCropBbox() ?? null
  }

  /**
   * Apply frame-level transforms to the already-extracted (and cropped) ImageData.
   * Add technique calls here as new preprocessing methods are introduced.
   *
   * @param frame    RGBA ImageData at processing resolution (already cropped + resized)
   * @param prevMask Float32Array mask [0, 1] from the previous frame, in full-frame space
   */
  apply(frame: ImageData, prevMask?: Float32Array): ImageData {
    void prevMask
    return frame
  }

  /**
   * Post-inference step: remap a crop-space mask back to full-frame space and update
   * any stateful preprocessors (e.g. RoiCropper's internal bbox state).
   *
   * If no spatial crop was active this frame (bbox is null / full frame), the mask is
   * returned unchanged.
   *
   * @param mask      Float32Array from the segmenter, in crop-bbox space
   * @param maskW     Width of that mask (= model input width)
   * @param maskH     Height of that mask (= model input height)
   * @param usedBbox  The bbox that was used for this frame's crop (from getNextCropBbox)
   */
  applyAfterInference(
    mask: Float32Array,
    maskW: number,
    maskH: number,
    usedBbox: BBox | null
  ): Float32Array {
    if (!this.roiCropper || !usedBbox) return mask

    const full = this.roiCropper.remapMask(mask, maskW, maskH, usedBbox, maskW, maskH)
    this.roiCropper.updateWithMask(full, maskW, maskH)
    return full
  }

  reset(): void {
    this.roiCropper?.reset()
  }
}
