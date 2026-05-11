import { PostProcessingConfig } from '..'
import { applySigmoid } from './Sigmoid'
import { applyMorphology } from './Morphology'
import { applyGuidedFilter } from './GuidedFilter'
import { TemporalEMA } from './TemporalEMA'

/**
 * Orchestrates the post-processing filters in a fixed order:
 *   raw_mask → [Sigmoid] → [Morphology] → [GuidedFilter] → [EMA] → final_mask
 * Filters not present in the config are skipped. Order is fixed by design:
 * sigmoid sharpens, morphology cleans speckles/holes, guided filter aligns
 * edges to the RGB guide, EMA stabilises temporally last (after spatial work).
 */
export class PostProcessingPipeline {
  private ema?: TemporalEMA

  constructor(private cfg: PostProcessingConfig) {
    if (cfg.ema) this.ema = new TemporalEMA(cfg.ema.alpha)
  }

  apply(
    mask: Float32Array,
    width: number,
    height: number,
    guide: ImageData
  ): Float32Array {
    let m = mask
    if (this.cfg.sigmoid) {
      m = applySigmoid(m, this.cfg.sigmoid.steepness, this.cfg.sigmoid.threshold)
    }
    if (this.cfg.morphology) {
      m = applyMorphology(
        m,
        width,
        height,
        this.cfg.morphology.op,
        this.cfg.morphology.kernelSize
      )
    }
    if (this.cfg.guidedFilter) {
      m = applyGuidedFilter(
        m,
        guide,
        this.cfg.guidedFilter.radius,
        this.cfg.guidedFilter.eps
      )
    }
    if (this.ema) {
      m = this.ema.apply(m)
    }
    return m
  }

  reset(): void {
    this.ema?.reset()
  }
}
