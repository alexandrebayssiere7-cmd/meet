import { SegmentationModel } from '..'
import { LandscapeSegmenter } from './LandscapeSegmenter'
import { MulticlassSegmenter } from './MulticlassSegmenter'
import { RVMSegmenter } from './RVMSegmenter'
import { Segmenter } from './Segmenter'

export type { Segmenter } from './Segmenter'
export { RVMSegmenter } from './RVMSegmenter'
export { probeMediapipeDelegate } from './Segmenter'

export interface CreateSegmenterOptions {
  rvmDownsampleRatio?: number
}

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
