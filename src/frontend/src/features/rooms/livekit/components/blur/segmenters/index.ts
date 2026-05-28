import { SegmentationModel } from '..'
import { DepthAnythingSegmenter } from './DepthAnythingSegmenter'
import { LandscapeSegmenter } from './LandscapeSegmenter'
import { MulticlassSegmenter } from './MulticlassSegmenter'
import { Segmenter } from './Segmenter'

export type { Segmenter } from './Segmenter'
export { probeMediapipeDelegate } from './Segmenter'
export { DepthAnythingSegmenter } from './DepthAnythingSegmenter'

export function createSegmenter(model?: SegmentationModel): Segmenter {
  if (model === SegmentationModel.MULTICLASS) {
    return new MulticlassSegmenter()
  }
  if (model === SegmentationModel.DEPTH_ANYTHING) {
    return new DepthAnythingSegmenter()
  }
  return new LandscapeSegmenter()
}
