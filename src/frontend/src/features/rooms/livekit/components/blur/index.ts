import { Track, TrackProcessor } from 'livekit-client'
import { AdvancedMattingProcessor } from './AdvancedMattingProcessor'
import { FaceLandmarksOptions } from './FaceLandmarksProcessor'

export enum ProcessorType {
  BLUR = 'blur',
  VIRTUAL = 'virtual',
  FACE_LANDMARKS = 'faceLandmarks',
}

export enum SegmentationModel {
  LANDSCAPE = 'landscape',
  MULTICLASS = 'multiclass',
}

export type MorphologyOp = 'erosion' | 'dilation' | 'opening' | 'closing'

export type PostProcessingConfig = {
  sigmoid?: { steepness: number; threshold: number }
  morphology?: { op: MorphologyOp; kernelSize: 3 | 5 | 7 }
  guidedFilter?: { radius: number; eps: number }
  ema?: { alpha: number }
}

export type ProcessorConfig =
  | {
      type: ProcessorType.BLUR
      blurRadius: number
      model?: SegmentationModel
      postProcessing?: PostProcessingConfig
    }
  | {
      type: ProcessorType.VIRTUAL
      imagePath: string
      fileId?: string
      model?: SegmentationModel
      postProcessing?: PostProcessingConfig
    }
  | ({ type: ProcessorType.FACE_LANDMARKS } & FaceLandmarksOptions)

export interface BackgroundProcessorInterface extends TrackProcessor<Track.Kind> {
  update(opts: ProcessorConfig): Promise<void>
  options: ProcessorConfig
}

export class BackgroundProcessorFactory {
  static hasModernApiSupport() {
    return true
  }

  static isSupported() {
    return true
  }

  static getProcessor(
    config: ProcessorConfig
  ): BackgroundProcessorInterface | undefined {
    if (
      config.type !== ProcessorType.BLUR &&
      config.type !== ProcessorType.VIRTUAL
    ) {
      return undefined
    }
    return new AdvancedMattingProcessor(config)
  }

  static fromProcessorConfig(data?: ProcessorConfig) {
    if (data) {
      return BackgroundProcessorFactory.getProcessor(data)
    }
    return undefined
  }
}
