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

export type PostProcessingConfig = {
  sigmoid?: { steepness: number; threshold: number }
  erosion?: { pixels: number }
  ema?: { alpha: number }
}

export type UpsamplingConfig = {
  method?: 'bilinear' | 'guided'
  radius?: number
  eps?: number
}

export type PreProcessingConfig = {
  roiCropping?: { enabled: boolean }
}

export type ProcessorConfig =
  | {
      type: ProcessorType.BLUR
      blurRadius: number
      model?: SegmentationModel
      preProcessing?: PreProcessingConfig
      postProcessing?: PostProcessingConfig
      upsampling?: UpsamplingConfig
    }
  | {
      type: ProcessorType.VIRTUAL
      imagePath: string
      fileId?: string
      model?: SegmentationModel
      preProcessing?: PreProcessingConfig
      postProcessing?: PostProcessingConfig
      upsampling?: UpsamplingConfig
    }
  | ({ type: ProcessorType.FACE_LANDMARKS } & FaceLandmarksOptions)

export interface BackgroundProcessorInterface extends TrackProcessor<Track.Kind> {
  update(opts: ProcessorConfig): Promise<void>
  waitForReady?(): Promise<void>
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
