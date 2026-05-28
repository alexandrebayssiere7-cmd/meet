import { Track, TrackProcessor } from 'livekit-client'
import { AdvancedMattingProcessor } from './AdvancedMattingProcessor'
import { FaceLandmarksOptions } from './FaceLandmarksProcessor'

export enum ProcessorType {
  BLUR = 'blur',
  VIRTUAL = 'virtual',
  FACE_LANDMARKS = 'faceLandmarks',
}

export enum SegmentationModel {
  AUTO = 'auto',
  LANDSCAPE = 'landscape',
  MULTICLASS = 'multiclass',
}

export type PostProcessingConfig = {
  erosion?: { pixels: number }
  opening?: { radius: number }
  closing?: { radius: number }
  ema?: { alpha: number }
}

export type UpsamplingConfig = {
  radius?: number
  eps?: number
}

export type PreProcessingConfig = {
  roiCropping?: { enabled: boolean }
}

export type LatencyMode = 0

export type MaskBlendMode = 'frameLock' | 'live' | 'blend'

export type ProcessorConfig =
  | {
      type: ProcessorType.BLUR
      blurRadius: number
      model?: SegmentationModel
      preProcessing?: PreProcessingConfig
      postProcessing?: PostProcessingConfig
      upsampling?: UpsamplingConfig
      latencyMode?: LatencyMode
    }
  | {
      type: ProcessorType.VIRTUAL
      imagePath: string
      fileId?: string
      model?: SegmentationModel
      preProcessing?: PreProcessingConfig
      postProcessing?: PostProcessingConfig
      upsampling?: UpsamplingConfig
      latencyMode?: LatencyMode
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
