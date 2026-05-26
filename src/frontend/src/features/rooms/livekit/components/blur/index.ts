import { Track, TrackProcessor } from 'livekit-client'
import { AdvancedMattingProcessor } from './AdvancedMattingProcessor'
import { FaceLandmarksOptions } from './FaceLandmarksProcessor'

/**
 * Types of background processors supported by the application.
 */
export enum ProcessorType {
  /** Blurs the background behind the user. */
  BLUR = 'blur',
  /** Replaces the background with a virtual image. */
  VIRTUAL = 'virtual',
  /** Superimposes funny assets (glasses, beret, moustache) using facial landmarks. */
  FACE_LANDMARKS = 'faceLandmarks',
}

/**
 * Segmentation models available for background removal.
 */
export enum SegmentationModel {
  /** Automatically select the best model based on device benchmarks. */
  AUTO = 'auto',
  /** Lightweight, fast binary model (256x144) optimized for landscapes and lower-end GPUs. */
  LANDSCAPE = 'landscape',
  /** Sophisticated multiclass model (256x256) with detailed class segmentation (hair, skin, clothes). */
  MULTICLASS = 'multiclass',
  /** Robust Video Matting — ONNX-based model, high quality, higher latency. */
  RVM = 'rvm',
}

/** Supported 2D morphological operations. */
export type MorphologyOp = 'erosion' | 'dilation' | 'opening' | 'closing'

/**
 * Configuration options for GPU/CPU mask post-processing.
 */
export type PostProcessingConfig = {
  sigmoid?: { steepness: number; threshold: number }
  /** Applies erosion (shrinking) to the mask in pixels. Useful to reduce halo artifacts. */
  erosion?: { pixels: number }
  /** Fills small holes in the mask by applying a dilation followed by an erosion of the specified radius. */
  closing?: { radius: number }
  /** Temporal Exponential Moving Average smoothing factor. Helps reduce edge flickering between frames. */
  ema?: { alpha: number }
}

/**
 * Configuration options for mask upsampling.
 */
export type UpsamplingConfig = {
  /** Upsampling method: standard bilinear interpolation or edge-preserving guided filter. */
  method?: 'bilinear' | 'guided'
  /** Radius of the guided filter window in pixels. */
  radius?: number
  /** Regularization parameter epsilon for the guided filter. Prevents division by zero and controls detail sharpness. */
  eps?: number
}

/**
 * Configuration options for preprocessing full video frames before segmentation.
 */
export type PreProcessingConfig = {
  /** Configures region-of-interest cropping to keep the segmenter focused on the person. */
  roiCropping?: { enabled: boolean }
}

/**
 * Discriminated union of processor configurations.
 */
export type ProcessorConfig =
  | {
    type: ProcessorType.BLUR
    blurRadius: number
    model?: SegmentationModel
    preProcessing?: PreProcessingConfig
    rvmDownsampleRatio?: number
    postProcessing?: PostProcessingConfig
    upsampling?: UpsamplingConfig
    maxFrameOffset?: number
  }
  | {
    type: ProcessorType.VIRTUAL
    imagePath: string
    fileId?: string
    model?: SegmentationModel
    preProcessing?: PreProcessingConfig
    rvmDownsampleRatio?: number
    postProcessing?: PostProcessingConfig
    upsampling?: UpsamplingConfig
    maxFrameOffset?: number
  }
  | ({ type: ProcessorType.FACE_LANDMARKS } & FaceLandmarksOptions)

/**
 * Interface that all background track processors must implement.
 */
export interface BackgroundProcessorInterface extends TrackProcessor<Track.Kind> {
  /** Update the processor options dynamically without rebuilding the entire pipeline. */
  update(opts: ProcessorConfig): Promise<void>
  /** Resolves when the processor is initialized and producing frames. */
  waitForReady?(): Promise<void>
  /** Active options configured on this processor. */
  options: ProcessorConfig
}

/**
 * Factory class to query support and instantiate the appropriate background processor.
 */
export class BackgroundProcessorFactory {
  /** Returns true if the browser supports modern track processor APIs. */
  static hasModernApiSupport() {
    return true
  }

  /** Returns true if background processing is supported on the current device. */
  static isSupported() {
    return true
  }

  /**
   * Instantiates a background processor matching the provided config.
   * Note: Face landmarks processor is instantiated separately in effects UI.
   */
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

  /**
   * Helper to instantiate a processor from a nullable configuration object.
   */
  static fromProcessorConfig(data?: ProcessorConfig) {
    if (data) {
      return BackgroundProcessorFactory.getProcessor(data)
    }
    return undefined
  }
}
