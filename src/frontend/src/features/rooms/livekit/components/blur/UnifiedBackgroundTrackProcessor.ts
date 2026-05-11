import { ProcessorOptions, Track } from 'livekit-client'
import {
  ProcessorWrapper,
  VirtualBackground,
} from '@livekit/track-processors'
import { createLiveKitBlurProcessor, LiveKitBlurProcessor } from 'gregblur/livekit'
import { ProcessorConfig, BackgroundProcessorInterface, ProcessorType } from '.'

export class UnifiedBackgroundTrackProcessor implements BackgroundProcessorInterface {
  processor: ProcessorWrapper<{ imagePath?: string; blurRadius?: number }> | LiveKitBlurProcessor
  opts: ProcessorConfig
  processorType: ProcessorType

  constructor(opts: ProcessorConfig) {
    this.opts = opts
    this.processorType = opts.type

    if (opts.type === 'virtual') {
      this.processor = VirtualBackground(opts.imagePath)
    } else if (opts.type === 'blur') {
      console.log('🚀 Initializing High-Quality Blur with Gregblur')
      this.processor = createLiveKitBlurProcessor({
        blurRadius: opts.blurRadius,
        initialEnabled: true,
        segmentationModel: 'selfie-multiclass-256',
      })
    } else {
      throw new Error(
        'Must provide either imagePath for virtual background or blurRadius for blur'
      )
    }
  }

  async init(opts: ProcessorOptions<Track.Kind>) {
    const settings = opts.track.getSettings()
    console.log(
      `[UnifiedProcessor] 📱 Camera Dimensions: ${settings.width}x${settings.height} (Aspect: ${
        (settings.width || 0) / (settings.height || 1)
      })`
    )

    if ((settings.width || 0) < (settings.height || 0)) {
      console.log('[UnifiedProcessor] 🔄 PORTRAIT mode detected')
    }

    return this.processor.init(opts)
  }

  async restart(opts: ProcessorOptions<Track.Kind>) {
    console.log('[UnifiedProcessor] 🔄 Restarting processor...')
    return this.processor.restart(opts)
  }

  async destroy() {
    return this.processor.destroy()
  }

  async update(opts: ProcessorConfig): Promise<void> {
    this.opts = opts

    // Since EffectsConfiguration recreates the processor on type or blur radius changes,
    // this update() is primarily for updating virtual background images seamlessly.
    if (this.processorType === ProcessorType.VIRTUAL && opts.type === 'virtual') {
      const wrapper = this.processor as ProcessorWrapper<{ imagePath?: string; blurRadius?: number }>
      await wrapper.updateTransformerOptions({ imagePath: opts.imagePath })
    }
  }

  get name() {
    return this.processor.name
  }

  get processedTrack() {
    return this.processor.processedTrack
  }

  get options() {
    return this.opts
  }
}
