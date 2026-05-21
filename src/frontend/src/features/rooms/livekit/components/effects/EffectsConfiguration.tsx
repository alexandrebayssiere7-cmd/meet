import { LocalVideoTrack, Track } from 'livekit-client'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  BackgroundProcessorFactory,
  BackgroundProcessorInterface,
  PostProcessingConfig,
  PreProcessingConfig,
  UpsamplingConfig,
  ProcessorConfig,
  ProcessorType,
  SegmentationModel,
} from '../blur'
import { css } from '@/styled-system/css'
import { Button, Dialog, H, P, Text, ToggleButton } from '@/primitives'
import { VisualOnlyTooltip } from '@/primitives/VisualOnlyTooltip'
import { HStack, styled } from '@/styled-system/jsx'
import { BlurOn } from '@/components/icons/BlurOn'
import { BlurOnStrong } from '@/components/icons/BlurOnStrong'
import { useTrackToggle } from '@livekit/components-react'
import { Loader } from '@/primitives/Loader'
import { useSyncAfterDelay } from '@/hooks/useSyncAfterDelay'
import { FunnyEffects } from './FunnyEffects'
import { useHasFunnyEffectsAccess } from '../../hooks/useHasFunnyEffectsAccess'
import { useScreenReaderAnnounce } from '@/hooks/useScreenReaderAnnounce'
import {
  ListFilesParams,
  useListMyFiles,
} from '@/features/files/api/listFiles.ts'
import { useCreateFile } from '@/features/files/api/createFile.ts'
import { FileTrigger } from 'react-aria-components'
import { RiDeleteBinLine, RiImageAddFill } from '@remixicon/react'
import { useDeleteFile } from '@/features/files/api/deleteFile.ts'
import { useUser } from '@/features/auth'
import { ApiFileItem } from '@/features/files/api/types.ts'
import { useConfig } from '@/api/useConfig.ts'
import { usePersistentUserChoices } from '@/features/rooms/livekit/hooks/usePersistentUserChoices.ts'
import { proxy, useSnapshot } from 'valtio'
import { Spinner } from '@/primitives/Spinner.tsx'
import { useMattingErrors } from '../blur/errors/MattingErrorStore'
import { useMattingStats } from '../blur/stats/MattingStatsStore'

enum BlurRadius {
  NONE = 0,
  LIGHT = 5,
  NORMAL = 10,
}

const isSupported = BackgroundProcessorFactory.isSupported()

const Information = styled('div', {
  base: {
    backgroundColor: 'orange.50',
    borderRadius: '4px',
    padding: '0.75rem 0.75rem',
    alignItems: 'start',
  },
})

export type EffectsConfigurationProps = {
  isDisabled?: boolean
  videoTrack: LocalVideoTrack
  layout?: 'vertical' | 'horizontal'
}

const listFilesQueryParams: ListFilesParams = {
  filters: {
    type: 'background_image',
    upload_state: 'ready',
    is_creator_me: true,
    is_deleted: false,
  },
  pagination: {
    page: 1,
    pageSize: 20,
  },
}

function deriveIdFromProcessorConfig(config: ProcessorConfig) {
  if (config.type === ProcessorType.BLUR) {
    return `blur-${config.blurRadius}`
  } else if (config.type === ProcessorType.VIRTUAL) {
    // the imagePath is not stable for custom backgrounds
    // so we try first with the fileId
    if (config.fileId) {
      return `virtual-${config.fileId}`
    }
    return `virtual-${config.imagePath}`
  } else if (config.type === ProcessorType.FACE_LANDMARKS) {
    return 'face-landmarks'
  }
  throw new Error(`Unknown config type in config: ${config}`)
}

type SliderRowProps = {
  label: string
  displayValue: string
  value: number
  min: number
  max: number
  step: number
  disabled?: boolean
  onChange: (v: number) => void
}

const SliderRow = ({
  label,
  displayValue,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: SliderRowProps) => (
  <label
    className={css({
      display: 'flex',
      flexDirection: 'column',
      gap: '0.15rem',
      fontSize: 'sm',
      paddingLeft: '1.4rem',
      marginTop: '0.15rem',
    })}
    style={{ opacity: disabled ? 0.5 : 1 }}
  >
    <span>
      {label} : <strong>{displayValue}</strong>
    </span>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))}
      className={css({
        width: '100%',
        cursor: 'pointer',
        _disabled: { cursor: 'not-allowed' },
      })}
    />
  </label>
)

const modelLabel = (m: SegmentationModel | null): string => {
  switch (m) {
    case SegmentationModel.LANDSCAPE: return 'Landscape'
    case SegmentationModel.MULTICLASS: return 'Multiclass'
    case SegmentationModel.RVM: return 'RVM'
    case SegmentationModel.AUTO: return 'Auto'
    default: return '—'
  }
}

const MattingDiagnostics = () => {
  const { t } = useTranslation('rooms')
  const stats = useMattingStats()
  if (!stats.active) return null
  const showActive = stats.configuredModel === SegmentationModel.AUTO
  return (
    <div
      className={css({
        marginBottom: '1rem',
        padding: '0.6rem 0.75rem',
        backgroundColor: 'greyscale.50',
        borderRadius: '4px',
        border: '1px solid greyscale.250',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.2rem',
      })}
    >
      <Text
        variant="bodyXsBold"
        className={css({ marginBottom: '0.15rem' })}
      >
        {t('effects.stats.title', 'Diagnostics matting')}
      </Text>
      {showActive && (
        <Text variant="sm">
          {t('effects.stats.activeModel', 'Active model')} :{' '}
          <strong>{modelLabel(stats.currentModel)}</strong>
        </Text>
      )}
      <Text variant="sm">
        {t('effects.stats.captureLatency', 'Capture→display latency')} :{' '}
        <strong>{stats.captureToDisplayLatencyMs.toFixed(1)} ms</strong>
      </Text>
      <Text variant="sm">
        {t('effects.stats.maskGap', 'Mask→render gap')} :{' '}
        <strong>{stats.maskFrameGapMs.toFixed(1)} ms</strong>{' '}
        <span className={css({ color: 'greyscale.500' })}>
          ({t('effects.stats.windowAvg', 'avg of last {{n}}', { n: stats.samples })})
        </span>
      </Text>
      <Text variant="sm">
        {t('effects.stats.inference', 'Segmenter inference')} :{' '}
        <strong>{stats.segmenterInferenceMs.toFixed(1)} ms</strong>
      </Text>
      <Text variant="sm">
        {t('effects.stats.fps', 'FPS')} :{' '}
        <strong>{stats.renderFps.toFixed(1)}</strong>{' '}
        <span className={css({ color: 'greyscale.500' })}>
          ({t('effects.stats.fpsRender', 'render')}) ·{' '}
          <strong>{stats.segmenterFps.toFixed(1)}</strong>{' '}
          ({t('effects.stats.fpsSegmenter', 'segmenter')})
        </span>
      </Text>
    </div>
  )
}

// We use a valtio store so that the state is persisted between the join room
// and the actual room
const uploadNotPossibleLocalState = proxy({
  imageBackgroundConfig: null as null | {
    type: ProcessorType.VIRTUAL
    imagePath: string
    label: string
  },
})

export const EffectsConfiguration = ({
  isDisabled,
  videoTrack,
  layout = 'horizontal',
}: EffectsConfigurationProps) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const blurLightRef = useRef<HTMLButtonElement | null>(null)
  const { t } = useTranslation('rooms', { keyPrefix: 'effects' })
  const { toggle, enabled } = useTrackToggle({ source: Track.Source.Camera })
  const [processorPending, setProcessorPending] = useState(false)
  const processorPendingReveal = useSyncAfterDelay(processorPending)
  const hasFunnyEffectsAccess = useHasFunnyEffectsAccess()
  const announce = useScreenReaderAnnounce()
  const effectAnnouncementTimeout = useRef<ReturnType<
    typeof setTimeout
  > | null>(null)
  const effectAnnouncementId = useRef(0)

  const {
    saveProcessorConfig,
    userChoices: { processorConfig },
  } = usePersistentUserChoices()

  // ----- Advanced matting settings (model + pre/post-processing toggles) -----
  const initialModel: SegmentationModel =
    (processorConfig &&
      (processorConfig.type === ProcessorType.BLUR ||
        processorConfig.type === ProcessorType.VIRTUAL) &&
      processorConfig.model) ||
    SegmentationModel.AUTO
  const initialPP: PostProcessingConfig =
    (processorConfig &&
      (processorConfig.type === ProcessorType.BLUR ||
        processorConfig.type === ProcessorType.VIRTUAL) &&
      processorConfig.postProcessing) ||
    {}
  const initialUP: UpsamplingConfig =
    (processorConfig &&
      (processorConfig.type === ProcessorType.BLUR ||
        processorConfig.type === ProcessorType.VIRTUAL) &&
      processorConfig.upsampling) ||
    {}
  const initialPRE: PreProcessingConfig =
    (processorConfig &&
      (processorConfig.type === ProcessorType.BLUR ||
        processorConfig.type === ProcessorType.VIRTUAL) &&
      processorConfig.preProcessing) ||
    {}
  const [roiCroppingEnabled, setRoiCroppingEnabled] = useState(
    !!initialPRE.roiCropping?.enabled
  )
  const [model, setModel] = useState<SegmentationModel>(initialModel)
  const initialRvmRatio: number | undefined =
    processorConfig &&
    (processorConfig.type === ProcessorType.BLUR ||
      processorConfig.type === ProcessorType.VIRTUAL)
      ? processorConfig.rvmDownsampleRatio
      : undefined
  const [rvmManual, setRvmManual] = useState<boolean>(
    initialRvmRatio !== undefined
  )
  const [rvmRatio, setRvmRatio] = useState<number>(initialRvmRatio ?? 0.25)
  const [sigmoidEnabled, setSigmoidEnabled] = useState(!!initialPP.sigmoid)
  const [sigmoidSteepness, setSigmoidSteepness] = useState<number>(
    initialPP.sigmoid?.steepness ?? 10
  )
  const [sigmoidThreshold, setSigmoidThreshold] = useState<number>(
    initialPP.sigmoid?.threshold ?? 0.5
  )
  const [erosionEnabled, setErosionEnabled] = useState(!!initialPP.erosion)
  const [erosionPixels, setErosionPixels] = useState<number>(
    initialPP.erosion?.pixels ?? 2
  )
  const [upsamplingGuided, setUpsamplingGuided] = useState(
    initialUP.method === 'guided'
  )
  const [upsamplingRadius, setUpsamplingRadius] = useState<number>(
    initialUP.radius ?? 8
  )
  const [upsamplingEpsLog, setUpsamplingEpsLog] = useState<number>(
    Math.log10(initialUP.eps ?? 0.01)
  )
  const [emaEnabled, setEmaEnabled] = useState(!!initialPP.ema)
  const [emaAlpha, setEmaAlpha] = useState<number>(
    initialPP.ema?.alpha ?? 0.5
  )
  const [closingEnabled, setClosingEnabled] = useState(!!initialPP.closing)
  const [closingRadius, setClosingRadius] = useState<number>(
    initialPP.closing?.radius ?? 0
  )
  const initialMaxFrameOffset: number =
    (processorConfig &&
      (processorConfig.type === ProcessorType.BLUR ||
        processorConfig.type === ProcessorType.VIRTUAL) &&
      processorConfig.maxFrameOffset) ||
    0
  const [maxFrameOffset, setMaxFrameOffset] =
    useState<number>(initialMaxFrameOffset)

  // Continuous blur radius slider; only meaningful when a blur effect is selected.
  const initialBlurRadius =
    processorConfig?.type === ProcessorType.BLUR
      ? processorConfig.blurRadius
      : 10
  const [blurRadiusValue, setBlurRadiusValue] =
    useState<number>(initialBlurRadius)

  const buildPreProcessing = useCallback((): PreProcessingConfig => {
    const cfg: PreProcessingConfig = {}
    if (roiCroppingEnabled) cfg.roiCropping = { enabled: true }
    return cfg
  }, [roiCroppingEnabled])

  const buildPostProcessing = useCallback((): PostProcessingConfig => {
    const cfg: PostProcessingConfig = {}
    if (sigmoidEnabled)
      cfg.sigmoid = { steepness: sigmoidSteepness, threshold: sigmoidThreshold }
    if (erosionEnabled && erosionPixels > 0)
      cfg.erosion = { pixels: erosionPixels }
    if (emaEnabled) cfg.ema = { alpha: emaAlpha }
    if (closingEnabled) cfg.closing = { radius: closingRadius }
    return cfg
  }, [
    sigmoidEnabled,
    sigmoidSteepness,
    sigmoidThreshold,
    erosionEnabled,
    erosionPixels,
    emaEnabled,
    emaAlpha,
    closingEnabled,
    closingRadius,
  ])

  const buildUpsampling = useCallback((): UpsamplingConfig => {
    if (!upsamplingGuided) return { method: 'bilinear' }
    return {
      method: 'guided',
      radius: upsamplingRadius,
      eps: Math.pow(10, upsamplingEpsLog),
    }
  }, [upsamplingGuided, upsamplingRadius, upsamplingEpsLog])

  const withAdvanced = useCallback(
    (config: ProcessorConfig): ProcessorConfig => {
      if (
        config.type === ProcessorType.BLUR ||
        config.type === ProcessorType.VIRTUAL
      ) {
        return {
          ...config,
          model,
          preProcessing: buildPreProcessing(),
          rvmDownsampleRatio:
            model === SegmentationModel.RVM && rvmManual
              ? rvmRatio
              : undefined,
          postProcessing: buildPostProcessing(),
          upsampling: buildUpsampling(),
          maxFrameOffset,
        }
      }
      return config
    },
    [model, buildPreProcessing, rvmManual, rvmRatio, buildPostProcessing, buildUpsampling, maxFrameOffset]
  )

  const selectedId = useMemo(
    () =>
      processorConfig ? deriveIdFromProcessorConfig(processorConfig) : 'none',
    [processorConfig]
  )

  const uploadNotPossibleSnap = useSnapshot(uploadNotPossibleLocalState)
  const mattingErrors = useMattingErrors()

  const announceEffectStatusMessage = useCallback(
    (message: string) => {
      effectAnnouncementId.current += 1
      const currentId = effectAnnouncementId.current

      if (effectAnnouncementTimeout.current) {
        clearTimeout(effectAnnouncementTimeout.current)
      }

      effectAnnouncementTimeout.current = setTimeout(() => {
        if (currentId !== effectAnnouncementId.current) return
        announce(message)
      }, 80)
    },
    [announce]
  )

  const getVirtualBackgroundName = useCallback(
    (imagePath?: string) => {
      if (!imagePath) return ''
      const match = imagePath.match(/\/backgrounds\/(\d+)\.jpg$/)
      if (!match) return ''
      const index = Number(match[1]) - 1
      if (Number.isNaN(index)) return ''
      return t(`virtual.presets.descriptions.${index}`)
    },
    [t]
  )

  const updateEffectStatusMessage = useCallback(
    (config: ProcessorConfig, wasSelectedBeforeToggle: boolean) => {
      if (wasSelectedBeforeToggle) {
        announceEffectStatusMessage(t('blur.status.none'))
        return
      }

      if (config.type === ProcessorType.BLUR) {
        const message =
          config.blurRadius === BlurRadius.LIGHT
            ? t('blur.status.light')
            : t('blur.status.strong')
        announceEffectStatusMessage(message)
        return
      }

      if (config.type === ProcessorType.VIRTUAL) {
        const backgroundName = getVirtualBackgroundName(config.imagePath)
        if (backgroundName) {
          announceEffectStatusMessage(
            `${t('virtual.selectedLabel')} ${backgroundName}`
          )
          return
        }
      }
    },
    [announceEffectStatusMessage, getVirtualBackgroundName, t]
  )

  const toggleEffect = useCallback(
    async (rawConfig: ProcessorConfig) => {
      const config = withAdvanced(rawConfig)
      setProcessorPending(true)
      const wasSelectedBeforeToggle =
        selectedId === deriveIdFromProcessorConfig(config)

      if (!videoTrack) {
        /**
         * Special case: if no video track is available, then we must pass directly the processor into the
         * toggle call. Otherwise, the rest of the function below would not have a videoTrack to call
         * setProcessor on.
         *
         * We arrive in this condition when we enter the room with the camera already off.
         */
        const newProcessorTmp = BackgroundProcessorFactory.getProcessor(config)!
        await toggle(true, {
          processor: newProcessorTmp,
        })
        setTimeout(() => setProcessorPending(false))
        return
      }

      if (!enabled) {
        await toggle(true)
      }

      const processor =
        videoTrack?.getProcessor() as BackgroundProcessorInterface
      try {
        if (wasSelectedBeforeToggle) {
          // Stop processor.
          await videoTrack.stopProcessor()
          saveProcessorConfig(undefined)
        } else if (
          !processor ||
          (processor.options.type !== config.type &&
            !BackgroundProcessorFactory.hasModernApiSupport())
        ) {
          // Change processor.
          const newProcessor = BackgroundProcessorFactory.getProcessor(config)!
          // IMPORTANT: Must explicitly stop previous processor before setting a new one
          // in browsers without modern API support to prevent UI crashes.
          // This workaround is needed until this issue is resolved:
          // https://github.com/livekit/track-processors-js/issues/85
          if (!BackgroundProcessorFactory.hasModernApiSupport()) {
            await videoTrack.stopProcessor()
          }
          await videoTrack.setProcessor(newProcessor)
          saveProcessorConfig(config)
        } else {
          await processor?.update(config)
          saveProcessorConfig(config)
        }

        updateEffectStatusMessage(config, wasSelectedBeforeToggle)
      } catch (error) {
        console.error('Error applying effect:', error)
      } finally {
        // Without setTimeout the DOM is not refreshing when updating the options.
        setTimeout(() => setProcessorPending(false))
      }
    },
    [
      enabled,
      saveProcessorConfig,
      selectedId,
      toggle,
      updateEffectStatusMessage,
      videoTrack,
      withAdvanced,
    ]
  )

  const applyAdvancedSettings = useCallback(async () => {
    if (!videoTrack || !processorConfig) return
    if (
      processorConfig.type !== ProcessorType.BLUR &&
      processorConfig.type !== ProcessorType.VIRTUAL
    )
      return
    const processor =
      videoTrack.getProcessor() as BackgroundProcessorInterface | undefined
    if (!processor) return
    const newConfig = withAdvanced(processorConfig)
    setProcessorPending(true)
    try {
      await processor.update(newConfig)
      // Wait until the new segmenter is fully loaded before clearing the spinner.
      await processor.waitForReady?.()
      saveProcessorConfig(newConfig)
    } finally {
      setTimeout(() => setProcessorPending(false))
    }
  }, [videoTrack, processorConfig, withAdvanced, saveProcessorConfig])

  // Live blur radius slider: debounced apply that doesn't go through toggleEffect
  // (which would stop the processor when slider sits on the same value as selected).
  const blurDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const applyBlurRadius = useCallback(
    (radius: number) => {
      setBlurRadiusValue(radius)
      if (blurDebounceRef.current) clearTimeout(blurDebounceRef.current)
      blurDebounceRef.current = setTimeout(async () => {
        const config = withAdvanced({
          type: ProcessorType.BLUR,
          blurRadius: radius,
        })
        const processor = videoTrack?.getProcessor() as
          | BackgroundProcessorInterface
          | undefined
        if (processor && processor.options.type === ProcessorType.BLUR) {
          await processor.update(config)
          saveProcessorConfig(config)
        } else {
          toggleEffect(config)
        }
      }, 200)
    },
    [videoTrack, withAdvanced, saveProcessorConfig, toggleEffect]
  )

  // Keep the slider in sync when the user picks a preset button (Light/Strong)
  // or when blur is disabled.
  useEffect(() => {
    if (processorConfig?.type === ProcessorType.BLUR) {
      setBlurRadiusValue(processorConfig.blurRadius)
    }
  }, [processorConfig])

  const { data: appConfig } = useConfig()
  const { isLoggedIn } = useUser()
  const canUploadBackground =
    isLoggedIn === true &&
    appConfig?.background_image?.upload_is_enabled === true
  // We split the error state in 2 parts so that there are no visual glitches when closing the alert
  const [personalBackgroundHasError, setPersonalBackgroundHasError] =
    useState<boolean>(false)
  const [personalBackgroundError, setPersonalBackgroundError] = useState<
    'file_too_large' | 'invalid_file_type' | null
  >(null)
  const createFileMutation = useCreateFile()
  const fileBeingUploadedObjectUrlRef = useRef<string | null>(null)
  useEffect(() => {
    return () => {
      if (fileBeingUploadedObjectUrlRef.current) {
        URL.revokeObjectURL(fileBeingUploadedObjectUrlRef.current)
        fileBeingUploadedObjectUrlRef.current = null
      }
    }
  }, [])
  const deleteFileMutation = useDeleteFile()
  const filesQ = useListMyFiles(listFilesQueryParams)
  const hasReachedMaxNbBackgrounds =
    (canUploadBackground &&
      appConfig &&
      filesQ.data &&
      filesQ.data.count >= appConfig.background_image.max_count_by_user) ??
    false

  const getHandleSelectChangeFile = useCallback(
    (file: ApiFileItem) => {
      return async () => {
        await toggleEffect({
          type: ProcessorType.VIRTUAL,
          imagePath: file.url!,
          fileId: file.id,
        })
      }
    },
    [toggleEffect]
  )

  const handleNewBackgroundFilePicked = useCallback(
    async (file: File) => {
      if (
        !(
          appConfig?.background_image?.allowed_mimetypes?.includes(file.type) ??
          false
        )
      ) {
        setPersonalBackgroundError('invalid_file_type')
        setPersonalBackgroundHasError(true)
        return
      }
      if (file.size > (appConfig?.background_image?.max_size ?? 0)) {
        setPersonalBackgroundError('file_too_large')
        setPersonalBackgroundHasError(true)
        return
      }

      // When the user is not logged in, we fallback to just loading that image
      if (!canUploadBackground) {
        // For the preview to work, we somehow need to create a data URL from the raw file.
        // revoking is handled with userChoicesStore
        if (uploadNotPossibleSnap.imageBackgroundConfig) {
          URL.revokeObjectURL(
            uploadNotPossibleSnap.imageBackgroundConfig.imagePath
          )
        }
        const imagePath = URL.createObjectURL(file)

        // We concatenate with the image path so that the constructed file-id
        // is unique for local files.
        const fileId = `local-image-${imagePath}`
        await toggleEffect({
          type: ProcessorType.VIRTUAL,
          imagePath,
          fileId,
        })
        uploadNotPossibleLocalState.imageBackgroundConfig = {
          type: ProcessorType.VIRTUAL,
          label: file.name.split('.')[0],
          imagePath,
        }
      } else {
        // Otherwise we create the file in the backend and automatically select it
        // when it's uploaded.
        // We create a local version so that we can quickly display it in the frontend.
        if (fileBeingUploadedObjectUrlRef.current) {
          URL.revokeObjectURL(fileBeingUploadedObjectUrlRef.current)
        }
        fileBeingUploadedObjectUrlRef.current = URL.createObjectURL(file)
        createFileMutation.mutate(
          {
            file,
            onProgress: (progress) => {
              console.debug('upload-progress', progress)
            },
          },
          {
            onSuccess: (file) => {
              // We automatically select that created file
              getHandleSelectChangeFile(file)()
            },
          }
        )
      }
    },
    [
      appConfig?.background_image?.allowed_mimetypes,
      appConfig?.background_image?.max_size,
      canUploadBackground,
      createFileMutation,
      getHandleSelectChangeFile,
      toggleEffect,
      uploadNotPossibleSnap.imageBackgroundConfig,
    ]
  )

  const filePickerErrorContext = useMemo(
    () => ({
      allowedExtension: appConfig?.background_image?.allowed_extensions ?? [],
      maxSize: (appConfig?.background_image?.max_size ?? 0) / (1024 * 1024),
    }),
    [appConfig?.background_image]
  )

  const processorOptions = useMemo<{
    isDisabled: boolean
    blurBased: {
      radius: BlurRadius
      Icon: React.FC
      ref?: React.Ref<HTMLButtonElement>
      tooltip: string
      id: string
      config: ProcessorConfig
      isSelected: boolean
    }[]
    virtualBackgrounds: {
      id: string
      config: ProcessorConfig
      isSelected: boolean
      tooltip: string
      ariaLabel: string
      thumbnailPath: string
      index: number
    }[]
    remoteCustomVirtualBackgrounds: {
      id: string
      config: ProcessorConfig
      isSelected: boolean
      tooltip: string
      file: ApiFileItem
    }[]
  }>(() => {
    return {
      isDisabled: (processorPendingReveal || isDisabled) ?? false,
      blurBased: [
        {
          key: 'light',
          radius: BlurRadius.LIGHT,
          icon: BlurOn,
          ref: blurLightRef,
        },
        {
          key: 'normal',
          radius: BlurRadius.NORMAL,
          icon: BlurOnStrong,
          ref: undefined,
        },
      ].map((item) => {
        const config: ProcessorConfig = {
          type: ProcessorType.BLUR,
          blurRadius: item.radius,
        }
        const id = deriveIdFromProcessorConfig(config)
        return {
          id,
          tooltip: t(`blur.light.${selectedId === id ? 'clear' : 'apply'}`),
          radius: item.radius,
          isSelected: selectedId === id,
          Icon: item.icon,
          ref: item.ref,
          config,
        }
      }),
      virtualBackgrounds: [...Array(8).keys()].map((index) => {
        const imagePath = `/assets/backgrounds/${index + 1}.jpg`
        const thumbnailPath = `/assets/backgrounds/thumbnails/${index + 1}.jpg`
        const config: ProcessorConfig = {
          type: ProcessorType.VIRTUAL,
          imagePath,
        }
        const id = deriveIdFromProcessorConfig(config)
        const isSelected = selectedId === id
        const prefix = isSelected ? 'selectedLabel' : 'apply'
        const backgroundName = t(`virtual.presets.descriptions.${index}`)
        const ariaLabel = `${t(`virtual.presets.${prefix}`)} ${backgroundName}`

        return {
          tooltip: backgroundName,
          id,
          config,
          isSelected: selectedId === id,
          thumbnailPath,
          ariaLabel,
          index,
        }
      }),
      remoteCustomVirtualBackgrounds: (filesQ.data?.results ?? [])
        .filter((file) => file.url)
        .map((file) => {
          const config: ProcessorConfig = {
            type: ProcessorType.VIRTUAL,
            imagePath: file.url!,
            fileId: file.id,
          }

          const id = deriveIdFromProcessorConfig(config)

          return {
            tooltip: file.title,
            id,
            config,
            isSelected: selectedId === id,
            file,
          }
        }),
    }
  }, [processorPendingReveal, isDisabled, filesQ.data?.results, t, selectedId])

  useEffect(() => {
    const videoElement = videoRef.current
    if (!videoElement) return

    const attachVideoTrack = async () => videoTrack?.attach(videoElement)
    attachVideoTrack()

    return () => {
      if (!videoElement) return
      videoTrack.detach(videoElement)
    }
  }, [videoTrack, videoTrack?.isMuted])

  useEffect(() => {
    if (!blurLightRef.current) return

    const rafId = requestAnimationFrame(() => {
      blurLightRef.current?.focus({ preventScroll: true })
    })

    return () => {
      cancelAnimationFrame(rafId)
    }
  }, [])

  useEffect(
    () => () => {
      if (effectAnnouncementTimeout.current) {
        clearTimeout(effectAnnouncementTimeout.current)
      }
    },
    []
  )

  return (
    <div
      className={css(
        layout === 'vertical'
          ? {
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem',
            }
          : {
              display: 'flex',
              gap: '1.5rem',
              flexDirection: 'column',
              md: {
                flexDirection: 'row',
                overflow: 'hidden',
              },
            }
      )}
    >
      <div
        className={css({
          width: '100%',
          aspectRatio: 16 / 9,
          position: 'relative',
          overflow: 'hidden',
          borderRadius: '8px',
        })}
      >
        {videoTrack && !videoTrack.isMuted ? (
          <video
            ref={videoRef}
            width="100%"
            muted
            style={{
              transform: 'rotateY(180deg)',
              [layout === 'vertical' ? 'height' : 'minHeight']: '175px',
              borderRadius: '8px',
            }}
          />
        ) : (
          <div
            style={{
              width: '100%',
              height: '100%',
              display: 'flex',
              backgroundColor: 'black',
              justifyContent: 'center',
              flexDirection: 'column',
            }}
          >
            <P
              style={{
                color: 'white',
                textAlign: 'center',
                textWrap: 'balance',
                marginBottom: 0,
              }}
            >
              {t(isDisabled ? 'cameraDisabled' : 'activateCamera')}
            </P>
          </div>
        )}
        {processorPendingReveal && (
          <div
            className={css({
              position: 'absolute',
              right: '8px',
              bottom: '8px',
            })}
          >
            <Loader />
          </div>
        )}
      </div>
      <div
        className={css(
          layout === 'horizontal'
            ? {
                md: {
                  borderLeft: '1px solid greyscale.250',
                  paddingLeft: '1.5rem',
                  width: '420px',
                  flexShrink: 0,
                },
              }
            : {}
        )}
      >
        {hasFunnyEffectsAccess && (
          <FunnyEffects
            videoTrack={videoTrack}
            isPending={processorPendingReveal}
            onPending={setProcessorPending}
          />
        )}
        {isSupported ? (
          <div>
            {mattingErrors.filter(e => e.level === 'error').map(e => (
              <Information key={e.code} style={{ marginBottom: '1rem' }}>
                <Text variant="bodyXsMedium">
                  {t(`matting.errors.${e.code}`, { defaultValue: e.detail ?? e.code })}
                </Text>
              </Information>
            ))}
            <div>
              <H
                lvl={2}
                style={{
                  marginBottom: '1rem',
                }}
                variant="bodyXsBold"
              >
                {t('blur.title')}
              </H>
              <div>
                <div
                  className={css({
                    display: 'flex',
                    gap: '1.25rem',
                  })}
                >
                  {processorOptions.blurBased.map(({ Icon, ...option }) => (
                    <ToggleButton
                      key={option.id}
                      ref={option.ref}
                      variant="bigSquare"
                      aria-label={option.tooltip}
                      tooltip={option.tooltip}
                      isDisabled={processorOptions.isDisabled}
                      onChange={() => toggleEffect(option.config)}
                      isSelected={option.isSelected}
                      data-attr={`toggle-${option.id}`}
                    >
                      <Icon />
                    </ToggleButton>
                  ))}
                </div>
                <div className={css({ marginTop: '0.6rem' })}>
                  <SliderRow
                    label={t('advanced.params.blurRadius')}
                    displayValue={`${blurRadiusValue} px`}
                    value={blurRadiusValue}
                    min={1}
                    max={50}
                    step={1}
                    disabled={processorOptions.isDisabled}
                    onChange={applyBlurRadius}
                  />
                </div>
              </div>

              <div className={css({ marginTop: '1.5rem' })}>
                <H
                  lvl={2}
                  style={{
                    marginBottom: '0.6rem',
                  }}
                  variant="bodyXsBold"
                >
                  {t('virtual.title')}
                </H>
              </div>

              <div
                className={css({
                  marginBottom: '1rem',
                })}
              >
                <H
                  lvl={2}
                  style={{
                    marginBottom: '0.4rem',
                  }}
                  variant="bodyXsMedium"
                >
                  {t('virtual.personal.title')}
                </H>

                <div
                  className={css({
                    display: 'flex',
                    gap: '1.25rem',
                    paddingBottom: '0.5rem',
                    flexWrap: 'wrap',
                  })}
                >
                  {createFileMutation.isPending &&
                    fileBeingUploadedObjectUrlRef.current && (
                      <VisualOnlyTooltip
                        tooltip={t('virtual.personal.uploadInProgress')}
                      >
                        <div className={css({ position: 'relative' })}>
                          <ToggleButton
                            variant="bigSquare"
                            aria-label={t('virtual.personal.uploadInProgress')}
                            isDisabled={true}
                            data-attr={`virtual-upload-in-progress`}
                            className={css({
                              bgSize: 'cover',
                              filter: 'grayscale(1)',
                            })}
                            style={{
                              backgroundImage: `url(${fileBeingUploadedObjectUrlRef.current})`,
                            }}
                          />
                          <div
                            className={css({
                              position: 'absolute',
                              top: '50%',
                              left: '50%',
                              transform: 'translate(-50%, -50%)',
                            })}
                          >
                            <Spinner size={24} />
                          </div>
                        </div>
                      </VisualOnlyTooltip>
                    )}
                  {canUploadBackground &&
                    processorOptions.remoteCustomVirtualBackgrounds.map(
                      (option) => (
                        <div
                          key={option.id}
                          className={
                            'hoverGroup ' + css({ position: 'relative' })
                          }
                        >
                          <VisualOnlyTooltip tooltip={option.tooltip}>
                            <ToggleButton
                              variant="bigSquare"
                              aria-label={option.tooltip}
                              isDisabled={processorOptions.isDisabled}
                              onChange={getHandleSelectChangeFile(option.file)}
                              isSelected={option.isSelected}
                              className={css({
                                bgSize: 'cover',
                              })}
                              style={{
                                backgroundImage: `url(${option.file.url!})`,
                              }}
                              data-attr={`toggle-virtual-${option.file.id}`}
                            />
                          </VisualOnlyTooltip>
                          <Button
                            className={
                              'hoverGroupChild ' +
                              css({
                                position: 'absolute',
                                top: '-8px',
                                right: '-8px',
                                transition: 'opacity 0.2s ease-in-out',
                              })
                            }
                            size={'xs'}
                            variant={'tertiary'}
                            onClick={() => {
                              if (option.isSelected) {
                                // we remove the current effect
                                toggleEffect(option.config)
                              }
                              deleteFileMutation.mutate({
                                fileId: option.file.id,
                              })
                            }}
                            isDisabled={deleteFileMutation.isPending}
                          >
                            <RiDeleteBinLine size={16} />
                          </Button>
                        </div>
                      )
                    )}
                  {!canUploadBackground &&
                    uploadNotPossibleSnap.imageBackgroundConfig && (
                      <VisualOnlyTooltip
                        tooltip={
                          uploadNotPossibleSnap.imageBackgroundConfig.label
                        }
                      >
                        <ToggleButton
                          variant="bigSquare"
                          aria-label={
                            uploadNotPossibleSnap.imageBackgroundConfig.label
                          }
                          isDisabled={
                            processorOptions.isDisabled ||
                            createFileMutation.isPending
                          }
                          onChange={() => {
                            toggleEffect(
                              uploadNotPossibleSnap.imageBackgroundConfig!
                            )
                          }}
                          isSelected={
                            deriveIdFromProcessorConfig(
                              uploadNotPossibleSnap.imageBackgroundConfig
                            ) === selectedId
                          }
                          className={css({
                            bgSize: 'cover',
                          })}
                          style={{
                            backgroundImage: `url(${uploadNotPossibleSnap.imageBackgroundConfig.imagePath})`,
                          }}
                          data-attr={`toggle-virtual-local`}
                        />
                      </VisualOnlyTooltip>
                    )}
                  <FileTrigger
                    acceptedFileTypes={
                      appConfig?.background_image?.allowed_mimetypes ?? [
                        'image/png',
                        'image/jpeg',
                      ]
                    }
                    onSelect={(e) => {
                      if (e && e.item(0)) {
                        const file = e.item(0) as File
                        handleNewBackgroundFilePicked(file)
                      }
                    }}
                  >
                    <Button
                      variant="bigSquare"
                      aria-label={t('virtual.personal.selectFileTooltip')}
                      tooltip={t('virtual.personal.selectFileTooltip')}
                      isDisabled={
                        (canUploadBackground &&
                          filesQ.data &&
                          filesQ.data.count >=
                            (appConfig?.background_image?.max_count_by_user ??
                              0)) ||
                        processorOptions.isDisabled ||
                        createFileMutation.isPending
                      }
                      data-attr="input-file-select-personal-background"
                    >
                      <RiImageAddFill />
                    </Button>
                  </FileTrigger>
                </div>
                {!isLoggedIn && (
                  <Text variant="xsNote">
                    {t('virtual.personal.notLoggedInWarning')}
                  </Text>
                )}
                {!canUploadBackground && isLoggedIn && (
                  <Text variant="xsNote">
                    {t('virtual.personal.warningUploadDisabled')}
                  </Text>
                )}
                {hasReachedMaxNbBackgrounds && (
                  <Text variant="xsNote">
                    {t('virtual.personal.uploadLimitReached')}
                  </Text>
                )}
              </div>
              <div
                className={css({
                  marginTop: '0.4rem',
                })}
              >
                <H
                  lvl={2}
                  style={{
                    marginBottom: '0.4rem',
                  }}
                  variant="bodyXsMedium"
                >
                  {t('virtual.presets.title')}
                </H>
                <div
                  className={css({
                    display: 'flex',
                    gap: '1.25rem',
                    paddingBottom: '0.5rem',
                    flexWrap: 'wrap',
                  })}
                >
                  {processorOptions.virtualBackgrounds.map((option) => (
                    <VisualOnlyTooltip key={option.id} tooltip={option.tooltip}>
                      <ToggleButton
                        variant="bigSquare"
                        aria-label={option.ariaLabel}
                        isDisabled={processorOptions.isDisabled}
                        onChange={() => toggleEffect(option.config)}
                        isSelected={option.isSelected}
                        className={css({
                          bgSize: 'cover',
                        })}
                        style={{
                          backgroundImage: `url(${option.thumbnailPath})`,
                        }}
                        data-attr={`toggle-virtual-preset-${option.index}`}
                      />
                    </VisualOnlyTooltip>
                  ))}
                </div>
              </div>

              {/* Advanced matting settings */}
              <div
                className={css({
                  marginTop: '1.5rem',
                  paddingTop: '1rem',
                  borderTop: '1px solid greyscale.250',
                })}
              >
                <H
                  lvl={2}
                  style={{ marginBottom: '0.6rem' }}
                  variant="bodyXsBold"
                >
                  {t('advanced.title')}
                </H>

                <H
                  lvl={3}
                  style={{ marginBottom: '0.4rem' }}
                  variant="bodyXsMedium"
                >
                  {t('advanced.model.title')}
                </H>
                <div
                  className={css({
                    display: 'flex',
                    gap: '1rem',
                    marginBottom: '1rem',
                  })}
                >
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="radio"
                      name="matting-model"
                      checked={model === SegmentationModel.AUTO}
                      onChange={() => setModel(SegmentationModel.AUTO)}
                    />
                    <Text variant="sm">{t('advanced.model.auto')}</Text>
                  </label>
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="radio"
                      name="matting-model"
                      checked={model === SegmentationModel.LANDSCAPE}
                      onChange={() => setModel(SegmentationModel.LANDSCAPE)}
                    />
                    <Text variant="sm">{t('advanced.model.landscape')}</Text>
                  </label>
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="radio"
                      name="matting-model"
                      checked={model === SegmentationModel.MULTICLASS}
                      onChange={() => setModel(SegmentationModel.MULTICLASS)}
                    />
                    <Text variant="sm">{t('advanced.model.multiclass')}</Text>
                  </label>
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="radio"
                      name="matting-model"
                      checked={model === SegmentationModel.RVM}
                      onChange={() => setModel(SegmentationModel.RVM)}
                    />
                    <Text variant="sm">{t('advanced.model.rvm')}</Text>
                  </label>
                </div>

                <MattingDiagnostics />

                {model === SegmentationModel.RVM && (
                  <div
                    className={css({
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.35rem',
                      marginBottom: '0.8rem',
                    })}
                  >
                    <label
                      className={css({
                        display: 'flex',
                        gap: '0.4rem',
                        alignItems: 'center',
                        cursor: 'pointer',
                      })}
                    >
                      <input
                        type="checkbox"
                        checked={rvmManual}
                        onChange={(e) => setRvmManual(e.target.checked)}
                      />
                      <Text variant="sm">
                        {t('advanced.rvmDownsample.manual')}
                      </Text>
                    </label>
                    <SliderRow
                      label={t('advanced.rvmDownsample.label')}
                      displayValue={rvmRatio.toFixed(3)}
                      value={rvmRatio}
                      min={0.125}
                      max={1.0}
                      step={0.125}
                      disabled={!rvmManual}
                      onChange={setRvmRatio}
                    />
                  </div>
                )}

                <H
                  lvl={3}
                  style={{ marginBottom: '0.4rem' }}
                  variant="bodyXsMedium"
                >
                  {t('advanced.sync.title')}
                </H>
                <div
                  className={css({
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                    marginBottom: '1rem',
                  })}
                >
                  <SliderRow
                    label={t('advanced.params.maxFrameOffset')}
                    displayValue={
                      maxFrameOffset === 0
                        ? t('advanced.params.maxFrameOffsetStrict')
                        : `${maxFrameOffset}`
                    }
                    value={maxFrameOffset}
                    min={0}
                    max={10}
                    step={1}
                    onChange={setMaxFrameOffset}
                  />
                </div>

                <H
                  lvl={3}
                  style={{ marginBottom: '0.4rem' }}
                  variant="bodyXsMedium"
                >
                  {t('advanced.preProcessing.title')}
                </H>
                <div
                  className={css({
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                    marginBottom: '1rem',
                  })}
                >
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="checkbox"
                      checked={roiCroppingEnabled}
                      onChange={(e) => setRoiCroppingEnabled(e.target.checked)}
                    />
                    <Text variant="sm">{t('advanced.preProcessing.roiCropping')}</Text>
                  </label>
                </div>

                <H
                  lvl={3}
                  style={{ marginBottom: '0.4rem' }}
                  variant="bodyXsMedium"
                >
                  {t('advanced.postProcessing.title')}
                </H>
                <div
                  className={css({
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                  })}
                >
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="checkbox"
                      checked={sigmoidEnabled}
                      onChange={(e) => setSigmoidEnabled(e.target.checked)}
                    />
                    <Text variant="sm">{t('advanced.postProcessing.sigmoid')}</Text>
                  </label>
                  <SliderRow
                    label={t('advanced.params.sigmoidSteepness')}
                    displayValue={sigmoidSteepness.toFixed(1)}
                    value={sigmoidSteepness}
                    min={0.5}
                    max={20}
                    step={0.5}
                    disabled={!sigmoidEnabled}
                    onChange={setSigmoidSteepness}
                  />
                  <SliderRow
                    label={t('advanced.params.sigmoidThreshold')}
                    displayValue={sigmoidThreshold.toFixed(2)}
                    value={sigmoidThreshold}
                    min={0.2}
                    max={0.8}
                    step={0.05}
                    disabled={!sigmoidEnabled}
                    onChange={setSigmoidThreshold}
                  />
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="checkbox"
                      checked={erosionEnabled}
                      onChange={(e) => setErosionEnabled(e.target.checked)}
                    />
                    <Text variant="sm">{t('advanced.postProcessing.erosion')}</Text>
                  </label>
                  <SliderRow
                    label={t('advanced.params.erosionPixels')}
                    displayValue={`${erosionPixels} px`}
                    value={erosionPixels}
                    min={0}
                    max={6}
                    step={1}
                    disabled={!erosionEnabled}
                    onChange={setErosionPixels}
                  />
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="checkbox"
                      checked={emaEnabled}
                      onChange={(e) => setEmaEnabled(e.target.checked)}
                    />
                    <Text variant="sm">{t('advanced.postProcessing.ema')}</Text>
                  </label>
                  <SliderRow
                    label={t('advanced.params.emaAlpha')}
                    displayValue={emaAlpha.toFixed(2)}
                    value={emaAlpha}
                    min={0.05}
                    max={1.0}
                    step={0.05}
                    disabled={!emaEnabled}
                    onChange={setEmaAlpha}
                  />
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="checkbox"
                      checked={closingEnabled}
                      onChange={(e) => setClosingEnabled(e.target.checked)}
                    />
                    <Text variant="sm">{t('advanced.postProcessing.holeFilling')}</Text>
                  </label>
                  <SliderRow
                    label={t('advanced.params.holeFillingRadius')}
                    displayValue={`${closingRadius} px`}
                    value={closingRadius}
                    min={0}
                    max={8}
                    step={1}
                    disabled={!closingEnabled}
                    onChange={setClosingRadius}
                  />
                </div>
                <H
                  lvl={3}
                  style={{ marginBottom: '0.4rem', marginTop: '0.75rem' }}
                  variant="bodyXsMedium"
                >
                  {t('advanced.upsampling.title')}
                </H>
                <div
                  className={css({
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                  })}
                >
                  <label
                    className={css({
                      display: 'flex',
                      gap: '0.4rem',
                      alignItems: 'center',
                      cursor: 'pointer',
                    })}
                  >
                    <input
                      type="checkbox"
                      checked={upsamplingGuided}
                      onChange={(e) => setUpsamplingGuided(e.target.checked)}
                    />
                    <Text variant="sm">{t('advanced.upsampling.guidedFilter')}</Text>
                  </label>
                  <SliderRow
                    label={t('advanced.params.upsamplingRadius')}
                    displayValue={String(upsamplingRadius)}
                    value={upsamplingRadius}
                    min={2}
                    max={32}
                    step={1}
                    disabled={!upsamplingGuided}
                    onChange={setUpsamplingRadius}
                  />
                  <SliderRow
                    label={t('advanced.params.upsamplingEps')}
                    displayValue={Math.pow(10, upsamplingEpsLog).toExponential(1)}
                    value={upsamplingEpsLog}
                    min={-4}
                    max={-1.3}
                    step={0.1}
                    disabled={!upsamplingGuided}
                    onChange={setUpsamplingEpsLog}
                  />
                </div>
                {processorConfig &&
                  (processorConfig.type === ProcessorType.BLUR ||
                    processorConfig.type === ProcessorType.VIRTUAL) && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onPress={applyAdvancedSettings}
                      isDisabled={processorOptions.isDisabled}
                      style={{ marginTop: '0.75rem' }}
                    >
                      {t('advanced.apply')}
                    </Button>
                  )}
              </div>
            </div>
          </div>
        ) : (
          <Information>
            <Text variant="sm">{t('notAvailable')}</Text>
          </Information>
        )}
      </div>
      <Dialog
        isOpen={personalBackgroundHasError}
        type="alert"
        title={t(`virtual.personal.errors.${personalBackgroundError}.title`)}
        aria-label={t(
          `virtual.personal.errors.${personalBackgroundError}.title`
        )}
        onClose={() => setPersonalBackgroundHasError(false)}
        onOpenChange={() => setPersonalBackgroundHasError(false)}
      >
        <P>
          {t(
            `virtual.personal.errors.${personalBackgroundError}.description`,
            filePickerErrorContext
          )}
        </P>
        <HStack justifyContent="end" direction="row">
          <Button
            variant="text"
            size="sm"
            onPress={() => setPersonalBackgroundHasError(false)}
          >
            {t('virtual.personal.errors.close')}
          </Button>
        </HStack>
      </Dialog>
    </div>
  )
}
