import { proxy, useSnapshot } from 'valtio'

export type MattingErrorCode =
  | 'MEDIAPIPE_INIT_FAILED'
  | 'MEDIAPIPE_GPU_FALLBACK_TO_CPU'
  | 'WEBGL2_INIT_FAILED'
  | 'WEBGPU_FALLBACK'
  | 'CAPTURESTREAM_UNSUPPORTED'
  | 'SEGMENTER_TIMEOUT_PASSTHROUGH'
  | 'SEGMENTER_PRODUCING_DEGENERATE_MASK'
  | 'GUIDED_FILTER_INIT_FAILED'
  | 'POSTPROCESS_SHADER_COMPILE_FAILED'
  | 'VIRTUAL_BG_LOAD_FAILED'
  | 'RVM_INIT_FAILED'
  | 'RVM_INFERENCE_FAILED'

export type MattingErrorLevel = 'info' | 'warn' | 'error'

export interface MattingError {
  code: MattingErrorCode
  level: MattingErrorLevel
  detail?: string
}

interface MattingErrorState {
  errors: MattingError[]
}

export const mattingErrorStore = proxy<MattingErrorState>({ errors: [] })

export function pushMattingError(e: MattingError) {
  const i = mattingErrorStore.errors.findIndex((x) => x.code === e.code)
  if (i >= 0) {
    mattingErrorStore.errors[i] = e
  } else {
    mattingErrorStore.errors.push(e)
  }
  // Surface every problem in the JS console too — easier to debug from Safari.
  const fn =
    e.level === 'error'
      ? console.error
      : e.level === 'warn'
        ? console.warn
        : console.info
  fn(`[matting:${e.code}]`, e.detail ?? '')
}

export function dismissMattingError(code: MattingErrorCode) {
  const i = mattingErrorStore.errors.findIndex((x) => x.code === code)
  if (i >= 0) mattingErrorStore.errors.splice(i, 1)
}

export function clearMattingErrors() {
  mattingErrorStore.errors.length = 0
}

export function useMattingErrors(): readonly MattingError[] {
  const snap = useSnapshot(mattingErrorStore)
  return snap.errors as readonly MattingError[]
}
