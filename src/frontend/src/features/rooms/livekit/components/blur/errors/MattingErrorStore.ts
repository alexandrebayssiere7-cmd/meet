import { proxy, useSnapshot } from 'valtio'

export type MattingErrorCode =
  | 'MEDIAPIPE_INIT_FAILED'
  | 'WEBGL2_INIT_FAILED'
  | 'CAPTURESTREAM_UNSUPPORTED'
  | 'POSTPROCESS_SHADER_COMPILE_FAILED'
  | 'VIRTUAL_BG_LOAD_FAILED'
  | 'CANVAS2D_FALLBACK'

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

export function useMattingErrors(): readonly MattingError[] {
  const snap = useSnapshot(mattingErrorStore)
  return snap.errors as readonly MattingError[]
}
