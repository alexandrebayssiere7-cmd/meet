import { proxy, useSnapshot } from 'valtio'

/**
 * Enumeration of all error codes that can arise during background matting.
 * Each code corresponds to a distinct failure mode in the pipeline.
 */
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

/**
 * Severity level of a matting error, mapped to the corresponding console method.
 * - `info`: informational/non-critical degradation (e.g. GPU→CPU fallback)
 * - `warn`: partial feature loss (e.g. guided filter unavailable)
 * - `error`: hard failure that prevents the effect from rendering
 */
export type MattingErrorLevel = 'info' | 'warn' | 'error'

/**
 * Represents a single error event in the matting pipeline.
 */
export interface MattingError {
  /** Unique error code identifying the failure mode. */
  code: MattingErrorCode
  /** Severity of the error. */
  level: MattingErrorLevel
  /** Optional human-readable message with context (e.g. exception message). */
  detail?: string
}

/**
 * Internal shape of the reactive Valtio store.
 */
interface MattingErrorState {
  /** Ordered list of active errors. At most one entry per code (upsert). */
  errors: MattingError[]
}

/**
 * Reactive Valtio proxy holding the current set of active matting errors.
 * Subscribe via `useMattingErrors()` inside React components.
 */
export const mattingErrorStore = proxy<MattingErrorState>({ errors: [] })

/**
 * Push or update a matting error in the store.
 * If an error with the same code already exists it is replaced (upsert),
 * preventing duplicates. The error is also surfaced in the browser console
 * at the appropriate severity level for easier debugging.
 *
 * @param e - The error to record.
 */
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

/**
 * Remove a specific error from the store by code.
 * No-op if no error with that code is currently active.
 *
 * @param code - The error code to remove.
 */
export function dismissMattingError(code: MattingErrorCode) {
  const i = mattingErrorStore.errors.findIndex((x) => x.code === code)
  if (i >= 0) mattingErrorStore.errors.splice(i, 1)
}

/**
 * Remove all active matting errors from the store.
 * Typically called when the processor is destroyed or restarted.
 */
export function clearMattingErrors() {
  mattingErrorStore.errors.length = 0
}

/**
 * React hook that returns the current list of active matting errors as a
 * read-only reactive snapshot. Re-renders the component on every change.
 *
 * @returns A read-only array of active `MattingError` objects.
 */
export function useMattingErrors(): readonly MattingError[] {
  const snap = useSnapshot(mattingErrorStore)
  return snap.errors as readonly MattingError[]
}
