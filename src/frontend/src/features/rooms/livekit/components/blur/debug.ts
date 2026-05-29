/**
 * Debug logging for the matting pipeline. Silent by default in production.
 * Opted in via the `?mattingDebug=1` query string 
 */
let cached: boolean | null = null

export const mattingDebugEnabled = (): boolean => {
  if (cached !== null) return cached
  if (typeof window === 'undefined') {
    cached = false
    return cached
  }
  try {
    const params = new URL(window.location.href).searchParams
    const on = (key: string) => {
      const v = params.get(key)
      return v === '1' || v === 'true'
    }
    cached = on('mattingDebug')
  } catch {
    cached = false
  }
  return cached
}

export const debugLog = (...args: unknown[]): void => {
  if (mattingDebugEnabled()) console.log(...args)
}

export const debugWarn = (...args: unknown[]): void => {
  if (mattingDebugEnabled()) console.warn(...args)
}