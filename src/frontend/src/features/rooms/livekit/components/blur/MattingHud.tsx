import { css } from '@/styled-system/css'
import { SegmentationModel } from '.'
import { useMattingStats } from './stats/MattingStatsStore'

const shortModel = (m: SegmentationModel | null): string => {
  switch (m) {
    case SegmentationModel.LANDSCAPE: return 'LS'
    case SegmentationModel.MULTICLASS: return 'MC'
    case SegmentationModel.RVM: return 'RVM'
    case SegmentationModel.AUTO: return 'AUTO'
    default: return '—'
  }
}

export interface MattingHudProps {
  /**
   * When true the HUD is rendered regardless of `?mattingHud=1`. Useful in
   * dev/integration scenarios. Defaults to false → HUD is opt-in via URL.
   */
  alwaysVisible?: boolean
}

const isHudEnabled = (): boolean => {
  if (typeof window === 'undefined') return false
  try {
    const url = new URL(window.location.href)
    const v = url.searchParams.get('mattingHud')
    return v === '1' || v === 'true'
  } catch {
    return false
  }
}

/**
 * Tiny overlay showing live matting diagnostics on the local video tile.
 * Renders nothing unless a matting processor is active AND the HUD has been
 * opted in via the `?mattingHud=1` query string (or `alwaysVisible` prop).
 */
export const MattingHud = ({ alwaysVisible = false }: MattingHudProps) => {
  const stats = useMattingStats()
  if (!stats.active) return null
  if (!alwaysVisible && !isHudEnabled()) return null

  const isAuto = stats.configuredModel === SegmentationModel.AUTO
  const modelStr = isAuto
    ? `${shortModel(stats.currentModel)} (AUTO)`
    : shortModel(stats.currentModel ?? stats.configuredModel)

  return (
    <div
      className={css({
        position: 'absolute',
        top: '8px',
        right: '8px',
        zIndex: 10,
        pointerEvents: 'none',
        background: 'rgba(0, 0, 0, 0.55)',
        color: '#f8fafc',
        fontFamily: 'monospace',
        fontSize: '11px',
        lineHeight: '1.35',
        padding: '4px 7px',
        borderRadius: '4px',
        whiteSpace: 'nowrap',
        userSelect: 'none',
      })}
    >
      <div>M:{modelStr}</div>
      <div>lat {stats.captureToDisplayLatencyMs.toFixed(0)}ms</div>
      <div>Δ {stats.maskFrameGapMs.toFixed(0)}ms</div>
      <div>inf {stats.segmenterInferenceMs.toFixed(0)}ms</div>
      <div>
        cam {stats.cameraFps.toFixed(0)} · r {stats.renderFps.toFixed(0)} · s {stats.segmenterFps.toFixed(0)} fps
      </div>
      {stats.effectiveLatencyMode !== null && (
        <div>
          mode {stats.effectiveLatencyMode}
          {stats.predictionActive ? ' · pred' : ''}
        </div>
      )}
      {stats.effectiveLatencyMode !== null && (
        <div>motion {stats.motionScoreUvPerSec.toFixed(2)}</div>
      )}
    </div>
  )
}
