import * as React from 'react'
import type { TrackReferenceOrPlaceholder } from '@livekit/components-core'
import {
  TrackLoop,
  usePagination,
  UseParticipantsOptions,
  useSwipe,
  TrackRefContext,
} from '@livekit/components-react'
import { mergeProps } from '@/utils/mergeProps'
import { PaginationIndicator } from './PaginationIndicator'
import { useGridLayout } from '../hooks/useGridLayout'
import { PaginationControl } from './PaginationControl'

/** @public */
export interface GridLayoutProps
  extends
    React.HTMLAttributes<HTMLDivElement>,
    Pick<UseParticipantsOptions, 'updateOnlyOn'> {
  children: React.ReactNode
  tracks: TrackReferenceOrPlaceholder[]
}

// Robust fallback layout when pagination/stability hooks of LiveKit crash
function FallbackGridLayout({ tracks, children, ...props }: GridLayoutProps) {
  const gridEl = React.useRef<HTMLDivElement>(null)
  const elementProps = React.useMemo(
    () => mergeProps(props, { className: 'lk-grid-layout' }),
    [props]
  )
  const { layout } = useGridLayout(gridEl, tracks.length)

  // Robust mapping function that doesn't crash on undefined or weird references
  const getTrackKey = (track: TrackReferenceOrPlaceholder) => {
    const participantId = track.participant?.identity || 'unknown'
    const source = track.source || 'camera'
    const trackSid = track.publication?.trackSid || 'placeholder'
    return `${participantId}_${source}_${trackSid}`
  }

  return (
    <div ref={gridEl} {...elementProps}>
      {tracks.map((track) => (
        <TrackRefContext.Provider value={track} key={getTrackKey(track)}>
          {children}
        </TrackRefContext.Provider>
      ))}
    </div>
  )
}

class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error: any, errorInfo: any) {
    console.warn('[GridLayout] usePagination crashed, falling back to robust direct rendering:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback
    }
    return this.props.children
  }
}

function BaseGridLayout({ tracks, ...props }: GridLayoutProps) {
  const gridEl = React.useRef<HTMLDivElement>(null)

  const elementProps = React.useMemo(
    () => mergeProps(props, { className: 'lk-grid-layout' }),
    [props]
  )
  const { layout } = useGridLayout(gridEl, tracks.length)
  const pagination = usePagination(layout.maxTiles, tracks)

  useSwipe(gridEl, {
    onLeftSwipe: pagination.nextPage,
    onRightSwipe: pagination.prevPage,
  })

  return (
    <div
      ref={gridEl}
      data-lk-pagination={pagination.totalPageCount > 1}
      {...elementProps}
    >
      <TrackLoop tracks={pagination.tracks}>{props.children}</TrackLoop>
      {tracks.length > layout.maxTiles && (
        <>
          <PaginationIndicator
            totalPageCount={pagination.totalPageCount}
            currentPage={pagination.currentPage}
          />
          <PaginationControl pagesContainer={gridEl} {...pagination} />
        </>
      )}
    </div>
  )
}

export function GridLayout(props: GridLayoutProps) {
  return (
    <ErrorBoundary fallback={<FallbackGridLayout {...props} />}>
      <BaseGridLayout {...props} />
    </ErrorBoundary>
  )
}
