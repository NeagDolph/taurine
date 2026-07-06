import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { AmbientList } from './components/AmbientList'
import { PinnedStack } from './components/PinnedStack'
import { SessionControls, type ListeningState } from './components/SessionControls'
import { StatusBar } from './components/StatusBar'
import { PcmAudioCapture } from './lib/audio'
import {
  SidecarClient,
  type AmbientItem,
  type AsrState,
  type Card,
  type IndexStatus,
  type RetrievalEvent,
  type SidecarConnectionState,
  type TranscriptEvent
} from './lib/sidecar'
import { FeedView, type FeedTranscript } from './views/FeedView'
import { MapView } from './views/MapView'

function App(): React.JSX.Element {
  const clientRef = useRef<SidecarClient | null>(null)
  const audioRef = useRef<PcmAudioCapture | null>(null)
  const transcriptIdRef = useRef(0)

  const [activeView, setActiveView] = useState<'feed' | 'map'>('feed')
  const [connection, setConnection] = useState<SidecarConnectionState>('disconnected')
  const [retryInMs, setRetryInMs] = useState<number | undefined>()
  const [asr, setAsr] = useState<AsrState>('disconnected')
  const [index, setIndex] = useState<IndexStatus>({ notes: 0, last_sync: null })
  const [statusDetail, setStatusDetail] = useState('')
  const [listeningState, setListeningState] = useState<ListeningState>('idle')
  const [micError, setMicError] = useState('')
  const [syncing, setSyncing] = useState(false)
  const [syncError, setSyncError] = useState('')
  const [chunks, setChunks] = useState<FeedTranscript[]>([])
  const [interimsByChannel, setInterimsByChannel] = useState<Record<number, TranscriptEvent>>({})
  const [ambientItems, setAmbientItems] = useState<AmbientItem[]>([])
  const [pinnedCards, setPinnedCards] = useState<Card[]>([])
  const [pinnedNoteIds, setPinnedNoteIds] = useState<string[]>([])
  const [lastRetrieval, setLastRetrieval] = useState<RetrievalEvent | null>(null)

  useEffect(() => {
    const client = new SidecarClient()
    clientRef.current = client

    const unsubscribe = [
      client.onConnection((event) => {
        setConnection(event.state)
        setRetryInMs(event.retryInMs)
      }),
      client.onTranscript((event) => {
        if (event.final) {
          transcriptIdRef.current += 1
          setChunks((current) =>
            [
              ...current,
              {
                ...event,
                id: `${event.ts}-${transcriptIdRef.current}`
              }
            ].slice(-200)
          )
          setInterimsByChannel((current) => {
            const next = { ...current }
            delete next[event.channel]
            return next
          })
          return
        }

        setInterimsByChannel((current) => ({
          ...current,
          [event.channel]: event
        }))
      }),
      client.onAmbient((event) => {
        setAmbientItems(event.items)
      }),
      client.onPin((event) => {
        setPinnedNoteIds((current) => (current.includes(event.card.note_id) ? current : [...current, event.card.note_id]))
        setPinnedCards((current) => {
          const withoutDuplicate = current.filter(
            (card) => card.card_id !== event.card.card_id && card.note_id !== event.card.note_id
          )
          return [event.card, ...withoutDuplicate].slice(0, 3)
        })
      }),
      client.onUnpin((event) => {
        setPinnedCards((current) => current.filter((card) => card.card_id !== event.card_id))
      }),
      client.onRetrieval((event) => {
        setLastRetrieval(event)
      }),
      client.onStatus((event) => {
        setAsr(event.asr)
        setIndex(event.index)
        setStatusDetail(event.detail)
      })
    ]

    client.connect()

    return () => {
      audioRef.current?.stop()
      audioRef.current = null
      client.stop()
      unsubscribe.forEach((remove) => remove())
      client.disconnect()
      clientRef.current = null
    }
  }, [])

  const interims = useMemo(() => Object.values(interimsByChannel), [interimsByChannel])
  const latestFinalText = chunks.length > 0 ? chunks[chunks.length - 1].text : ''

  const startListening = useCallback(async () => {
    const client = clientRef.current
    if (!client || listeningState === 'starting' || listeningState === 'listening') {
      return
    }

    setMicError('')
    setListeningState('starting')
    client.start([{ id: 0, label: 'mic' }])

    const capture = new PcmAudioCapture((frame) => {
      clientRef.current?.sendAudioFrame(frame)
    })
    audioRef.current = capture

    try {
      await capture.start()
      setListeningState('listening')
    } catch (error) {
      capture.stop()
      audioRef.current = null
      client.stop()
      setMicError(error instanceof Error ? error.message : 'Microphone capture failed')
      setListeningState('error')
    }
  }, [listeningState])

  const stopListening = useCallback(() => {
    setListeningState('stopping')
    audioRef.current?.stop()
    audioRef.current = null
    clientRef.current?.stop()
    setListeningState('idle')
  }, [])

  const dismissCard = useCallback((cardId: string) => {
    clientRef.current?.dismiss(cardId)
    setPinnedCards((current) => current.filter((card) => card.card_id !== cardId))
  }, [])

  const openNote = useCallback((noteId: string) => {
    clientRef.current?.openNote(noteId)
  }, [])

  const syncIndex = useCallback(async () => {
    const client = clientRef.current
    if (!client || syncing) {
      return
    }

    setSyncing(true)
    setSyncError('')
    try {
      await client.syncIndex()
    } catch (error) {
      setSyncError(error instanceof Error ? error.message : 'Index sync failed')
    } finally {
      setSyncing(false)
    }
  }, [syncing])

  const showDisconnectedBanner = connection !== 'connected'

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <h1>Taurine</h1>
          <span>Live meeting copilot</span>
        </div>
        <SessionControls state={listeningState} error={micError} onStart={startListening} onStop={stopListening} />
      </header>

      {showDisconnectedBanner ? (
        <div className="disconnect-banner" role="status">
          Sidecar {connection}
          {retryInMs ? `, retrying in ${Math.ceil(retryInMs / 1000)}s` : ''}
        </div>
      ) : null}

      <PinnedStack cards={pinnedCards} onOpenNote={openNote} onDismiss={dismissCard} />

      <div className="workspace">
        <main className="main-pane">
          <div className="view-switch" role="tablist" aria-label="Main view">
            <button
              className={activeView === 'feed' ? 'active' : ''}
              type="button"
              role="tab"
              aria-selected={activeView === 'feed'}
              onClick={() => setActiveView('feed')}
            >
              Feed
            </button>
            <button
              className={activeView === 'map' ? 'active' : ''}
              type="button"
              role="tab"
              aria-selected={activeView === 'map'}
              onClick={() => setActiveView('map')}
            >
              Map
            </button>
          </div>

          <div style={viewStackStyle}>
            <div aria-hidden={activeView !== 'feed'} style={viewLayerStyle(activeView === 'feed')}>
              <FeedView chunks={chunks} interims={interims} />
            </div>
            <div aria-hidden={activeView !== 'map'} style={viewLayerStyle(activeView === 'map')}>
              <MapView
                latestFinalText={latestFinalText}
                lastSync={index.last_sync}
                pinnedNoteIds={pinnedNoteIds}
                retrieval={lastRetrieval}
                visible={activeView === 'map'}
              />
            </div>
          </div>
        </main>

        <AmbientList items={ambientItems} />
      </div>

      <StatusBar
        asr={asr}
        connection={connection}
        detail={statusDetail}
        index={index}
        isSyncing={syncing}
        syncError={syncError}
        onSync={syncIndex}
      />
    </div>
  )
}

const viewStackStyle: CSSProperties = {
  height: '100%',
  minHeight: 0,
  overflow: 'hidden',
  position: 'relative'
}

function viewLayerStyle(active: boolean): CSSProperties {
  return {
    display: 'grid',
    gridTemplateRows: 'minmax(0, 1fr)',
    inset: 0,
    minHeight: 0,
    opacity: active ? 1 : 0,
    pointerEvents: active ? 'auto' : 'none',
    position: 'absolute',
    transition: 'opacity 120ms ease',
    visibility: active ? 'visible' : 'hidden'
  }
}

export default App
