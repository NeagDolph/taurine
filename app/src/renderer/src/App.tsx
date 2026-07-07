import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { AmbientList } from './components/AmbientList'
import { PinnedStack } from './components/PinnedStack'
import { SessionControls, type ListeningState } from './components/SessionControls'
import { StatusBar } from './components/StatusBar'
import { TriggerInspector } from './components/TriggerInspector'
import { PcmAudioCapture } from './lib/audio'
import {
  SidecarClient,
  type AmbientItem,
  type AsrState,
  type Card,
  type IndexStatus,
  type RetrievalEvent,
  type SidecarConnectionState,
  type TriggerEvent,
  type TranscriptEvent
} from './lib/sidecar'
import { ConnectorsView } from './views/ConnectorsView'
import { DataView } from './views/DataView'
import { FeedView, type FeedTranscript } from './views/FeedView'

type MainTab = 'live' | 'data' | 'connectors'

const MAIN_TABS: Array<{ id: MainTab; label: string }> = [
  { id: 'live', label: 'Live' },
  { id: 'data', label: 'Data' },
  { id: 'connectors', label: 'Connectors' }
]

function App(): React.JSX.Element {
  const clientRef = useRef<SidecarClient | null>(null)
  const audioRef = useRef<PcmAudioCapture | null>(null)
  const transcriptIdRef = useRef(0)

  const [activeTab, setActiveTab] = useState<MainTab>('live')
  const [mapRefreshKey, setMapRefreshKey] = useState(0)
  const [connection, setConnection] = useState<SidecarConnectionState>('disconnected')
  const [retryInMs, setRetryInMs] = useState<number | undefined>()
  const [asr, setAsr] = useState<AsrState>('disconnected')
  const [index, setIndex] = useState<IndexStatus>({ notes: 0, last_sync: null })
  const [statusDetail, setStatusDetail] = useState('')
  const [listeningState, setListeningState] = useState<ListeningState>('idle')
  const [micError, setMicError] = useState('')
  const [syncing, setSyncing] = useState(false)
  const [syncError, setSyncError] = useState('')
  const [reenriching, setReenriching] = useState(false)
  const [reenrichError, setReenrichError] = useState('')
  const [chunks, setChunks] = useState<FeedTranscript[]>([])
  const [interimsByChannel, setInterimsByChannel] = useState<Record<number, TranscriptEvent>>({})
  const [triggersBySeq, setTriggersBySeq] = useState<Map<number, TriggerEvent[]>>(new Map())
  const [inspectedTrigger, setInspectedTrigger] = useState<TriggerEvent | null>(null)
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
          setChunks((current) => appendFinalChunk(current, event, transcriptIdRef.current))
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
        setPinnedNoteIds((current) =>
          current.includes(event.card.note_id) ? current : [...current, event.card.note_id]
        )
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
      client.onTrigger((event) => {
        setTriggersBySeq((current) => registerTrigger(current, event))
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

  useEffect(() => {
    setTriggersBySeq((current) => pruneTriggersByChunks(current, chunks))
  }, [chunks])

  const interims = useMemo(() => Object.values(interimsByChannel), [interimsByChannel])
  const latestFinalText = chunks.length > 0 ? chunks[chunks.length - 1].text : ''
  const isReenriching = reenriching || Boolean(index.reenrich?.running)

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
    if (!client || syncing || index.reenrich?.running) {
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
  }, [index.reenrich?.running, syncing])

  const reenrichIndex = useCallback(async () => {
    const client = clientRef.current
    if (!client || reenriching || syncing || index.reenrich?.running) {
      return
    }

    setReenriching(true)
    setReenrichError('')
    try {
      const status = await client.reenrichIndex()
      setIndex((current) => ({
        ...current,
        reenrich: status
      }))
    } catch (error) {
      setReenrichError(error instanceof Error ? error.message : 'Re-enrich failed')
    } finally {
      setReenriching(false)
    }
  }, [index.reenrich?.running, reenriching, syncing])

  const showDisconnectedBanner = connection !== 'connected'

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <h1>Taurine</h1>
          <span>Live meeting copilot</span>
        </div>

        <nav className="app-tabs" role="tablist" aria-label="Main view">
          {MAIN_TABS.map((tab) => (
            <button
              key={tab.id}
              className={activeTab === tab.id ? 'active' : ''}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <SessionControls
          state={listeningState}
          error={micError}
          onStart={startListening}
          onStop={stopListening}
        />
      </header>

      {showDisconnectedBanner ? (
        <div className="disconnect-banner" role="status">
          Sidecar {connection}
          {retryInMs ? `, retrying in ${Math.ceil(retryInMs / 1000)}s` : ''}
        </div>
      ) : null}

      <div className="workspace-stack" style={viewStackStyle}>
        <div aria-hidden={activeTab !== 'live'} style={viewLayerStyle(activeTab === 'live')}>
          <div className="page live-page">
            <PinnedStack cards={pinnedCards} onOpenNote={openNote} onDismiss={dismissCard} />
            <div className="workspace">
              <main className="main-pane live-main">
                <FeedView
                  chunks={chunks}
                  interims={interims}
                  triggersBySeq={triggersBySeq}
                  onInspectTrigger={setInspectedTrigger}
                />
              </main>

              <AmbientList items={ambientItems} />
            </div>
          </div>
        </div>

        <div aria-hidden={activeTab !== 'data'} style={viewLayerStyle(activeTab === 'data')}>
          <DataView
            latestFinalText={latestFinalText}
            lastSync={index.last_sync}
            pinnedNoteIds={pinnedNoteIds}
            refreshKey={mapRefreshKey}
            retrieval={lastRetrieval}
            visible={activeTab === 'data'}
          />
        </div>

        <div
          aria-hidden={activeTab !== 'connectors'}
          style={viewLayerStyle(activeTab === 'connectors')}
        >
          <ConnectorsView
            onDataChanged={() => setMapRefreshKey((current) => current + 1)}
            onReenrich={reenrichIndex}
            reenrichDisabled={syncing || isReenriching}
            reenrichError={reenrichError}
            reenriching={isReenriching}
            visible={activeTab === 'connectors'}
          />
        </div>
      </div>

      <TriggerInspector
        trigger={inspectedTrigger}
        onClose={() => setInspectedTrigger(null)}
        onOpenNote={openNote}
      />

      <StatusBar
        asr={asr}
        connection={connection}
        detail={statusDetail}
        index={index}
        isSyncing={syncing}
        isReenriching={isReenriching}
        syncError={syncError}
        onSync={syncIndex}
      />
    </div>
  )
}

const MAX_FINAL_CHUNKS = 500

function appendFinalChunk(
  current: FeedTranscript[],
  event: TranscriptEvent,
  transcriptId: number
): FeedTranscript[] {
  const chunk = {
    ...event,
    id: event.seq > 0 ? `seq-${event.seq}` : `${event.ts}-${transcriptId}`
  }

  const next =
    event.seq > 0 ? [...current.filter((item) => item.seq !== event.seq), chunk] : [...current, chunk]
  return next.sort(compareTranscriptChunks).slice(-MAX_FINAL_CHUNKS)
}

function compareTranscriptChunks(left: FeedTranscript, right: FeedTranscript): number {
  if (left.seq > 0 && right.seq > 0) {
    return left.seq - right.seq
  }

  return left.ts - right.ts
}

function registerTrigger(current: Map<number, TriggerEvent[]>, trigger: TriggerEvent): Map<number, TriggerEvent[]> {
  const next = new Map(current)

  for (const seq of trigger.seqs) {
    if (seq <= 0) {
      continue
    }

    const existing = next.get(seq) ?? []
    next.set(seq, [...existing.filter((item) => item.trigger_id !== trigger.trigger_id), trigger])
  }

  return next
}

function pruneTriggersByChunks(
  current: Map<number, TriggerEvent[]>,
  chunks: FeedTranscript[]
): Map<number, TriggerEvent[]> {
  const retainedSeqs = new Set(chunks.map((chunk) => chunk.seq).filter((seq) => seq > 0))
  let changed = false
  const next = new Map<number, TriggerEvent[]>()

  for (const [seq, triggers] of current) {
    if (retainedSeqs.has(seq)) {
      next.set(seq, triggers)
    } else {
      changed = true
    }
  }

  return changed ? next : current
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
