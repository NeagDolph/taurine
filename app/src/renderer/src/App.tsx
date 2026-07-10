import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { ActionableRail } from './components/ActionableRail'
import { AmbientList } from './components/AmbientList'
import { SessionControls, type ListeningState } from './components/SessionControls'
import { StatusBar } from './components/StatusBar'
import { SummaryPanel } from './components/SummaryPanel'
import { TriggerInspector } from './components/TriggerInspector'
import { PcmAudioCapture } from './lib/audio'
import {
  SidecarClient,
  fetchStartupStatus,
  type Actionable,
  type AsrState,
  type ConnectorInfo,
  type IndexStatus,
  type ProfileState,
  type RetrievalEvent,
  type LiveGroundingEvent,
  type SidecarConnectionState,
  type StartupStatus,
  type SummaryEvent,
  type TriggerEvent,
  type TranscriptEvent
} from './lib/sidecar'
import { UpdatesClient } from './lib/updates'
import { ChatView } from './views/ChatView'
import { ConnectorsView } from './views/ConnectorsView'
import { DataView } from './views/DataView'
import { FeedView, type FeedTranscript } from './views/FeedView'

type MainTab = 'live' | 'chat' | 'data' | 'connectors'

const MAIN_TABS: Array<{ id: MainTab; label: string }> = [
  { id: 'live', label: 'Live' },
  { id: 'chat', label: 'Chat' },
  { id: 'data', label: 'Data' },
  { id: 'connectors', label: 'Connectors' }
]

function App(): React.JSX.Element {
  const clientRef = useRef<SidecarClient | null>(null)
  const audioRef = useRef<PcmAudioCapture | null>(null)
  const transcriptIdRef = useRef(0)

  const [activeTab, setActiveTab] = useState<MainTab>('live')
  const [connectors, setConnectors] = useState<ConnectorInfo[] | null>(null)
  const [graphVersion, setGraphVersion] = useState(0)
  const [profile, setProfile] = useState<ProfileState | undefined>(undefined)
  const [connection, setConnection] = useState<SidecarConnectionState>('disconnected')
  const [retryInMs, setRetryInMs] = useState<number | undefined>()
  const [startup, setStartup] = useState<StartupStatus | null>(null)
  const [asr, setAsr] = useState<AsrState>('disconnected')
  const [index, setIndex] = useState<IndexStatus>({ notes: 0, last_sync: null })
  const [statusDetail, setStatusDetail] = useState('')
  const [listeningState, setListeningState] = useState<ListeningState>('idle')
  const [micError, setMicError] = useState('')
  const [syncing, setSyncing] = useState(false)
  const [syncError, setSyncError] = useState('')
  const [chunks, setChunks] = useState<FeedTranscript[]>([])
  const [interimsByChannel, setInterimsByChannel] = useState<Record<number, TranscriptEvent>>({})
  const [triggersBySeq, setTriggersBySeq] = useState<Map<number, TriggerEvent[]>>(new Map())
  const [inspectGrounding, setInspectGrounding] = useState(false)
  const [liveGrounding, setLiveGrounding] = useState<LiveGroundingEvent | null>(null)
  const [summary, setSummary] = useState<SummaryEvent | null>(null)
  const [actionables, setActionables] = useState<Actionable[]>([])
  const [groundedNoteIds, setGroundedNoteIds] = useState<string[]>([])
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
      client.onGrounding((event) => {
        setLiveGrounding(event)
        setGroundedNoteIds((current) => {
          const next = new Set(current)
          event.notes.forEach((item) => next.add(item.note_id))
          return next.size === current.length ? current : [...next]
        })
      }),
      client.onSummary((event) => {
        setSummary(event)
        setActionables(event.actionables)
        setGroundedNoteIds((current) => {
          const next = new Set(current)
          event.evidence?.notes.forEach((item) => next.add(item.note_id))
          return next.size === current.length ? current : [...next]
        })
      }),
      client.onActionable((event) => {
        setActionables((current) => {
          const withoutItem = current.filter((item) => item.id !== event.item.id)
          if (event.op === 'dismiss') {
            return withoutItem
          }
          return [...withoutItem, event.item]
        })
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

    // all non-live state (connectors, sync progress, re-enrich, graph
    // version, profile) arrives over the /updates websocket — no polling
    const updates = new UpdatesClient()
    const unsubscribeUpdates = updates.onUpdate((event) => {
      switch (event.type) {
        case 'connectors':
          setConnectors(event.connectors)
          break
        case 'sync':
          setIndex((current) => ({ ...current, sync: event.status }))
          break
        case 'graph':
          setGraphVersion(event.version)
          break
        case 'profile':
          setProfile({ card: event.card, generation: event.generation })
          break
      }
    })
    updates.connect()

    return () => {
      audioRef.current?.stop()
      audioRef.current = null
      client.stop()
      unsubscribe.forEach((remove) => remove())
      client.disconnect()
      clientRef.current = null
      unsubscribeUpdates()
      updates.disconnect()
    }
  }, [])

  // While the sidecar websocket is down, poll /startup so the banner can show
  // build progress (model loading, source extraction, knowledge index build)
  // instead of a bare reconnect countdown. A fetch failure means the server
  // process has not bound the port yet.
  useEffect(() => {
    if (connection === 'connected') {
      return
    }

    let cancelled = false
    const poll = async (): Promise<void> => {
      try {
        const status = await fetchStartupStatus()
        if (!cancelled) {
          setStartup(status)
        }
      } catch {
        if (!cancelled) {
          setStartup(null)
        }
      }
    }

    void poll()
    const timer = window.setInterval(() => void poll(), 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [connection])

  useEffect(() => {
    setTriggersBySeq((current) => pruneTriggersByChunks(current, chunks))
  }, [chunks])

  const interims = useMemo(() => Object.values(interimsByChannel), [interimsByChannel])
  const latestFinalText = chunks.length > 0 ? chunks[chunks.length - 1].text : ''
  const isSyncing = syncing || Boolean(index.sync?.running)

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

  const actionableStatus = useCallback((id: string, status: 'confirmed' | 'dismissed') => {
    clientRef.current?.actionableStatus(id, status)
    setActionables((current) =>
      status === 'dismissed'
        ? current.filter((item) => item.id !== id)
        : current.map((item) => (item.id === id ? { ...item, status } : item))
    )
  }, [])

  const openNote = useCallback((noteId: string) => {
    clientRef.current?.openNote(noteId)
  }, [])

  // The single global sync action: the POST returns 202 immediately and
  // progress streams back over /updates and status events.
  const syncIndex = useCallback(async () => {
    const client = clientRef.current
    if (!client || syncing || index.sync?.running) {
      return
    }

    setSyncing(true)
    setSyncError('')
    try {
      const status = await client.syncIndex()
      setIndex((current) => ({ ...current, sync: status }))
    } catch (error) {
      setSyncError(error instanceof Error ? error.message : 'Sync failed')
    } finally {
      setSyncing(false)
    }
  }, [index.sync?.running, syncing])

  const cancelSync = useCallback(async () => {
    const client = clientRef.current
    if (!client || !index.sync?.running) {
      return
    }
    try {
      const status = await client.cancelSync()
      if (status) {
        setIndex((current) => ({ ...current, sync: status }))
      }
    } catch (error) {
      setSyncError(error instanceof Error ? error.message : 'Cancel failed')
    }
  }, [index.sync?.running])

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
          {startupBannerText(connection, retryInMs, startup)}
        </div>
      ) : null}

      <div className="workspace-stack" style={viewStackStyle}>
        <div aria-hidden={activeTab !== 'live'} style={viewLayerStyle(activeTab === 'live')}>
          <div className="page live-page">
            <div className="summary-row">
              <SummaryPanel summary={summary} onOpenNote={openNote} />
              <ActionableRail
                actionables={actionables}
                evidence={summary?.evidence ?? null}
                onStatus={actionableStatus}
                onOpenNote={openNote}
              />
            </div>
            <div className="workspace">
              <main className="main-pane live-main">
                <FeedView
                  chunks={chunks}
                  interims={interims}
                  triggersBySeq={triggersBySeq}
                  onInspectTrigger={() => undefined}
                />
              </main>

              <AmbientList
                notes={liveGrounding?.display_notes ?? []}
                passages={liveGrounding?.display_passages ?? []}
                noNewEvidence={liveGrounding?.no_new_evidence}
                onInspect={liveGrounding ? () => setInspectGrounding(true) : undefined}
                onOpenNote={openNote}
              />
            </div>
          </div>
        </div>

        <div aria-hidden={activeTab !== 'chat'} style={viewLayerStyle(activeTab === 'chat')}>
          <ChatView visible={activeTab === 'chat'} onOpenNote={openNote} />
        </div>

        <div aria-hidden={activeTab !== 'data'} style={viewLayerStyle(activeTab === 'data')}>
          <DataView
            latestFinalText={latestFinalText}
            groundedNoteIds={groundedNoteIds}
            profile={profile}
            onProfileChange={setProfile}
            refreshKey={graphVersion}
            retrieval={lastRetrieval}
            visible={activeTab === 'data'}
          />
        </div>

        <div
          aria-hidden={activeTab !== 'connectors'}
          style={viewLayerStyle(activeTab === 'connectors')}
        >
          <ConnectorsView
            connectors={connectors}
            onSync={syncIndex}
            onCancelSync={cancelSync}
            syncing={isSyncing}
            syncStatus={index.sync ?? null}
            syncError={syncError}
            visible={activeTab === 'connectors'}
          />
        </div>
      </div>

      <TriggerInspector
        grounding={inspectGrounding ? liveGrounding : null}
        onClose={() => setInspectGrounding(false)}
        onOpenNote={openNote}
      />

      <StatusBar
        asr={asr}
        connection={connection}
        detail={statusDetail}
        index={index}
        isSyncing={isSyncing}
        syncError={syncError}
        onSync={syncIndex}
      />
    </div>
  )
}

const STARTUP_PHASE_LABELS: Record<string, string> = {
  starting: 'starting',
  loading_models: 'loading models',
  opening_index: 'opening knowledge index',
  extracting_sources: 'reading sources',
  building_index: 'building knowledge index',
  finalizing: 'finalizing'
}

function startupBannerText(
  connection: SidecarConnectionState,
  retryInMs: number | undefined,
  startup: StartupStatus | null
): string {
  if (startup && !startup.ready) {
    if (startup.error) {
      return `Core startup failed: ${startup.error}`
    }
    const label = STARTUP_PHASE_LABELS[startup.phase] ?? startup.phase
    const detail = startup.detail ? ` (${startup.detail})` : ''
    const elapsed = startup.elapsed_s >= 1 ? ` · ${Math.round(startup.elapsed_s)}s` : ''
    return `Core starting — ${label}${detail}${elapsed}`
  }

  const retry = retryInMs ? `, retrying in ${Math.ceil(retryInMs / 1000)}s` : ''
  return `Sidecar ${connection}${retry}`
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
    event.seq > 0
      ? [...current.filter((item) => item.seq !== event.seq), chunk]
      : [...current, chunk]
  return next.sort(compareTranscriptChunks).slice(-MAX_FINAL_CHUNKS)
}

function compareTranscriptChunks(left: FeedTranscript, right: FeedTranscript): number {
  if (left.seq > 0 && right.seq > 0) {
    return left.seq - right.seq
  }

  return left.ts - right.ts
}

function registerTrigger(
  current: Map<number, TriggerEvent[]>,
  trigger: TriggerEvent
): Map<number, TriggerEvent[]> {
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
