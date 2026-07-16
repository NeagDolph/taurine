import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { SessionControls, type ListeningState } from './components/SessionControls'
import { StatusBar } from './components/StatusBar'
import { PcmAudioCapture } from './lib/audio'
import { SystemAudioCapture } from './lib/sysaudio'
import {
  createMeeting as createMeetingRecord,
  deleteMeeting as deleteMeetingRecord,
  listMeetings,
  patchMeeting,
  type MeetingListRow
} from './lib/meetings'
import { DebouncedNotepadPush } from './lib/notepad-sync'
import {
  SidecarClient,
  fetchStartupStatus,
  type Actionable,
  type AsrState,
  type ConnectorInfo,
  type FinalSummaryEvent,
  type FinalSummaryStatusEvent,
  type IndexStatus,
  type ProfileState,
  type RetrievalEvent,
  type LiveGroundingEvent,
  type SidecarConnectionState,
  type StartupStatus,
  type SummaryEvent,
  type TranscriptEvent
} from './lib/sidecar'
import { UpdatesClient } from './lib/updates'
import { ChatView, type ChatSeedRequest } from './views/ChatView'
import { ConnectorsView } from './views/ConnectorsView'
import { DataView } from './views/DataView'
import type { FeedTranscript } from './views/FeedView'
import { MeetingsView, type MeetingsCoordinator } from './views/meetings/MeetingsView'
import { MapView, type MapFocusRequest } from './views/map/MapView'

type MainTab = 'meetings' | 'chat' | 'map' | 'data' | 'connectors'

const MAIN_TABS: Array<{ id: MainTab; label: string }> = [
  { id: 'meetings', label: 'Meetings' },
  { id: 'chat', label: 'Chat' },
  { id: 'map', label: 'Map' },
  { id: 'data', label: 'Data' },
  { id: 'connectors', label: 'Connectors' }
]


function App(): React.JSX.Element {
  const clientRef = useRef<SidecarClient | null>(null)
  const audioRef = useRef<PcmAudioCapture | null>(null)
  const systemAudioRef = useRef<SystemAudioCapture | null>(null)
  const transcriptIdRef = useRef(0)
  const liveMeetingIdRef = useRef<string | null>(null)
  const notesFlushRef = useRef<(() => Promise<void>) | null>(null)
  const notesSavesRef = useRef<Map<string, Promise<void>>>(new Map())
  const notepadPushRef = useRef<DebouncedNotepadPush | null>(null)

  const [activeTab, setActiveTab] = useState<MainTab>('meetings')
  const [meetingsIndex, setMeetingsIndex] = useState<MeetingListRow[]>([])
  const [selectedMeetingId, setSelectedMeetingId] = useState<string | null>(null)
  const [liveMeetingId, setLiveMeetingId] = useState<string | null>(null)
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
  const [liveGrounding, setLiveGrounding] = useState<LiveGroundingEvent | null>(null)
  const [summary, setSummary] = useState<SummaryEvent | null>(null)
  const [finalSummaryStatuses, setFinalSummaryStatuses] = useState<
    Record<string, FinalSummaryStatusEvent>
  >({})
  const [latestFinalSummary, setLatestFinalSummary] = useState<FinalSummaryEvent | null>(null)
  const [actionables, setActionables] = useState<Actionable[]>([])
  const [groundedNoteIds, setGroundedNoteIds] = useState<string[]>([])
  const [lastRetrieval, setLastRetrieval] = useState<RetrievalEvent | null>(null)
  const [mapFocus, setMapFocus] = useState<MapFocusRequest | null>(null)
  const [chatSeed, setChatSeed] = useState<ChatSeedRequest | null>(null)

  // Call detection lives in the main process now (OS-level always-on-top
  // panel, visible even when this window is hidden). The renderer only
  // reports listening state and reacts to the panel's "Start notes".
  useEffect(() => {
    window.api.session.reportListening(listeningState !== 'idle')
  }, [listeningState])

  useEffect(() => {
    let disposed = false
    const client = new SidecarClient()
    clientRef.current = client
    const notepadPush = new DebouncedNotepadPush((meetingId, textMd) => {
      client.sendNotepad(meetingId, textMd)
    })
    notepadPushRef.current = notepadPush

    const refreshMeetings = async (): Promise<void> => {
      try {
        const meetings = await listMeetings()
        if (!disposed) {
          setMeetingsIndex(meetings)
          setSelectedMeetingId((current) => selectedMeetingForIndex(current, meetings))
        }
      } catch {
        // The updates websocket will supply the list once core startup completes.
      }
    }

    const unsubscribe = [
      client.onConnection((event) => {
        setConnection(event.state)
        setRetryInMs(event.retryInMs)
        if (event.state === 'connected') void refreshMeetings()
      }),
      client.onTranscript((event) => {
        if (!liveMeetingIdRef.current) return
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
        if (!liveMeetingIdRef.current) return
        setLiveGrounding(event)
        setGroundedNoteIds((current) => {
          const next = new Set(current)
          event.notes.forEach((item) => next.add(item.note_id))
          return next.size === current.length ? current : [...next]
        })
      }),
      client.onSummary((event) => {
        if (!liveMeetingIdRef.current) return
        setSummary(event)
        setActionables(event.actionables)
        setGroundedNoteIds((current) => {
          const next = new Set(current)
          event.evidence?.notes.forEach((item) => next.add(item.note_id))
          return next.size === current.length ? current : [...next]
        })
      }),
      client.onFinalSummaryStatus((event) => {
        setFinalSummaryStatuses((current) => ({ ...current, [event.meetingId]: event }))
      }),
      client.onFinalSummary((event) => {
        setLatestFinalSummary(event)
        setMeetingsIndex((current) =>
          current.map((meeting) =>
            meeting.meetingId === event.meetingId
              ? { ...meeting, title: event.title, status: 'done' }
              : meeting
          )
        )
      }),
      client.onActionable((event) => {
        if (!liveMeetingIdRef.current) return
        setActionables((current) => {
          const withoutItem = current.filter((item) => item.id !== event.item.id)
          if (event.op === 'dismiss') {
            return withoutItem
          }
          return [...withoutItem, event.item]
        })
      }),
      client.onRetrieval((event) => {
        if (!liveMeetingIdRef.current) return
        setLastRetrieval(event)
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
        case 'meetings':
          setMeetingsIndex(event.meetings)
          setSelectedMeetingId((current) => selectedMeetingForIndex(current, event.meetings))
          break
      }
    })
    updates.connect()
    void refreshMeetings()

    return () => {
      disposed = true
      systemAudioRef.current?.stop()
      systemAudioRef.current = null
      audioRef.current?.stop()
      audioRef.current = null
      notepadPush.cancel()
      if (notepadPushRef.current === notepadPush) notepadPushRef.current = null
      client.stop()
      unsubscribe.forEach((remove) => remove())
      client.disconnect()
      clientRef.current = null
      liveMeetingIdRef.current = null
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

  const latestFinalText = chunks.length > 0 ? chunks[chunks.length - 1].text : ''
  const isSyncing = syncing || Boolean(index.sync?.running)

  const selectMeeting = useCallback((id: string | null) => {
    setSelectedMeetingId(id)
  }, [])

  const createMeeting = useCallback(async (): Promise<string> => {
    const meeting = await createMeetingRecord()
    setMeetingsIndex((current) => upsertMeetingRow(current, meeting))
    setSelectedMeetingId(meeting.meetingId)
    return meeting.meetingId
  }, [])

  const deleteMeeting = useCallback(
    async (id: string): Promise<void> => {
      if (id === liveMeetingId) throw new Error('Stop the live meeting before deleting it')
      await deleteMeetingRecord(id)
      const remaining = meetingsIndex.filter((meeting) => meeting.meetingId !== id)
      setMeetingsIndex(remaining)
      if (selectedMeetingId === id) setSelectedMeetingId(remaining[0]?.meetingId ?? null)
    },
    [liveMeetingId, meetingsIndex, selectedMeetingId]
  )

  const renameMeeting = useCallback(async (id: string, title: string): Promise<void> => {
    const meeting = await patchMeeting(id, { title })
    setMeetingsIndex((current) => upsertMeetingRow(current, meeting))
  }, [])

  const saveNotes = useCallback((id: string, notesMd: string): Promise<void> => {
    const prior = notesSavesRef.current.get(id) ?? Promise.resolve()
    const task = prior
      .catch(() => undefined)
      .then(async () => {
        const meeting = await patchMeeting(id, { notesMd })
        setMeetingsIndex((current) => upsertMeetingRow(current, meeting))
      })
    notesSavesRef.current.set(id, task)
    task.then(
      () => {
        if (notesSavesRef.current.get(id) === task) notesSavesRef.current.delete(id)
      },
      () => {
        if (notesSavesRef.current.get(id) === task) notesSavesRef.current.delete(id)
      }
    )
    return task
  }, [])

  const registerNotesFlush = useCallback((flush: () => Promise<void>): (() => void) => {
    notesFlushRef.current = flush
    return () => {
      if (notesFlushRef.current === flush) notesFlushRef.current = null
    }
  }, [])

  const queueNotepadPush = useCallback(
    (id: string, textMd: string): void => {
      if (listeningState !== 'listening' || id !== liveMeetingId) return
      notepadPushRef.current?.schedule(id, textMd)
    },
    [listeningState, liveMeetingId]
  )

  const flushNotepadPush = useCallback((id: string, textMd: string): void => {
    if (id !== liveMeetingIdRef.current) return
    notepadPushRef.current?.flush(id, textMd)
  }, [])

  const startMeeting = useCallback(
    async (id?: string): Promise<void> => {
      const client = clientRef.current
      if (!client || listeningState === 'starting' || listeningState === 'listening') {
        return
      }

      let meetingId = id
      if (!meetingId) {
        try {
          meetingId = await createMeeting()
        } catch (error) {
          setMicError(error instanceof Error ? error.message : 'Meeting could not be created')
          setListeningState('error')
          return
        }
      }
      setSelectedMeetingId(meetingId)
      setActiveTab('meetings')
      setMicError('')
      setListeningState('starting')
      liveMeetingIdRef.current = meetingId
      setLiveMeetingId(meetingId)
      transcriptIdRef.current = 0
      setChunks([])
      setInterimsByChannel({})
      setLiveGrounding(null)
      setSummary(null)
      setActionables([])
      setGroundedNoteIds([])
      setLastRetrieval(null)
      setFinalSummaryStatuses((current) => {
        if (!(meetingId in current)) return current
        const next = { ...current }
        delete next[meetingId]
        return next
      })
      setLatestFinalSummary((current) => (current?.meetingId === meetingId ? null : current))
      notepadPushRef.current?.cancel()

      let systemAudioAvailable = false
      try {
        systemAudioAvailable = await window.api.sysAudio.available()
      } catch {
        // Unsupported platforms and missing helpers retain the existing mic-only path.
      }
      client.start(
        systemAudioAvailable
          ? [
              { id: 0, label: 'mic' },
              { id: 1, label: 'system' }
            ]
          : [{ id: 0, label: 'mic' }],
        meetingId
      )

      const capture = new PcmAudioCapture((frame) => {
        clientRef.current?.sendAudioFrame(frame)
      })
      audioRef.current = capture

      let systemCapture: SystemAudioCapture | null = null
      if (systemAudioAvailable) {
        systemCapture = new SystemAudioCapture(
          (frame) => clientRef.current?.sendAudioFrame(frame),
          () => {
            const failedCapture = systemCapture
            if (!failedCapture || systemAudioRef.current !== failedCapture) return
            failedCapture.stop()
            systemAudioRef.current = null
            if (liveMeetingIdRef.current === meetingId) {
              setMicError('System audio stopped — mic still live')
            }
          }
        )
      }
      systemAudioRef.current = systemCapture

      try {
        await capture.start()
        setListeningState('listening')
      } catch (error) {
        systemCapture?.stop()
        systemAudioRef.current = null
        capture.stop()
        audioRef.current = null
        client.stop()
        notepadPushRef.current?.cancel()
        liveMeetingIdRef.current = null
        setLiveMeetingId(null)
        setMicError(error instanceof Error ? error.message : 'Microphone capture failed')
        setListeningState('error')
        return
      }

      if (systemCapture) {
        try {
          await systemCapture.start()
        } catch {
          systemCapture.stop()
          if (systemAudioRef.current === systemCapture) {
            systemAudioRef.current = null
          }
          if (liveMeetingIdRef.current === meetingId) {
            setMicError('System audio stopped — mic still live')
          }
        }
      }
    },
    [createMeeting, listeningState]
  )

  const stopMeeting = useCallback(async (): Promise<void> => {
    if (listeningState !== 'listening' && listeningState !== 'starting') return
    setListeningState('stopping')
    try {
      await notesFlushRef.current?.()
      await Promise.all([...notesSavesRef.current.values()])
      notepadPushRef.current?.flushPending()
    } catch (error) {
      setMicError(
        error instanceof Error
          ? `Could not save notes before stopping: ${error.message}`
          : 'Could not save notes before stopping'
      )
      setListeningState('listening')
      return
    }
    systemAudioRef.current?.stop()
    systemAudioRef.current = null
    audioRef.current?.stop()
    audioRef.current = null
    clientRef.current?.stop()
    notepadPushRef.current?.cancel()
    liveMeetingIdRef.current = null
    setLiveMeetingId(null)
    setListeningState('idle')
  }, [listeningState])

  // OS-level notification panel (or its countdown) chose "Start notes".
  useEffect(
    () =>
      window.api.callNotif.onStart(() => {
        setActiveTab('meetings')
        void startMeeting()
      }),
    [startMeeting]
  )

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

  // citation chips / chat source pills → center + pulse that node on the map
  const showInMap = useCallback((noteId: string) => {
    setMapFocus((current) => ({ noteId, nonce: (current?.nonce ?? 0) + 1 }))
    setActiveTab('map')
  }, [])

  // map "Ask about this" → new chat seeded with the note, composer prefilled
  const askAboutNote = useCallback((noteId: string, title: string) => {
    setChatSeed((current) => ({ noteId, title, nonce: (current?.nonce ?? 0) + 1 }))
    setActiveTab('chat')
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

  const meetingsCoordinator = useMemo<MeetingsCoordinator>(
    () => ({
      meetings: meetingsIndex,
      selectedMeetingId,
      liveMeetingId,
      finalSummaryStatus: selectedMeetingId
        ? (finalSummaryStatuses[selectedMeetingId] ?? null)
        : null,
      latestFinalSummary,
      listeningState,
      micError: micError || null,
      live: {
        chunks,
        interims: interimsByChannel,
        summary,
        actionables,
        liveGrounding
      },
      selectMeeting,
      createMeeting,
      deleteMeeting,
      renameMeeting,
      startMeeting,
      stopMeeting,
      saveNotes,
      registerNotesFlush,
      queueNotepadPush,
      flushNotepadPush,
      setActionableStatus: actionableStatus,
      openNoteInMap: showInMap
    }),
    [
      showInMap,
      actionables,
      actionableStatus,
      chunks,
      createMeeting,
      deleteMeeting,
      interimsByChannel,
      listeningState,
      liveGrounding,
      liveMeetingId,
      finalSummaryStatuses,
      latestFinalSummary,
      meetingsIndex,
      micError,
      flushNotepadPush,
      queueNotepadPush,
      registerNotesFlush,
      renameMeeting,
      saveNotes,
      selectMeeting,
      selectedMeetingId,
      startMeeting,
      stopMeeting,
      summary
    ]
  )

  const showDisconnectedBanner = connection !== 'connected'

  return (
    <div className="relative flex h-screen flex-col gap-2.5 p-3">
      <header className="flex min-h-[52px] flex-none items-center justify-between gap-4 rounded-[10px] border border-white/[0.08] bg-surface-1/70 px-3.5 py-2 backdrop-blur-md">
        <div className="flex min-w-0 items-center gap-2.5">
          <h1 className="text-[15px] font-bold leading-none tracking-tight text-fg before:mr-2 before:inline-block before:h-3.5 before:w-1 before:translate-y-px before:rounded-sm before:bg-accent before:content-['']">
            Taurine
          </h1>
          <span className="whitespace-nowrap text-[11px] font-medium text-fg/40">
            Live meeting copilot
          </span>
        </div>

        <nav
          className="flex items-center gap-0.5 rounded-full border border-white/[0.08] bg-white/[0.03] p-0.5"
          role="tablist"
          aria-label="Main view"
        >
          {MAIN_TABS.map((tab) => (
            <button
              key={tab.id}
              className={`h-6 min-w-[76px] rounded-full border-0 px-3.5 text-xs transition-colors ${
                activeTab === tab.id
                  ? 'bg-accent font-semibold text-accent-ink'
                  : 'bg-transparent font-medium text-fg/55 hover:text-fg'
              }`}
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
          onStart={() => void startMeeting()}
          onStop={() => void stopMeeting()}
        />
      </header>

      {showDisconnectedBanner ? (
        <div
          className="flex-none rounded-[10px] border border-accent/50 bg-accent/10 px-3.5 py-2 text-xs font-medium text-accent-hi backdrop-blur-md"
          role="status"
        >
          {startupBannerText(connection, retryInMs, startup)}
        </div>
      ) : null}

      <div className="min-h-0 flex-1" style={viewStackStyle}>
        <div
          aria-hidden={activeTab !== 'meetings'}
          style={viewLayerStyle(activeTab === 'meetings')}
        >
          <MeetingsView
            key={selectedMeetingId ?? 'no-meeting-selected'}
            coordinator={meetingsCoordinator}
          />
        </div>

        <div aria-hidden={activeTab !== 'chat'} style={viewLayerStyle(activeTab === 'chat')}>
          <ChatView
            visible={activeTab === 'chat'}
            onOpenNote={showInMap}
            onShowInMap={showInMap}
            seedRequest={chatSeed}
          />
        </div>

        <div aria-hidden={activeTab !== 'map'} style={viewLayerStyle(activeTab === 'map')}>
          <MapView
            latestFinalText={latestFinalText}
            groundedNoteIds={groundedNoteIds}
            refreshKey={graphVersion}
            retrieval={lastRetrieval}
            visible={activeTab === 'map'}
            focusRequest={mapFocus}
            onOpenNote={openNote}
            onAskAbout={askAboutNote}
          />
        </div>

        <div aria-hidden={activeTab !== 'data'} style={viewLayerStyle(activeTab === 'data')}>
          <DataView profile={profile} onProfileChange={setProfile} />
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

function upsertMeetingRow(current: MeetingListRow[], meeting: MeetingListRow): MeetingListRow[] {
  const row: MeetingListRow = {
    meetingId: meeting.meetingId,
    title: meeting.title,
    createdAt: meeting.createdAt,
    updatedAt: meeting.updatedAt,
    startedAt: meeting.startedAt,
    endedAt: meeting.endedAt,
    status: meeting.status
  }
  return [row, ...current.filter((item) => item.meetingId !== meeting.meetingId)].sort(
    (left, right) => right.createdAt - left.createdAt
  )
}

function selectedMeetingForIndex(
  current: string | null,
  meetings: MeetingListRow[]
): string | null {
  if (current && meetings.some((row) => row.meetingId === current)) return current
  return meetings[0]?.meetingId ?? null
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
