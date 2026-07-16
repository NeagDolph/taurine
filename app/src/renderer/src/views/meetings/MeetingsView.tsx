import { useCallback, useEffect, useRef, useState } from 'react'
import type { ListeningState } from '../../components/SessionControls'
import { getMeeting, patchMeeting, type Meeting, type MeetingListRow } from '../../lib/meetings'
import { appendSummaryToNotes, flushCurrentNotepad } from '../../lib/notepad-sync'
import { applySuggestion, revalidatePendingSuggestions } from '../../lib/suggestions'
import type {
  Actionable,
  FinalSummaryEvent,
  FinalSummaryStatusEvent,
  LiveGroundingEvent,
  SummaryEvent,
  TranscriptEvent
} from '../../lib/sidecar'
import { MeetingChat, type AskRequest } from './MeetingChat'
import { NotepadEditor } from './NotepadEditor'
import { SummaryCard } from './SummaryCard'
import { TranscriptOverlay } from './TranscriptOverlay'

export interface MeetingsCoordinator {
  meetings: MeetingListRow[]
  selectedMeetingId: string | null
  liveMeetingId: string | null
  listeningState: ListeningState
  micError: string | null
  finalSummaryStatus: FinalSummaryStatusEvent | null
  latestFinalSummary: FinalSummaryEvent | null
  live: {
    chunks: TranscriptEvent[]
    interims: Record<number, TranscriptEvent>
    summary: SummaryEvent | null
    actionables: Actionable[]
    liveGrounding: LiveGroundingEvent | null
  }
  selectMeeting(id: string | null): void
  createMeeting(): Promise<string>
  deleteMeeting(id: string): Promise<void>
  renameMeeting(id: string, title: string): Promise<void>
  startMeeting(id?: string): Promise<void>
  stopMeeting(): Promise<void>
  saveNotes(id: string, notesMd: string): Promise<void>
  registerNotesFlush(flush: () => Promise<void>): () => void
  queueNotepadPush(id: string, notesMd: string): void
  flushNotepadPush(id: string, notesMd: string): void
  setActionableStatus(id: string, status: 'confirmed' | 'dismissed'): void
  openNoteInMap(noteId: string): void
}

export function MeetingsView({
  coordinator
}: {
  coordinator: MeetingsCoordinator
}): React.JSX.Element {
  const {
    selectedMeetingId,
    saveNotes,
    registerNotesFlush,
    queueNotepadPush,
    flushNotepadPush,
    selectMeeting,
    createMeeting,
    deleteMeeting,
    renameMeeting,
    openNoteInMap
  } = coordinator
  const [meeting, setMeeting] = useState<Meeting | null>(null)
  const [notesMd, setNotesMd] = useState('')
  const [title, setTitle] = useState('')
  const [loading, setLoading] = useState(Boolean(selectedMeetingId))
  const [error, setError] = useState('')
  const [transcriptOpen, setTranscriptOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [askRequest, setAskRequest] = useState<AskRequest | null>(null)
  const [askDraft, setAskDraft] = useState('')
  const timerRef = useRef<number | null>(null)
  const currentMeetingIdRef = useRef<string | null>(null)
  const loadedMeetingIdRef = useRef<string | null>(null)
  const currentNotesRef = useRef('')
  const savedNotesRef = useRef('')

  const savePendingNotes = useCallback(async (): Promise<void> => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
    const id = currentMeetingIdRef.current
    const markdown = currentNotesRef.current
    if (!id || markdown === savedNotesRef.current) return
    await saveNotes(id, markdown)
    savedNotesRef.current = markdown
  }, [saveNotes])

  const flushNotes = useCallback(async (): Promise<void> => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
    const id = currentMeetingIdRef.current
    savedNotesRef.current = await flushCurrentNotepad({
      meetingId: loadedMeetingIdRef.current === id ? id : null,
      currentTextMd: currentNotesRef.current,
      savedTextMd: savedNotesRef.current,
      save: saveNotes,
      push: flushNotepadPush
    })
  }, [flushNotepadPush, saveNotes])

  useEffect(() => registerNotesFlush(flushNotes), [flushNotes, registerNotesFlush])

  const applyLoadedMeeting = useCallback((next: Meeting): void => {
    setMeeting(next)
    setNotesMd(next.notesMd)
    setTitle(next.title)
    currentNotesRef.current = next.notesMd
    savedNotesRef.current = next.notesMd
    loadedMeetingIdRef.current = next.meetingId
  }, [])

  // Reset per-meeting overlay state during render when the selection moves.
  const [overlaysFor, setOverlaysFor] = useState(selectedMeetingId)
  if (overlaysFor !== selectedMeetingId) {
    setOverlaysFor(selectedMeetingId)
    setChatOpen(false)
    setTranscriptOpen(false)
    setAskRequest(null)
    setLoading(Boolean(selectedMeetingId))
    if (!selectedMeetingId) setMeeting(null)
  }

  useEffect(() => {
    const controller = new AbortController()
    currentMeetingIdRef.current = selectedMeetingId
    loadedMeetingIdRef.current = null

    if (selectedMeetingId) {
      void getMeeting(selectedMeetingId, { signal: controller.signal })
        .then((next) => {
          if (controller.signal.aborted) return
          applyLoadedMeeting(next)
        })
        .catch((loadError: unknown) => {
          if (!controller.signal.aborted) {
            setError(loadError instanceof Error ? loadError.message : 'Meeting could not be loaded')
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false)
        })
    }

    return () => {
      controller.abort()
      void flushNotes().catch(() => undefined)
    }
  }, [applyLoadedMeeting, flushNotes, selectedMeetingId])

  useEffect(() => {
    const event = coordinator.latestFinalSummary
    if (!event || event.meetingId !== selectedMeetingId) return

    let cancelled = false
    const refetchAuthoritativeMeeting = async (): Promise<void> => {
      try {
        await flushNotes()
        const next = await getMeeting(event.meetingId)
        if (!cancelled) applyLoadedMeeting(next)
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error ? loadError.message : 'Final summary could not be loaded'
          )
        }
      }
    }
    void refetchAuthoritativeMeeting()
    return () => {
      cancelled = true
    }
  }, [applyLoadedMeeting, coordinator.latestFinalSummary, flushNotes, selectedMeetingId])

  const handleNotesChange = useCallback(
    (markdown: string): void => {
      setNotesMd(markdown)
      currentNotesRef.current = markdown
      queueNotepadPush(currentMeetingIdRef.current ?? '', markdown)
      if (markdown === savedNotesRef.current) return
      if (timerRef.current !== null) window.clearTimeout(timerRef.current)
      timerRef.current = window.setTimeout(() => {
        timerRef.current = null
        void savePendingNotes().catch((saveError: unknown) => {
          setError(saveError instanceof Error ? saveError.message : 'Notes could not be saved')
        })
      }, 800)
    },
    [queueNotepadPush, savePendingNotes]
  )

  const handleAddToNotes = useCallback(
    (summaryMd: string): void => {
      handleNotesChange(appendSummaryToNotes(currentNotesRef.current, summaryMd))
    },
    [handleNotesChange]
  )

  const handleResolveSuggestion = useCallback(
    (id: string, accept: boolean): void => {
      if (!meeting) return
      const edit = meeting.suggestions.find(
        (suggestion) => suggestion.id === id && suggestion.status === 'pending'
      )
      if (!edit) return

      const status = accept ? ('accepted' as const) : ('rejected' as const)
      const nextSuggestions = meeting.suggestions.map((suggestion) =>
        suggestion.id === id ? { ...suggestion, status } : suggestion
      )
      setMeeting((current) =>
        current?.meetingId === meeting.meetingId
          ? { ...current, suggestions: nextSuggestions }
          : current
      )
      if (accept) {
        handleNotesChange(applySuggestion(currentNotesRef.current, edit))
      }

      void patchMeeting(meeting.meetingId, { suggestions: nextSuggestions }).catch(
        (patchError: unknown) => {
          setMeeting((current) =>
            current?.meetingId === meeting.meetingId
              ? {
                  ...current,
                  suggestions: current.suggestions.map((suggestion) =>
                    suggestion.id === id ? { ...suggestion, status: 'pending' } : suggestion
                  )
                }
              : current
          )
          setError(
            patchError instanceof Error
              ? patchError.message
              : 'Suggestion status could not be saved'
          )
        }
      )
    },
    [handleNotesChange, meeting]
  )

  const handleSelect = useCallback(
    async (id: string | null): Promise<void> => {
      try {
        await flushNotes()
        selectMeeting(id)
      } catch (saveError) {
        setError(saveError instanceof Error ? saveError.message : 'Notes could not be saved')
      }
    },
    [flushNotes, selectMeeting]
  )

  const handleCreate = useCallback(async (): Promise<void> => {
    try {
      await flushNotes()
      await createMeeting()
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Meeting could not be created')
    }
  }, [createMeeting, flushNotes])

  const handleDelete = useCallback(
    async (id: string): Promise<void> => {
      try {
        if (id === selectedMeetingId) await flushNotes()
        await deleteMeeting(id)
      } catch (deleteError) {
        setError(
          deleteError instanceof Error ? deleteError.message : 'Meeting could not be deleted'
        )
      }
    },
    [deleteMeeting, flushNotes, selectedMeetingId]
  )

  const commitTitle = useCallback(async (): Promise<void> => {
    if (!meeting || title === meeting.title) return
    try {
      await renameMeeting(meeting.meetingId, title)
      setMeeting((current) => (current ? { ...current, title } : current))
    } catch (renameError) {
      setTitle(meeting.title)
      setError(renameError instanceof Error ? renameError.message : 'Meeting could not be renamed')
    }
  }, [meeting, renameMeeting, title])

  const ask = useCallback((text: string): void => {
    const question = text.trim()
    if (!question) return
    setAskDraft('')
    setTranscriptOpen(false)
    setChatOpen(true)
    setAskRequest((current) => ({ text: question, nonce: (current?.nonce ?? 0) + 1 }))
  }, [])

  const isLiveSelection = meeting?.meetingId === coordinator.liveMeetingId
  const isListening =
    isLiveSelection &&
    (coordinator.listeningState === 'listening' || coordinator.listeningState === 'starting')
  const suggestions =
    meeting && meeting.suggestionsBaseRev < meeting.notesRev
      ? revalidatePendingSuggestions(notesMd, meeting.suggestions)
      : (meeting?.suggestions ?? [])
  const finalSummaryStatus =
    coordinator.finalSummaryStatus ??
    (meeting?.status === 'summarizing'
      ? {
          type: 'final_summary_status' as const,
          meetingId: meeting.meetingId,
          state: 'running' as const,
          detail: ''
        }
      : null)

  if (!selectedMeetingId) {
    return (
      <HomeList
        meetings={coordinator.meetings}
        liveMeetingId={coordinator.liveMeetingId}
        onOpen={(id) => void handleSelect(id)}
        onCreate={() => void handleCreate()}
        onDelete={(id) => void handleDelete(id)}
      />
    )
  }

  return (
    <div className="relative h-full min-h-0 overflow-hidden">
      {loading ? (
        <div className="flex h-full items-center justify-center text-xs text-fg/45">
          Loading meeting…
        </div>
      ) : meeting ? (
        <>
          <div className="h-full min-h-0 overflow-y-auto pb-36">
            <div className="mx-auto w-full max-w-[680px] px-6">
              <div className="flex h-12 items-center justify-between pt-2">
                <button
                  className="icon-button"
                  type="button"
                  aria-label="All meetings"
                  title="All meetings"
                  onClick={() => void handleSelect(null)}
                >
                  ⌂
                </button>
                {!isLiveSelection && meeting.status !== 'summarizing' ? (
                  <button
                    className="secondary-button compact"
                    type="button"
                    disabled={coordinator.listeningState !== 'idle'}
                    onClick={() => void coordinator.startMeeting(meeting.meetingId)}
                  >
                    ● Record
                  </button>
                ) : null}
              </div>

              <input
                className="mt-4 w-full border-0 bg-transparent font-serif text-[34px] font-medium leading-tight text-fg/90 outline-none placeholder:text-fg/25"
                value={title}
                placeholder="New note"
                aria-label="Meeting title"
                onChange={(event) => setTitle(event.target.value)}
                onBlur={() => void commitTitle()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') event.currentTarget.blur()
                }}
              />

              <div className="mb-5 mt-2.5 flex items-center gap-2">
                <span className="meta-chip">🗓 {dateLabel(meeting.createdAt)}</span>
                {isListening ? (
                  <span className="meta-chip border-ok/30 text-ok">
                    <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ok align-middle" />
                    Recording
                  </span>
                ) : meeting.status === 'summarizing' ? (
                  <span className="meta-chip text-fg/60">Summarizing…</span>
                ) : null}
                {coordinator.micError && isLiveSelection ? (
                  <span className="meta-chip border-bad/30 text-bad" title={coordinator.micError}>
                    Mic issue
                  </span>
                ) : null}
              </div>

              <SummaryCard
                summary={isLiveSelection ? coordinator.live.summary : null}
                actionables={isLiveSelection ? coordinator.live.actionables : []}
                finalSummaryStatus={finalSummaryStatus}
                listeningState={isLiveSelection ? coordinator.listeningState : 'idle'}
                onActionableStatus={coordinator.setActionableStatus}
                onAddToNotes={handleAddToNotes}
                onOpenNote={openNoteInMap}
              />

              <NotepadEditor
                key={meeting.meetingId}
                valueMd={notesMd}
                onChangeMd={handleNotesChange}
                suggestions={suggestions}
                onResolveSuggestion={handleResolveSuggestion}
                onOpenNoteInMap={openNoteInMap}
                readOnly={isLiveSelection && coordinator.listeningState === 'stopping'}
              />
            </div>
          </div>

          <TranscriptOverlay
            open={transcriptOpen}
            meeting={meeting}
            isLive={isLiveSelection}
            live={coordinator.live}
            onClose={() => setTranscriptOpen(false)}
            onOpenNote={openNoteInMap}
          />

          <MeetingChat
            key={`${meeting.meetingId}:${meeting.chatId ?? 'new'}`}
            meeting={meeting}
            open={chatOpen}
            askRequest={askRequest}
            onClose={() => setChatOpen(false)}
            onOpenNote={openNoteInMap}
          />

          <div className="absolute bottom-5 left-1/2 z-40 flex w-[min(680px,92%)] -translate-x-1/2 items-center gap-2 rounded-full border border-white/[0.1] bg-surface-2/90 py-1.5 pl-2 pr-2 shadow-2xl backdrop-blur-xl">
            <button
              className={`flex h-9 items-center gap-1.5 rounded-full px-3 text-sm transition-colors ${
                transcriptOpen ? 'bg-white/[0.09]' : 'hover:bg-white/[0.05]'
              } ${isListening ? 'text-ok' : 'text-fg/60'}`}
              type="button"
              aria-pressed={transcriptOpen}
              title={isListening ? 'Live transcript' : 'Transcript'}
              aria-label="Toggle transcript"
              onClick={() => {
                setChatOpen(false)
                setTranscriptOpen((current) => !current)
              }}
            >
              <Waveform active={isListening} />
              {transcriptOpen ? '⌃' : '⌄'}
            </button>
            {isListening ? (
              <button
                className="flex h-9 w-9 flex-none items-center justify-center rounded-full text-fg/70 transition-colors hover:bg-bad/15 hover:text-bad"
                type="button"
                title="Stop recording"
                aria-label="Stop recording"
                onClick={() => void coordinator.stopMeeting()}
              >
                ◼
              </button>
            ) : null}
            <input
              className="h-9 min-w-0 flex-1 border-0 bg-transparent px-2 text-[13.5px] text-fg outline-none placeholder:text-fg/35"
              value={askDraft}
              placeholder="Ask anything"
              aria-label="Ask about this meeting"
              onChange={(event) => setAskDraft(event.target.value)}
              onFocus={() => {
                if (chatOpen) return
                setTranscriptOpen(false)
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') ask(askDraft)
              }}
            />
            <button
              className="flex h-9 flex-none items-center gap-1.5 rounded-full border border-white/[0.1] bg-white/[0.04] px-3.5 text-xs text-fg/75 transition-colors hover:bg-white/[0.08] hover:text-fg"
              type="button"
              onClick={() => ask('What did I miss in this meeting? Catch me up briefly.')}
            >
              ⚡ What did I miss
            </button>
          </div>
        </>
      ) : (
        <div className="flex h-full items-center justify-center p-6 text-center text-xs text-bad">
          {error || 'Meeting could not be loaded.'}
        </div>
      )}

      {error && meeting ? (
        <div className="pointer-events-none absolute bottom-20 left-1/2 z-50 -translate-x-1/2 rounded-lg border border-bad/30 bg-surface-2/95 px-3 py-2 text-xs text-bad shadow-lg">
          {error}
        </div>
      ) : null}
    </div>
  )
}

function HomeList({
  meetings,
  liveMeetingId,
  onOpen,
  onCreate,
  onDelete
}: {
  meetings: MeetingListRow[]
  liveMeetingId: string | null
  onOpen(id: string): void
  onCreate(): void
  onDelete(id: string): void
}): React.JSX.Element {
  return (
    <div className="h-full min-h-0 overflow-y-auto">
      <div className="mx-auto w-full max-w-[680px] px-6 pb-16">
        <div className="flex items-end justify-between pt-10">
          <h1 className="m-0 font-serif text-[30px] font-medium text-fg/90">Meetings</h1>
          <button className="primary-button" type="button" onClick={onCreate}>
            + New note
          </button>
        </div>

        {meetings.length === 0 ? (
          <div className="mt-16 text-center">
            <div className="text-3xl text-accent/50">✦</div>
            <p className="mt-3 text-sm text-fg/55">
              Notes and transcripts from your meetings live here.
            </p>
          </div>
        ) : (
          <ul className="mt-6 grid list-none gap-1 p-0">
            {meetings.map((row) => (
              <li key={row.meetingId} className="group relative">
                <button
                  className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 rounded-xl border border-transparent px-3.5 py-3 text-left transition-colors hover:border-white/[0.08] hover:bg-white/[0.03]"
                  type="button"
                  onClick={() => onOpen(row.meetingId)}
                >
                  <span className="min-w-0">
                    <span className="flex items-center gap-2">
                      {row.meetingId === liveMeetingId ? (
                        <span className="h-1.5 w-1.5 flex-none animate-pulse rounded-full bg-ok" />
                      ) : null}
                      <span className="truncate text-[14.5px] text-fg/90">
                        {row.title || 'New note'}
                      </span>
                    </span>
                  </span>
                  <span className="flex-none text-xs tabular-nums text-fg/40">
                    {dateLabel(row.createdAt)}
                  </span>
                </button>
                <button
                  className="absolute -right-9 top-1/2 hidden h-7 w-7 -translate-y-1/2 items-center justify-center rounded text-fg/35 hover:text-bad group-hover:flex"
                  type="button"
                  aria-label={`Delete ${row.title || 'meeting'}`}
                  onClick={() => onDelete(row.meetingId)}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function Waveform({ active }: { active: boolean }): React.JSX.Element {
  return (
    <span className={`flex h-4 items-end gap-[2px] ${active ? 'waveform-live' : ''}`} aria-hidden>
      <span className="w-[3px] rounded-full bg-current" style={{ height: '55%' }} />
      <span className="w-[3px] rounded-full bg-current" style={{ height: '100%' }} />
      <span className="w-[3px] rounded-full bg-current" style={{ height: '40%' }} />
      <span className="w-[3px] rounded-full bg-current" style={{ height: '75%' }} />
    </span>
  )
}

function dateLabel(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000)
  const today = new Date()
  const startOfDay = (value: Date): number =>
    new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime()
  const dayDelta = Math.round((startOfDay(today) - startOfDay(date)) / 86_400_000)
  if (dayDelta === 0) return 'Today'
  if (dayDelta === 1) return 'Yesterday'
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
