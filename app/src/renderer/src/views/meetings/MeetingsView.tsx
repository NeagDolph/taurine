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
import { MeetingSidebar } from './MeetingSidebar'
import { NotepadEditor } from './NotepadEditor'
import { SummaryCard } from './SummaryCard'
import { TranscriptPanel } from './TranscriptPanel'

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
    renameMeeting
  } = coordinator
  const [meeting, setMeeting] = useState<Meeting | null>(null)
  const [notesMd, setNotesMd] = useState('')
  const [title, setTitle] = useState('')
  const [loading, setLoading] = useState(Boolean(selectedMeetingId))
  const [error, setError] = useState('')
  const [transcriptOpen, setTranscriptOpen] = useState(true)
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
    async (id: string): Promise<void> => {
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

  const isLiveSelection = meeting?.meetingId === coordinator.liveMeetingId
  const selectedSummary = isLiveSelection ? coordinator.live.summary : null
  const selectedActionables = isLiveSelection ? coordinator.live.actionables : []
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

  return (
    <div className="grid h-full min-h-0 grid-cols-[220px_minmax(0,1fr)] overflow-hidden rounded-[10px] border border-white/[0.08] bg-surface-1/60">
      <MeetingSidebar
        meetings={coordinator.meetings}
        selectedMeetingId={selectedMeetingId}
        liveMeetingId={coordinator.liveMeetingId}
        onSelect={(id) => void handleSelect(id)}
        onCreate={() => void handleCreate()}
        onDelete={(id) => void handleDelete(id)}
      />

      <div
        className={`grid min-h-0 ${
          transcriptOpen && meeting
            ? 'grid-cols-[minmax(0,1fr)_clamp(280px,30vw,390px)]'
            : 'grid-cols-[minmax(0,1fr)]'
        }`}
      >
        {!selectedMeetingId ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
            <div className="text-4xl text-accent/60">✦</div>
            <div>
              <h2 className="text-base font-semibold text-fg">Start a meeting note</h2>
              <p className="mt-1 text-xs text-fg/45">
                Capture notes now and add live context when you need it.
              </p>
            </div>
            <button className="primary-button" type="button" onClick={() => void handleCreate()}>
              New note
            </button>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center text-xs text-fg/45">
            Loading meeting…
          </div>
        ) : meeting ? (
          <main className="grid min-h-0 grid-rows-[auto_auto_minmax(0,1fr)_auto] overflow-hidden">
            <div className="flex h-12 items-center gap-2 border-b border-white/[0.08] px-5">
              <input
                className="min-w-0 flex-1 border-0 bg-transparent text-[16px] font-semibold text-fg outline-none placeholder:text-fg/30"
                value={title}
                placeholder="Untitled meeting"
                aria-label="Meeting title"
                onChange={(event) => setTitle(event.target.value)}
                onBlur={() => void commitTitle()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') event.currentTarget.blur()
                }}
              />
              {!transcriptOpen ? (
                <button
                  className="secondary-button compact"
                  type="button"
                  onClick={() => setTranscriptOpen(true)}
                >
                  〽 Transcript
                </button>
              ) : null}
              {meeting.meetingId !== coordinator.liveMeetingId ? (
                <button
                  className="secondary-button compact"
                  type="button"
                  disabled={coordinator.listeningState !== 'idle'}
                  onClick={() => void coordinator.startMeeting(meeting.meetingId)}
                >
                  Start this meeting
                </button>
              ) : null}
            </div>

            <SummaryCard
              summary={selectedSummary}
              actionables={selectedActionables}
              finalSummaryMd={meeting.finalSummaryMd}
              finalSummaryStatus={finalSummaryStatus}
              listeningState={isLiveSelection ? coordinator.listeningState : 'idle'}
              onActionableStatus={coordinator.setActionableStatus}
              onAddToNotes={handleAddToNotes}
            />

            <div className="min-h-0 overflow-y-auto bg-surface-0/20">
              <NotepadEditor
                key={meeting.meetingId}
                valueMd={notesMd}
                onChangeMd={handleNotesChange}
                suggestions={suggestions}
                onResolveSuggestion={handleResolveSuggestion}
                readOnly={isLiveSelection && coordinator.listeningState === 'stopping'}
              />
            </div>

            <div className="border-t border-white/[0.08] p-3">
              <button
                className="flex h-9 w-full items-center rounded-lg border border-white/[0.08] bg-white/[0.025] px-3 text-left text-xs text-fg/35"
                type="button"
                disabled
              >
                Ask about this meeting
              </button>
            </div>
          </main>
        ) : (
          <div className="flex items-center justify-center p-6 text-center text-xs text-bad">
            {error || 'Meeting could not be loaded.'}
          </div>
        )}

        {meeting && transcriptOpen ? (
          <TranscriptPanel
            meeting={meeting}
            liveMeetingId={coordinator.liveMeetingId}
            live={coordinator.live}
            onClose={() => setTranscriptOpen(false)}
          />
        ) : null}
      </div>

      {error && meeting ? (
        <div className="pointer-events-none absolute bottom-16 left-1/2 z-20 -translate-x-1/2 rounded-lg border border-bad/30 bg-surface-2/95 px-3 py-2 text-xs text-bad shadow-lg">
          {error}
        </div>
      ) : null}
    </div>
  )
}
