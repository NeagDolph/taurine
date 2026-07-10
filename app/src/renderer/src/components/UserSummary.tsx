import { useCallback, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  generateProfile,
  saveProfile,
  type ProfileGenerationPhase,
  type ProfileGenerationStatus,
  type ProfileState
} from '../lib/sidecar'

interface UserSummaryProps {
  profile: ProfileState | undefined
  onProfileChange: (profile: ProfileState) => void
}

const RESEARCH_STEPS: Array<{
  phase: Exclude<ProfileGenerationPhase, 'idle' | 'failed'>
  label: string
}> = [
  { phase: 'surveying', label: 'Survey graph' },
  { phase: 'interests', label: 'Map interests' },
  { phase: 'projects_people', label: 'Projects & people' },
  { phase: 'career_studies', label: 'Career & studies' },
  { phase: 'synthesizing', label: 'Write summary' }
]

const STARTING_STATUS: ProfileGenerationStatus = {
  running: true,
  phase: 'surveying',
  activity: 'Starting graph research',
  nodes_read: 0,
  tool_calls: 0,
  started_at: null,
  elapsed_s: 0,
  error: ''
}

export function UserSummary({ profile, onProfileChange }: UserSummaryProps): React.JSX.Element {
  const [collapsed, setCollapsed] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [starting, setStarting] = useState(false)
  const [confirmingCard, setConfirmingCard] = useState<string | null>(null)
  const [requestError, setRequestError] = useState('')
  const [now, setNow] = useState(() => Date.now())

  const card = profile?.card ?? ''
  const generation = starting ? STARTING_STATUS : profile?.generation
  const generating = Boolean(generation?.running)
  const failed = generation?.phase === 'failed'
  const confirming = Boolean(card.trim()) && confirmingCard === card

  useEffect(() => {
    if (!generating) {
      return
    }
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [generating])

  const elapsed = !generation
    ? 0
    : generation.running && generation.started_at !== null
      ? Math.max(generation.elapsed_s, now / 1000 - generation.started_at)
      : generation.elapsed_s

  const beginEdit = useCallback(() => {
    setDraft(card)
    setRequestError('')
    setConfirmingCard(null)
    setEditing(true)
    setCollapsed(false)
  }, [card])

  const cancelEdit = useCallback(() => {
    setEditing(false)
    setRequestError('')
  }, [])

  const save = useCallback(async () => {
    if (!draft.trim() || saving) {
      return
    }
    setSaving(true)
    setRequestError('')
    try {
      const next = await saveProfile(draft.trim())
      onProfileChange(next)
      setEditing(false)
    } catch (saveError) {
      setRequestError(saveError instanceof Error ? saveError.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }, [draft, onProfileChange, saving])

  const startGeneration = useCallback(async () => {
    if (generating || starting) {
      return
    }
    setConfirmingCard(null)
    setEditing(false)
    setCollapsed(false)
    setRequestError('')
    setStarting(true)
    try {
      const next = await generateProfile()
      onProfileChange(next)
    } catch (generateError) {
      setRequestError(
        generateError instanceof Error ? generateError.message : 'Generation failed to start'
      )
    } finally {
      setStarting(false)
    }
  }, [generating, onProfileChange, starting])

  const requestGeneration = useCallback(() => {
    setCollapsed(false)
    setRequestError('')
    if (card.trim()) {
      setConfirmingCard(card)
      return
    }
    void startGeneration()
  }, [card, startGeneration])

  const loading = profile === undefined
  const ready = Boolean(card.trim())

  return (
    <section
      className="flex min-w-0 max-w-full flex-none flex-col rounded-[10px] border border-white/[0.08] bg-surface-1/60"
      aria-label="User summary"
    >
      <header className="flex min-h-[44px] items-center justify-between gap-2.5 border-b border-white/[0.08] px-3.5 py-2">
        <button
          type="button"
          className="inline-flex items-center gap-2 border-0 bg-transparent p-0 py-0.5 text-inherit"
          aria-expanded={!collapsed}
          onClick={() => setCollapsed((current) => !current)}
        >
          <span
            className={`inline-block h-[7px] w-[7px] border-r-2 border-t-2 border-fg/40 transition-transform duration-150 ${
              collapsed ? 'rotate-45' : 'rotate-[135deg]'
            }`}
            aria-hidden="true"
          />
          <span className="section-label">User summary</span>
        </button>
        <div className="flex gap-2">
          {editing ? (
            <>
              <button
                type="button"
                className="secondary-button compact"
                onClick={cancelEdit}
                disabled={saving}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary-button compact"
                onClick={() => void save()}
                disabled={saving || !draft.trim()}
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
            </>
          ) : generating ? (
            <button type="button" className="secondary-button compact" disabled>
              Generating…
            </button>
          ) : (
            <>
              <button
                type="button"
                className="primary-button compact"
                onClick={requestGeneration}
                disabled={loading}
              >
                <span aria-hidden="true">✦</span>
                Generate
              </button>
              {ready ? (
                <button type="button" className="secondary-button compact" onClick={beginEdit}>
                  Edit
                </button>
              ) : null}
            </>
          )}
        </div>
      </header>

      {collapsed ? null : (
        <>
          {confirming && !generating ? (
            <div
              className="flex items-center gap-2.5 border-b border-accent/50 bg-accent/10 px-3 py-2"
              role="alert"
            >
              <span className="text-xs text-accent" aria-hidden="true">
                ▲
              </span>
              <div className="flex min-w-0 flex-1 flex-col">
                <strong className="text-[11.5px] font-bold text-accent-hi">
                  Replace your current summary?
                </strong>
                <span className="text-[10.5px] text-fg/65">
                  Generation permanently overwrites the current content.
                </span>
              </div>
              <div className="flex flex-none gap-2">
                <button
                  type="button"
                  className="secondary-button compact"
                  onClick={() => setConfirmingCard(null)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="primary-button compact"
                  onClick={() => void startGeneration()}
                >
                  Overwrite & generate
                </button>
              </div>
            </div>
          ) : null}

          {editing ? (
            <Editor draft={draft} error={requestError} onChange={setDraft} />
          ) : loading ? (
            <div className="px-3.5 py-6 text-center text-xs text-fg/45" aria-live="polite">
              Loading user summary…
            </div>
          ) : generating && generation ? (
            <ResearchRail generation={generation} elapsed={elapsed} />
          ) : failed && generation ? (
            <GenerationFailure
              error={generation.error || requestError}
              hasExistingCard={ready}
              onRetry={() => void startGeneration()}
              onWrite={beginEdit}
            />
          ) : ready ? (
            <div className="min-w-0 max-w-full p-3.5">
              {requestError ? <p className="inline-error">{requestError}</p> : null}
              <div className="markdown-body min-w-0 max-w-full break-words text-[13px] leading-relaxed text-fg [overflow-wrap:anywhere]">
                <ReactMarkdown>{card}</ReactMarkdown>
              </div>
            </div>
          ) : (
            <EmptySummary
              error={requestError}
              onGenerate={() => void startGeneration()}
              onWrite={beginEdit}
            />
          )}
        </>
      )}
    </section>
  )
}

function Editor({
  draft,
  error,
  onChange
}: {
  draft: string
  error: string
  onChange: (value: string) => void
}): React.JSX.Element {
  return (
    <div className="min-w-0 max-w-full p-3.5">
      <textarea
        className="w-full resize-y rounded-md border border-white/15 bg-surface-0 p-3 font-sans text-[13px] leading-relaxed text-fg outline-none transition-colors focus:border-accent/60"
        value={draft}
        rows={14}
        spellCheck={false}
        autoFocus
        onChange={(event) => onChange(event.target.value)}
      />
      <div className="mt-2 flex gap-3 text-xs text-fg/45">
        <span>{draft.length.toLocaleString()} chars</span>
        {error ? <span className="inline-error">{error}</span> : null}
      </div>
    </div>
  )
}

function EmptySummary({
  error,
  onGenerate,
  onWrite
}: {
  error: string
  onGenerate: () => void
  onWrite: () => void
}): React.JSX.Element {
  return (
    <div className="flex min-h-[190px] flex-col items-center bg-white/[0.02] p-6 text-center">
      <div
        className="mb-2.5 flex h-9 w-9 items-center justify-center rounded-[10px] border border-accent/50 bg-accent/10 text-base text-accent"
        aria-hidden="true"
      >
        ✦
      </div>
      <h3 className="m-0 text-[13px] font-bold leading-snug text-fg">Build your user summary</h3>
      <p className="mb-3 mt-1 max-w-[430px] text-[11.5px] leading-normal text-fg/45">
        Let GLM-5.2 explore your knowledge graph to understand your work, interests, projects, and
        relationships.
      </p>
      <div className="flex items-center gap-3">
        <button type="button" className="primary-button compact" onClick={onGenerate}>
          Generate from graph
        </button>
        <button type="button" className="link-button" onClick={onWrite}>
          Write it by hand instead
        </button>
      </div>
      {error ? <span className="inline-error mt-2.5">{error}</span> : null}
    </div>
  )
}

function ResearchRail({
  generation,
  elapsed
}: {
  generation: ProfileGenerationStatus
  elapsed: number
}): React.JSX.Element {
  const activeIndex = RESEARCH_STEPS.findIndex((step) => step.phase === generation.phase)
  const progress = Math.max(8, ((Math.max(activeIndex, 0) + 0.55) / RESEARCH_STEPS.length) * 100)

  return (
    <div className="grid min-h-[242px] grid-cols-[176px_minmax(0,1fr)]">
      <aside
        className="border-r border-white/[0.08] bg-surface-2/60 px-3 py-3.5"
        aria-label="Profile research progress"
      >
        <span className="mb-2.5 block text-[9.5px] font-extrabold uppercase tracking-[0.09em] text-fg/45">
          Profile research
        </span>
        {RESEARCH_STEPS.map((step, index) => {
          const state = index < activeIndex ? 'done' : index === activeIndex ? 'active' : 'pending'
          return (
            <div
              className={`relative py-1.5 pl-5 text-[10.5px] leading-tight after:absolute after:-bottom-1 after:left-[5px] after:top-[19px] after:w-px after:bg-white/[0.08] after:content-[''] last:after:hidden ${
                state === 'done'
                  ? 'text-fg/65'
                  : state === 'active'
                    ? 'font-semibold text-accent-hi'
                    : 'text-fg/45'
              }`}
              key={step.phase}
            >
              <span
                className={`absolute left-0.5 top-[9px] h-[7px] w-[7px] rounded-full ${
                  state === 'done'
                    ? 'bg-accent shadow-[0_0_0_3px_rgba(245,197,24,0.13)]'
                    : state === 'active'
                      ? 'animate-[ring-pulse_1.4s_ease-in-out_infinite] bg-accent shadow-[0_0_0_4px_rgba(245,197,24,0.2)] [--ring:rgba(245,197,24,0.35)]'
                      : 'bg-fg/25'
                }`}
                aria-hidden="true"
              />
              {step.label}
            </div>
          )
        })}
      </aside>
      <div className="flex min-w-0 flex-col p-4">
        <div className="flex items-start justify-between gap-3.5">
          <div>
            <h3 className="m-0 text-[13px] font-bold leading-snug text-fg">
              {phaseTitle(generation.phase)}
            </h3>
            <p className="mb-0 mt-1 text-[10.5px] leading-normal text-fg/45">
              {generation.activity || 'Exploring the knowledge graph'}
            </p>
          </div>
          <span className="flex-none rounded-full border border-accent/30 bg-accent/10 px-2 py-1 text-[9.5px] font-extrabold text-accent-hi">
            {generation.nodes_read.toLocaleString()} nodes explored
          </span>
        </div>
        <div className="my-3.5 h-0.5 overflow-hidden rounded-full bg-white/[0.06]" aria-hidden="true">
          <span
            className="block h-full min-w-[8%] bg-gradient-to-r from-accent/60 to-accent transition-[width] duration-300 motion-reduce:transition-none"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div
          className="flex min-w-0 items-center gap-2 rounded-md border border-white/[0.08] bg-white/[0.03] px-2.5 py-2 text-[10.5px] text-fg/65"
          role="status"
          aria-live="polite"
        >
          <span
            className="h-[11px] w-[11px] flex-none animate-spin rounded-full border-2 border-white/15 border-t-accent motion-reduce:animate-none"
            aria-hidden="true"
          />
          <span className="truncate">{generation.activity || 'Researching your graph'}</span>
        </div>
        <div className="mt-auto flex justify-between pt-3.5 text-[10px] text-fg/45">
          <span>GLM-5.2 · agentic graph research</span>
          <span>
            {generation.tool_calls} tool {generation.tool_calls === 1 ? 'call' : 'calls'} ·{' '}
            {formatElapsed(elapsed)}
          </span>
        </div>
      </div>
    </div>
  )
}

function GenerationFailure({
  error,
  hasExistingCard,
  onRetry,
  onWrite
}: {
  error: string
  hasExistingCard: boolean
  onRetry: () => void
  onWrite: () => void
}): React.JSX.Element {
  return (
    <div
      className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-3 bg-bad/5 p-4"
      role="alert"
    >
      <div
        className="flex h-6 w-6 items-center justify-center rounded-full border border-bad/40 bg-bad/10 text-[11px] font-black text-bad"
        aria-hidden="true"
      >
        !
      </div>
      <div>
        <h3 className="m-0 text-[13px] font-bold leading-snug text-fg">
          Profile generation stopped
        </h3>
        <p className="mb-0 mt-1 text-[11px] leading-normal text-fg/65">
          {error || 'The profile agent could not finish this run.'}
        </p>
        {hasExistingCard ? (
          <p className="mb-0 mt-1 text-[11px] leading-normal text-fg/65">
            Your previous summary is still saved.
          </p>
        ) : null}
      </div>
      <div className="flex gap-2">
        <button type="button" className="primary-button compact" onClick={onRetry}>
          Try again
        </button>
        {!hasExistingCard ? (
          <button type="button" className="secondary-button compact" onClick={onWrite}>
            Write by hand
          </button>
        ) : null}
      </div>
    </div>
  )
}

function phaseTitle(phase: ProfileGenerationPhase): string {
  switch (phase) {
    case 'surveying':
      return 'Surveying your knowledge graph'
    case 'interests':
      return 'Mapping your interests'
    case 'projects_people':
      return 'Connecting projects to people'
    case 'career_studies':
      return 'Tracing career and studies'
    case 'synthesizing':
      return 'Writing your user profile'
    default:
      return 'Building your user profile'
  }
}

function formatElapsed(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(whole / 60)
  const remainder = whole % 60
  return minutes > 0 ? `${minutes}m ${remainder.toString().padStart(2, '0')}s` : `${remainder}s`
}
