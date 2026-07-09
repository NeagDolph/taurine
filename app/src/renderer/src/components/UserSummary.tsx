import { useCallback, useEffect, useMemo, useState } from 'react'
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

  const elapsed = useMemo(() => {
    if (!generation) {
      return 0
    }
    if (generation.running && generation.started_at !== null) {
      return Math.max(generation.elapsed_s, now / 1000 - generation.started_at)
    }
    return generation.elapsed_s
  }, [generation, now])

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
    <section className="panel-card user-summary" aria-label="User summary">
      <header className="panel-card-header user-summary-header">
        <button
          type="button"
          className="collapse-toggle"
          aria-expanded={!collapsed}
          onClick={() => setCollapsed((current) => !current)}
        >
          <span className={`chevron ${collapsed ? '' : 'open'}`} aria-hidden="true" />
          <span className="section-label">User summary</span>
        </button>
        <div className="panel-card-actions">
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
                className="primary-button compact user-summary-generate"
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
            <div className="user-summary-confirm" role="alert">
              <span className="user-summary-confirm-icon" aria-hidden="true">
                ▲
              </span>
              <div className="user-summary-confirm-copy">
                <strong>Replace your current summary?</strong>
                <span>Generation permanently overwrites the current content.</span>
              </div>
              <div className="user-summary-confirm-actions">
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
            <div className="user-summary-loading" aria-live="polite">
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
            <div className="user-summary-body">
              {requestError ? <p className="inline-error">{requestError}</p> : null}
              <div className="user-summary-text markdown-body">
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
    <div className="user-summary-body">
      <textarea
        className="user-summary-editor"
        value={draft}
        rows={14}
        spellCheck={false}
        autoFocus
        onChange={(event) => onChange(event.target.value)}
      />
      <div className="user-summary-meta">
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
    <div className="user-summary-empty">
      <div className="user-summary-empty-glyph" aria-hidden="true">
        ✦
      </div>
      <h3>Build your user summary</h3>
      <p>
        Let GLM-5.2 explore your knowledge graph to understand your work, interests, projects, and
        relationships.
      </p>
      <div className="user-summary-empty-actions">
        <button type="button" className="primary-button compact" onClick={onGenerate}>
          Generate from graph
        </button>
        <button type="button" className="link-button" onClick={onWrite}>
          Write it by hand instead
        </button>
      </div>
      {error ? <span className="inline-error">{error}</span> : null}
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
    <div className="user-summary-research" aria-live="polite">
      <aside className="user-summary-research-rail" aria-label="Profile research progress">
        <span className="user-summary-rail-label">Profile research</span>
        {RESEARCH_STEPS.map((step, index) => {
          const state = index < activeIndex ? 'done' : index === activeIndex ? 'active' : 'pending'
          return (
            <div className={`user-summary-research-step ${state}`} key={step.phase}>
              <span aria-hidden="true" />
              {step.label}
            </div>
          )
        })}
      </aside>
      <div className="user-summary-research-work">
        <div className="user-summary-research-heading">
          <div>
            <h3>{phaseTitle(generation.phase)}</h3>
            <p>{generation.activity || 'Exploring the knowledge graph'}</p>
          </div>
          <span className="user-summary-node-count">
            {generation.nodes_read.toLocaleString()} nodes explored
          </span>
        </div>
        <div className="user-summary-progress" aria-hidden="true">
          <span style={{ width: `${progress}%` }} />
        </div>
        <div className="user-summary-activity-row">
          <span className="user-summary-activity-spinner" aria-hidden="true" />
          <span>{generation.activity || 'Researching your graph'}</span>
        </div>
        <div className="user-summary-research-meta">
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
    <div className="user-summary-failure" role="alert">
      <div className="user-summary-failure-glyph" aria-hidden="true">
        !
      </div>
      <div>
        <h3>Profile generation stopped</h3>
        <p>{error || 'The profile agent could not finish this run.'}</p>
        {hasExistingCard ? <p>Your previous summary is still saved.</p> : null}
      </div>
      <div className="user-summary-failure-actions">
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
