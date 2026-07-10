import {
  deleteConnectorData,
  disableConnector,
  enableConnector,
  updateConnectorConfig,
  type ConnectorInfo,
  type SyncStatus
} from '../lib/sidecar'
import { sourceColor } from '../lib/graph'
import { useCallback, useMemo, useState } from 'react'

interface ConnectionsPanelProps {
  connectors: ConnectorInfo[] | null
  syncing: boolean
  syncStatus: SyncStatus | null
  syncError: string
  onSync: () => void
  onCancelSync: () => void
}

type PendingAction = 'enable' | 'disable' | 'delete' | 'config'
type ConnectorStatus = 'syncing' | 'attention' | 'active' | 'off'

const APPLE_NOTES_CONNECTOR_ID = 'apple_notes'
const FOLDERS_CONNECTOR_ID = 'local_folders'

export function ConnectionsPanel({
  connectors,
  syncing,
  syncStatus,
  syncError,
  onSync,
  onCancelSync
}: ConnectionsPanelProps): React.JSX.Element {
  const [rowErrors, setRowErrors] = useState<Record<string, string>>({})
  const [pendingById, setPendingById] = useState<Record<string, PendingAction | undefined>>({})
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set())
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [configDrafts, setConfigDrafts] = useState<Record<string, Record<string, string>>>({})

  const setRowError = useCallback((connectorId: string, message: string) => {
    setRowErrors((current) => ({ ...current, [connectorId]: message }))
  }, [])

  const clearRowError = useCallback((connectorId: string) => {
    setRowErrors((current) => {
      if (!current[connectorId]) {
        return current
      }
      const next = { ...current }
      delete next[connectorId]
      return next
    })
  }, [])

  // state refresh arrives over the /updates websocket; actions only fire the
  // request and surface errors
  const runConnectorAction = useCallback(
    async (connectorId: string, action: PendingAction, request: () => Promise<void>) => {
      setPendingById((current) => ({ ...current, [connectorId]: action }))
      clearRowError(connectorId)

      try {
        await request()
        setConfirmDeleteId(null)
      } catch (error) {
        setRowError(connectorId, error instanceof Error ? error.message : 'Connector action failed')
      } finally {
        setPendingById((current) => ({ ...current, [connectorId]: undefined }))
      }
    },
    [clearRowError, setRowError]
  )

  const toggleExpanded = useCallback((connectorId: string) => {
    setExpandedIds((current) => {
      const next = new Set(current)
      if (next.has(connectorId)) {
        next.delete(connectorId)
      } else {
        next.add(connectorId)
      }
      return next
    })
  }, [])

  const updateConfigDraft = useCallback((connectorId: string, key: string, value: string) => {
    setConfigDrafts((current) => ({
      ...current,
      [connectorId]: { ...current[connectorId], [key]: value }
    }))
  }, [])

  const saveConfig = useCallback(
    async (connector: ConnectorInfo) => {
      const drafts = configDrafts[connector.connector_id] ?? {}
      // PUT replaces the whole config dict, so merge drafts over current values
      const payload: Record<string, unknown> = { ...connector.config }
      for (const [key, draft] of Object.entries(drafts)) {
        payload[key] = parseConfigValue(draft, connector.config[key])
      }

      await runConnectorAction(connector.connector_id, 'config', async () => {
        await updateConnectorConfig(connector.connector_id, payload)
        setConfigDrafts((current) => {
          const next = { ...current }
          delete next[connector.connector_id]
          return next
        })
      })
    },
    [configDrafts, runConnectorAction]
  )

  const saveFolders = useCallback(
    async (connector: ConnectorInfo, folders: string[]) => {
      await runConnectorAction(connector.connector_id, 'config', () =>
        updateConnectorConfig(connector.connector_id, { ...connector.config, folders })
      )
    },
    [runConnectorAction]
  )

  const sortedConnectors = useMemo(() => {
    return [...(connectors ?? [])].sort((left, right) => {
      if (left.connector_id === APPLE_NOTES_CONNECTOR_ID) {
        return -1
      }
      if (right.connector_id === APPLE_NOTES_CONNECTOR_ID) {
        return 1
      }
      return left.label.localeCompare(right.label)
    })
  }, [connectors])

  const activeCount = useMemo(
    () => sortedConnectors.filter((c) => c.enabled && c.available).length,
    [sortedConnectors]
  )
  const totalDocs = useMemo(
    () => sortedConnectors.reduce((sum, c) => sum + Math.max(0, c.docs), 0),
    [sortedConnectors]
  )
  const syncRunning = syncing || sortedConnectors.some((c) => c.syncing)
  const staleIdle = !syncRunning && syncStatus?.stale === true

  return (
    <section
      className="flex min-h-0 flex-col rounded-[10px] border border-white/[0.08] bg-surface-1/60 p-3.5"
      aria-label="Connections"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-2.5">
          <span className="section-label">Sources</span>
          {connectors !== null ? (
            <span className="whitespace-nowrap text-[11px] font-medium text-fg/45">
              {activeCount} active · {formatCount(totalDocs)} docs indexed
            </span>
          ) : null}
        </div>
        <div className="flex flex-none gap-2">
          {syncRunning ? (
            <button
              className="secondary-button compact"
              type="button"
              onClick={onCancelSync}
              title="Stop the running sync; the previous index stays active"
            >
              Cancel
            </button>
          ) : null}
          <button
            className={`secondary-button compact${staleIdle ? ' sync-stale' : ''}`}
            type="button"
            disabled={syncRunning}
            onClick={onSync}
            title="Import every enabled source and rebuild the knowledge index"
          >
            {syncRunning ? <span className="inline-spinner" aria-hidden="true" /> : null}
            {syncRunning ? globalSyncLabel(syncStatus) : 'Sync'}
          </button>
        </div>
      </div>

      {syncError || syncStatus?.error ? (
        <div
          className="mt-2.5 rounded-md border border-bad/40 bg-[rgba(60,20,20,0.72)] px-2.5 py-2 text-xs font-medium leading-snug text-[#ffc7c2]"
          role="alert"
        >
          {syncError || (syncStatus?.error ? `Last sync failed: ${syncStatus.error}` : '')}
        </div>
      ) : null}

      {staleIdle ? (
        <div
          className="mt-2.5 rounded-md border border-accent/50 bg-accent/10 px-2.5 py-2 text-xs font-medium leading-snug text-accent-hi"
          role="status"
        >
          Graph is out of date with the enabled sources — press Sync to rebuild.
        </div>
      ) : null}

      {connectors === null ? (
        <div className="mt-3 flex h-fit flex-col gap-2" aria-hidden="true">
          {[0, 1, 2, 3].map((index) => (
            <div
              className="flex items-center gap-3 rounded-lg border border-white/[0.08] bg-surface-2/60 px-3.5 py-2.5"
              key={index}
            >
              <span className="h-8 w-8 flex-none animate-pulse rounded-lg border border-white/[0.08] bg-white/[0.04]" />
              <span className="grid flex-1 gap-1.5">
                <span className="h-2.5 animate-pulse rounded bg-white/[0.05]" />
                <span className="h-2.5 w-[45%] animate-pulse rounded bg-white/[0.05]" />
              </span>
            </div>
          ))}
        </div>
      ) : sortedConnectors.length === 0 ? (
        <div className="empty-surface mt-3">No connectors reported</div>
      ) : (
        <div className="mt-3 flex h-fit flex-col gap-2">
          {sortedConnectors.map((connector) => (
            <ConnectorRow
              configDrafts={configDrafts[connector.connector_id] ?? {}}
              connector={connector}
              confirmDelete={confirmDeleteId === connector.connector_id}
              error={rowErrors[connector.connector_id] ?? ''}
              expanded={expandedIds.has(connector.connector_id)}
              key={connector.connector_id}
              pendingAction={pendingById[connector.connector_id]}
              onCancelDelete={() => setConfirmDeleteId(null)}
              onConfirmDelete={() =>
                void runConnectorAction(connector.connector_id, 'delete', () =>
                  deleteConnectorData(connector.connector_id)
                )
              }
              onRequestDelete={() => setConfirmDeleteId(connector.connector_id)}
              onSaveConfig={() => void saveConfig(connector)}
              onSaveFolders={(folders) => void saveFolders(connector, folders)}
              onToggle={() =>
                void runConnectorAction(
                  connector.connector_id,
                  connector.enabled ? 'disable' : 'enable',
                  () =>
                    connector.enabled
                      ? disableConnector(connector.connector_id)
                      : enableConnector(connector.connector_id)
                )
              }
              onToggleExpanded={() => toggleExpanded(connector.connector_id)}
              onUpdateConfig={(key, value) => updateConfigDraft(connector.connector_id, key, value)}
            />
          ))}
        </div>
      )}
    </section>
  )
}

function ConnectorRow({
  connector,
  pendingAction,
  error,
  expanded,
  configDrafts,
  confirmDelete,
  onToggle,
  onToggleExpanded,
  onRequestDelete,
  onConfirmDelete,
  onCancelDelete,
  onUpdateConfig,
  onSaveConfig,
  onSaveFolders
}: {
  connector: ConnectorInfo
  pendingAction?: PendingAction
  error: string
  expanded: boolean
  configDrafts: Record<string, string>
  confirmDelete: boolean
  onToggle: () => void
  onToggleExpanded: () => void
  onRequestDelete: () => void
  onConfirmDelete: () => void
  onCancelDelete: () => void
  onUpdateConfig: (key: string, value: string) => void
  onSaveConfig: () => void
  onSaveFolders: (folders: string[]) => void
}): React.JSX.Element {
  const status = connectorStatus(connector)
  const signInRequired = connector.kind === 'oauth' && !connector.available
  const isFolders = connector.connector_id === FOLDERS_CONNECTOR_ID
  const toggleDisabled = pendingAction === 'enable' || pendingAction === 'disable'
  const deleteDisabled = pendingAction === 'delete' || connector.syncing || connector.docs === 0
  const configKeys = Object.keys(connector.config).filter(
    (key) => isEditableConfigValue(connector.config[key]) && !(isFolders && key === 'folders')
  )
  const configDirty = Object.entries(configDrafts).some(
    ([key, draft]) => draft !== formatConfigValue(connector.config[key])
  )
  const detailId = `connector-detail-${connector.connector_id}`
  const progressFraction =
    connector.syncing && connector.sync_total > 0
      ? Math.min(1, connector.sync_done / connector.sync_total)
      : null

  return (
    <article
      className={`overflow-hidden rounded-lg border bg-surface-2/60 transition-colors ${
        expanded ? 'border-white/15' : 'border-white/[0.08]'
      }`}
    >
      <div
        className="grid cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 px-3.5 py-2.5 hover:bg-white/[0.02] focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-accent/60"
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-controls={detailId}
        onClick={onToggleExpanded}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            onToggleExpanded()
          }
        }}
      >
        <span
          className="flex h-8 w-8 flex-none items-center justify-center rounded-lg border border-[color-mix(in_srgb,var(--glyph-color,#8a8a8a)_24%,transparent)] bg-[color-mix(in_srgb,var(--glyph-color,#8a8a8a)_12%,transparent)] text-[color:var(--glyph-color,#8a8a8a)]"
          style={{ '--glyph-color': sourceColor(connector.connector_id) } as React.CSSProperties}
          aria-hidden="true"
        >
          <ConnectorGlyph connectorId={connector.connector_id} />
        </span>

        <span className="flex min-w-0 flex-col gap-0.5">
          <span className="flex items-center gap-2 text-[13px] font-semibold leading-tight text-fg">
            {connector.label}
            {connector.kind === 'oauth' ? (
              <span className="rounded-full border border-white/[0.08] px-1.5 py-[3px] text-[9px] font-extrabold uppercase leading-none tracking-wider text-fg/45">
                OAuth
              </span>
            ) : null}
          </span>
          <span className="flex min-w-0 items-center gap-1.5 truncate text-xs text-fg/65">
            <span
              className={`h-[7px] w-[7px] flex-none rounded-full ${STATUS_DOT_CLASSES[status]}`}
              aria-hidden="true"
            />
            {subtitleFor(connector, status, signInRequired)}
          </span>
        </span>

        <span className="flex items-center gap-3">
          <button
            className={`relative h-5 w-9 flex-none cursor-pointer rounded-full border p-0 transition-colors ${
              connector.enabled ? 'border-ok/40 bg-ok/20' : 'border-white/[0.08] bg-white/[0.03]'
            }`}
            type="button"
            role="switch"
            aria-checked={connector.enabled}
            aria-label={`${connector.label} ${connector.enabled ? 'enabled' : 'disabled'}`}
            disabled={toggleDisabled}
            title={
              connector.enabled
                ? 'Disable — keeps stored data; the next sync drops it from the graph and retrieval'
                : 'Enable — the next sync imports and indexes this source'
            }
            onClick={(event) => {
              event.stopPropagation()
              onToggle()
            }}
          >
            <span
              className={`absolute left-0.5 top-0.5 h-3.5 w-3.5 rounded-full transition-transform ${
                connector.enabled ? 'translate-x-4 bg-ok' : 'bg-fg/40'
              }`}
              aria-hidden="true"
            />
          </button>
          <span
            className={`flex text-fg/45 transition-transform duration-150 ${expanded ? 'rotate-180' : ''}`}
            aria-hidden="true"
          >
            <svg viewBox="0 0 12 12" width="12" height="12">
              <path
                d="M3 4.5 6 7.5 9 4.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
        </span>
      </div>

      {connector.syncing ? (
        <div className="h-0.5 overflow-hidden bg-white/[0.06]" aria-hidden="true">
          <span
            className={`block h-full bg-accent transition-[width] duration-300 ${
              progressFraction === null
                ? 'w-2/5 animate-[indeterminate_1.1s_ease-in-out_infinite]'
                : ''
            }`}
            style={
              progressFraction === null
                ? undefined
                : { width: `${Math.round(progressFraction * 100)}%` }
            }
          />
        </div>
      ) : null}

      {expanded ? (
        <div className="grid gap-3 border-t border-white/[0.08] px-3.5 pb-3.5 pt-3" id={detailId}>
          {!signInRequired && connector.detail ? (
            <p
              className={`m-0 text-xs leading-normal ${
                status === 'attention' ? 'text-accent-hi' : 'text-fg/65'
              }`}
            >
              {connector.detail}
            </p>
          ) : null}

          {signInRequired ? <SignInBlock connector={connector} /> : null}

          {isFolders ? (
            <FolderManager
              connector={connector}
              pending={pendingAction === 'config'}
              onSaveFolders={onSaveFolders}
            />
          ) : null}

          {configKeys.length > 0 ? (
            <div className="grid gap-2">
              <span className={DETAIL_LABEL}>Configuration</span>
              {configKeys.map((key) => {
                const inputId = `connector-config-${connector.connector_id}-${key}`
                const meta = configFieldMeta(key)
                const draft = configDrafts[key] ?? formatConfigValue(connector.config[key])

                return (
                  <div
                    className="grid grid-cols-[190px_minmax(0,1fr)] items-center gap-2.5"
                    key={key}
                  >
                    <label className="text-[11px] font-semibold text-fg/45" htmlFor={inputId}>
                      {meta.label}
                    </label>
                    <input
                      className="min-w-0 rounded-md border border-white/[0.08] bg-white/[0.03] px-2.5 py-1.5 text-xs text-fg outline-none transition-colors focus:border-accent/60"
                      id={inputId}
                      type="text"
                      value={draft}
                      placeholder={meta.placeholder}
                      spellCheck={false}
                      onChange={(event) => onUpdateConfig(key, event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && configDirty && pendingAction !== 'config') {
                          onSaveConfig()
                        }
                      }}
                    />
                  </div>
                )
              })}
              <div className="flex justify-end">
                <button
                  className="secondary-button compact"
                  type="button"
                  disabled={!configDirty || pendingAction === 'config'}
                  onClick={onSaveConfig}
                >
                  {pendingAction === 'config' ? 'Saving' : 'Save changes'}
                </button>
              </div>
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-2">
            <span className="flex-1" />
            <button
              className="secondary-button compact danger"
              type="button"
              disabled={deleteDisabled}
              onClick={onRequestDelete}
              title={`Delete all ${formatCount(connector.docs)} stored ${connector.label} documents from the index`}
            >
              {pendingAction === 'delete' ? 'Deleting' : 'Delete data'}
            </button>
          </div>

          {confirmDelete ? (
            <div
              className="flex flex-wrap items-center gap-2 rounded-md border border-accent/50 bg-white/[0.03] p-2"
              role="group"
              aria-label={`Delete ${connector.label} data`}
            >
              <span className="mr-0.5 text-xs font-bold text-accent-hi">
                Delete all {formatCount(connector.docs)} stored document
                {connector.docs === 1 ? '' : 's'}? A future sync can re-import them.
              </span>
              <button
                className="secondary-button compact danger"
                type="button"
                onClick={onConfirmDelete}
              >
                Delete
              </button>
              <button
                className="icon-button"
                type="button"
                aria-label="Cancel delete"
                onClick={onCancelDelete}
              >
                &times;
              </button>
            </div>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <div
          className="rounded-md border border-bad/40 bg-[rgba(60,20,20,0.72)] px-2.5 py-2 text-xs font-medium leading-snug text-[#ffc7c2]"
          role="alert"
        >
          {error}
        </div>
      ) : null}
    </article>
  )
}

const STATUS_DOT_CLASSES: Record<ConnectorStatus, string> = {
  active: 'bg-ok shadow-[0_0_0_3px_rgba(87,217,138,0.12)]',
  attention: 'bg-accent-hi shadow-[0_0_0_3px_rgba(245,197,24,0.13)]',
  off: 'bg-fg/40',
  syncing: 'bg-accent animate-[ring-pulse_1.2s_ease-in-out_infinite] [--ring:rgba(245,197,24,0.2)]'
}

const DETAIL_LABEL = 'text-[10px] font-extrabold uppercase tracking-[0.07em] text-fg/45'

function FolderManager({
  connector,
  pending,
  onSaveFolders
}: {
  connector: ConnectorInfo
  pending: boolean
  onSaveFolders: (folders: string[]) => void
}): React.JSX.Element {
  const folders = folderList(connector.config)

  const addFolder = async (): Promise<void> => {
    const chosen = await chooseDirectory()
    const trimmed = chosen?.trim() ?? ''
    if (!trimmed || folders.includes(trimmed)) {
      return
    }
    onSaveFolders([...folders, trimmed])
  }

  return (
    <div className="grid gap-2">
      <span className={DETAIL_LABEL}>Attached folders</span>
      {folders.length === 0 ? (
        <p className="m-0 text-xs leading-normal text-fg/45">
          No folders attached yet. Documents inside attached folders (.md, .txt, .pdf, .docx, .rtf)
          are indexed into the corpus and re-synced automatically.
        </p>
      ) : (
        <ul className="m-0 grid list-none gap-1.5 p-0">
          {folders.map((folder) => (
            <li
              className="flex items-center gap-2 rounded-md border border-white/[0.08] bg-white/[0.03] py-1.5 pl-2.5 pr-2 text-fg/45"
              key={folder}
            >
              <svg viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
                <path
                  d="M2.5 6a2 2 0 0 1 2-2h3.2l1.6 2h6.2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                />
              </svg>
              <code
                className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-fg"
                title={folder}
              >
                {folder}
              </code>
              <button
                className="icon-button"
                type="button"
                aria-label={`Detach ${folder}`}
                title="Detach folder — already-indexed documents stay until the next sync"
                disabled={pending}
                onClick={() => onSaveFolders(folders.filter((item) => item !== folder))}
              >
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex">
        <button
          className="secondary-button compact"
          type="button"
          disabled={pending}
          onClick={() => void addFolder()}
        >
          {pending ? <span className="inline-spinner" aria-hidden="true" /> : null}
          {pending ? 'Saving' : 'Add folder…'}
        </button>
      </div>
    </div>
  )
}

async function chooseDirectory(): Promise<string | null> {
  const picker = window.api?.chooseDirectory
  if (picker) {
    try {
      return await picker()
    } catch {
      return null
    }
  }
  // non-Electron fallback (e.g. dev preview in a plain browser)
  return window.prompt('Folder path to attach (e.g. /Users/you/Documents/Research)')
}

function folderList(config: Record<string, unknown>): string[] {
  const raw = config.folders
  return Array.isArray(raw) ? raw.filter((item): item is string => typeof item === 'string') : []
}

function SignInBlock({ connector }: { connector: ConnectorInfo }): React.JSX.Element {
  const [copied, setCopied] = useState(false)
  const command = signInCommand(connector)

  return (
    <div className="grid gap-2.5 rounded-md border border-white/[0.08] bg-white/[0.03] px-3 py-2.5">
      <p className="m-0 text-xs leading-normal text-fg/65">
        Sign in runs a one-time local OAuth flow in your terminal and stores the tokens in{' '}
        <code className="text-[11.5px] text-fg">core/.env</code>. Restart the sidecar afterwards.
      </p>
      <div className="flex items-center gap-2">
        <code className="flex-1 overflow-x-auto whitespace-nowrap rounded-md border border-white/[0.08] bg-black/35 px-2.5 py-2 font-mono text-[11.5px] text-fg">
          {command}
        </code>
        <button
          className="secondary-button compact"
          type="button"
          onClick={() => {
            void navigator.clipboard.writeText(command).then(() => {
              setCopied(true)
              window.setTimeout(() => setCopied(false), 1600)
            })
          }}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </div>
  )
}

function ConnectorGlyph({ connectorId }: { connectorId: string }): React.JSX.Element {
  const common = {
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.6,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const
  }

  switch (connectorId) {
    case 'apple_notes':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <rect x="3.5" y="3" width="13" height="14" rx="2.5" {...common} />
          <path d="M6.5 7.5h7M6.5 10.5h7M6.5 13.5h4" {...common} />
        </svg>
      )
    case 'obsidian':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <path d="M10 2.5 15.5 7l-1.8 9.5H6.3L4.5 7 10 2.5Z" {...common} />
          <path d="M10 2.5 8.8 16.5" {...common} />
        </svg>
      )
    case 'imessage':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <path
            d="M10 3.5c-4 0-7 2.6-7 5.8 0 1.9 1 3.5 2.7 4.6-.1.9-.5 1.9-1.2 2.6 1.4-.2 2.6-.7 3.4-1.3.7.2 1.4.3 2.1.3 4 0 7-2.6 7-5.9s-3-6.1-7-6.1Z"
            {...common}
          />
        </svg>
      )
    case 'claude_mem':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <path d="M10 3v14M4 6.5l12 7M16 6.5l-12 7" {...common} />
        </svg>
      )
    case 'codex_mem':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <rect x="2.5" y="4" width="15" height="12" rx="2" {...common} />
          <path d="M6 8.5 8.5 10.5 6 12.5M10.5 13h3.5" {...common} />
        </svg>
      )
    case 'local_folders':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <path
            d="M2.5 6a2 2 0 0 1 2-2h3.2l1.6 2h6.2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2Z"
            {...common}
          />
          <path d="M2.5 9h15" {...common} />
        </svg>
      )
    case 'gmail':
    case 'outlook':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <rect x="2.5" y="4.5" width="15" height="11" rx="2" {...common} />
          <path d="m3.5 6.5 6.5 4.8 6.5-4.8" {...common} />
        </svg>
      )
    default:
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <circle cx="10" cy="10" r="6.5" {...common} />
          <path d="M10 6.5v3.5l2.5 2" {...common} />
        </svg>
      )
  }
}

function connectorStatus(connector: ConnectorInfo): ConnectorStatus {
  if (connector.syncing) {
    return 'syncing'
  }
  if (!connector.enabled) {
    return 'off'
  }
  return connector.available ? 'active' : 'attention'
}

function subtitleFor(
  connector: ConnectorInfo,
  status: ConnectorStatus,
  signInRequired: boolean
): string {
  if (status === 'syncing') {
    return syncProgressLabel(connector)
  }
  if (status === 'attention') {
    return signInRequired ? 'Sign in required' : 'Needs attention'
  }
  if (status === 'off') {
    return connector.docs > 0 ? `Off · ${formatCount(connector.docs)} docs stored` : 'Off'
  }
  if (
    connector.connector_id === FOLDERS_CONNECTOR_ID &&
    folderList(connector.config).length === 0
  ) {
    return 'No folders attached yet'
  }
  if (connector.docs === 0) {
    return 'Ready · no documents yet'
  }
  return `${formatCount(connector.docs)} docs · synced ${formatRelativeTime(connector.last_sync)}`
}

function signInCommand(connector: ConnectorInfo): string {
  if (connector.connector_id === 'gmail' || connector.connector_id === 'outlook') {
    return `cd core && uv run python -m taurine_core.connectors.email_auth ${connector.connector_id}`
  }

  const match = connector.detail.match(/run: (.+)$/)
  return match ? match[1] : ''
}

const CONFIG_FIELD_META: Record<string, { label: string; placeholder: string }> = {
  vault: { label: 'Vault path', placeholder: '/Users/name/Obsidian/Vault' },
  glob: { label: 'Memory file glob', placeholder: '~/.claude/**/MEMORY.md' },
  globs: { label: 'Memory file globs (comma-separated)', placeholder: '~/.codex/memory/*.md' },
  db: { label: 'Database path', placeholder: '~/Library/Messages/chat.db' },
  contacts_glob: {
    label: 'Contacts database glob',
    placeholder: '~/Library/Application Support/AddressBook/**/*.abcddb'
  },
  window_days: { label: 'Sync window (days)', placeholder: '30' }
}

function configFieldMeta(key: string): { label: string; placeholder: string } {
  const known = CONFIG_FIELD_META[key]
  if (known) {
    return known
  }

  const label = key
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part, index) => (index === 0 ? part.charAt(0).toUpperCase() + part.slice(1) : part))
    .join(' ')
  return { label, placeholder: '' }
}

function isEditableConfigValue(value: unknown): boolean {
  return (
    typeof value === 'string' ||
    typeof value === 'number' ||
    (Array.isArray(value) && value.every((item) => typeof item === 'string'))
  )
}

function formatConfigValue(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number') {
    return String(value)
  }
  if (Array.isArray(value)) {
    return value.join(', ')
  }
  return ''
}

/** Parse a draft back to the type of the value it replaces. */
function parseConfigValue(draft: string, current: unknown): unknown {
  const trimmed = draft.trim()
  if (typeof current === 'number') {
    const parsed = Number(trimmed)
    return Number.isFinite(parsed) ? parsed : current
  }
  if (Array.isArray(current)) {
    return trimmed.length === 0
      ? []
      : trimmed
          .split(',')
          .map((item) => item.trim())
          .filter(Boolean)
  }
  return trimmed
}

function formatCount(value: number): string {
  return new Intl.NumberFormat().format(Math.max(0, value))
}

function formatRelativeTime(epoch: number | null | undefined): string {
  if (!epoch) {
    return 'never'
  }

  const millis = epoch > 1_000_000_000_000 ? epoch : epoch * 1000
  const diffSeconds = Math.round((Date.now() - millis) / 1000)
  const absSeconds = Math.abs(diffSeconds)

  if (absSeconds < 60) {
    return 'just now'
  }

  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['year', 31536000],
    ['month', 2592000],
    ['week', 604800],
    ['day', 86400],
    ['hour', 3600],
    ['minute', 60]
  ]
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })

  for (const [unit, secondsPerUnit] of units) {
    if (absSeconds >= secondsPerUnit) {
      return formatter.format(-Math.round(diffSeconds / secondsPerUnit), unit)
    }
  }

  return 'just now'
}

function globalSyncLabel(status: SyncStatus | null): string {
  if (status?.phase === 'indexing') {
    return status.detail ? `Indexing — ${status.detail}` : 'Indexing'
  }
  if (status?.detail) {
    return `Syncing — ${status.detail}`
  }
  return 'Syncing'
}

function syncProgressLabel(connector: ConnectorInfo): string {
  if (connector.sync_total > 0) {
    const phase = connector.sync_phase === 'enriching' ? 'Enriching' : 'Indexing'
    return `${phase} ${connector.sync_done}/${connector.sync_total}`
  }
  if (connector.sync_phase === 'extracting') {
    return 'Extracting'
  }
  return 'Syncing'
}
