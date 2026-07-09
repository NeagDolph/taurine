import {
  deleteConnectorData,
  disableConnector,
  enableConnector,
  syncAllConnectors,
  syncConnector,
  updateConnectorConfig,
  type ConnectorInfo
} from '../lib/sidecar'
import { sourceColor } from '../lib/graph'
import { useCallback, useMemo, useState } from 'react'

interface ConnectionsPanelProps {
  connectors: ConnectorInfo[] | null
  reenriching: boolean
  reenrichError: string
  reenrichDisabled: boolean
  reenrichTarget: string | null
  onReenrich: (source?: string) => void
}

type PendingAction = 'enable' | 'disable' | 'delete' | 'sync' | 'config' | 'sync-all'
type ConnectorStatus = 'syncing' | 'attention' | 'active' | 'off'

const APPLE_NOTES_CONNECTOR_ID = 'apple_notes'
const FOLDERS_CONNECTOR_ID = 'local_folders'

export function ConnectionsPanel({
  connectors,
  reenriching,
  reenrichError,
  reenrichDisabled,
  reenrichTarget,
  onReenrich
}: ConnectionsPanelProps): React.JSX.Element {
  const [rowErrors, setRowErrors] = useState<Record<string, string>>({})
  const [pendingById, setPendingById] = useState<Record<string, PendingAction | undefined>>({})
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set())
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [configDrafts, setConfigDrafts] = useState<Record<string, Record<string, string>>>({})
  const [syncAllPending, setSyncAllPending] = useState(false)
  const [panelError, setPanelError] = useState('')

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

  const runSyncAll = useCallback(async () => {
    setSyncAllPending(true)
    setPanelError('')
    try {
      await syncAllConnectors()
    } catch (error) {
      setPanelError(error instanceof Error ? error.message : 'Sync failed')
    } finally {
      setSyncAllPending(false)
    }
  }, [])

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
  const anySyncing = sortedConnectors.some((c) => c.syncing)

  return (
    <section className="connections-panel" aria-label="Connections">
      <div className="connections-panel-header">
        <div className="connections-panel-heading">
          <span className="section-label">Sources</span>
          {connectors !== null ? (
            <span className="connections-panel-stats">
              {activeCount} active · {formatCount(totalDocs)} docs indexed
            </span>
          ) : null}
        </div>
        <div className="panel-card-actions">
          <button
            className="secondary-button compact"
            type="button"
            disabled={syncAllPending || anySyncing}
            onClick={() => void runSyncAll()}
            title="Sync every enabled connector now"
          >
            {syncAllPending || anySyncing ? (
              <span className="inline-spinner" aria-hidden="true" />
            ) : null}
            {syncAllPending || anySyncing ? 'Syncing' : 'Sync all'}
          </button>
          <button
            className="secondary-button compact"
            type="button"
            disabled={reenrichDisabled}
            onClick={() => onReenrich()}
            title="Regenerate the user card, then re-enrich every document from every connector"
          >
            {reenriching && reenrichTarget === 'all' ? (
              <span className="inline-spinner" aria-hidden="true" />
            ) : null}
            {reenriching && reenrichTarget === 'all' ? 'Re-enriching all' : 'Re-enrich all'}
          </button>
        </div>
      </div>

      {reenrichError || panelError ? (
        <div className="connections-panel-error" role="alert">
          {reenrichError || panelError}
        </div>
      ) : null}

      {connectors === null ? (
        <div className="connector-list" aria-hidden="true">
          {[0, 1, 2, 3].map((index) => (
            <div className="connector-row skeleton" key={index}>
              <span className="connector-glyph" />
              <span className="connector-skeleton-lines">
                <span />
                <span />
              </span>
            </div>
          ))}
        </div>
      ) : sortedConnectors.length === 0 ? (
        <div className="empty-surface">No connectors reported</div>
      ) : (
        <div className="connector-list">
          {sortedConnectors.map((connector) => (
            <ConnectorRow
              configDrafts={configDrafts[connector.connector_id] ?? {}}
              connector={connector}
              confirmDelete={confirmDeleteId === connector.connector_id}
              error={rowErrors[connector.connector_id] ?? ''}
              expanded={expandedIds.has(connector.connector_id)}
              key={connector.connector_id}
              pendingAction={pendingById[connector.connector_id]}
              reenrichDisabled={reenrichDisabled}
              reenriching={reenriching && reenrichTarget === connector.connector_id}
              onCancelDelete={() => setConfirmDeleteId(null)}
              onConfirmDelete={() =>
                void runConnectorAction(connector.connector_id, 'delete', () =>
                  deleteConnectorData(connector.connector_id)
                )
              }
              onRequestDelete={() => setConfirmDeleteId(connector.connector_id)}
              onSaveConfig={() => void saveConfig(connector)}
              onSaveFolders={(folders) => void saveFolders(connector, folders)}
              onSync={() =>
                void runConnectorAction(connector.connector_id, 'sync', async () => {
                  await syncConnector(connector.connector_id)
                })
              }
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
              onReenrich={() => onReenrich(connector.connector_id)}
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
  reenriching,
  reenrichDisabled,
  onToggle,
  onToggleExpanded,
  onRequestDelete,
  onConfirmDelete,
  onCancelDelete,
  onSync,
  onUpdateConfig,
  onSaveConfig,
  onSaveFolders,
  onReenrich
}: {
  connector: ConnectorInfo
  pendingAction?: PendingAction
  error: string
  expanded: boolean
  configDrafts: Record<string, string>
  confirmDelete: boolean
  reenriching: boolean
  reenrichDisabled: boolean
  onToggle: () => void
  onToggleExpanded: () => void
  onRequestDelete: () => void
  onConfirmDelete: () => void
  onCancelDelete: () => void
  onSync: () => void
  onUpdateConfig: (key: string, value: string) => void
  onSaveConfig: () => void
  onSaveFolders: (folders: string[]) => void
  onReenrich: () => void
}): React.JSX.Element {
  const status = connectorStatus(connector)
  const signInRequired = connector.kind === 'oauth' && !connector.available
  const isFolders = connector.connector_id === FOLDERS_CONNECTOR_ID
  const syncRunning = connector.syncing || pendingAction === 'sync'
  const syncDisabled = syncRunning || !connector.available || !connector.enabled
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
    <article className={`connector-row status-${status}${expanded ? ' expanded' : ''}`}>
      <div
        className="connector-row-head"
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
          className="connector-glyph"
          style={{ '--glyph-color': sourceColor(connector.connector_id) } as React.CSSProperties}
          aria-hidden="true"
        >
          <ConnectorGlyph connectorId={connector.connector_id} />
        </span>

        <span className="connector-identity">
          <span className="connector-name">
            {connector.label}
            {connector.kind === 'oauth' ? (
              <span className="connector-oauth-badge">OAuth</span>
            ) : null}
          </span>
          <span className="connector-subtitle">
            <span className={`connector-status-dot ${status}`} aria-hidden="true" />
            {subtitleFor(connector, status, signInRequired)}
          </span>
        </span>

        <span className="connector-head-actions">
          <button
            className={`connector-switch${connector.enabled ? ' on' : ''}`}
            type="button"
            role="switch"
            aria-checked={connector.enabled}
            aria-label={`${connector.label} ${connector.enabled ? 'enabled' : 'disabled'}`}
            disabled={toggleDisabled}
            title={
              connector.enabled
                ? 'Disable — keeps stored data, removes it from the graph and retrieval'
                : 'Enable — stored data rejoins the graph and retrieval'
            }
            onClick={(event) => {
              event.stopPropagation()
              onToggle()
            }}
          >
            <span aria-hidden="true" />
          </button>
          <span className={`connector-chevron${expanded ? ' open' : ''}`} aria-hidden="true">
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
        <div className="connector-progress" aria-hidden="true">
          <span
            className={progressFraction === null ? 'indeterminate' : ''}
            style={
              progressFraction === null
                ? undefined
                : { width: `${Math.round(progressFraction * 100)}%` }
            }
          />
        </div>
      ) : null}

      {expanded ? (
        <div className="connector-detail" id={detailId}>
          {!signInRequired && connector.detail ? (
            <p className={`connector-detail-note${status === 'attention' ? ' attention' : ''}`}>
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
            <div className="connector-config">
              <span className="connector-detail-label">Configuration</span>
              {configKeys.map((key) => {
                const inputId = `connector-config-${connector.connector_id}-${key}`
                const meta = configFieldMeta(key)
                const draft = configDrafts[key] ?? formatConfigValue(connector.config[key])

                return (
                  <div className="connector-config-field" key={key}>
                    <label htmlFor={inputId}>{meta.label}</label>
                    <input
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
              <div className="connector-config-actions">
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

          <div className="connector-detail-actions">
            <button
              className="secondary-button compact"
              type="button"
              disabled={syncDisabled}
              onClick={onSync}
            >
              {syncRunning ? <span className="inline-spinner" aria-hidden="true" /> : null}
              {syncRunning ? syncProgressLabel(connector) : 'Sync now'}
            </button>
            <button
              className="secondary-button compact"
              type="button"
              disabled={reenrichDisabled || !connector.enabled || connector.docs === 0}
              onClick={onReenrich}
              title={`Re-run enrichment for the ${connector.docs} ${connector.label} document${connector.docs === 1 ? '' : 's'} already indexed`}
            >
              {reenriching ? <span className="inline-spinner" aria-hidden="true" /> : null}
              {reenriching ? 'Re-enriching' : 'Re-enrich'}
            </button>
            <span className="connector-detail-spacer" />
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
              className="connector-confirm"
              role="group"
              aria-label={`Delete ${connector.label} data`}
            >
              <span>
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
        <div className="connector-error" role="alert">
          {error}
        </div>
      ) : null}
    </article>
  )
}

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
    <div className="connector-folders">
      <span className="connector-detail-label">Attached folders</span>
      {folders.length === 0 ? (
        <p className="connector-folders-empty">
          No folders attached yet. Documents inside attached folders (.md, .txt, .pdf, .docx, .rtf)
          are indexed into the corpus and re-synced automatically.
        </p>
      ) : (
        <ul className="connector-folder-list">
          {folders.map((folder) => (
            <li key={folder}>
              <svg viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
                <path
                  d="M2.5 6a2 2 0 0 1 2-2h3.2l1.6 2h6.2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                />
              </svg>
              <code title={folder}>{folder}</code>
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
      <div className="connector-folders-actions">
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
    <div className="connector-signin">
      <p>
        Sign in runs a one-time local OAuth flow in your terminal and stores the tokens in{' '}
        <code>core/.env</code>. Restart the sidecar afterwards.
      </p>
      <div className="connector-command">
        <code>{command}</code>
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
