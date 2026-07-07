import {
  deleteConnectorData,
  disableConnector,
  enableConnector,
  syncConnector,
  updateConnectorConfig,
  type ConnectorInfo
} from '../lib/sidecar'
import { useCallback, useMemo, useState } from 'react'

interface ConnectionsPanelProps {
  visible: boolean
  connectors: ConnectorInfo[] | null
  reenriching: boolean
  reenrichError: string
  reenrichDisabled: boolean
  reenrichTarget: string | null
  onReenrich: (source?: string) => void
}

type PendingAction = 'enable' | 'disable' | 'delete' | 'sync' | 'config'

const APPLE_NOTES_CONNECTOR_ID = 'apple_notes'
const OBSIDIAN_CONNECTOR_ID = 'obsidian'

export function ConnectionsPanel({
  visible,
  connectors,
  reenriching,
  reenrichError,
  reenrichDisabled,
  reenrichTarget,
  onReenrich
}: ConnectionsPanelProps): React.JSX.Element {
  const [cardErrors, setCardErrors] = useState<Record<string, string>>({})
  const [pendingById, setPendingById] = useState<Record<string, PendingAction | undefined>>({})
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [configDrafts, setConfigDrafts] = useState<Record<string, string>>({})
  const [dirtyConfigIds, setDirtyConfigIds] = useState<Set<string>>(() => new Set())

  void visible

  const setCardError = useCallback((connectorId: string, message: string) => {
    setCardErrors((current) => ({ ...current, [connectorId]: message }))
  }, [])

  const clearCardError = useCallback((connectorId: string) => {
    setCardErrors((current) => {
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
      clearCardError(connectorId)

      try {
        await request()
        setConfirmDeleteId(null)
      } catch (error) {
        setCardError(connectorId, error instanceof Error ? error.message : 'Connector action failed')
      } finally {
        setPendingById((current) => ({ ...current, [connectorId]: undefined }))
      }
    },
    [clearCardError, setCardError]
  )

  const updateConfigDraft = useCallback((connectorId: string, value: string) => {
    setConfigDrafts((current) => ({ ...current, [connectorId]: value }))
    setDirtyConfigIds((current) => new Set(current).add(connectorId))
  }, [])

  const saveObsidianConfig = useCallback(
    async (connectorId: string) => {
      const vaultPath = configDrafts[connectorId] ?? ''
      await runConnectorAction(connectorId, 'config', async () => {
        await updateConnectorConfig(connectorId, { vault_path: vaultPath.trim() })
        setDirtyConfigIds((current) => {
          const next = new Set(current)
          next.delete(connectorId)
          return next
        })
      })
    },
    [configDrafts, runConnectorAction]
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

  return (
    <section className="connections-panel" aria-label="Connections">
      <div className="connections-panel-header">
        <span className="section-label">Sources</span>
        <div className="panel-card-actions">
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

      {reenrichError ? (
        <div className="connections-panel-error" role="alert">
          {reenrichError}
        </div>
      ) : null}

      {connectors === null ? (
        <div className="empty-surface">Loading connections</div>
      ) : sortedConnectors.length === 0 ? (
        <div className="empty-surface">No connectors reported</div>
      ) : (
        <div className="connector-list">
          {sortedConnectors.map((connector) => (
            <ConnectorCard
              configDirty={dirtyConfigIds.has(connector.connector_id)}
              configDraft={
                dirtyConfigIds.has(connector.connector_id)
                  ? configDrafts[connector.connector_id] ?? ''
                  : configString(connector, 'vault_path')
              }
              connector={connector}
              confirmDelete={confirmDeleteId === connector.connector_id}
              error={cardErrors[connector.connector_id] ?? ''}
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
              onReenrich={() => onReenrich(connector.connector_id)}
              onSaveConfig={() => void saveObsidianConfig(connector.connector_id)}
              onSync={() =>
                void runConnectorAction(connector.connector_id, 'sync', async () => {
                  await syncConnector(connector.connector_id)
                })
              }
              onUpdateConfig={(value) => updateConfigDraft(connector.connector_id, value)}
            />
          ))}
        </div>
      )}
    </section>
  )
}

function ConnectorCard({
  connector,
  pendingAction,
  error,
  configDraft,
  configDirty,
  confirmDelete,
  reenriching,
  reenrichDisabled,
  onToggle,
  onRequestDelete,
  onConfirmDelete,
  onCancelDelete,
  onSync,
  onUpdateConfig,
  onSaveConfig,
  onReenrich
}: {
  connector: ConnectorInfo
  pendingAction?: PendingAction
  error: string
  configDraft: string
  configDirty: boolean
  confirmDelete: boolean
  reenriching: boolean
  reenrichDisabled: boolean
  onToggle: () => void
  onRequestDelete: () => void
  onConfirmDelete: () => void
  onCancelDelete: () => void
  onSync: () => void
  onUpdateConfig: (value: string) => void
  onSaveConfig: () => void
  onReenrich: () => void
}): React.JSX.Element {
  const isObsidian = connector.connector_id === OBSIDIAN_CONNECTOR_ID
  const syncRunning = connector.syncing || pendingAction === 'sync'
  const syncDisabled = syncRunning || !connector.available || !connector.enabled
  const toggleDisabled = pendingAction === 'enable' || pendingAction === 'disable'
  const deleteDisabled =
    pendingAction === 'delete' || connector.syncing || connector.docs === 0
  const configInputId = `connector-config-${connector.connector_id}`

  return (
    <article className="connector-card">
      <div className="connector-card-main">
        <div className="connector-title-block">
          <div className="connector-label-row">
            <h2>{connector.label}</h2>
            <span className="connector-kind-badge">{connector.kind}</span>
          </div>
          <div className={`connector-availability ${connector.available ? 'available' : 'unavailable'}`}>
            <span aria-hidden="true" />
            <p>{connector.available ? 'Available' : connector.detail || 'Unavailable'}</p>
          </div>
        </div>

        <div className="connector-stats">
          <span>
            <strong>{formatCount(connector.docs)}</strong>
            {connector.enabled ? 'docs' : 'docs stored'}
          </span>
          <span>
            <strong>{formatRelativeTime(connector.last_sync)}</strong>
            last sync
          </span>
        </div>
      </div>

      {isObsidian ? (
        <div className="connector-config-row">
          <label htmlFor={configInputId}>Vault path</label>
          <input
            id={configInputId}
            type="text"
            value={configDraft}
            placeholder="/Users/name/Obsidian/Vault"
            onChange={(event) => onUpdateConfig(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && configDirty && pendingAction !== 'config') {
                onSaveConfig()
              }
            }}
          />
          <button
            className="secondary-button compact"
            type="button"
            disabled={!configDirty || pendingAction === 'config'}
            onClick={onSaveConfig}
          >
            {pendingAction === 'config' ? 'Saving' : 'Save'}
          </button>
        </div>
      ) : null}

      <div className="connector-actions">
        <button
          className={`connector-toggle${connector.enabled ? ' enabled' : ''}`}
          type="button"
          aria-pressed={connector.enabled}
          disabled={toggleDisabled}
          onClick={onToggle}
          title={
            connector.enabled
              ? 'Disable — keeps stored data, removes it from the graph and retrieval'
              : 'Enable — stored data rejoins the graph and retrieval'
          }
        >
          <span aria-hidden="true" />
          {connector.enabled ? 'Enabled' : 'Disabled'}
        </button>

        <button className="secondary-button compact" type="button" disabled={syncDisabled} onClick={onSync}>
          {syncRunning ? (
            <span className="inline-spinner" aria-hidden="true" />
          ) : null}
          {syncRunning ? syncProgressLabel(connector) : 'Sync'}
        </button>

        <button
          className="secondary-button compact"
          type="button"
          disabled={reenrichDisabled || !connector.enabled || connector.docs === 0}
          onClick={onReenrich}
          title={`Re-run enrichment for the ${connector.docs} ${connector.label} document${connector.docs === 1 ? '' : 's'} already indexed`}
        >
          {reenriching ? (
            <span className="inline-spinner" aria-hidden="true" />
          ) : null}
          {reenriching ? 'Re-enriching' : 'Re-enrich'}
        </button>

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
        <div className="connector-confirm" role="group" aria-label={`Delete ${connector.label} data`}>
          <span>
            Delete all {formatCount(connector.docs)} stored document
            {connector.docs === 1 ? '' : 's'}? A future sync can re-import them.
          </span>
          <button className="secondary-button compact danger" type="button" onClick={onConfirmDelete}>
            Delete
          </button>
          <button className="icon-button" type="button" aria-label="Cancel delete" onClick={onCancelDelete}>
            &times;
          </button>
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

function configString(connector: ConnectorInfo, key: string): string {
  const value = connector.config[key]
  return typeof value === 'string' ? value : ''
}

function formatCount(value: number): string {
  return new Intl.NumberFormat().format(Math.max(0, value))
}

function formatRelativeTime(epoch: number | null | undefined): string {
  if (!epoch) {
    return 'Never'
  }

  const millis = epoch > 1_000_000_000_000 ? epoch : epoch * 1000
  const diffSeconds = Math.round((Date.now() - millis) / 1000)
  const absSeconds = Math.abs(diffSeconds)

  if (absSeconds < 60) {
    return diffSeconds >= 0 ? 'Just now' : 'Soon'
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

  return 'Just now'
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
