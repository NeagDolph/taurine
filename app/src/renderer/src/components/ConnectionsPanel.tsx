import {
  disableConnector,
  enableConnector,
  fetchConnectors,
  syncConnector,
  updateConnectorConfig,
  type ConnectorInfo
} from '../lib/sidecar'
import { useCallback, useEffect, useMemo, useState } from 'react'

interface ConnectionsPanelProps {
  visible: boolean
  reenriching: boolean
  reenrichError: string
  reenrichDisabled: boolean
  onDataChanged: () => void
  onReenrich: () => void
}

type PendingAction = 'enable' | 'disable' | 'purge' | 'sync' | 'config'

const CONNECTOR_POLL_MS = 5000
const APPLE_NOTES_CONNECTOR_ID = 'apple_notes'
const OBSIDIAN_CONNECTOR_ID = 'obsidian'

export function ConnectionsPanel({
  visible,
  reenriching,
  reenrichError,
  reenrichDisabled,
  onDataChanged,
  onReenrich
}: ConnectionsPanelProps): React.JSX.Element {
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [panelError, setPanelError] = useState('')
  const [cardErrors, setCardErrors] = useState<Record<string, string>>({})
  const [pendingById, setPendingById] = useState<Record<string, PendingAction | undefined>>({})
  const [confirmDisableId, setConfirmDisableId] = useState<string | null>(null)
  const [configDrafts, setConfigDrafts] = useState<Record<string, string>>({})
  const [dirtyConfigIds, setDirtyConfigIds] = useState<Set<string>>(() => new Set())

  const refreshConnectors = useCallback(
    async (silent = false) => {
      if (!silent) {
        setLoading(true)
      }

      try {
        const nextConnectors = await fetchConnectors()
        setConnectors(nextConnectors)
        setPanelError('')
        setConfigDrafts((current) => {
          const nextDrafts = { ...current }
          for (const connector of nextConnectors) {
            if (connector.connector_id === OBSIDIAN_CONNECTOR_ID && !dirtyConfigIds.has(connector.connector_id)) {
              nextDrafts[connector.connector_id] = configString(connector, 'vault_path')
            }
          }
          return nextDrafts
        })
      } catch (error) {
        setPanelError(error instanceof Error ? error.message : 'Failed to load connectors')
      } finally {
        setLoading(false)
      }
    },
    [dirtyConfigIds]
  )

  useEffect(() => {
    if (!visible) {
      return
    }

    void refreshConnectors(connectors.length > 0)
    const interval = window.setInterval(() => {
      void refreshConnectors(true)
    }, CONNECTOR_POLL_MS)

    return () => window.clearInterval(interval)
  }, [connectors.length, refreshConnectors, visible])

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

  const runConnectorAction = useCallback(
    async (connectorId: string, action: PendingAction, request: () => Promise<void>) => {
      setPendingById((current) => ({ ...current, [connectorId]: action }))
      clearCardError(connectorId)

      try {
        await request()
        setConfirmDisableId(null)
        await refreshConnectors(true)
        onDataChanged()
      } catch (error) {
        setCardError(connectorId, error instanceof Error ? error.message : 'Connector action failed')
      } finally {
        setPendingById((current) => ({ ...current, [connectorId]: undefined }))
      }
    },
    [clearCardError, onDataChanged, refreshConnectors, setCardError]
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
    return [...connectors].sort((left, right) => {
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
        <button className="secondary-button compact" type="button" onClick={() => void refreshConnectors()}>
          Refresh
        </button>
      </div>

      {panelError ? (
        <div className="connections-panel-error" role="alert">
          {panelError}
        </div>
      ) : null}

      {loading && connectors.length === 0 ? (
        <div className="empty-surface">Loading connections</div>
      ) : sortedConnectors.length === 0 ? (
        <div className="empty-surface">No connectors reported</div>
      ) : (
        <div className="connector-list">
          {sortedConnectors.map((connector) => (
            <ConnectorCard
              configDirty={dirtyConfigIds.has(connector.connector_id)}
              configDraft={configDrafts[connector.connector_id] ?? ''}
              connector={connector}
              confirmDisable={confirmDisableId === connector.connector_id}
              error={cardErrors[connector.connector_id] ?? ''}
              key={connector.connector_id}
              pendingAction={pendingById[connector.connector_id]}
              reenrichDisabled={reenrichDisabled}
              reenrichError={connector.connector_id === APPLE_NOTES_CONNECTOR_ID ? reenrichError : ''}
              reenriching={reenriching}
              onCancelDisable={() => setConfirmDisableId(null)}
              onConfirmDisable={(purge) =>
                void runConnectorAction(connector.connector_id, purge ? 'purge' : 'disable', () =>
                  disableConnector(connector.connector_id, purge)
                )
              }
              onDisable={() => setConfirmDisableId(connector.connector_id)}
              onEnable={() =>
                void runConnectorAction(connector.connector_id, 'enable', () =>
                  enableConnector(connector.connector_id)
                )
              }
              onReenrich={onReenrich}
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
  confirmDisable,
  reenriching,
  reenrichError,
  reenrichDisabled,
  onEnable,
  onDisable,
  onConfirmDisable,
  onCancelDisable,
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
  confirmDisable: boolean
  reenriching: boolean
  reenrichError: string
  reenrichDisabled: boolean
  onEnable: () => void
  onDisable: () => void
  onConfirmDisable: (purge: boolean) => void
  onCancelDisable: () => void
  onSync: () => void
  onUpdateConfig: (value: string) => void
  onSaveConfig: () => void
  onReenrich: () => void
}): React.JSX.Element {
  const isAppleNotes = connector.connector_id === APPLE_NOTES_CONNECTOR_ID
  const isObsidian = connector.connector_id === OBSIDIAN_CONNECTOR_ID
  const syncRunning = connector.syncing || pendingAction === 'sync'
  const syncDisabled = syncRunning || !connector.available || !connector.enabled
  const toggleDisabled = pendingAction === 'enable' || pendingAction === 'disable' || pendingAction === 'purge'
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
            docs
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
          onClick={connector.enabled ? onDisable : onEnable}
        >
          <span aria-hidden="true" />
          {connector.enabled ? 'Enabled' : 'Disabled'}
        </button>

        <button className="secondary-button compact" type="button" disabled={syncDisabled} onClick={onSync}>
          {syncRunning ? (
            <span className="inline-spinner" aria-hidden="true" />
          ) : null}
          {syncRunning ? 'Syncing' : 'Sync'}
        </button>

        {isAppleNotes ? (
          <button
            className="secondary-button compact"
            type="button"
            disabled={reenrichDisabled}
            onClick={onReenrich}
          >
            {reenriching ? (
              <span className="inline-spinner" aria-hidden="true" />
            ) : null}
            {reenriching ? 'Re-enriching' : 'Re-enrich'}
          </button>
        ) : null}
      </div>

      {confirmDisable ? (
        <div className="connector-confirm" role="group" aria-label={`Disable ${connector.label}`}>
          <span>Disable connector</span>
          <button className="secondary-button compact" type="button" onClick={() => onConfirmDisable(false)}>
            Keep data
          </button>
          <button className="secondary-button compact danger" type="button" onClick={() => onConfirmDisable(true)}>
            Remove indexed data
          </button>
          <button className="icon-button" type="button" aria-label="Cancel disable" onClick={onCancelDisable}>
            &times;
          </button>
        </div>
      ) : null}

      {error ? (
        <div className="connector-error" role="alert">
          {error}
        </div>
      ) : null}
      {reenrichError ? (
        <div className="connector-error" role="alert">
          {reenrichError}
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
