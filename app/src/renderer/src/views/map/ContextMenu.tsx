import { useEffect, useRef, useState } from 'react'
import { fetchFacts, type FactRecord, type MapNode, type StaticGraphLink } from '../../lib/graph'
import { endpointNode } from './render'

export type MenuTarget =
  | { kind: 'node'; x: number; y: number; node: MapNode }
  | { kind: 'edge'; x: number; y: number; link: StaticGraphLink }

interface ContextMenuProps {
  target: MenuTarget
  isPinned: boolean
  onClose: () => void
  onOpenNote: (noteId: string) => void
  onFocus: (noteId: string) => void
  onAskAbout: (noteId: string, title: string) => void
  onTogglePin: (noteId: string) => void
  onHideNode: (noteId: string) => void
}

export function MapContextMenu({
  target,
  isPinned,
  onClose,
  onOpenNote,
  onFocus,
  onAskAbout,
  onTogglePin,
  onHideNode
}: ContextMenuProps): React.JSX.Element {
  const menuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        onClose()
      }
    }
    const onPointerDown = (event: PointerEvent): void => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('pointerdown', onPointerDown, true)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('pointerdown', onPointerDown, true)
    }
  }, [onClose])

  const run = (action: () => void): void => {
    action()
    onClose()
  }

  return (
    <div
      className="map-context-menu"
      ref={menuRef}
      role="menu"
      style={{ left: target.x, top: target.y }}
    >
      {target.kind === 'node' ? (
        <>
          <div className="map-menu-title">{target.node.title}</div>
          {target.node.source === 'apple_notes' ? (
            <button
              className="map-menu-item"
              type="button"
              role="menuitem"
              onClick={() => run(() => onOpenNote(target.node.id))}
            >
              Open note
            </button>
          ) : null}
          <button
            className="map-menu-item"
            type="button"
            role="menuitem"
            onClick={() => run(() => onFocus(target.node.id))}
          >
            Focus here
          </button>
          <button
            className="map-menu-item"
            type="button"
            role="menuitem"
            onClick={() => run(() => onAskAbout(target.node.id, target.node.title))}
          >
            Ask about this
          </button>
          <button
            className="map-menu-item"
            type="button"
            role="menuitem"
            onClick={() => run(() => onTogglePin(target.node.id))}
          >
            {isPinned ? 'Unpin' : 'Pin position'}
          </button>
          <button
            className="map-menu-item map-menu-danger"
            type="button"
            role="menuitem"
            onClick={() => run(() => onHideNode(target.node.id))}
          >
            Hide node
          </button>
        </>
      ) : (
        <EdgeFacts key={target.link.id} link={target.link} />
      )}
    </div>
  )
}

type FactsState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; facts: FactRecord[] }
  | { status: 'error' }

// NOTE: rendered with key={link.id}, so a different edge remounts this
// component — the loading state is the initial state, not an effect update.
function EdgeFacts({ link }: { link: StaticGraphLink }): React.JSX.Element {
  const [state, setState] = useState<FactsState>(() =>
    link.factIds.length > 0 ? { status: 'loading' } : { status: 'idle' }
  )

  useEffect(() => {
    if (link.factIds.length === 0) {
      return undefined
    }

    let cancelled = false
    fetchFacts(link.factIds)
      .then((facts) => {
        if (!cancelled) {
          setState({ status: 'ready', facts })
        }
      })
      .catch(() => {
        if (!cancelled) {
          setState({ status: 'error' })
        }
      })

    return () => {
      cancelled = true
    }
  }, [link])

  const source = endpointNode(link.source)
  const target = endpointNode(link.target)
  const mechanisms = Object.entries(link.mechanisms).filter(([, weight]) => weight > 0)

  return (
    <div className="map-edge-facts">
      <div className="map-menu-title">
        {link.edgeType} connection · w {link.weight.toFixed(2)}
      </div>
      {source && target ? (
        <div className="map-edge-endpoints">
          {source.title} ↔ {target.title}
        </div>
      ) : null}
      {mechanisms.length > 0 ? (
        <div className="map-edge-mechanisms">
          {mechanisms.map(([name, weight]) => (
            <span className="map-chip" key={name}>
              {name} {weight.toFixed(2)}
            </span>
          ))}
        </div>
      ) : null}

      {state.status === 'loading' ? (
        <div className="map-edge-state">Loading facts…</div>
      ) : state.status === 'error' ? (
        <div className="map-edge-state">Facts unavailable</div>
      ) : state.status === 'ready' && state.facts.length > 0 ? (
        <ul className="map-fact-list">
          {state.facts.map((fact) => (
            <li key={fact.fact_id}>
              <span className="map-fact-text">
                <strong>{fact.subject}</strong> {fact.predicate} <strong>{fact.object}</strong>
              </span>
              {fact.confidence > 0 ? (
                <span className="map-fact-confidence">{Math.round(fact.confidence * 100)}%</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : link.fact ? (
        <p className="map-edge-legacy-fact">{link.fact}</p>
      ) : link.shared.length > 0 ? (
        <p className="map-edge-legacy-fact">Shared: {link.shared.join(', ')}</p>
      ) : (
        <div className="map-edge-state">No supporting facts recorded</div>
      )}
    </div>
  )
}
