import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { sourceColor, sourceLabel, type MapNode } from '../../lib/graph'

export interface SearchResult {
  node: MapNode
  /** what matched, for the dropdown hint */
  via: 'title' | 'entity' | 'gist'
}

interface MapSearchProps {
  query: string
  results: SearchResult[]
  onQueryChange: (query: string) => void
  onPick: (noteId: string) => void
}

export function MapSearch({
  query,
  results,
  onQueryChange,
  onPick
}: MapSearchProps): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const [highlightIndex, setHighlightIndex] = useState(0)
  const [lastQuery, setLastQuery] = useState(query)
  const rootRef = useRef<HTMLDivElement | null>(null)

  // reset the highlight when the query changes — adjusted during render
  if (query !== lastQuery) {
    setLastQuery(query)
    setHighlightIndex(0)
  }

  useEffect(() => {
    if (!open) {
      return undefined
    }
    const onPointerDown = (event: PointerEvent): void => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    window.addEventListener('pointerdown', onPointerDown, true)
    return () => window.removeEventListener('pointerdown', onPointerDown, true)
  }, [open])

  const showDropdown = open && query.trim().length > 0 && results.length > 0

  const pick = (noteId: string): void => {
    onPick(noteId)
    setOpen(false)
  }

  return (
    <div className="map-search" ref={rootRef}>
      <label className="map-search-field">
        <span className="map-sr-only">Search notes by title, gist, or entity</span>
        <input
          type="search"
          value={query}
          placeholder="Search titles, gists, entities"
          onChange={(event) => {
            onQueryChange(event.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && results.length > 0) {
              event.preventDefault()
              pick(results[Math.min(highlightIndex, results.length - 1)].node.id)
            } else if (event.key === 'ArrowDown' && results.length > 0) {
              event.preventDefault()
              setOpen(true)
              setHighlightIndex((index) => Math.min(index + 1, results.length - 1))
            } else if (event.key === 'ArrowUp' && results.length > 0) {
              event.preventDefault()
              setHighlightIndex((index) => Math.max(index - 1, 0))
            } else if (event.key === 'Escape') {
              if (open) {
                setOpen(false)
              } else {
                onQueryChange('')
              }
            }
          }}
        />
      </label>

      {showDropdown ? (
        <ol className="map-search-results" role="listbox" aria-label="Search results">
          {results.map((result, index) => (
            <li key={result.node.id}>
              <button
                className={`map-search-result${index === highlightIndex ? ' highlighted' : ''}`}
                type="button"
                role="option"
                aria-selected={index === highlightIndex}
                onMouseEnter={() => setHighlightIndex(index)}
                // mousedown beats the input blur so the click always lands
                onMouseDown={(event) => {
                  event.preventDefault()
                  pick(result.node.id)
                }}
              >
                <span
                  className="map-search-dot"
                  style={{ '--map-source-color': sourceColor(result.node.source) } as CSSProperties}
                  aria-hidden="true"
                />
                <span className="map-search-main">
                  <strong>{result.node.title}</strong>
                  <span>
                    {sourceLabel(result.node.source)}
                    {result.via !== 'title' ? ` · ${result.via} match` : ''}
                    {` · ${result.node.degree} links`}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  )
}
