import type { Card } from '../lib/sidecar'

interface PinnedStackProps {
  cards: Card[]
  onOpenNote: (noteId: string) => void
  onDismiss: (cardId: string) => void
}

const provenanceLabels: Record<Card['provenance'], string> = {
  exact: 'Exact',
  search: 'Search',
  linked: 'Linked'
}

export function PinnedStack({ cards, onOpenNote, onDismiss }: PinnedStackProps): React.JSX.Element {
  return (
    <section className="pinned-region" aria-label="Pinned notes">
      <div className="section-label">Pinned</div>
      <div className="pinned-stack">
        {cards.length === 0 ? (
          <div className="empty-surface">No pinned notes</div>
        ) : (
          cards.slice(0, 3).map((card) => (
            <article className="pinned-card" key={card.card_id}>
              <div className="card-header">
                <div>
                  <h2>{card.title}</h2>
                  <span className={`provenance-chip provenance-${card.provenance}`}>
                    {provenanceLabels[card.provenance]}
                  </span>
                </div>
                <button
                  className="icon-button"
                  type="button"
                  aria-label={`Dismiss ${card.title}`}
                  onClick={() => onDismiss(card.card_id)}
                >
                  &times;
                </button>
              </div>
              <p>{card.summary}</p>
              <div className="card-actions">
                <button className="secondary-button" type="button" onClick={() => onOpenNote(card.note_id)}>
                  Open in Notes
                </button>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  )
}
