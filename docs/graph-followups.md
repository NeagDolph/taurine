# Graph View — Deferred Follow-ups

Deferred from the 2026-07-09 graph rehaul (see `graph-rehaul-spec.md`). Roughly ordered.

1. **Edge-type / mechanism filtering (B10)** — legend rows become click-to-filter
   controls; toggles for semantic-only / entity-mediated / fact-backed edges using the
   per-edge `mechanisms` weights now in the payload.
2. **Keyboard support (B13)** — `+`/`-` zoom, arrows pan, `/` focuses search, `Esc`
   clears selection/filters, `f` zoom-to-fit, `[`/`]` back/forward in ego history.
3. **Session subgraph view (D19)** — one-click filter to only the notes this live
   session retrieved/grounded/traversed (`RetrievalEvent` hits + expansions +
   `groundedNoteIds`); effectively an automatic ego graph of the meeting.
4. **Chat answer → graph highlighting (E24)** — "view sources in map" on an assistant
   turn renders the chat retrieval subgraph with the same rank-heat + traversal-glow
   language the Live view uses.
5. **Multi-select → chat (E23)** — lasso/shift-click a set of nodes → "summarize these /
   find the connection" chat action; selection sets also become the unit for hide/export.
6. **Structural gap bridging (C16, InfraNodus-style)** — find weakly connected community
   pairs and their few bridge notes; surface "these two areas barely touch" with a chat
   prompt to explore the bridge.
7. **LLM-assisted clustering** — can LLMs beat pure label propagation? Yes, three
   compounding levels, all build-time (cheap, cached per `build_id`):
   - **Naming (do first, biggest UX win/$):** LP communities are anonymous ints; feed
     each community's top entities + a sample of note gists to a small model to produce
     a 2–4 word label rendered at low zoom (Nomic-Atlas-style topic atlas).
   - **Boundary refinement:** LP over the degree-capped projection misassigns
     low-degree/bridge notes. For notes with weak membership (few intra-community
     edges), ask an LLM to pick the best-fitting community given the note gist and
     candidate community labels — hybrid graph+semantic assignment.
   - **Semantic clustering as an alternative lens:** cluster note embeddings directly
     (HDBSCAN/k-means over passage-embedding averages) and reconcile with graph
     communities; where they disagree, an LLM adjudicates. This can produce *better*
     clusterings than connection-based LP because it groups notes that are similar but
     not yet linked — and disagreement itself is a signal (candidate missing edges).
8. **Tailwind-ify map UI** — migrate the map's self-contained CSS to Tailwind once the
   app-wide migration is merged and settled.
9. **Edge provenance → transcript segment (E26)** — `/facts` gives source_note_id today;
   extend to passage/char offsets and eventually the originating transcript segment,
   with a "this extraction is wrong" correction affordance.
10. **UMAP-seeded layout (F29)** — average passage embeddings per note, use UMAP coords
    as force-layout initialization/weak gravity for a stable semantic geography across
    builds.
