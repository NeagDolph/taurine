# Taurine Retrieval Stack Audit — 2026-07-06

Scope: full retrieval loop (GLiNER2 → query formulation & RRF fusion → graph
expansion → Cohere rerank → slate → presenter → cadence), verified against the
session JSONL logs in `core/sessions/`. Ground truth is the instrumented
sessions `20260706-191353-3f4b958a.jsonl` (8 C2 passes) and
`20260706-190957-8771ddd0.jsonl` (2 passes); the pre-instrumentation sessions
lack `salient`/`gliner_spans`/`dense_*` fields and were used only for window
text. All code references are `file:line` in `core/taurine_core/`.

Corpus stats (from `core/taurine.db`): 281 notes, 1137 edges, 765 aliases,
2612 embeddings (2001 `utterance`, 330 `chunk`, 281 `gist`).

The scenario across the main session: user speaks "Hi Ashish … Shopify
integration … Armand". Correct primary hit is **p572** ("Ashish Parik –
Integration expert for shopify"); the ASR-drift target is **p577**
("Armaan Talk", user says "Armand"). p572 is retrieved and presented
correctly. p577 never reaches the slate — but the seed triage (rerank demotion
+ C2_TOP_K cutoff) is only *part* of why. This audit isolates the other causes.

**Out of scope (already being fixed by other agents, per brief):** C2_TOP_K→8,
top-3 fused-protect, rapidfuzz span→alias snapping, Deepgram ASR, agentic
presenter with `read_note`. Config already shows these partially landed
(`config.py:48,50,55-58`). Findings below are the *remaining* defects.

---

## Executive summary — top 5 findings by retrieval-accuracy impact

1. **RRF dilutes single-head hits into oblivion (HIGH).** RRF gives a note
   credit only for the lists it appears in. A note strong in `dense` but absent
   from `bm25`/`dense_salient`/`dense_gliner` gets ≈1/(60+rank) total and sinks
   below mediocre notes that appear in 3 lists. p577 was **dense rank 4** in
   passes 5–7 yet **fused rank 17, 17, 20** because it was in *only* the dense
   list. This is upstream of the rerank/cutoff fixes and defeats fused-protect
   whenever a good note is dense-only. (`ambient.py:162-172`)

2. **The salient query systematically selects ASR garbage (HIGH).** Salience =
   `1/(1+doc_frequency(term))` (`ambient.py:191`). Misspellings and nonsense
   tokens have df=0 → score 1.0 → they always win. Verified df: `mored`=0,
   `shopif`=0, `armand`=0, `regarding`=0, while the *useful* term `shopify`=3
   scores lower. Real salient queries were `"ashish earlier regarding bigger
   mored armand"`. `dense_salient` never once surfaced p577 and adds noise to
   fusion.

3. **Entity/topic graph edges are the weakest edges, so the meaningful links
   contribute least to expansion (HIGH).** Mean edge weight by type: semantic
   0.683, entity 0.469, topic 0.402 (mixed 0.881). Expansion score =
   `seed_score × edge_weight × HOP_DECAY(0.5)` (`ambient.py:248`), so the
   "shared Shopify/Ashish entity" edges that carry real relevance produce the
   *smallest* arrivals (~0.006–0.02). No graph-expanded note survived to the
   slate in 7 of 8 passes; the one that did (pass 7) surfaced off-topic notes.

4. **The reranker never sees the text that actually won dense retrieval
   (MEDIUM-HIGH).** `dense_search` collapses all embedding rows (incl. 2001
   synthetic utterances) to a per-note max (`store.py:170-172`), but the rerank
   doc is `title — gist\n best_chunk`, and `best_chunk` only considers
   `kind=="chunk"` rows (`store.py:239`). Verified: for p572/p592/p97 the
   winning dense embedding is an *utterance*, not a chunk — so Cohere reranks a
   document that omits the sentence that matched. This is a dense↔rerank
   representation mismatch that makes rerank scores noisier than they should be.

5. **Gists are hard-truncated to 120 chars mid-word, degrading both rerank docs
   and the presenter prompt (MEDIUM).** `enrich.py:41` `gist=gist[:120]`. p572's
   stored gist ends `"…Anil and Dilip are responsible f"`. That same string is
   the rerank document tail (`ambient.py:283`) and the presenter slate line
   (`presenter.py:141`). The single most information-dense field the LLM sees is
   chopped.

---

## Per-layer findings

### Layer 1 — GLiNER2 schema & extraction quality

GLiNER2 extraction is **the healthiest layer.** Across every instrumented pass
it extracted the right entities: `Ashish`/`person`, `Shopify`/`organization`,
`Shopify integration`/`topic`, `friday deadline`/`event`, and it correctly
caught `Armand`/`person` in passes 5–7 even under ASR drift
(session 3f4b958a, passes 1–7; session 8771ddd0 both passes).

Verified issues, lower severity:

- **F1.1 (schema noise / low precision on `topic`).** The `topic` label
  ("The concrete subject matter … e.g. 'checkout flow', 'Q3 roadmap'",
  `config.GLINER2_SCHEMA`) yielded `"high level"` as a topic (pass 2) — a filler
  phrase, not a subject. `topic` and `organization` also overlap ("Shopify" was
  tagged organization while "Shopify integration" was topic — fine here, but the
  descriptions don't disambiguate a bare product name from a topic).
- **F1.2 (uniform 1.0 span scores discard GLiNER confidence).** `spans.py:74`
  hard-codes score 1.0 for every span; `GLINER_MAX_SPANS=4` then truncates by
  *schema order*, not model confidence (`spans.py:75`, `_normalize_result`).
  A low-confidence `event` can outrank a high-confidence `person` purely because
  `event` never appears (person is first in schema) — but the cap can still drop
  a genuinely salient later-schema span. Impact is small at 4 spans but the
  mechanism is fragile.
- **F1.3 (the real GLiNER value is unrealized downstream).** `dense_gliner`
  embeds each span and max-pools (`ambient.py:212-225`), but because the spans
  are the *misspelled ASR text* ("Armand"), the span-dense query misses p577 —
  exactly the seed finding. GLiNER extracts perfectly; the failure is that its
  output is fed to a semantic embedder that can't bridge the spelling gap. The
  rapidfuzz span→alias snap (already being added, `config.py:55-58`) addresses
  this at the alias layer but **not** the `dense_gliner` embedding path — the
  snapped alias should also replace the span text used for `dense_gliner`.

### Layer 2 — Query formulation & RRF fusion

**F2.1 — RRF single-list starvation (top finding #1).**
`_rrf_scores` (`ambient.py:162-172`) fuses four ranked lists (`dense`,
`dense_salient`, `dense_gliner`, `bm25`) with `1/(RRF_K+rank)`, `RRF_K=60`.
A note present in one list at rank r scores `1/(60+r)` ≈ 0.016; a note present
in three lists at rank ~15 each scores ≈ 0.048 — 3× higher despite being a
weaker individual match. p577 tracking (session 3f4b958a):

| pass | dense | fused | ordered | in slate |
|------|-------|-------|---------|----------|
| 1 | 5 | 6 | 6 | no |
| 2 | 1 | 2 | 8 | no |
| 5 | 4 | **17** | 7 | no |
| 6 | 4 | **17** | 7 | no |
| 7 | 4 | **20** | — | no |

In passes 5–7 p577 was verified present in **only** the `dense` head
(absent from `dense_salient`, `dense_gliner`, `bm25`), which is exactly why
fusion buried it at rank 17–20. Because fused-protect (top-3) operates on the
*fused* order, a dense-only good hit is invisible to it.

**F2.2 — Salient query prefers rare = garbage (top finding #2).** Verified df
values above. Selection is `sorted(scored, key=-score)[:6]` (`ambient.py:194`).
Stopwords are filtered (`config.ALIAS_STOPWORDS`) but ASR misspellings and
partial words ("shopif", "mored") are not — and they are maximally rare, so they
are *preferentially* chosen. The salient head therefore contributes a query of
mostly-noise terms and its `dense_salient` list rarely overlaps the useful
notes, feeding F2.1 (it becomes a fourth list that good notes are absent from,
worsening RRF dilution rather than helping).

**F2.3 — three of four fusion lists are dense variants over the same encoder.**
`dense`, `dense_salient`, `dense_gliner` all call `index.dense_search` with the
same bge-small model. When they agree they triple-count the same signal;
when the query text is noisy (salient/gliner spans) they inject noise. Only
`bm25` is an independent signal. RRF's independence assumption is violated.

### Layer 3 — Graph leverage

**F3.1 — meaningful edges are the weakest (top finding #3).** Edge-weight means
by type (verified): semantic 0.683 / entity 0.469 / topic 0.402 / mixed 0.881.
Entity/topic weight = `Σ _term_weight(freq)` where
`_term_weight = 1/(1+log(1+note_count))` (`graph.py:188`) — capped low and
shrinking with term popularity, so a shared "Shopify" tag (appears in several
notes) is heavily discounted. Semantic edges (raw cosine ≥0.60) dominate the
graph: 734 of 1137 edges are pure `semantic`.

**F3.2 — expansion never influences the slate.** Verified across all 8 passes:
graph-expanded notes appear in `ordered` occasionally (0–5 per pass) but reached
`slate_adds` in exactly **one** pass (pass 7), and there the survivors were
**off-topic** (p536 "Logistic Regression … NLP homework", p569 "Joel") arriving
via p542/p229 seeds. Expansion arrival scores are ~0.006–0.02
(`expansions` in every c2_pass), vs rerank/fused scores 0.03–0.5, so expanded
notes are essentially always at the bottom of the pool and die at the floor or
the C2_TOP_K cut. **Graph expansion currently adds latency and off-topic
candidates with ~zero positive contribution.**

**F3.3 — the target's own graph is useless for the target.** p577 ("Armaan
Talk") neighbors are all `semantic` edges to generic "talk/meeting with X" notes
(Sharon, Sumit, Greg, Vineet) with `shared=[]` — none connect to Ashish/Shopify.
p577 is enrichment-poor, so it has no topic/entity edges at all. Meanwhile p572
*does* have the right entity edges (→ Wink Todos via "Shopify", → Vineet via
"Vineet") but at weight 0.419–0.477 (per F3.1). The seed rule (expand only from
top-5 fused, skip direct hits — `ambient.py:243-246`) further guarantees p577
gets nothing: it is never a top-5 fused seed, and the notes that *are* seeds
don't link to it.

**F3.4 — no graph-vote / co-arrival boost.** Expansion overwrites with `max`
(`ambient.py:249-251`); a note reached from *multiple* seeds gets no cumulative
credit. A note pointed at by 3 of the top-5 seeds (a strong graph signal) is
scored identically to one reached from a single seed. Graph consensus is
discarded.

### Layer 4 — Rerank behavior

**F4.1 — rerank doc omits the winning embedding text (top finding #4).**
Detailed above (F #4). `best_chunk` restricts to `kind=="chunk"`
(`store.py:238-239`); the utterance/gist that actually maximized dense
similarity is never shown to Cohere. The reranker is judging a different
document than the one that earned the note its place in the pool.

**F4.2 — rerank reordering is heavy and sometimes right, sometimes wrong.**
`ordered[1]`'s prior fused rank across passes: 1,1,7,2,1,1,1,6. When Cohere
promotes a deep fused item it is usually correct here (pass 2: p572 fused-7 →
ordered-1, the right note; pass 7: p173 fused-6 → ordered-1, but p173 is the
off-topic "dreams" note — **wrong**). So rerank both rescues good RRF-diluted
notes *and* elevates off-topic ones. Net: rerank is compensating for F2.1's
fusion damage rather than adding independent signal.

**F4.3 — fallback tail can reintroduce floor-failing notes.**
`_reranked_or_fallback` appends the fused fallback tail for any id Cohere didn't
return (`ambient.py:314`). Combined with `RERANK_MAX_CANDIDATES=20` docs, this
is minor, but it means the "ordered" list is a rerank head + raw-fused tail,
partially undermining the reranker's judgement at the boundary.

### Layer 5 — Slate mechanics

**F5.1 — search-lane quota (8) == C2_TOP_K (8), so every survivor floods one
lane.** `QUOTA_SEARCH=8` (`config.py:94`), `C2_TOP_K=8` (`config.py:48`). Each
C2 pass can add 8 search items, immediately at quota; the next pass's adds
trigger evictions (`_evict_over_quota`, `slate.py:118`). Verified 4 `evict/quota`
events; evicted notes include p250, p499, p212, p545. Eviction key is
`(views, -added_rev)` (`slate.py:124`) — it evicts the *oldest unviewed* item,
which can be a still-relevant note pushed out by a newer, weaker one. Score is
not consulted in eviction.

**F5.2 — decay counts *views not shown-picks*, evicting good notes fast.**
`mark_viewed` increments `views` for every slate item each time the slate is
viewed and kicks at `views>=decay` with `DECAY_SEARCH=2` (`slate.py:92-105`,
`config.py:95`). Verified 7 `kick/decayed` events. A relevant note the presenter
hasn't picked yet is evicted after two view cycles regardless of relevance.
p577, even if it ever reached the slate, would have ~2 view cycles to be picked.

**F5.3 — `shown` set is permanent within a session (never re-show).**
`self.shown` (`slate.py:33,80`) blocks any re-add for the whole session
(7 `shown_drop` events). Correct per spec, but combined with F5.2 means a note
shown once and dismissed cannot return even when the conversation circles back.

### Layer 6 — Presenter inputs

**F6.1 — presenter got the right info when it mattered and picked correctly.**
4 presenter calls (session 3f4b958a): rev5 → [] (empty, valid), rev12 → p572,
rev30 → p565, rev39 → p97. p572 is the correct primary pick. No presenter errors.
The presenter is **not** the bottleneck — it never received p577 because
retrieval never delivered it.

**F6.2 — truncated gists reach the prompt (top finding #5).** Slate lines are
`{provenance} {note_id} "{title}" - {gist}` (`presenter.py:141`) with the
120-char mid-word-cut gist. The transcript budget is only
`PRESENTER_TRANSCRIPT_CHUNKS=15` (`config.py:107`) and last-15 chunks
(`presenter.py:111`) — adequate here but with the truncated gists the model's
per-card evidence is thin.

**F6.3 — provenance line for search items shows `heard` = full window.**
`_slate_line` sets `[search: matched "{item.heard}"]` where `heard` is the
entire rolling window (`ambient.py:355`, `SlateItem.heard=window_text`). For
long windows this is a paragraph, not the matched span, so the "why this card"
signal to the LLM is diffuse. GLiNER spans / best_chunk would be a sharper
`matched` value.

### Layer 7 — Cadence

**F7.1 — cadence is the dominant gate on presenter frequency, but correctly
so.** Decisions (session 3f4b958a): 40 cooldown, 25 debounce, 4 allowed.
`COOLDOWN_S=20`, `DEBOUNCE_S=3` (`config.py:100-101`). With 8 C2 passes packed
into a short window, cooldown blocking most is expected and not a retrieval
defect. No `empty_backoff` or `rate_cap` blocks were hit. **No evidence cadence
blocked a good moment** — the presenter fired on all 4 revisions it was allowed,
and the missing card (p577) was missing from retrieval, not from cadence.

---

## Recommended fixes — ordered by value / effort

1. **Weighted-sum fusion floor, or dense-rank rescue, instead of pure RRF.**
   (Fixes F2.1, biggest win.) Cheapest: after RRF, guarantee the top-K by raw
   `dense` cosine are also protected into the pool (analogous to fused-protect
   but on the dense list) — a dense-only note at cos 0.74 rank 4 should never be
   fused-buried at 17. Better: replace RRF with a weighted score combining
   normalized dense cosine + normalized BM25 + presence bonuses, so a single
   strong signal isn't punished for being alone. Small change in
   `ambient.py:162-172` / `_ordered_candidates`.

2. **Fix the salient query to prefer *mid-rare real* terms, not df=0 garbage.**
   (Fixes F2.2.) Drop terms with `doc_frequency==0` (they match nothing and are
   almost always ASR noise), and/or gate salient terms on being real tokens
   (in an alias/vocab set or df≥1). Trivial edit in `_salient_query`
   (`ambient.py:190-201`). Consider dropping the salient head entirely and
   relying on GLiNER spans — it currently contributes mostly noise (F2.3).

3. **Make `best_chunk` chunk-agnostic (use the actual winning embedding text).**
   (Fixes F4.1.) Let `best_chunk` consider `chunk`, `gist`, *and* `utterance`
   rows (or return the specific embedding row that maximized dense for that note)
   so the rerank doc contains the text that earned the note its rank. Change the
   `kind == "chunk"` filter in `store.py:238-239`.

4. **Also embed the *snapped* alias in `dense_gliner`, not the raw ASR span.**
   (Fixes F1.3.) The rapidfuzz snap already being added should feed the snapped
   surface form into the `dense_gliner` query (`ambient.py:212-225`), not just
   the alias exact-match table — otherwise the span-dense head keeps missing
   spelling-drifted notes like p577.

5. **Rebalance graph edge weights and add graph-vote.** (Fixes F3.1/F3.4.)
   Raise entity/topic weight relative to semantic (they carry more precise
   relevance), e.g. drop the log discount or add a per-edge-type multiplier so
   "shared Shopify entity" isn't the weakest edge. Add cumulative credit when a
   note is reached from multiple seeds (`ambient.py:249` `max` → additive or
   `1-∏(1-s)`), turning graph consensus into signal. If expansion still yields
   nothing above the floor after this, consider gating it off to save latency
   (F3.2 shows it currently contributes ~zero).

6. **Lengthen gists (or stop truncating mid-word).** (Fixes F5.2-adjacent /
   F6.2.) Raise the 120-char cap (`enrich.py:41,79`) to ~200 and cut on a word
   boundary; both rerank docs and presenter cards improve for free.

7. **Decouple `QUOTA_SEARCH` from `C2_TOP_K` and make eviction score-aware.**
   (Fixes F5.1/F5.2.) Either raise `QUOTA_SEARCH` above `C2_TOP_K` so a single
   pass can't fill and immediately churn the lane, or make `_evict_over_quota`
   consider `score` (evict lowest-score-unviewed, not oldest-unviewed) so a
   strong note isn't dropped for a fresh weak one (`slate.py:118-124`).

8. **Sharpen the `matched` evidence shown to the presenter.** (Fixes F6.3.)
   Set `SlateItem.heard` to the best chunk / top GLiNER span rather than the full
   window (`ambient.py:355`) so the presenter's per-card justification is
   concrete.

## Notes on what is NOT broken
- GLiNER2 extraction accuracy (Layer 1 core) — extracted correct entities every
  pass, including under ASR drift.
- Presenter selection quality (Layer 6) — picked the correct primary note when
  retrieval delivered it; zero errors.
- Cadence (Layer 7) — no good moment was blocked; the missing card was a
  retrieval miss, not a cadence miss.
- The synthetic-utterance *concept* is sound (spec's inverse-HyDE bridge); the
  defect is operational (F4.1: their winning text never reaches the reranker),
  not the idea itself.
