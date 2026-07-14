"""Chat policy adapter for the shared evidence retrieval engine."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import replace
from typing import Any, Callable, Literal, Mapping, Sequence

from taurine_core.indexer.store import Index
from taurine_core.ports import SpanExtractorPort
from taurine_core.retrieval.evidence import SharedRetrievalEngine
from taurine_core.retrieval.evidence_payloads import (
    FACT_APPEND_FIELDS,
    fact_group_payload,
    fact_source_identity,
    passage_payload,
)
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.rerank import PassageRerankCandidate
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    EvidencePath,
    RetrievalQuery,
    TemporalIntent,
)


EngineFactory = Callable[[EvidenceRegistry], SharedRetrievalEngine]
TemporalFilter = Literal["current", "historical", "any"]
Direction = Literal["outgoing", "incoming", "both"]

_DEPENDENT = re.compile(
    r"(?:\b(?:it|its|they|them|their|that|this|those|these|he|she|his|her)\b|"
    r"^(?:and|also|what about|how about|when|where|why|who)\b)",
    re.IGNORECASE,
)
_PRONOUN = re.compile(
    r"\b(?:it|its|they|them|their|that|this|those|these|he|she|his|her)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)
_QUERY_STOPWORDS = frozenset(
    {"a", "an", "and", "about", "does", "how", "is", "me", "of", "the", "to", "what", "when", "where", "why", "who"}
)

# M6: fact-support exposure caps — at most one non-selected support passage
# per fact and at most four extra passages per pack.
_FACT_SUPPORT_PER_FACT = 1
_FACT_SUPPORT_MAX_TOTAL = 4
_FACT_SUPPORT_SNIPPET_CHARS = 200


class ChatRetriever:
    """One per chat, owning stable evidence identities and accepted-turn state."""

    def __init__(
        self,
        index: Index,
        chat_id: str,
        *,
        engine_factory: EngineFactory,
        tool_engine_factory: EngineFactory,
        span_extractor: SpanExtractorPort | None = None,
    ) -> None:
        self.index = index
        self.chat_id = chat_id
        self._engine_factory = engine_factory
        self._tool_engine_factory = tool_engine_factory
        self._span_extractor = span_extractor
        persisted = index.chat_registry(chat_id)
        current = persisted.get("current") if isinstance(persisted, dict) else None
        initial = current if isinstance(current, dict) else persisted
        self.registry = _restore_registry(initial)
        self._accepted_reset = _float_map(
            initial.get("accepted_reset") if isinstance(initial, dict) else None
        )
        self._accepted_turn = _integer(
            initial.get("accepted_turn") if isinstance(initial, dict) else None
        )
        self._seed_reset = _float_map(
            initial.get("seed_reset") if isinstance(initial, dict) else None
        )
        pre_turn = persisted.get("pre_turn") if isinstance(persisted, dict) else None
        self._persisted_pre_turn = pre_turn if isinstance(pre_turn, dict) else None
        self._pre_turn: dict[str, object] | None = None
        self._turn_base: EvidencePack | None = None
        self._turn_read_ids: list[str] = []
        # H7: accumulator changes stay pending until the turn is finalized as
        # accepted (commit_turn) — failed/cancelled turns never advance state.
        self._pending_reset: dict[str, float] | None = None
        self.engine = engine_factory(self.registry)
        self._tool_engine: SharedRetrievalEngine | None = None
        self._fact_variant_cache: dict[str, tuple[Any, ...]] | None = None

    def context_pack(
        self,
        message: str,
        *,
        now: float,
        conversation_context: Sequence[str] = (),
        rewritten_query: str | None = None,
    ) -> EvidencePack:
        """Run Stage A; accumulator advancement stays pending until commit_turn."""

        self.begin_turn()
        context = tuple(str(row) for row in conversation_context[-2:] if str(row).strip())
        rewritten = (
            str(rewritten_query).strip()
            if rewritten_query and self.is_dependent_query(message, context)
            else _rewrite_query(message, context)
        )
        dependent = rewritten != message
        query = RetrievalQuery(
            text=rewritten,
            mode="chat",
            now=now,
            conversation_context=context,
            temporal_intent=TemporalIntent("any"),
        )
        session_reset = dict(self._seed_reset)
        if dependent and self._accepted_reset:
            for passage_id, score in self._accepted_reset.items():
                session_reset[passage_id] = max(session_reset.get(passage_id, 0.0), score)
        pack = self.engine.retrieve(
            query,
            entity_spans=self._entity_spans(message),
            session_reset=session_reset or None,
        )
        public_query = replace(query, text=message)
        pack = replace(
            pack,
            query=public_query,
            trace=replace(
                pack.trace,
                original_query=message,
                rewritten_query=rewritten,
            ),
        )
        self._turn_base = pack
        self._record_turn_read(pack.trace.read_ids)
        if pack.decision.accepted:
            self._pending_reset = {
                passage.passage_id: float(passage.rerank_score)
                for passage in pack.passages
            }
        elif not dependent:
            self._pending_reset = {}
        return pack

    def begin_turn(self) -> dict[str, float]:
        """Snapshot accepted state before Stage A or speculative tool search."""
        self._pre_turn = self.snapshot()
        self._turn_base = None
        self._turn_read_ids = []
        self._pending_reset = None
        return dict(self._seed_reset)

    def commit_turn(self) -> None:
        """Advance accepted-turn state after the turn finalized successfully.

        Called by the engine only after an accepted (non-cancelled,
        non-errored) finalization, per §14.5 accepted-turn keying (H7)."""

        if self._turn_base is None:
            return
        if self._pending_reset is not None:
            self._accepted_reset = self._pending_reset
        self._accepted_turn += 1
        self._seed_reset = {}  # graph-handoff seeds bias exactly one accepted turn
        self._pending_reset = None

    def discard_turn(self) -> None:
        """Restore the pre-turn snapshot after a cancelled/errored turn (H7).

        Same mechanism regeneration uses: the registry and accumulator return
        exactly to their state before this turn's Stage A ran."""

        if self._pre_turn is not None:
            self.restore(self._pre_turn)

    def seed_notes(self, note_ids: Sequence[str]) -> None:
        """Pre-populate the seed accumulator so the next turn is biased
        toward these notes ("Ask about this" from the map). Unknown note ids
        match no passages and are silently ignored."""

        requested = {str(value) for value in note_ids if str(value).strip()}
        if not requested:
            return
        for row in self.engine.store.passage_contexts(self.engine.snapshot.passage_ids):
            if str(row.note_id) in requested:
                passage_id = str(row.passage_id)
                self._seed_reset[passage_id] = max(
                    self._seed_reset.get(passage_id, 0.0), 1.0
                )

    def is_dependent_query(self, message: str, context: Sequence[str]) -> bool:
        normalized = tuple(str(row) for row in context[-2:] if str(row).strip())
        return _rewrite_query(message, normalized) != message

    def fallback_rewrite(self, message: str, context: Sequence[str]) -> str:
        normalized = tuple(str(row) for row in context[-2:] if str(row).strip())
        return _rewrite_query(message, normalized)

    def search_evidence(
        self,
        query: str,
        *,
        k: int,
        temporal: TemporalFilter,
        now: float,
        session_reset_override: Mapping[str, float] | None = None,
    ) -> EvidencePack:
        """Run a tool query without advancing accepted-turn state."""

        intent = TemporalIntent(temporal)
        request = RetrievalQuery(
            text=query,
            mode="tool",
            now=now,
            temporal_intent=intent,
        )
        if self._tool_engine is None:
            self._tool_engine = self._tool_engine_factory(self.registry)
        # Mid-turn tool queries see this turn's pending accepted reset (if
        # Stage A accepted) without committing it (H7).
        session_reset = (
            dict(session_reset_override)
            if session_reset_override is not None
            else (
                self._pending_reset
                if self._pending_reset is not None
                else self._accepted_reset
            )
        )
        pack = self._tool_engine.retrieve(
            request,
            entity_spans=self._entity_spans(query),
            session_reset=session_reset or None,
            passage_limit=max(0, int(k)),
        )
        self._record_turn_read(pack.trace.read_ids)
        if self._turn_base is None:
            self._turn_base = pack
        elif not self._turn_base.decision.accepted and pack.decision.accepted:
            stage_a = self._turn_base
            self._turn_base = replace(
                pack,
                trace=replace(
                    pack.trace,
                    original_query=stage_a.trace.original_query,
                    rewritten_query=stage_a.trace.rewritten_query,
                    debug={
                        **dict(pack.trace.debug),
                        "stage_a_abstention": stage_a.decision.reason,
                        "tool_query": query,
                    },
                ),
            )
        return pack

    def resolve_note(self, ref: str) -> str | None:
        record = self.registry.resolve(ref)
        if record is None or record.kind != "note":
            return None
        return record.source_id

    def _entity_spans(self, text: str) -> tuple[tuple[str, str | None], ...]:
        if self._span_extractor is None:
            return ()
        try:
            rows = self._span_extractor.extract(text)
        except Exception:
            return ()
        return tuple(
            (str(surface), str(etype) if etype else None)
            for surface, confidence, etype in rows
            if str(surface).strip() and float(confidence) > 0.0
        )

    def turn_pack(self) -> EvidencePack | None:
        """Materialize every passage exposed during this turn for auditing/UI."""

        base = self._turn_base
        if base is None:
            return None
        passages = {row.evidence_id: row for row in base.passages}
        notes = {row.note_ref: row for row in base.notes}
        facts = {row.fact_ref: row for row in base.facts}
        paths = {row.path_id: row for row in base.paths}
        read_refs = tuple(self._turn_read_ids)

        missing_passage_records = [
            (ref, record)
            for ref in read_refs
            if (record := self.registry.resolve(ref)) is not None
            and record.kind == "passage"
            and ref not in passages
        ]
        original_ids = tuple(
            str(record.payload.get("passage_id") or record.source_id)
            for _ref, record in missing_passage_records
        )
        context_by_id = {
            str(row.passage_id): row
            for row in self.engine.store.passage_contexts(original_ids)
        }
        for ref, record in missing_passage_records:
            passage_id = str(record.payload.get("passage_id") or record.source_id)
            row = context_by_id.get(passage_id)
            if row is None:
                continue
            is_excerpt = bool(record.payload.get("excerpt"))
            passages[ref] = EvidencePassage(
                evidence_id=ref,
                passage_id=passage_id,
                note_id=str(row.note_id),
                title=str(row.title),
                source=str(row.source),
                text=(
                    str(record.payload.get("text") or "")
                    if is_excerpt
                    else str(row.text)
                ),
                char_start=(
                    int(record.payload.get("char_start") or 0)
                    if is_excerpt
                    else int(row.char_start)
                ),
                char_end=(
                    int(record.payload.get("char_end") or 0)
                    if is_excerpt
                    else int(row.char_end)
                ),
                created=float(row.created),
                modified=float(row.modified),
                retrieval_score=0.0,
                rerank_score=0.0,
            )

        missing_note_ids = [
            record.source_id
            for ref in read_refs
            if (record := self.registry.resolve(ref)) is not None
            and record.kind == "note"
            and ref not in notes
        ]
        evidence_by_note: dict[str, list[str]] = {}
        for passage in passages.values():
            evidence_by_note.setdefault(passage.note_id, []).append(passage.evidence_id)
        for row in self.engine.store.note_contexts(missing_note_ids):
            ref = self.registry.ref_for("note", str(row.note_id))
            if ref is None:
                continue
            notes[ref] = EvidenceNote(
                note_ref=ref,
                note_id=str(row.note_id),
                title=str(row.title),
                source=str(row.source),
                modified=float(row.modified),
                score=0.0,
                evidence_ids=tuple(evidence_by_note.get(str(row.note_id), ())),
            )

        for ref in read_refs:
            record = self.registry.resolve(ref)
            if record is None:
                continue
            payload = record.payload
            if record.kind == "fact" and ref not in facts:
                logical = payload.get("logical_key")
                if not isinstance(logical, list) or len(logical) != 3:
                    continue
                facts[ref] = EvidenceFact(
                    fact_ref=ref,
                    fact_ids=tuple(str(value) for value in payload.get("fact_ids", ())),
                    subject=str(payload.get("subject") or logical[0]),
                    predicate=str(payload.get("predicate") or logical[1]),
                    object=str(payload.get("object") or logical[2]),
                    direction=str(payload.get("direction") or "forward"),
                    status=str(payload.get("status") or "candidate"),
                    observed_at=float(payload.get("observed_at") or 0.0),
                    valid_from=_optional_float(payload.get("valid_from")),
                    valid_to=_optional_float(payload.get("valid_to")),
                    confidence=0.0,
                    source_evidence_ids=tuple(str(value) for value in payload.get("source_evidence_ids", ())),
                )
            elif record.kind == "path" and ref not in paths:
                nodes = tuple(str(value) for value in payload.get("node_ids", ()))
                paths[ref] = EvidencePath(
                    path_id=ref,
                    node_ids=nodes,
                    edge_kinds=tuple(str(value) for value in payload.get("edge_kinds", ())),
                    fact_ids=tuple(str(value) for value in payload.get("fact_ids", ())),
                    weight=0.0,
                    rendered=str(payload.get("rendered") or " -> ".join(nodes)),
                )
        # M5: selected_ids stays the Stage A candidate set from the engine
        # trace (retrieved); read_ids carries everything consulted this turn.
        return replace(
            base,
            passages=tuple(passages[ref] for ref in read_refs if ref in passages),
            facts=tuple(facts[ref] for ref in read_refs if ref in facts),
            paths=tuple(paths[ref] for ref in read_refs if ref in paths),
            notes=tuple(notes[ref] for ref in read_refs if ref in notes),
            trace=replace(base.trace, read_ids=read_refs),
        )

    def expose_fact_support(
        self, pack: EvidencePack
    ) -> dict[str, tuple[str, str, str]]:
        """M6: make non-selected fact-support passages citable.

        For each pack fact, at most one support ref that is not already a
        rendered passage is marked read (so citing it validates) and returned
        as ``fact_ref → (support_ref, title, snippet)`` for prompt rendering.
        Bounded to ``_FACT_SUPPORT_MAX_TOTAL`` extra passages per pack; only
        refs whose registry payload carries text are exposed."""

        rendered = {row.evidence_id for row in pack.passages}
        exposed: dict[str, tuple[str, str, str]] = {}
        used_refs: set[str] = set()
        for fact in pack.facts:
            if len(exposed) >= _FACT_SUPPORT_MAX_TOTAL:
                break
            per_fact = 0
            for ref in fact.source_evidence_ids:
                if per_fact >= _FACT_SUPPORT_PER_FACT:
                    break
                if ref in rendered or ref in used_refs:
                    continue
                record = self.registry.resolve(ref)
                if record is None or record.kind != "passage":
                    continue
                text = str(record.payload.get("text") or "").strip()
                if not text:
                    continue
                passage_id = str(record.payload.get("passage_id") or record.source_id)
                rows = self.engine.store.passage_contexts((passage_id,))
                title = str(rows[0].title) if rows else ""
                self.registry.mark_read(ref)
                self._record_turn_read((ref,))
                exposed[fact.fact_ref] = (
                    ref,
                    title,
                    text[:_FACT_SUPPORT_SNIPPET_CHARS],
                )
                used_refs.add(ref)
                per_fact += 1
        return exposed

    def evidence_sufficient(self) -> bool:
        """M11: True when some evidence set read this turn cleared its floor.

        Stage A acceptance or an accepted tool pack (which promotes the turn
        base) both count; fallback/abstained-only turns do not. Deterministic —
        reuses the calibrated decisions already on the packs."""

        base = self._turn_base
        return base is not None and bool(base.decision.accepted)

    def read_note(
        self,
        note_ref: str,
        *,
        query: str,
        cursor: str | None = None,
        max_chars: int,
    ) -> dict[str, object]:
        """Return registered passages, ranked by query or paged sequentially."""

        note_id = self.resolve_note(note_ref)
        if note_id is None:
            raise ValueError(f"could not resolve note: {note_ref}")
        all_rows = tuple(
            row
            for row in self.engine.store.passage_contexts(self.engine.snapshot.passage_ids)
            if str(row.note_id) == note_id
        )
        offset, sequential, char_offset, cursor_order = _decode_cursor(cursor)
        if cursor_order:
            row_by_id = {str(row.passage_id): row for row in all_rows}
            ordered = [row_by_id[row_id] for row_id in cursor_order if row_id in row_by_id]
        elif sequential:
            ordered = sorted(all_rows, key=lambda row: (int(row.char_start), str(row.passage_id)))
        else:
            terms = _terms(query)
            lexical = sorted(
                all_rows,
                key=lambda row: (
                    -len(terms & _terms(str(row.text))),
                    int(row.char_start),
                    str(row.passage_id),
                ),
            )
            ordered = lexical
            if self._tool_engine is None:
                self._tool_engine = self._tool_engine_factory(self.registry)
            reranker = getattr(self._tool_engine, "reranker", None)
            if reranker is not None:
                candidates = tuple(
                    PassageRerankCandidate(
                        passage_id=str(row.passage_id),
                        note_id=str(row.note_id),
                        title=str(row.title),
                        source=str(row.source),
                        text=str(row.text),
                        recall_score=float(len(terms & _terms(str(row.text)))) / max(1, len(terms)),
                    )
                    for row in lexical
                )
                reranked = reranker.rerank(query, candidates)
                if not reranked.degraded:
                    row_by_id = {str(row.passage_id): row for row in all_rows}
                    ordered = [
                        row_by_id[decision.candidate.passage_id]
                        for decision in reranked.decisions
                    ]
                    ranked_ids = {str(row.passage_id) for row in ordered}
                    ordered.extend(
                        row for row in lexical if str(row.passage_id) not in ranked_ids
                    )
        budget = max(1, int(max_chars))
        consumed = 0
        passages: list[dict[str, object]] = []
        index = offset
        while index < len(ordered):
            row = ordered[index]
            full_text = str(row.text)
            local_start = char_offset if index == offset else 0
            if local_start >= len(full_text):
                index += 1
                char_offset = 0
                continue
            remaining_budget = budget - consumed
            if remaining_budget <= 0:
                break
            if passages and len(full_text) - local_start > remaining_budget:
                break
            text = full_text[local_start : local_start + remaining_budget]
            absolute_start = int(row.char_start) + local_start
            absolute_end = absolute_start + len(text)
            partial = local_start > 0 or len(text) < len(full_text)
            if partial:
                # Excerpts keep their own offset-suffixed identity and shape.
                evidence_ref = self.registry.register(
                    "passage",
                    f"{row.passage_id}:excerpt:{absolute_start}:{absolute_end}",
                    {
                        "passage_id": str(row.passage_id),
                        "note_id": note_id,
                        "text": text,
                        "char_start": absolute_start,
                        "char_end": absolute_end,
                        "excerpt": True,
                    },
                )
            else:
                # C2: a full passage shares the engine's identity — reuse the
                # Stage A ref when present, else register the canonical payload.
                evidence_ref = self.registry.ref_for("passage", str(row.passage_id))
                if evidence_ref is None:
                    evidence_ref = self.registry.register(
                        "passage", str(row.passage_id), passage_payload(row)
                    )
            self.registry.mark_read(note_ref)
            self.registry.mark_read(evidence_ref)
            self._record_turn_read((note_ref, evidence_ref))
            passages.append(
                {
                    "evidence_id": evidence_ref,
                    "text": text,
                    "char_start": absolute_start,
                    "char_end": absolute_end,
                }
            )
            consumed += len(text)
            local_end = local_start + len(text)
            if local_end < len(full_text):
                char_offset = local_end
                break
            index += 1
            char_offset = 0
            if consumed >= budget:
                break
        truncated = char_offset > 0 or index < len(ordered)
        return {
            "note": note_ref,
            "title": str(all_rows[0].title) if all_rows else "",
            "source": str(all_rows[0].source) if all_rows else "",
            "passages": passages,
            "truncated": truncated,
            "next_cursor": (
                _encode_cursor(
                    index,
                    sequential=True,
                    char_offset=char_offset,
                    order=tuple(str(row.passage_id) for row in ordered),
                )
                if truncated
                else None
            ),
        }

    def expand_evidence(
        self,
        ref: str,
        *,
        direction: Direction,
        max_hops: int,
        limit: int,
    ) -> dict[str, object]:
        """Traverse the directed base graph and return only registered evidence."""

        record = self.registry.resolve(ref)
        if record is None:
            raise ValueError(f"could not resolve evidence: {ref}")
        if record.kind not in {"note", "passage", "fact"}:
            raise ValueError("expansion requires a registered N#, E#, or F# reference")
        if record.kind == "note":
            seeds = tuple(
                str(row.passage_id)
                for row in self.engine.store.passage_contexts(self.engine.snapshot.passage_ids)
                if str(row.note_id) == record.source_id
            )
        else:
            seeds = _record_graph_seeds(record, direction)
        if not seeds:
            return {"from": ref, "results": [], "paths": []}
        edges = tuple(self.engine.snapshot.base_graph.edges)
        queue = [(seed, (seed,), (), ()) for seed in seeds]
        visited = set(seeds)
        results: list[dict[str, object]] = []
        paths: list[dict[str, object]] = []
        supporting_passages: dict[str, dict[str, object]] = {}
        passage_ids = set(self.engine.snapshot.passage_ids)
        max_hops = min(4, max(1, int(max_hops)))
        while queue and len(results) < max(0, int(limit)):
            node, nodes, kinds, fact_ids = queue.pop(0)
            if len(kinds) >= max_hops:
                continue
            for edge in edges:
                target = _directed_target(edge, node, direction)
                if target is None or target in visited:
                    continue
                visited.add(target)
                next_nodes = (*nodes, target)
                next_kinds = (*kinds, str(edge.kind))
                next_facts = (*fact_ids, *(str(value) for value in edge.fact_ids))
                queue.append((target, next_nodes, next_kinds, next_facts))
                if target not in passage_ids:
                    continue
                context = self.engine.store.passage_contexts((target,))
                if not context:
                    continue
                row = context[0]
                evidence_ref = self.registry.register(
                    "passage", target,
                    {"passage_id": target, "note_id": str(row.note_id), "text": str(row.text), "char_start": int(row.char_start), "char_end": int(row.char_end)},
                )
                note_ref = self.registry.ref_for("note", str(row.note_id))
                if note_ref is None:
                    note_ref = self.registry.register("note", str(row.note_id), {"note_id": str(row.note_id)})
                matching_facts, fact_support, fact_contexts = self._register_traversed_facts(
                    next_facts
                )
                supporting_passages.update(
                    {str(item["evidence_id"]): item for item in fact_support}
                )
                rendered_path = _render_expansion_path(
                    next_nodes, next_kinds, fact_contexts, self.engine.store
                )
                path_key = "\x1f".join((*next_nodes, *next_kinds, *next_facts))
                path_ref = self.registry.register(
                    "path", path_key,
                    {"node_ids": list(next_nodes), "edge_kinds": list(next_kinds), "fact_ids": list(next_facts), "rendered": rendered_path},
                )
                for cite_ref in (evidence_ref, note_ref, path_ref):
                    self.registry.mark_read(cite_ref)
                for fact_ref in matching_facts:
                    self.registry.mark_read(fact_ref)
                self._record_turn_read(
                    (evidence_ref, note_ref, path_ref, *matching_facts)
                )
                results.append(
                    {"evidence_id": evidence_ref, "note": note_ref, "title": str(row.title), "source": str(row.source), "text": str(row.text), "fact_refs": matching_facts, "path": path_ref}
                )
                paths.append({"path_id": path_ref, "rendered": rendered_path, "node_ids": list(next_nodes), "edge_kinds": list(next_kinds), "fact_refs": matching_facts})
                if len(results) >= limit:
                    break
        return {"from": ref, "results": results, "supporting_passages": list(supporting_passages.values()), "paths": paths}

    def _fact_variant_groups(self) -> dict[str, tuple[Any, ...]]:
        """fact_id → its variant's full provenance group, in the engine's
        grouping order (logical key + status + validity, rows sorted by
        fact_id). Built once per retriever from the snapshot fact index."""

        if self._fact_variant_cache is None:
            reader = getattr(self.engine.store, "fact_contexts", None)
            contexts = reader() if callable(reader) else ()
            grouped: dict[tuple[Any, ...], list[Any]] = {}
            for context in contexts:
                variant = (
                    tuple(context.logical_key),
                    str(context.status),
                    getattr(context, "valid_from", None),
                    getattr(context, "valid_to", None),
                )
                grouped.setdefault(variant, []).append(context)
            cache: dict[str, tuple[Any, ...]] = {}
            for rows in grouped.values():
                ordered = tuple(sorted(rows, key=lambda row: str(row.fact_id)))
                for row in ordered:
                    cache[str(row.fact_id)] = ordered
            self._fact_variant_cache = cache
        return self._fact_variant_cache

    def _register_traversed_facts(
        self, fact_ids: Sequence[str]
    ) -> tuple[list[str], list[dict[str, object]], tuple[Any, ...]]:
        """C2: register traversed facts under the engine's grouped identity.

        The payload carries the full provenance group from the snapshot fact
        index (all fact_ids of the logical variant, observed_at = max), built
        by the same shared helpers the engine uses, so a fact the engine
        already registered is reused via ref_for instead of conflicting."""

        groups = self._fact_variant_groups()
        traversed = list(dict.fromkeys(str(value) for value in fact_ids))
        refs: list[str] = _fact_refs(self.registry, set(traversed))
        support: list[dict[str, object]] = []
        contexts_out: list[Any] = []
        registered_variants: set[tuple[Any, ...]] = set()
        for fact_id in traversed:
            group = groups.get(fact_id)
            if group is None:
                continue
            contexts_out.extend(
                row for row in group if str(row.fact_id) == fact_id
            )
            first = group[0]
            variant = (
                tuple(first.logical_key),
                str(first.status),
                getattr(first, "valid_from", None),
                getattr(first, "valid_to", None),
            )
            if variant in registered_variants:
                continue
            registered_variants.add(variant)
            source_refs: list[str] = []
            for row in group:
                passage_rows = self.engine.store.passage_contexts(
                    (str(row.source_passage_id),)
                )
                if not passage_rows:
                    continue
                passage = passage_rows[0]
                evidence_ref = self.registry.ref_for(
                    "passage", str(passage.passage_id)
                )
                if evidence_ref is None:
                    evidence_ref = self.registry.register(
                        "passage",
                        str(passage.passage_id),
                        passage_payload(passage),
                    )
                note_ref = self.registry.ref_for("note", str(passage.note_id))
                if note_ref is None:
                    note_ref = self.registry.register(
                        "note", str(passage.note_id), {"note_id": str(passage.note_id)}
                    )
                source_refs.append(evidence_ref)
                for current_ref in (note_ref, evidence_ref):
                    self.registry.mark_read(current_ref)
                self._record_turn_read((note_ref, evidence_ref))
                support.append(
                    {
                        "evidence_id": evidence_ref,
                        "note": note_ref,
                        "title": str(passage.title),
                        "source": str(passage.source),
                        "text": str(passage.text),
                    }
                )
            identity = fact_source_identity(
                self.engine.snapshot.build_id,
                first.logical_key,
                str(first.status),
                getattr(first, "valid_from", None),
                getattr(first, "valid_to", None),
            )
            fact_ref = self.registry.ref_for("fact", identity)
            if fact_ref is None:
                fact_ref = self.registry.register(
                    "fact",
                    identity,
                    fact_group_payload(
                        first.logical_key,
                        str(first.status),
                        getattr(first, "valid_from", None),
                        getattr(first, "valid_to", None),
                        group,
                    ),
                )
            self.registry.merge_payload(
                fact_ref,
                {
                    "fact_ids": [str(row.fact_id) for row in group],
                    "source_evidence_ids": list(dict.fromkeys(source_refs)),
                },
                append_list_fields=FACT_APPEND_FIELDS,
            )
            self.registry.mark_read(fact_ref)
            self._record_turn_read((fact_ref,))
            if fact_ref not in refs:
                refs.append(fact_ref)
        return refs, support, tuple(contexts_out)

    def snapshot(self) -> dict[str, object]:
        return {
            "version": 1,
            "registry": self.registry.snapshot(),
            "accepted_reset": dict(sorted(self._accepted_reset.items())),
            "accepted_turn": self._accepted_turn,
            "seed_reset": dict(sorted(self._seed_reset.items())),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        self.registry = _restore_registry(state)
        self._accepted_reset = _float_map(state.get("accepted_reset"))
        self._accepted_turn = _integer(state.get("accepted_turn"))
        self._seed_reset = _float_map(state.get("seed_reset"))
        self.engine = self._engine_factory(self.registry)
        self._tool_engine = None
        self._turn_base = None
        self._turn_read_ids = []
        self._pending_reset = None

    def restore_pre_turn(self) -> None:
        state = self._persisted_pre_turn
        if state is not None:
            self.restore(state)

    def save(self) -> None:
        self.index.set_chat_registry(
            self.chat_id,
            {
                "version": 1,
                "current": self.snapshot(),
                "pre_turn": self._pre_turn or self.snapshot(),
            },
        )

    def _record_turn_read(self, refs: Sequence[str]) -> None:
        for ref in refs:
            value = str(ref)
            if value and value not in self._turn_read_ids:
                self._turn_read_ids.append(value)


def _rewrite_query(message: str, context: tuple[str, ...]) -> str:
    if not context or _DEPENDENT.search(message) is None:
        return message
    if _PRONOUN.search(message) is None:
        message_terms = _terms(message) - _QUERY_STOPWORDS
        context_terms = _terms("\n".join(context)) - _QUERY_STOPWORDS
        if message_terms and not message_terms.intersection(context_terms):
            return message
    if not message.strip():
        return message
    return "\n".join((*context, message))


def _restore_registry(state: object) -> EvidenceRegistry:
    if not isinstance(state, Mapping):
        return EvidenceRegistry()
    snapshot = state.get("registry")
    if not isinstance(snapshot, Mapping):
        return EvidenceRegistry()
    return EvidenceRegistry.from_snapshot(snapshot)


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(score) for key, score in value.items()}


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _terms(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN.finditer(text)}


def _encode_cursor(
    offset: int,
    *,
    sequential: bool,
    char_offset: int = 0,
    order: tuple[str, ...] = (),
) -> str:
    payload = json.dumps(
        {
            "offset": offset,
            "sequential": sequential,
            "char_offset": char_offset,
            "order": list(order),
        },
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> tuple[int, bool, int, tuple[str, ...]]:
    if not cursor:
        return 0, False, 0, ()
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        order = payload.get("order", ())
        if not isinstance(order, list) or not all(isinstance(row, str) for row in order):
            raise ValueError
        return (
            max(0, int(payload["offset"])),
            bool(payload.get("sequential", True)),
            max(0, int(payload.get("char_offset", 0))),
            tuple(order),
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("invalid read cursor") from None


def _record_graph_seeds(record: Any, direction: Direction) -> tuple[str, ...]:
    if record.kind == "passage":
        return (str(record.payload.get("passage_id") or record.source_id),)
    if record.kind == "fact":
        subject = str(record.payload.get("subject_entity_id") or "")
        object_id = str(record.payload.get("object_entity_id") or "")
        if direction == "outgoing":
            return (subject,) if subject else ()
        if direction == "incoming":
            return (object_id,) if object_id else ()
        return tuple(value for value in (subject, object_id) if value)
    return ()


def _directed_target(edge: Any, node: str, direction: Direction) -> str | None:
    if direction in {"outgoing", "both"} and str(edge.src_id) == node:
        return str(edge.dst_id)
    if direction in {"incoming", "both"} and str(edge.dst_id) == node:
        return str(edge.src_id)
    return None


def _fact_refs(registry: EvidenceRegistry, fact_ids: set[str]) -> list[str]:
    refs: list[str] = []
    for ref in registry.refs("retrieved"):
        record = registry.resolve(ref)
        if record is None or record.kind != "fact":
            continue
        registered = record.payload.get("fact_ids")
        if isinstance(registered, list) and fact_ids.intersection(str(value) for value in registered):
            refs.append(ref)
    return refs


def _render_expansion_path(
    nodes: Sequence[str],
    kinds: Sequence[str],
    facts: Sequence[Any],
    store: Any,
) -> str:
    passage_rows = store.passage_contexts(tuple(nodes))
    passage_labels = {
        str(row.passage_id): f'"{row.title}" passage' for row in passage_rows
    }
    entity_labels: dict[str, str] = {}
    for fact in facts:
        entity_labels[str(fact.subject_entity_id)] = str(fact.subject)
        entity_labels[str(fact.object_entity_id)] = str(fact.object)

    def label(node: str) -> str:
        return passage_labels.get(node, entity_labels.get(node, node))

    if not nodes:
        return ""
    rendered = label(str(nodes[0]))
    for index, kind in enumerate(kinds):
        left, right = str(nodes[index]), str(nodes[index + 1])
        relation = str(kind)
        matching = next(
            (
                fact
                for fact in facts
                if {str(fact.subject_entity_id), str(fact.object_entity_id)}
                == {left, right}
            ),
            None,
        )
        if matching is not None:
            relation = str(matching.predicate)
        elif relation == "contains":
            relation = "mentions"
        elif relation == "semantic":
            relation = "semantically_related"
        rendered += f" --{relation}--> {label(right)}"
    return rendered
