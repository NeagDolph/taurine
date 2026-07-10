"""Evidence-native live grounding accumulator and display policy."""

from __future__ import annotations

from dataclasses import dataclass

from taurine_core import config
from taurine_core.retrieval.registry import EvidenceRegistry
from taurine_core.retrieval.types import (
    EvidenceFact,
    EvidenceNote,
    EvidencePack,
    EvidencePassage,
    EvidencePath,
    RetrievalDecision,
    RetrievalTrace,
)


@dataclass(frozen=True)
class GroundingUpdate:
    """Accumulated support plus a display-only view for one live retrieval pass."""

    decision: RetrievalDecision
    passages: tuple[EvidencePassage, ...]
    facts: tuple[EvidenceFact, ...]
    paths: tuple[EvidencePath, ...]
    notes: tuple[EvidenceNote, ...]
    display_passages: tuple[EvidencePassage, ...]
    display_notes: tuple[EvidenceNote, ...]
    no_new_evidence: bool
    new_support_ids: tuple[str, ...]
    revision: int
    trace: RetrievalTrace | None = None
    # Only evidence accepted for the current retrieval pass may support the
    # current summary update. The accumulated fields above are session/UI state.
    support_passages: tuple[EvidencePassage, ...] = ()
    support_facts: tuple[EvidenceFact, ...] = ()
    support_paths: tuple[EvidencePath, ...] = ()
    support_notes: tuple[EvidenceNote, ...] = ()


class GroundingEngine:
    """Accumulate accepted registered evidence; keep hysteresis out of support."""

    def __init__(
        self,
        registry: EvidenceRegistry,
        *,
        display_limit: int = config.LIVE_EVIDENCE_PASSAGES,
        hysteresis: float | None = config.GROUNDING_HYSTERESIS,
    ) -> None:
        if display_limit < 0:
            raise ValueError("display limit must be non-negative")
        if hysteresis is not None and hysteresis < 1.0:
            raise ValueError("display hysteresis must be at least one")
        self.registry = registry
        self._display_limit = int(display_limit)
        self._hysteresis = hysteresis
        self._passages: dict[str, EvidencePassage] = {}
        self._facts: dict[str, EvidenceFact] = {}
        self._paths: dict[str, EvidencePath] = {}
        self._notes: dict[str, EvidenceNote] = {}
        self._display_ids: tuple[str, ...] = ()
        self._revision = 0
        self._trace: RetrievalTrace | None = None

    def on_pack(
        self,
        pack: EvidencePack,
        *,
        topic_shift: bool = False,
    ) -> GroundingUpdate:
        self._trace = pack.trace
        reset_changed = False
        if topic_shift:
            reset_changed = bool(self.evidence_ids() or self._display_ids)
            self._clear()

        if not pack.decision.accepted:
            if reset_changed:
                self._revision += 1
            return self._update(
                pack.decision,
                no_new_evidence=True,
                new_support_ids=(),
                support=None,
            )

        self._validate_pack(pack)
        prior_support = set(self._support_ids())
        for row in pack.passages:
            self._passages[row.evidence_id] = row
        for row in pack.facts:
            self._facts[row.fact_ref] = row
        for row in pack.paths:
            self._paths[row.path_id] = row
        for row in pack.notes:
            self._notes[row.note_ref] = row
        new_support_ids = _ordered_refs(set(self._support_ids()) - prior_support)

        self._display_ids = self._select_display(pack.passages)
        if reset_changed or new_support_ids:
            self._revision += 1
        return self._update(
            pack.decision,
            no_new_evidence=not new_support_ids,
            new_support_ids=new_support_ids,
            support=pack,
        )

    def evidence_ids(self) -> tuple[str, ...]:
        return _ordered_refs(
            set(self._passages) | set(self._facts) | set(self._paths) | set(self._notes)
        )

    def session_reset(self) -> dict[str, float]:
        """Passage-node reset mass for the next shared retrieval pass."""

        return {
            row.passage_id: max(0.0, float(row.rerank_score))
            for row in self._ordered_passages()
            if row.rerank_score > 0.0
        }

    def _validate_pack(self, pack: EvidencePack) -> None:
        rows = (
            *((row.evidence_id, "passage") for row in pack.passages),
            *((row.fact_ref, "fact") for row in pack.facts),
            *((row.path_id, "path") for row in pack.paths),
            *((row.note_ref, "note") for row in pack.notes),
        )
        for ref, expected_kind in rows:
            record = self.registry.resolve(ref)
            if record is None or record.kind != expected_kind:
                raise ValueError(
                    f"accepted evidence {ref} is missing from the shared registry"
                )
            if "read" not in record.states:
                raise ValueError(
                    f"accepted evidence {ref} was not supplied/read"
                )
        for fact in pack.facts:
            for source_ref in fact.source_evidence_ids:
                record = self.registry.resolve(source_ref)
                if (
                    record is None
                    or record.kind != "passage"
                    or "read" not in record.states
                ):
                    raise ValueError(
                        f"fact {fact.fact_ref} source {source_ref} was not supplied/read"
                    )

    def _select_display(
        self,
        current: tuple[EvidencePassage, ...],
    ) -> tuple[str, ...]:
        if self._display_limit == 0:
            return ()
        current_by_id = {row.evidence_id: row for row in current}
        ordered_current = sorted(
            current_by_id,
            key=lambda ref: (-_display_score(current_by_id[ref]), ref),
        )
        selected: list[str] = []
        used: set[str] = set()
        for slot in range(self._display_limit):
            incumbent_id = (
                self._display_ids[slot] if slot < len(self._display_ids) else None
            )
            incumbent = (
                self._passages.get(incumbent_id) if incumbent_id is not None else None
            )
            challenger_id = next(
                (ref for ref in ordered_current if ref not in used), None
            )
            challenger = (
                current_by_id.get(challenger_id)
                if challenger_id is not None
                else None
            )
            if incumbent is not None and incumbent_id not in used:
                if challenger is None or challenger_id == incumbent_id:
                    chosen = incumbent_id
                elif self._hysteresis is None or _display_score(challenger) > (
                    _display_score(incumbent) * self._hysteresis
                ):
                    chosen = challenger_id
                else:
                    chosen = incumbent_id
            else:
                chosen = challenger_id
            if chosen is None:
                break
            selected.append(chosen)
            used.add(chosen)
        for ref in ordered_current:
            if len(selected) >= self._display_limit:
                break
            if ref not in used:
                selected.append(ref)
                used.add(ref)
        return tuple(selected)

    def _ordered_passages(self) -> tuple[EvidencePassage, ...]:
        return tuple(
            sorted(
                self._passages.values(),
                key=lambda row: (-_display_score(row), row.evidence_id),
            )
        )

    def _update(
        self,
        decision: RetrievalDecision,
        *,
        no_new_evidence: bool,
        new_support_ids: tuple[str, ...],
        support: EvidencePack | None,
    ) -> GroundingUpdate:
        passages = self._ordered_passages()
        facts = tuple(
            sorted(
                self._facts.values(),
                key=lambda row: (-row.confidence, row.fact_ref),
            )
        )
        paths = tuple(
            sorted(
                self._paths.values(),
                key=lambda row: (-row.weight, row.path_id),
            )
        )
        notes = tuple(
            sorted(
                self._notes.values(),
                key=lambda row: (-row.score, row.note_ref),
            )
        )
        display_passages = tuple(
            self._passages[ref]
            for ref in self._display_ids
            if ref in self._passages
        )
        display_note_ids = {row.note_id for row in display_passages}
        display_notes = tuple(row for row in notes if row.note_id in display_note_ids)
        return GroundingUpdate(
            decision=decision,
            passages=passages,
            facts=facts,
            paths=paths,
            notes=notes,
            display_passages=display_passages,
            display_notes=display_notes,
            no_new_evidence=no_new_evidence,
            new_support_ids=new_support_ids,
            revision=self._revision,
            trace=self._trace,
            support_passages=support.passages if support is not None else (),
            support_facts=support.facts if support is not None else (),
            support_paths=support.paths if support is not None else (),
            support_notes=support.notes if support is not None else (),
        )

    def _support_ids(self) -> tuple[str, ...]:
        return _ordered_refs(set(self._passages) | set(self._facts))

    def _clear(self) -> None:
        self._passages.clear()
        self._facts.clear()
        self._paths.clear()
        self._notes.clear()
        self._display_ids = ()


def _display_score(row: EvidencePassage) -> float:
    return max(0.0, float(row.rerank_score))


def _ordered_refs(refs: set[str]) -> tuple[str, ...]:
    prefix_order = {prefix: index for index, prefix in enumerate("NEFPT")}

    def key(ref: str) -> tuple[int, int, str]:
        try:
            number = int(ref[1:])
        except ValueError:
            number = 0
        return (prefix_order.get(ref[:1], 99), number, ref)

    return tuple(sorted(refs, key=key))
