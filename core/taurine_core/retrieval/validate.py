"""Citation validity, availability, fact provenance, and temporal validation.

Per-sentence citation-coverage and NLI-entailment stages were removed by
decision on 2026-07-10: they rejected honest absence answers (entailment
cannot prove an absence) and no hallucinated citations were ever observed.
See docs/graph-followups.md.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal

from taurine_core.retrieval.registry import EvidenceRecord, EvidenceRegistry


ValidationMode = Literal["live", "chat", "tool"]
_CITATION_RE = re.compile(r"\[([A-Z])(\d+)\]")
_COVERAGE_PREFIXES = frozenset({"E", "F", "T", "M"})
_HISTORICAL_QUALIFIERS = re.compile(
    r"\b(?:formerly|previously|historically|used to|at the time|was|were|had|did)\b",
    flags=re.IGNORECASE,
)
_INACTIVE_FACT_STATUSES = frozenset(
    {"historical", "superseded", "conflicted", "candidate"}
)



@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    accepted_output: str
    unsupported_sentences: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    repair_required: bool
    cited_ids: tuple[str, ...]
    degraded_reason: str
    evidence_excerpt_required: bool = False
    # (sentence, reason) per unsupported sentence, in rejection order (L31).
    unsupported_details: tuple[tuple[str, str], ...] = ()

    @property
    def requires_evidence_excerpt(self) -> bool:
        return self.evidence_excerpt_required


@dataclass(frozen=True)
class _Sentence:
    rendered: str
    claim: str
    citations: tuple[str, ...]


class EvidenceValidator:
    def __init__(self, registry: EvidenceRegistry) -> None:
        self.registry = registry

    def validate(
        self,
        output: str,
        *,
        mode: ValidationMode,
        available_ids: Collection[str],
        supplied_ids: Collection[str] = (),
    ) -> ValidationResult:
        """Validate ``output``; ``available_ids`` is the caller's explicit
        citable set (L13: no session-cumulative default — callers must scope
        availability to the current turn/pass themselves)."""
        if mode not in {"live", "chat", "tool"}:
            raise ValueError(f"unsupported validation mode: {mode}")
        sentences, visible_text = _sentences(output)
        cited_ids = _ordered_refs(
            match.group(0)[1:-1] for match in _CITATION_RE.finditer(visible_text)
        )
        records = {ref: self.registry.resolve(ref) for ref in cited_ids}
        invalid: set[str] = {ref for ref, record in records.items() if record is None}
        available = set(available_ids)
        unavailable = {
            ref
            for ref, record in records.items()
            if record is not None
            and record.kind in {"passage", "transcript", "meeting"}
            and ref not in available
        }
        invalid.update(unavailable)

        fact_sources: dict[str, tuple[str, ...]] = {}
        provenance_invalid: set[str] = set()
        for ref, record in records.items():
            if record is None or record.kind != "fact":
                continue
            sources = _fact_source_refs(record)
            citeable_sources = tuple(
                source
                for source in sources
                if source in available
                and (source_record := self.registry.resolve(source)) is not None
                and source_record.kind == "passage"
            )
            fact_sources[ref] = citeable_sources
            if not citeable_sources:
                provenance_invalid.add(ref)
        invalid.update(provenance_invalid)

        unsupported: list[str] = []
        details: list[tuple[str, str]] = []
        reason = ""

        def reject(sentence: _Sentence, sentence_reason: str) -> None:
            nonlocal reason
            unsupported.append(sentence.rendered)
            details.append((sentence.rendered, sentence_reason))
            reason = reason or sentence_reason

        for sentence in sentences:
            sentence_refs = set(sentence.citations)
            if not sentence_refs:
                continue  # uncited prose is the model's judgment, not ours
            if sentence_refs & invalid:
                if sentence_refs & provenance_invalid:
                    reject(sentence, "fact_provenance")
                elif sentence_refs & unavailable:
                    reject(sentence, "evidence_unavailable")
                else:
                    reject(sentence, "invalid_citation")
                continue
            if _temporally_inconsistent(sentence, records):
                reject(sentence, "temporal_inconsistency")

        if invalid and not reason:
            reason = "invalid_citation"
        accepted = not invalid and not unsupported
        if accepted:
            for ref in cited_ids:
                self.registry.mark_cited(ref)
        return ValidationResult(
            accepted=accepted,
            accepted_output=output if accepted else "",
            unsupported_sentences=tuple(dict.fromkeys(unsupported)),
            invalid_citations=_ordered_refs(invalid),
            repair_required=bool(not accepted and mode != "live"),
            cited_ids=cited_ids,
            degraded_reason=reason,
            unsupported_details=tuple(dict.fromkeys(details)),
        )


def validate_evidence_output(
    output: str,
    registry: EvidenceRegistry,
    *,
    mode: ValidationMode,
    available_ids: Collection[str],
    supplied_ids: Collection[str] = (),
) -> ValidationResult:
    return EvidenceValidator(registry).validate(
        output,
        mode=mode,
        available_ids=available_ids,
        supplied_ids=supplied_ids,
    )


def _sentences(output: str) -> tuple[tuple[_Sentence, ...], str]:
    rows: list[_Sentence] = []
    visible: list[str] = []
    in_code = False
    for raw_line in output.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        visible.append(raw_line)
        for rendered in _split_line(stripped):
            citations = _ordered_refs(
                match.group(0)[1:-1] for match in _CITATION_RE.finditer(rendered)
            )
            claim = _CITATION_RE.sub("", rendered).strip()
            claim = re.sub(r"^[-*+]\s+", "", claim).strip()
            rows.append(_Sentence(rendered, claim, citations))
    return tuple(rows), "\n".join(visible)


def _split_line(line: str) -> tuple[str, ...]:
    matches = re.findall(
        r".+?(?:[.!?](?:\s*\[[A-Z]\d+\])*(?=\s+|$)|$)",
        line,
    )
    return tuple(part.strip() for part in matches if part.strip())


def _fact_source_refs(record: EvidenceRecord) -> tuple[str, ...]:
    for key in ("source_evidence_ids", "evidence_ids", "source_refs"):
        value = record.payload.get(key)
        if isinstance(value, (list, tuple)):
            return _ordered_refs(str(ref) for ref in value)
    return ()


def _temporally_inconsistent(
    sentence: _Sentence,
    records: dict[str, EvidenceRecord | None],
) -> bool:
    facts = [
        records[ref]
        for ref in sentence.citations
        if ref.startswith("F") and records.get(ref) is not None
    ]
    if not facts or _HISTORICAL_QUALIFIERS.search(sentence.claim):
        return False
    statuses = {
        str(record.payload.get("status", "active")).casefold()
        for record in facts
        if record is not None
    }
    return bool(statuses) and statuses <= _INACTIVE_FACT_STATUSES


def _payload_text(record: EvidenceRecord) -> str:
    for key in ("text", "body", "transcript", "content", "quote"):
        value = record.payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ordered_refs(refs: Iterable[str]) -> tuple[str, ...]:
    unique = set(refs)

    def key(ref: str) -> tuple[int, int, str]:
        prefix_order = {prefix: index for index, prefix in enumerate("NEFPTM")}
        try:
            number = int(ref[1:])
        except ValueError:
            number = 0
        return (prefix_order.get(ref[:1], 99), number, ref)

    return tuple(sorted(unique, key=key))
