"""Sentence-level evidence, provenance, NLI, and temporal validation."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from taurine_core.retrieval.registry import EvidenceRecord, EvidenceRegistry


ValidationMode = Literal["live", "chat", "tool"]
_CITATION_RE = re.compile(r"\[([A-Z])(\d+)\]")
_COVERAGE_PREFIXES = frozenset({"E", "F", "T"})
_HISTORICAL_QUALIFIERS = re.compile(
    r"\b(?:formerly|previously|historically|used to|at the time|was|were|had|did)\b",
    flags=re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"\b(?:may|might|perhaps|possibly|uncertain|unsure|not sure|"
    r"could be|I (?:do not|don't) know|I (?:think|believe|suspect))\b",
    flags=re.IGNORECASE,
)
_CONVERSATIONAL = re.compile(
    r"^(?:thanks|thank you|here(?:'s| is) what I found|"
    r"I can help|let me know|no new evidence)\b",
    flags=re.IGNORECASE,
)
_INACTIVE_FACT_STATUSES = frozenset(
    {"historical", "superseded", "conflicted", "candidate"}
)


class NLIScorer(Protocol):
    """Fake-friendly local entailment seam: evidence premise, generated claim."""

    def score(self, premise: str, hypothesis: str) -> float: ...


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

    @property
    def requires_evidence_excerpt(self) -> bool:
        return self.evidence_excerpt_required


@dataclass(frozen=True)
class _Sentence:
    rendered: str
    claim: str
    citations: tuple[str, ...]
    factual: bool


class EvidenceValidator:
    def __init__(
        self,
        registry: EvidenceRegistry,
        nli_scorer: NLIScorer | None,
        *,
        entailment_floor: float,
    ) -> None:
        if not 0.0 <= entailment_floor <= 1.0:
            raise ValueError("entailment floor must be within [0, 1]")
        self.registry = registry
        self.nli_scorer = nli_scorer
        self.entailment_floor = float(entailment_floor)

    def validate(
        self,
        output: str,
        *,
        mode: ValidationMode,
        supplied_ids: Collection[str] = (),
    ) -> ValidationResult:
        if mode not in {"live", "chat", "tool"}:
            raise ValueError(f"unsupported validation mode: {mode}")
        sentences, visible_text = _sentences(output)
        cited_ids = _ordered_refs(
            match.group(0)[1:-1] for match in _CITATION_RE.finditer(visible_text)
        )
        records = {ref: self.registry.resolve(ref) for ref in cited_ids}
        invalid: set[str] = {ref for ref, record in records.items() if record is None}
        available = set(supplied_ids) | set(self.registry.refs("read"))
        unavailable = {
            ref
            for ref, record in records.items()
            if record is not None
            and record.kind in {"passage", "transcript"}
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

        factual = tuple(sentence for sentence in sentences if sentence.factual)
        unsupported: list[str] = []
        reason = ""
        nli_candidates: list[tuple[_Sentence, tuple[str, ...]]] = []
        for sentence in factual:
            sentence_refs = set(sentence.citations)
            coverage = {
                ref for ref in sentence_refs if ref[:1] in _COVERAGE_PREFIXES
            }
            if not coverage:
                unsupported.append(sentence.rendered)
                reason = reason or "citation_coverage"
                continue
            if sentence_refs & invalid:
                unsupported.append(sentence.rendered)
                if sentence_refs & provenance_invalid:
                    reason = reason or "fact_provenance"
                elif sentence_refs & unavailable:
                    reason = reason or "evidence_unavailable"
                else:
                    reason = reason or "invalid_citation"
                continue
            if _temporally_inconsistent(sentence, records):
                unsupported.append(sentence.rendered)
                reason = reason or "temporal_inconsistency"
                continue
            premises = _premises(
                coverage,
                records=records,
                fact_sources=fact_sources,
                registry=self.registry,
            )
            if not premises:
                unsupported.append(sentence.rendered)
                reason = reason or "evidence_text_unavailable"
                continue
            nli_candidates.append((sentence, premises))

        if nli_candidates and self.nli_scorer is None:
            return self._nli_unavailable(
                output, factual, invalid, cited_ids, mode
            )

        for sentence, premises in nli_candidates:
            try:
                scores = tuple(
                    float(self.nli_scorer.score(premise, sentence.claim))
                    for premise in premises
                )
            except Exception:
                return self._nli_unavailable(
                    output, factual, invalid, cited_ids, mode
                )
            if not any(
                math.isfinite(score) and score >= self.entailment_floor
                for score in scores
            ):
                unsupported.append(sentence.rendered)
                reason = reason or "entailment_below_floor"

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
        )

    def _nli_unavailable(
        self,
        output: str,
        factual: tuple[_Sentence, ...],
        invalid: set[str],
        cited_ids: tuple[str, ...],
        mode: ValidationMode,
    ) -> ValidationResult:
        del output
        is_live = mode == "live"
        return ValidationResult(
            accepted=False,
            accepted_output="",
            unsupported_sentences=tuple(sentence.rendered for sentence in factual),
            invalid_citations=_ordered_refs(invalid),
            repair_required=False,
            cited_ids=cited_ids,
            degraded_reason=(
                "nli_unavailable_live_fail_closed"
                if is_live
                else "nli_unavailable_chat_excerpt_required"
            ),
            evidence_excerpt_required=not is_live,
        )


def validate_evidence_output(
    output: str,
    registry: EvidenceRegistry,
    *,
    mode: ValidationMode,
    nli_scorer: NLIScorer | None,
    entailment_floor: float,
    supplied_ids: Collection[str] = (),
) -> ValidationResult:
    return EvidenceValidator(
        registry,
        nli_scorer,
        entailment_floor=entailment_floor,
    ).validate(output, mode=mode, supplied_ids=supplied_ids)


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
        is_heading = bool(re.match(r"^\s{0,3}#{1,6}\s+", raw_line))
        for rendered in _split_line(stripped):
            citations = _ordered_refs(
                match.group(0)[1:-1] for match in _CITATION_RE.finditer(rendered)
            )
            claim = _CITATION_RE.sub("", rendered).strip()
            claim = re.sub(r"^[-*+]\s+", "", claim).strip()
            factual = _is_factual(claim, is_heading=is_heading)
            rows.append(_Sentence(rendered, claim, citations, factual))
    return tuple(rows), "\n".join(visible)


def _split_line(line: str) -> tuple[str, ...]:
    matches = re.findall(
        r".+?(?:[.!?](?:\s*\[[A-Z]\d+\])*(?=\s+|$)|$)",
        line,
    )
    return tuple(part.strip() for part in matches if part.strip())


def _is_factual(claim: str, *, is_heading: bool) -> bool:
    if not claim or is_heading or claim.endswith("?"):
        return False
    if _UNCERTAINTY.search(claim) or _CONVERSATIONAL.search(claim):
        return False
    return bool(re.search(r"[A-Za-z0-9]", claim))


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


def _premises(
    coverage: set[str],
    *,
    records: dict[str, EvidenceRecord | None],
    fact_sources: dict[str, tuple[str, ...]],
    registry: EvidenceRegistry,
) -> tuple[str, ...]:
    source_refs: set[str] = set()
    for ref in coverage:
        record = records.get(ref)
        if record is None:
            continue
        if record.kind in {"passage", "transcript"}:
            source_refs.add(ref)
        elif record.kind == "fact":
            source_refs.update(fact_sources.get(ref, ()))
    texts: list[str] = []
    for ref in _ordered_refs(source_refs):
        record = registry.resolve(ref)
        if record is None:
            continue
        text = _payload_text(record)
        if text and text not in texts:
            texts.append(text)
    return tuple(texts)


def _payload_text(record: EvidenceRecord) -> str:
    for key in ("text", "body", "transcript", "content", "quote"):
        value = record.payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ordered_refs(refs: Iterable[str]) -> tuple[str, ...]:
    unique = set(refs)

    def key(ref: str) -> tuple[int, int, str]:
        prefix_order = {prefix: index for index, prefix in enumerate("NEFPT")}
        try:
            number = int(ref[1:])
        except ValueError:
            number = 0
        return (prefix_order.get(ref[:1], 99), number, ref)

    return tuple(sorted(unique, key=key))
