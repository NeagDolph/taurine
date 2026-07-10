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
    r"\b(?:might|perhaps|possibly|uncertain|unsure|not sure|"
    r"could be|I (?:do not|don't) know|I (?:think|believe|suspect))\b",
    flags=re.IGNORECASE,
)
# "may" hedges only in lowercase — "May 2"/"signed in May" are dates, and
# exempting them would let factual date claims skip citation (M12).
_MAY_HEDGE = re.compile(r"\bmay\b")
_CONVERSATIONAL = re.compile(
    r"^(?:thanks|thank you|here(?:'s| is) what I found|"
    r"I can help|let me know|no new evidence)\b",
    flags=re.IGNORECASE,
)
# Absence-of-evidence statements ("couldn't find a return date", "no record
# of X") assert nothing positive from the knowledge base; entailment cannot
# prove an absence from a passage, so validating them only shreds honest
# "not found" answers into the excerpt fallback.
_ABSENCE = re.compile(
    r"(?:\b(?:couldn't|can't|cannot|could not|did not|didn't|do not|don't|doesn't|was unable to)\s+"
    r"(?:find|locate|see|surface)\b"
    r"|\bfound\s+no\b"
    r"|\b(?:no|not any|none of the|nothing in)\b"
    r".{0,60}?\b(?:evidence|record(?:s|ed)?|mention(?:s|ed)?|information|"
    r"date(?:s)?|detail(?:s)?|reference(?:s)?|indication|answer|passage(?:s)?|note(?:s)?)\b"
    r"|\b(?:is|are|was|were)n?(?:'t| not)\s+(?:recorded|mentioned|documented|available|stated)\b)",
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
        details: list[tuple[str, str]] = []
        reason = ""
        nli_candidates: list[tuple[_Sentence, tuple[str, ...]]] = []

        def reject(sentence: _Sentence, sentence_reason: str) -> None:
            nonlocal reason
            unsupported.append(sentence.rendered)
            details.append((sentence.rendered, sentence_reason))
            reason = reason or sentence_reason

        for sentence in factual:
            sentence_refs = set(sentence.citations)
            coverage = {
                ref for ref in sentence_refs if ref[:1] in _COVERAGE_PREFIXES
            }
            if not coverage:
                reject(sentence, "citation_coverage")
                continue
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
                continue
            premises = _premises(
                coverage,
                records=records,
                fact_sources=fact_sources,
                registry=self.registry,
            )
            if not premises:
                reject(sentence, "evidence_text_unavailable")
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
                reject(sentence, "entailment_below_floor")

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
            unsupported_details=tuple(
                (sentence.rendered, "nli_unavailable") for sentence in factual
            ),
        )


def validate_evidence_output(
    output: str,
    registry: EvidenceRegistry,
    *,
    mode: ValidationMode,
    nli_scorer: NLIScorer | None,
    entailment_floor: float,
    available_ids: Collection[str],
    supplied_ids: Collection[str] = (),
) -> ValidationResult:
    return EvidenceValidator(
        registry,
        nli_scorer,
        entailment_floor=entailment_floor,
    ).validate(
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


def _is_factual(claim: str, *, is_heading: bool, _depth: int = 0) -> bool:
    if not claim or is_heading or claim.endswith("?"):
        return False
    if not re.search(r"[A-Za-z0-9]", claim):
        return False
    # short structural labels ("What's in the record:") frame, not claim
    if claim.endswith(":") and len(claim.split()) <= 8:
        return False
    match = (
        _CONVERSATIONAL.search(claim)
        or _UNCERTAINTY.search(claim)
        or _MAY_HEDGE.search(claim)
        or _ABSENCE.search(claim)
    )
    if match is None:
        return True
    if _depth >= 1:
        return False
    # M12: a conversational/uncertainty prefix only exempts the sentence when
    # NO factual-claim tail follows a colon/comma boundary after the matched
    # phrase ("Here's what I found: <claim>" must not bypass validation).
    # Split once and classify the remainder independently.
    boundary = re.search(r"[:,]", claim[match.end() :])
    if boundary is None:
        return False
    tail = claim[match.end() + boundary.end() :].strip()
    if not tail:
        return False
    return _is_factual(tail, is_heading=False, _depth=1)


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
