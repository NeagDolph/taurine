"""Deterministic, provenance-preserving canonical entity resolution."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from taurine_core import config
from taurine_core.schemas import ExtractedMention


ResolutionStatus = Literal["resolved", "ambiguous", "missing"]
_CORPORATE_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
}
_PRONOUNS = {"i", "me", "we", "us", "you", "he", "she", "they", "them"}


@dataclass(frozen=True)
class MentionInput:
    mention_id: str
    note_id: str
    passage_id: str
    mention: ExtractedMention


@dataclass(frozen=True)
class ResolvedMention:
    mention_id: str
    note_id: str
    passage_id: str
    entity_id: str
    mention: ExtractedMention


@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: str
    canonical: str
    normalized: str
    etype: str
    df: int
    traversal_eligible: bool
    traversal_reason: str
    ambiguity_key: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    score: float
    reason: str
    canonical: str
    etype: str


@dataclass(frozen=True)
class EntityResolution:
    status: ResolutionStatus
    candidates: tuple[EntityCandidate, ...]


@dataclass(frozen=True)
class ResolutionResult:
    entities: tuple[ResolvedEntity, ...]
    mentions: tuple[ResolvedMention, ...]
    mention_to_entity: dict[str, str]


class EntityResolver:
    def __init__(
        self,
        *,
        total_notes: int,
        owner_aliases: set[str] | None = None,
        similarity: Callable[[str, str], float] | None = None,
        similarity_threshold: float = 0.95,
    ) -> None:
        self.total_notes = max(1, int(total_notes))
        self.owner_aliases = {value.casefold() for value in (owner_aliases or set())}
        self.similarity = similarity
        self.similarity_threshold = float(similarity_threshold)
        self._entities: dict[str, ResolvedEntity] = {}
        self._alias_index: dict[str, set[str]] = {}
        self._surface_index: dict[str, set[str]] = {}

    def resolve_mentions(self, mentions: Sequence[MentionInput]) -> ResolutionResult:
        ordered = sorted(
            mentions,
            key=lambda row: (
                row.mention.etype,
                _normalize(row.mention.surface),
                row.mention.surface.casefold(),
                row.mention_id,
            ),
        )
        # Stages 1-3 (exact / casefold / deterministic normalization) are one
        # O(n) bucketing pass: casefold-equal surfaces normalize equally, and
        # digits/type are constant within a (etype, normalized) bucket.
        bucket_index: dict[tuple[str, str], int] = {}
        bucket_members: list[list[int]] = []
        for index, row in enumerate(ordered):
            key = (row.mention.etype, _normalize(row.mention.surface))
            bucket = bucket_index.get(key)
            if bucket is None:
                bucket = len(bucket_members)
                bucket_index[key] = bucket
                bucket_members.append([])
            bucket_members[bucket].append(index)

        parent = list(range(len(bucket_members)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        def bucket_etype(bucket: int) -> str:
            return ordered[bucket_members[bucket][0]].mention.etype

        # Stage 4 (H1): map every expansion's acronym BEFORE any union, so a
        # short form merges only when it names exactly one expansion group in
        # the whole batch, and unions run short<->expansion only — union-find
        # transitivity can never bridge two expansions through a shared
        # acronym (§7.2 uniqueness guard).
        expansions: dict[tuple[str, str], set[int]] = {}
        for bucket, members in enumerate(bucket_members):
            etype = bucket_etype(bucket)
            for index in members:
                acronym = _acronym(ordered[index].mention.surface)
                if len(acronym) >= 2:
                    expansions.setdefault((etype, acronym.casefold()), set()).add(bucket)
        for bucket, members in enumerate(bucket_members):
            etype = bucket_etype(bucket)
            targets: set[int] = set()
            for index in members:
                surface = ordered[index].mention.surface
                compact = _compact(surface)
                if not 2 <= len(compact) <= 5:
                    continue
                groups_for_acronym = expansions.get((etype, compact), set()) - {bucket}
                if len(groups_for_acronym) != 1:
                    continue  # unknown or ambiguous acronym: stays its own entity
                target = next(iter(groups_for_acronym))
                if any(
                    _acronym(ordered[other].mention.surface).casefold() == compact
                    and _digits(ordered[other].mention.surface) == _digits(surface)
                    for other in bucket_members[target]
                ):
                    targets.add(target)
            if len(targets) == 1:
                union(bucket, targets.pop())

        # Stages 5-6: embedding similarity generates candidates, never merges
        # by itself — lexical/digit/type guards (and the person last-name
        # containment rule) must all agree.
        if self.similarity is not None:
            representatives = [
                max(
                    (ordered[index].mention.surface.strip() for index in members),
                    key=lambda value: (len(value), value.casefold(), value),
                )
                for members in bucket_members
            ]
            buckets_by_etype: dict[str, list[int]] = {}
            for bucket in range(len(bucket_members)):
                buckets_by_etype.setdefault(bucket_etype(bucket), []).append(bucket)
            for etype, buckets in sorted(buckets_by_etype.items()):
                for position, left in enumerate(buckets):
                    for right in buckets[position + 1 :]:
                        if find(left) == find(right):
                            continue
                        left_rep = representatives[left]
                        right_rep = representatives[right]
                        if _digits(left_rep) != _digits(right_rep):
                            continue
                        if not _lexically_compatible(left_rep, right_rep):
                            continue
                        if etype == "person" and not _person_merge_allowed(
                            left_rep, right_rep
                        ):
                            continue
                        if self.similarity(left_rep, right_rep) >= self.similarity_threshold:
                            union(left, right)

        groups: dict[int, list[MentionInput]] = {}
        for bucket, members in enumerate(bucket_members):
            groups.setdefault(find(bucket), []).extend(ordered[index] for index in members)

        entities: list[ResolvedEntity] = []
        mention_to_entity: dict[str, str] = {}
        resolved_mentions: list[ResolvedMention] = []
        for members in groups.values():
            entity = self._entity_from_members(members)
            entities.append(entity)
            for member in members:
                mention_to_entity[member.mention_id] = entity.entity_id
                resolved_mentions.append(
                    ResolvedMention(
                        mention_id=member.mention_id,
                        note_id=member.note_id,
                        passage_id=member.passage_id,
                        entity_id=entity.entity_id,
                        mention=member.mention,
                    )
                )

        entities.sort(key=lambda row: row.entity_id)
        resolved_mentions.sort(key=lambda row: row.mention_id)
        self._entities = {row.entity_id: row for row in entities}
        self._rebuild_indexes(entities)
        return ResolutionResult(
            entities=tuple(entities),
            mentions=tuple(resolved_mentions),
            mention_to_entity=dict(sorted(mention_to_entity.items())),
        )

    def lookup_query_surface(
        self,
        surface: str,
        optional_type: str | None,
        query_context: str,
    ) -> EntityResolution:
        del query_context
        surface_cf = surface.strip().casefold()
        exact_ids = self._surface_index.get(surface, set())
        if exact_ids:
            ids = exact_ids
            reason = "exact_alias"
        else:
            ids = self._alias_index.get(surface_cf, set())
            reason = "casefold_alias"
        candidates = [self._entities[entity_id] for entity_id in ids]
        if optional_type is not None:
            candidates = [row for row in candidates if row.etype == optional_type]
        rendered = tuple(
            EntityCandidate(
                entity_id=row.entity_id,
                score=1.0,
                reason=reason,
                canonical=row.canonical,
                etype=row.etype,
            )
            for row in sorted(candidates, key=lambda row: (row.canonical.casefold(), row.entity_id))
        )
        if not rendered:
            return EntityResolution(status="missing", candidates=())
        return EntityResolution(
            status="resolved" if len(rendered) == 1 else "ambiguous",
            candidates=rendered,
        )

    def query_traversal_ids(
        self,
        surface: str,
        optional_type: str | None,
        query_context: str,
    ) -> tuple[str, ...]:
        resolution = self.lookup_query_surface(surface, optional_type, query_context)
        if resolution.status != "resolved":
            return ()
        return tuple(candidate.entity_id for candidate in resolution.candidates)

    def _entity_from_members(self, members: list[MentionInput]) -> ResolvedEntity:
        surfaces = [row.mention.surface.strip() for row in members]
        counts: dict[str, int] = {}
        confidence: dict[str, float] = {}
        for row in members:
            surface = row.mention.surface.strip()
            counts[surface] = counts.get(surface, 0) + 1
            confidence[surface] = max(confidence.get(surface, 0.0), row.mention.confidence)
        # An acronym never canonicalizes a group that holds its expansion
        # (§7.2): prefer the most frequent expansion, longest on ties.
        acronym_surfaces = {
            surface
            for surface in counts
            if any(
                other != surface and _compact(surface) == _acronym(other).casefold()
                for other in counts
            )
        }
        expansion_surfaces = [surface for surface in counts if surface not in acronym_surfaces]
        if acronym_surfaces and expansion_surfaces:
            canonical = min(
                expansion_surfaces,
                key=lambda value: (
                    -counts[value],
                    -confidence[value],
                    -len(value),
                    value.casefold(),
                    value,
                ),
            )
        else:
            canonical = min(
                counts,
                key=lambda value: (
                    -counts[value],
                    -confidence[value],
                    len(value),
                    value.casefold(),
                    value,
                ),
            )
        etype = members[0].mention.etype
        normalized = _normalize(canonical)
        member_identity = "|".join(sorted({f"{etype}:{_normalize(value)}" for value in surfaces}))
        entity_id = "entity-" + hashlib.sha256(member_identity.encode("utf-8")).hexdigest()[:24]
        aliases = set(surfaces)
        for surface in surfaces:
            acronym = _acronym(surface)
            if len(acronym) >= 2:
                aliases.add(acronym)
        df = len({row.note_id for row in members})
        eligible, reason = self._traversal_policy(canonical, etype, df)
        return ResolvedEntity(
            entity_id=entity_id,
            canonical=canonical,
            normalized=normalized,
            etype=etype,
            df=df,
            traversal_eligible=eligible,
            traversal_reason=reason,
            ambiguity_key=canonical.casefold(),
            aliases=tuple(sorted(aliases, key=lambda value: (value.casefold(), value))),
        )

    def _traversal_policy(self, canonical: str, etype: str, df: int) -> tuple[bool, str]:
        canonical_cf = canonical.casefold()
        if etype == "date_or_deadline":
            return False, "date_default_off"
        if canonical_cf in _PRONOUNS:
            return False, "pronoun"
        if etype == "concept" and canonical_cf in config.CONCEPT_STOPLIST:
            return False, "concept_stoplist"
        if canonical_cf in self.owner_aliases and df >= max(3, math.ceil(self.total_notes * 0.5)):
            return False, "owner_alias_hub"
        cap = max(5, math.ceil(self.total_notes * (0.5 if etype == "concept" else 0.35)))
        if df > cap:
            return False, "document_frequency_hub"
        return True, "eligible"

    def _rebuild_indexes(self, entities: list[ResolvedEntity]) -> None:
        self._alias_index = {}
        self._surface_index = {}
        for entity in entities:
            for alias in entity.aliases:
                self._alias_index.setdefault(alias.casefold(), set()).add(entity.entity_id)
                self._surface_index.setdefault(alias, set()).add(entity.entity_id)


def _person_merge_allowed(left: str, right: str) -> bool:
    """Embedding merges of people are the known failure mode: require the same
    last name AND token containment, so "Dr. Maya Chen"/"Maya Chen" merge but
    two distinct full names sharing a surname never do (§7.2)."""
    left_tokens = _normalize(left).split()
    right_tokens = _normalize(right).split()
    if not left_tokens or not right_tokens or left_tokens[-1] != right_tokens[-1]:
        return False
    left_set, right_set = set(left_tokens), set(right_tokens)
    return left_set <= right_set or right_set <= left_set


def _normalize(surface: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", surface.casefold())
    while tokens and tokens[-1] in _CORPORATE_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _compact(surface: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", surface.casefold()))


def _acronym(surface: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", surface)
    if len(tokens) < 2:
        return ""
    return "".join(token[0].upper() for token in tokens if token)


def _digits(surface: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", surface))


def _lexically_compatible(left: str, right: str) -> bool:
    left_tokens = set(_normalize(left).split())
    right_tokens = set(_normalize(right).split())
    if not left_tokens or not right_tokens:
        return False
    return bool(left_tokens & right_tokens) or _compact(left) in _compact(right) or _compact(
        right
    ) in _compact(left)
