"""Deterministic, provenance-preserving canonical entity resolution."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Literal, Mapping, Sequence

from taurine_core import config
from taurine_core.schemas import ExtractedMention


_LOG = logging.getLogger("taurine.entities")


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
class GrayZonePair:
    """A candidate merge the deterministic resolver declined (edge-quality
    item 6): near-threshold embedding pairs, ambiguous acronyms, and
    token-subset name variants. The grounded merge judge decides; digit/type
    guards stay absolute regardless of its verdict."""

    etype: str
    left_entity_id: str
    right_entity_id: str
    left_surface: str
    right_surface: str
    left_normalized: str
    right_normalized: str
    combined_df: int
    reason: str


@dataclass(frozen=True)
class ResolutionResult:
    entities: tuple[ResolvedEntity, ...]
    mentions: tuple[ResolvedMention, ...]
    mention_to_entity: dict[str, str]
    gray_zone_pairs: tuple[GrayZonePair, ...] = ()


class EntityResolver:
    def __init__(
        self,
        *,
        total_notes: int,
        owner_aliases: set[str] | None = None,
        similarity: Callable[[str, str], float] | None = None,
        similarity_threshold: float = 0.95,
        note_sources: Mapping[str, str] | None = None,
    ) -> None:
        self.total_notes = max(1, int(total_notes))
        self.owner_aliases = {value.casefold() for value in (owner_aliases or set())}
        self.similarity = similarity
        self.similarity_threshold = float(similarity_threshold)
        # note_id → connector source; enables owner auto-detection (an owner
        # candidate must span ≥3 sources, not just dominate one group chat)
        self.note_sources = dict(note_sources or {})
        self._entities: dict[str, ResolvedEntity] = {}
        self._alias_index: dict[str, set[str]] = {}
        self._surface_index: dict[str, set[str]] = {}

    def resolve_mentions(
        self,
        mentions: Sequence[MentionInput],
        *,
        forced_merges: Sequence[tuple[str, str, str]] = (),
    ) -> ResolutionResult:
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
        # bucket pairs the deterministic stages decline to merge but a
        # grounded judge may (item 6); resolved to entities after grouping
        gray_candidates: list[tuple[int, int, str]] = []
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
            ambiguous_targets: set[int] = set()
            for index in members:
                surface = ordered[index].mention.surface
                compact = _compact(surface)
                if not 2 <= len(compact) <= 5:
                    continue
                groups_for_acronym = expansions.get((etype, compact), set()) - {bucket}
                if len(groups_for_acronym) > 1:
                    # H1 keeps an ambiguous short form unmerged; surface each
                    # candidate expansion for the merge judge
                    ambiguous_targets.update(groups_for_acronym)
                    continue
                if len(groups_for_acronym) != 1:
                    continue  # unknown acronym: stays its own entity
                target = next(iter(groups_for_acronym))
                if any(
                    _acronym(ordered[other].mention.surface).casefold() == compact
                    and _digits(ordered[other].mention.surface) == _digits(surface)
                    for other in bucket_members[target]
                ):
                    targets.add(target)
            for target in sorted(ambiguous_targets):
                gray_candidates.append((bucket, target, "acronym_ambiguous"))
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
            band = float(config.MERGE_GRAY_ZONE_EMBED_BAND)
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
                        score = self.similarity(left_rep, right_rep)
                        if score < self.similarity_threshold - band:
                            continue
                        if score >= self.similarity_threshold and (
                            etype != "person"
                            or _person_merge_allowed(left_rep, right_rep)
                        ):
                            union(left, right)
                        else:
                            # just under the merge floor, or above it but held
                            # back by the person guard: not merged, judgeable
                            gray_candidates.append(
                                (left, right, "embedding_gray_zone")
                            )

        # Same-type pairs sharing a full token with different token counts
        # ("Neil" vs "Neil Agrawal") never merge deterministically; surface
        # them for the merge judge (item 6).
        normalized_by_bucket = {
            bucket: normalized for (_, normalized), bucket in bucket_index.items()
        }
        token_buckets: dict[tuple[str, str], list[int]] = {}
        for (etype, normalized), bucket in bucket_index.items():
            for token in sorted(set(normalized.split())):
                token_buckets.setdefault((etype, token), []).append(bucket)
        for (_etype, _token), buckets in sorted(token_buckets.items()):
            for position, left in enumerate(buckets):
                for right in buckets[position + 1 :]:
                    if find(left) == find(right):
                        continue
                    left_norm = normalized_by_bucket[left]
                    right_norm = normalized_by_bucket[right]
                    if len(left_norm.split()) == len(right_norm.split()):
                        continue
                    if _digits(left_norm) != _digits(right_norm):
                        continue
                    gray_candidates.append((left, right, "token_overlap"))

        # Judge-approved gray-zone merges from a prior pass (build.py hook).
        # Digit/type guards hold no matter what the judge said: pairs are
        # keyed by (etype, normalized, normalized), so the type guard is
        # structural, and mismatched digits never union.
        for etype, left_norm, right_norm in forced_merges:
            left_bucket = bucket_index.get((etype, str(left_norm)))
            right_bucket = bucket_index.get((etype, str(right_norm)))
            if left_bucket is None or right_bucket is None:
                continue
            if _digits(str(left_norm)) != _digits(str(right_norm)):
                continue
            union(left_bucket, right_bucket)

        groups: dict[int, list[MentionInput]] = {}
        for bucket, members in enumerate(bucket_members):
            groups.setdefault(find(bucket), []).extend(ordered[index] for index in members)

        entities: list[ResolvedEntity] = []
        entity_by_root: dict[int, ResolvedEntity] = {}
        members_by_entity: dict[str, list[MentionInput]] = {}
        mention_to_entity: dict[str, str] = {}
        resolved_mentions: list[ResolvedMention] = []
        for root, members in groups.items():
            entity = self._entity_from_members(members)
            entity_by_root[root] = entity
            entities.append(entity)
            members_by_entity[entity.entity_id] = members
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

        gray_zone_pairs = self._gray_zone_pairs(gray_candidates, find, entity_by_root)
        entities = self._apply_owner_detection(entities, members_by_entity)
        entities.sort(key=lambda row: row.entity_id)
        resolved_mentions.sort(key=lambda row: row.mention_id)
        self._entities = {row.entity_id: row for row in entities}
        self._rebuild_indexes(entities)
        return ResolutionResult(
            entities=tuple(entities),
            mentions=tuple(resolved_mentions),
            mention_to_entity=dict(sorted(mention_to_entity.items())),
            gray_zone_pairs=gray_zone_pairs,
        )

    def _gray_zone_pairs(
        self,
        candidates: Sequence[tuple[int, int, str]],
        find: Callable[[int], int],
        entity_by_root: Mapping[int, ResolvedEntity],
    ) -> tuple[GrayZonePair, ...]:
        """Bucket-pair candidates → entity pairs, deduped, df-prioritized,
        capped at MERGE_GRAY_ZONE_CAP per build."""

        seen: dict[tuple[str, str, str], GrayZonePair] = {}
        for left_bucket, right_bucket, reason in candidates:
            left_root, right_root = find(left_bucket), find(right_bucket)
            if left_root == right_root:
                continue  # a later stage merged them after all
            left = entity_by_root[left_root]
            right = entity_by_root[right_root]
            if left.etype != right.etype:
                continue
            if (left.normalized, left.entity_id) > (right.normalized, right.entity_id):
                left, right = right, left
            key = (left.etype, left.normalized, right.normalized)
            if key in seen:
                continue
            seen[key] = GrayZonePair(
                etype=left.etype,
                left_entity_id=left.entity_id,
                right_entity_id=right.entity_id,
                left_surface=left.canonical,
                right_surface=right.canonical,
                left_normalized=left.normalized,
                right_normalized=right.normalized,
                combined_df=left.df + right.df,
                reason=reason,
            )
        ranked = sorted(
            seen.values(),
            key=lambda pair: (
                -pair.combined_df,
                pair.etype,
                pair.left_normalized,
                pair.right_normalized,
            ),
        )
        return tuple(ranked[: max(0, int(config.MERGE_GRAY_ZONE_CAP))])

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

    def _apply_owner_detection(
        self,
        entities: list[ResolvedEntity],
        members_by_entity: Mapping[str, Sequence[MentionInput]],
    ) -> list[ResolvedEntity]:
        """Flag the corpus owner (and same-name variants) as ``owner_alias_hub``.

        Explicit ``config.OWNER_ALIASES`` always wins: when configured aliases
        exist, auto-detection is skipped entirely. Otherwise the owner is
        auto-detected as the most frequent person entity whose df is at least
        2× the runner-up person (name variants of the leader cannot be the
        runner-up) and which appears across ≥3 connector sources. Any entity
        whose casefolded surface is a token-subset or -superset of a
        detected/configured owner alias is flagged with the same reason
        ("Neil Agrawal" when "Neil" is the owner).
        """

        owner_token_sets = [
            tokens
            for alias in sorted(self.owner_aliases)
            if (tokens := set(_normalize(alias).split()))
        ]
        detected_id: str | None = None
        if not owner_token_sets:
            detected = self._autodetect_owner(entities, members_by_entity)
            if detected is not None:
                detected_id = detected.entity_id
                tokens = set(_normalize(detected.canonical).split())
                if tokens:
                    owner_token_sets.append(tokens)
        if not owner_token_sets:
            return entities

        updated: list[ResolvedEntity] = []
        for entity in entities:
            tokens = set(_normalize(entity.canonical).split())
            related = entity.entity_id == detected_id or (
                bool(tokens)
                and any(
                    tokens <= owner or owner <= tokens
                    for owner in owner_token_sets
                )
            )
            if not related or entity.traversal_reason == "owner_alias_hub":
                updated.append(entity)
                continue
            if entity.entity_id != detected_id:
                _LOG.info(
                    "owner alias hub: %r (%s, df=%d) is a token subset/superset "
                    "of a %s owner alias",
                    entity.canonical,
                    entity.etype,
                    entity.df,
                    "detected" if detected_id is not None else "configured",
                )
            updated.append(
                replace(
                    entity,
                    traversal_eligible=False,
                    traversal_reason="owner_alias_hub",
                )
            )
        return updated

    def _autodetect_owner(
        self,
        entities: list[ResolvedEntity],
        members_by_entity: Mapping[str, Sequence[MentionInput]],
    ) -> ResolvedEntity | None:
        if not self.note_sources:
            return None
        persons = sorted(
            (row for row in entities if row.etype == "person"),
            key=lambda row: (-row.df, row.entity_id),
        )
        if not persons:
            return None
        top = persons[0]
        top_tokens = set(_normalize(top.canonical).split())
        runner_up_df = 0
        for row in persons[1:]:
            tokens = set(_normalize(row.canonical).split())
            if tokens and top_tokens and (tokens <= top_tokens or top_tokens <= tokens):
                continue  # "Neil Agrawal" must not block detection of "Neil"
            runner_up_df = row.df
            break
        sources = {
            self.note_sources.get(member.note_id)
            for member in members_by_entity.get(top.entity_id, ())
        } - {None}
        if top.df >= 3 and top.df >= 2 * runner_up_df and len(sources) >= 3:
            _LOG.info(
                "owner auto-detected: %r df=%d runner_up_df=%d sources=%d "
                "reason=owner_alias_hub",
                top.canonical,
                top.df,
                runner_up_df,
                len(sources),
            )
            return top
        return None

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
