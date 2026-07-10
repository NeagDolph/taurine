"""Session-scoped stable evidence identities and usage-state tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from threading import RLock
from types import MappingProxyType
from typing import Literal, Mapping


EvidenceKind = Literal["note", "passage", "fact", "path", "transcript"]
EvidenceState = Literal["retrieved", "read", "cited"]
_PREFIX: dict[EvidenceKind, str] = {
    "note": "N",
    "passage": "E",
    "fact": "F",
    "path": "P",
    "transcript": "T",
}
_PREFIX_ORDER = {prefix: index for index, prefix in enumerate(("N", "E", "F", "P", "T"))}


@dataclass(frozen=True)
class EvidenceRecord:
    ref: str
    kind: EvidenceKind
    source_id: str
    payload: Mapping[str, object]
    states: frozenset[EvidenceState]


class EvidenceRegistry:
    """Assign each source identity once; state only advances retrieved -> read -> cited."""

    def __init__(self) -> None:
        # Engines run in worker threads while validators/ledgers mutate on the
        # event loop; one reentrant lock guards every check-then-act sequence.
        self._lock = RLock()
        self._by_identity: dict[tuple[EvidenceKind, str], str] = {}
        self._records: dict[str, EvidenceRecord] = {}
        self._next = {prefix: 1 for prefix in _PREFIX.values()}

    def register(
        self,
        kind: EvidenceKind,
        source_id: str,
        payload: Mapping[str, object],
    ) -> str:
        if kind not in _PREFIX:
            raise ValueError(f"unknown evidence kind: {kind}")
        identity = (kind, str(source_id))
        frozen_payload = _freeze_payload(payload)
        with self._lock:
            existing_ref = self._by_identity.get(identity)
            if existing_ref is not None:
                if self._records[existing_ref].payload != frozen_payload:
                    raise ValueError(
                        "evidence identity was registered with a different payload"
                    )
                return existing_ref
            prefix = _PREFIX[kind]
            ref = f"{prefix}{self._next[prefix]}"
            self._next[prefix] += 1
            record = EvidenceRecord(
                ref=ref,
                kind=kind,
                source_id=str(source_id),
                payload=frozen_payload,
                states=frozenset({"retrieved"}),
            )
            self._records[ref] = record
            self._by_identity[identity] = ref
            return ref

    def resolve(self, ref: str) -> EvidenceRecord | None:
        with self._lock:
            return self._records.get(ref)

    def ref_for(self, kind: EvidenceKind, source_id: str) -> str | None:
        with self._lock:
            return self._by_identity.get((kind, str(source_id)))

    def mark_read(self, ref: str) -> None:
        self._mark(ref, "read")

    def mark_cited(self, ref: str) -> None:
        self._mark(ref, "read")
        self._mark(ref, "cited")

    def merge_payload(
        self,
        ref: str,
        payload: Mapping[str, object],
        *,
        append_list_fields: frozenset[str] = frozenset(),
    ) -> None:
        """Merge stable metadata while allowing explicit append-only provenance lists."""

        with self._lock:
            record = self._records.get(ref)
            if record is None:
                raise KeyError(f"unknown evidence reference: {ref}")
            current = _thaw_payload(record.payload)
            incoming = _thaw_payload(_freeze_payload(payload))
            for key, value in incoming.items():
                if key in append_list_fields:
                    existing = current.get(key, [])
                    if not isinstance(existing, list) or not isinstance(value, list):
                        raise ValueError(
                            f"append-only evidence field {key!r} must be a list"
                        )
                    current[key] = list(dict.fromkeys((*existing, *value)))
                elif key in current and current[key] != value:
                    raise ValueError(f"evidence payload field {key!r} cannot be replaced")
                else:
                    current[key] = value
            self._records[ref] = replace(record, payload=_freeze_payload(current))

    def refs(self, state: EvidenceState) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        ref
                        for ref, record in self._records.items()
                        if state in record.states
                    ),
                    key=_ref_key,
                )
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "version": 1,
                "next": dict(sorted(self._next.items())),
                "entries": [
                    {
                        "ref": record.ref,
                        "kind": record.kind,
                        "source_id": record.source_id,
                        "payload": _thaw_payload(record.payload),
                        "states": sorted(record.states),
                    }
                    for record in sorted(
                        self._records.values(), key=lambda row: _ref_key(row.ref)
                    )
                ],
            }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> "EvidenceRegistry":
        if snapshot.get("version") != 1:
            raise ValueError("unsupported evidence registry snapshot")
        registry = cls()
        entries = snapshot.get("entries")
        counters = snapshot.get("next")
        if not isinstance(entries, list) or not isinstance(counters, dict):
            raise ValueError("invalid evidence registry snapshot")
        for value in entries:
            if not isinstance(value, dict):
                raise ValueError("invalid evidence registry entry")
            kind = value.get("kind")
            ref = str(value.get("ref", ""))
            source_id = str(value.get("source_id", ""))
            payload = value.get("payload")
            states = value.get("states")
            if kind not in _PREFIX or not ref.startswith(_PREFIX[kind]):
                raise ValueError("invalid evidence registry reference")
            if not isinstance(payload, dict) or not isinstance(states, list):
                raise ValueError("invalid evidence registry entry")
            state_set = frozenset(states)
            if not state_set <= {"retrieved", "read", "cited"} or "retrieved" not in state_set:
                raise ValueError("invalid evidence registry states")
            if "cited" in state_set and "read" not in state_set:
                raise ValueError("invalid evidence registry state hierarchy")
            record = EvidenceRecord(
                ref=ref,
                kind=kind,
                source_id=source_id,
                payload=_freeze_payload(payload),
                states=state_set,
            )
            if ref in registry._records or (kind, source_id) in registry._by_identity:
                raise ValueError("duplicate evidence registry identity")
            registry._records[ref] = record
            registry._by_identity[(kind, source_id)] = ref
        for prefix in registry._next:
            value = counters.get(prefix)
            if not isinstance(value, int) or value < 1:
                raise ValueError("invalid evidence registry counter")
            highest = max(
                (int(ref[1:]) for ref in registry._records if ref.startswith(prefix)),
                default=0,
            )
            if value <= highest:
                raise ValueError("evidence registry counter would reuse an ID")
            registry._next[prefix] = value
        return registry

    def _mark(self, ref: str, state: EvidenceState) -> None:
        with self._lock:
            record = self._records.get(ref)
            if record is None:
                raise KeyError(f"unknown evidence reference: {ref}")
            self._records[ref] = replace(record, states=record.states | {state})


def _freeze_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    normalized = json.loads(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")))
    return MappingProxyType(normalized)


def _thaw_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")))


def _ref_key(ref: str) -> tuple[int, int, str]:
    prefix = ref[:1]
    try:
        number = int(ref[1:])
    except ValueError:
        number = 0
    return (_PREFIX_ORDER.get(prefix, len(_PREFIX_ORDER)), number, ref)
