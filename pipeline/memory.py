"""Evidence-backed domain memory, separate from generic message compaction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

MEMORY_MODES = frozenset({"off", "on", "compressed"})
_LIST_FIELDS = (
    "confirmed_papers",
    "entities",
    "constraints",
    "comparison_targets",
    "unresolved_questions",
    "citations",
    "verified_facts",
)


def _marker(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def _unique(values: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        marker = _marker(value)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _normalize(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def _is_traceable(value: str, source: str) -> bool:
    needle = _normalize(value)
    return bool(needle and f" {needle} " in f" {source} ")


def _valid_evidence(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        chunk_id = item.get("chunk_id")
        paper_id = item.get("paper_id")
        title = item.get("title")
        text = item.get("text")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (chunk_id, paper_id, title, text)
        ):
            continue
        assert isinstance(chunk_id, str)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        valid.append(dict(item))
    return valid


@dataclass
class ResearchMemory:
    """Structured research memory with evidence-only writes and bounded compression."""

    mode: str = "compressed"
    confirmed_papers: list[dict[str, Any]] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    comparison_targets: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    verified_facts: list[dict[str, Any]] = field(default_factory=list)
    compressed: bool = False
    update_count: int = 0
    compression_count: int = 0
    compression_token_threshold: int = 512
    hard_item_limit: int = 1024

    def __post_init__(self) -> None:
        if self.mode not in MEMORY_MODES:
            raise ValueError(f"memory mode must be one of {sorted(MEMORY_MODES)}")
        for name in ("compression_token_threshold", "hard_item_limit"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.update_count, int) or self.update_count < 0:
            raise ValueError("update_count must be a non-negative integer")
        if not isinstance(self.compression_count, int) or self.compression_count < 0:
            raise ValueError("compression_count must be a non-negative integer")
        for name in _LIST_FIELDS:
            if not isinstance(getattr(self, name), list):
                raise TypeError(f"{name} must be a list")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any] | None,
        mode: str = "compressed",
    ) -> ResearchMemory:
        if value is not None and not isinstance(value, Mapping):
            raise TypeError("memory value must be a mapping or None")
        payload = dict(value or {})
        payload["mode"] = mode
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in payload.items() if key in allowed})

    def _traceable_values(
        self,
        values: Any,
        source: str,
    ) -> list[str]:
        if not isinstance(values, list):
            return []
        return _unique(
            [
                item.strip()
                for item in values
                if isinstance(item, str)
                and item.strip()
                and _is_traceable(item, source)
            ]
        )

    def _enforce_hard_limit(self) -> None:
        for name in _LIST_FIELDS:
            values = getattr(self, name)
            setattr(self, name, values[-self.hard_item_limit :])

    def update(
        self,
        *,
        user_message: str,
        nlu: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        citations: Sequence[str],
        unresolved: bool = False,
        traceable_context: Sequence[Any] = (),
    ) -> None:
        """Update from traceable NLU values and evidence actually cited by generation."""
        if self.mode == "off":
            return
        if not isinstance(nlu, Mapping):
            raise TypeError("nlu must be a mapping")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise TypeError("evidence must be a sequence")
        if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
            raise TypeError("citations must be a sequence")

        valid_evidence = _valid_evidence(
            [item for item in evidence if isinstance(item, Mapping)]
        )
        evidence_by_id = {item["chunk_id"]: item for item in valid_evidence}
        valid_citations = _unique(
            [
                citation.strip()
                for citation in citations
                if isinstance(citation, str) and citation.strip() in evidence_by_id
            ]
        )
        used = [evidence_by_id[citation] for citation in valid_citations]

        source_payload = {
            "user_message": user_message,
            "prior_memory": self.context(),
            "evidence": valid_evidence,
            "history": list(traceable_context),
        }
        source = _normalize(
            json.dumps(source_payload, ensure_ascii=False, sort_keys=True, default=str)
        )
        self.entities = _unique(
            self.entities
            + self._traceable_values(nlu.get("resolved_references", []), source)
        )
        self.constraints = _unique(
            self.constraints
            + self._traceable_values(nlu.get("constraints", []), source)
        )
        self.comparison_targets = _unique(
            self.comparison_targets
            + self._traceable_values(nlu.get("comparison_targets", []), source)
        )
        self.confirmed_papers = _unique(
            self.confirmed_papers
            + [{"paper_id": item["paper_id"], "title": item["title"]} for item in used]
        )
        self.citations = _unique(self.citations + valid_citations)
        self.verified_facts = _unique(
            self.verified_facts
            + [
                {
                    "paper_id": item["paper_id"],
                    "chunk_id": item["chunk_id"],
                    "excerpt": " ".join(str(item["text"]).split())[:240],
                }
                for item in used
            ]
        )
        message = (user_message or "").strip()
        if unresolved and message:
            self.unresolved_questions = _unique(self.unresolved_questions + [message])
        elif message:
            self.unresolved_questions = [
                question
                for question in self.unresolved_questions
                if question != message
            ]
        self.update_count += 1
        self._enforce_hard_limit()
        self.maybe_compress()

    def maybe_compress(self) -> bool:
        """Compress only when compressed mode exceeds its configured threshold."""
        if (
            self.mode != "compressed"
            or self.estimated_tokens() <= self.compression_token_threshold
        ):
            return False
        return self.compress()

    def compress(
        self,
        *,
        max_papers: int = 24,
        max_entities: int = 32,
        max_constraints: int = 24,
        max_comparison_targets: int = 24,
        max_unresolved: int = 8,
        max_citations: int = 48,
        max_facts: int = 12,
    ) -> bool:
        """Bound all research fields while retaining recent keys and citation links."""
        if self.mode != "compressed":
            return False
        limits = {
            "confirmed_papers": max_papers,
            "entities": max_entities,
            "constraints": max_constraints,
            "comparison_targets": max_comparison_targets,
            "unresolved_questions": max_unresolved,
            "citations": max_citations,
            "verified_facts": max_facts,
        }
        for name, limit in limits.items():
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ValueError(f"{name} compression limit must be a positive integer")
            values = getattr(self, name)
            setattr(self, name, values[-limit:])
        self.verified_facts = [
            {**fact, "excerpt": str(fact.get("excerpt", ""))[:160]}
            for fact in self.verified_facts
        ]
        self.compressed = True
        self.compression_count += 1
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def context(self) -> dict[str, Any]:
        if self.mode == "off":
            return {}
        return {
            "confirmed_papers": [dict(item) for item in self.confirmed_papers],
            "entities": list(self.entities),
            "constraints": list(self.constraints),
            "comparison_targets": list(self.comparison_targets),
            "unresolved_questions": list(self.unresolved_questions),
            "citations": list(self.citations),
            "verified_facts": [dict(item) for item in self.verified_facts],
        }

    def estimated_tokens(self) -> int:
        if self.mode == "off":
            return 0
        serialized = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, default=str
        )
        return max(1, (len(serialized) + 3) // 4)
