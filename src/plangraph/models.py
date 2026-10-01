"""Internal data models for PlanGraph entities and evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Node:
    """Represents a code or concept entity within the graph."""

    id: str
    label: str
    file_type: str = "code"
    source_file: str = ""
    source_location: str | None = None
    community: int | None = None
    community_name: str | None = None
    is_callable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Construct a Node from a dictionary (e.g., from graph.json)."""
        node_id = str(data.get("id", ""))
        label = str(data.get("label", node_id))
        file_type = str(data.get("file_type", "code"))
        source_file = str(data.get("source_file", ""))
        source_location = data.get("source_location")
        if source_location is not None:
            source_location = str(source_location)

        community = data.get("community")
        if community is not None:
            try:
                community = int(community)
            except (ValueError, TypeError):
                community = None

        community_name = data.get("community_name")
        if community_name is not None:
            community_name = str(community_name)

        is_callable = bool(data.get("_callable", False))

        # Store any remaining attributes as extra metadata
        known_keys = {
            "id",
            "label",
            "file_type",
            "source_file",
            "source_location",
            "community",
            "community_name",
            "_callable",
        }
        metadata = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            id=node_id,
            label=label,
            file_type=file_type,
            source_file=source_file,
            source_location=source_location,
            community=community,
            community_name=community_name,
            is_callable=is_callable,
            metadata=metadata,
        )


@dataclass(frozen=True)
class Edge:
    """Represents a directed relationship between two entities."""

    source: str
    target: str
    relation: str
    confidence: str = "EXTRACTED"
    source_file: str = ""
    source_location: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        """Construct an Edge from a dictionary (e.g., from graph.json)."""
        source = str(data.get("source", ""))
        target = str(data.get("target", ""))
        relation = str(data.get("relation", ""))
        confidence = str(data.get("confidence", "EXTRACTED"))
        source_file = str(data.get("source_file", ""))
        source_location = data.get("source_location")
        if source_location is not None:
            source_location = str(source_location)
        context = data.get("context")
        if context is not None:
            context = str(context)

        known_keys = {
            "source",
            "target",
            "relation",
            "confidence",
            "source_file",
            "source_location",
            "context",
        }
        metadata = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            source=source,
            target=target,
            relation=relation,
            confidence=confidence,
            source_file=source_file,
            source_location=source_location,
            context=context,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ImpactEvidence:
    """Represents deterministic evidence explaining why an entity or edge was included."""

    source_id: str
    target_id: str
    relation: str
    source_file: str = ""
    source_location: str | None = None
    description: str = ""

    def format(self) -> str:
        """Format evidence into a concise human-readable citation."""
        loc = f":{self.source_location}" if self.source_location else ""
        file_ref = f" [{self.source_file}{loc}]" if self.source_file else ""
        desc = f" ({self.description})" if self.description else ""
        return f"{self.source_id} --[{self.relation}]--> {self.target_id}{file_ref}{desc}"
