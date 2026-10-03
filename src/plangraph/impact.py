"""Impact analysis layer for PlanGraph.

Analyzes resolved plan entities against Graphify code graph relationships
to produce codebase-aware evidence of direct impacts and dependencies without
generating code or replacing contributor reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx

from plangraph.exceptions import PlanGraphError
from plangraph.graph import get_node, load_graph
from plangraph.models import Node
from plangraph.plan import CandidateEntity, CandidateKind, ParsedPlan, parse_plan
from plangraph.resolver import (
    GraphIndex,
    ResolutionResult,
    ResolutionStatus,
    ResolvedEntity,
    resolve_plan,
)

# -----------------------------------------------------------------------------
# Relationship Filtering Policy
# -----------------------------------------------------------------------------

# Exclude documentation / LLM rationale metadata nodes and edges from code impact
EXCLUDED_RELATIONS: frozenset[str] = frozenset({"rationale_for"})
EXCLUDED_FILE_TYPES: frozenset[str] = frozenset({"rationale"})

# Mapping of incoming relations to clear directional descriptors
INCOMING_RELATION_MAP: dict[str, str] = {
    "calls": "called_by",
    "indirect_call": "called_by",
    "imports": "imported_by",
    "imports_from": "imported_by",
    "dynamic_import": "imported_by",
    "method": "defined_in",
    "contains": "contained_in",
    "defines": "defined_in",
    "inherits": "inherited_by",
    "extends": "extended_by",
    "implements": "implemented_by",
    "specializes": "specialized_by",
    "mixes_in": "mixed_into",
    "references": "referenced_by",
    "references_constant": "referenced_by",
    "uses": "used_by",
    "uses_config": "used_by",
    "uses_static_prop": "used_by",
    "instantiates": "instantiated_by",
    "reads_from": "read_by",
    "listened_by": "listens_to",
    "bound_to": "binds",
    "re_exports": "re_exported_by",
    "dispatches_to": "dispatched_from",
}


MAX_CONTAINMENT_DISPLAY: int = 10


# -----------------------------------------------------------------------------
# Phase 4 Domain Models
# -----------------------------------------------------------------------------

class ImpactDirection(str, Enum):
    """Direction of the relationship relative to the resolved entity."""

    OUTGOING = "outgoing"  # Resolved entity -> Related entity
    INCOMING = "incoming"  # Related entity -> Resolved entity


@dataclass(frozen=True)
class ImpactRelationship:
    """A direct directed relationship between a resolved entity and a connected graph entity."""

    relation: str
    direction: ImpactDirection
    target_node_id: str
    target_label: str
    target_file: str = ""
    target_location: str | None = None
    target_file_type: str = "code"
    confidence: str = "EXTRACTED"
    context: str | None = None
    is_external: bool = False

    @property
    def display_relation(self) -> str:
        """Human-readable display relation taking direction into account."""
        if self.direction == ImpactDirection.OUTGOING:
            return self.relation
        return INCOMING_RELATION_MAP.get(self.relation, f"incoming_{self.relation}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "relation": self.relation,
            "display_relation": self.display_relation,
            "direction": self.direction.value,
            "target_node_id": self.target_node_id,
            "target_label": self.target_label,
            "target_file": self.target_file,
            "target_location": self.target_location,
            "target_file_type": self.target_file_type,
            "confidence": self.confidence,
            "context": self.context,
            "is_external": self.is_external,
        }


@dataclass(frozen=True)
class ImpactedEntity:
    """A graph entity touched or referenced by the plan, with its direct relationships."""

    node_id: str
    label: str
    source_file: str
    source_location: str | None = None
    file_type: str = "code"
    is_callable: bool = False
    originating_candidates: tuple[CandidateEntity, ...] = field(default_factory=tuple)
    relationships: tuple[ImpactRelationship, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "source_file": self.source_file,
            "source_location": self.source_location,
            "file_type": self.file_type,
            "is_callable": self.is_callable,
            "originating_candidates": [c.to_dict() for c in self.originating_candidates],
            "relationships": [r.to_dict() for r in self.relationships],
        }


@dataclass(frozen=True)
class ImpactSummary:
    """Summary counts of candidates, resolved entities, and discovered relationships."""

    total_candidates: int
    resolved_candidates: int
    ambiguous_candidates: int
    unresolved_candidates: int
    not_code_candidates: int
    unique_impacted_entities: int
    relationships_by_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "total_candidates": self.total_candidates,
            "resolved_candidates": self.resolved_candidates,
            "ambiguous_candidates": self.ambiguous_candidates,
            "unresolved_candidates": self.unresolved_candidates,
            "not_code_candidates": self.not_code_candidates,
            "unique_impacted_entities": self.unique_impacted_entities,
            "relationships_by_type": dict(self.relationships_by_type),
        }


@dataclass(frozen=True)
class ImpactAnalysis:
    """The aggregate impact analysis result of a contributor plan."""

    plan_title: str
    plan_file: str | None
    graph_file: str | None
    resolution_result: ResolutionResult
    impacted_entities: tuple[ImpactedEntity, ...] = field(default_factory=tuple)
    ambiguous_entities: tuple[ResolvedEntity, ...] = field(default_factory=tuple)
    unresolved_entities: tuple[ResolvedEntity, ...] = field(default_factory=tuple)
    not_code_entities: tuple[ResolvedEntity, ...] = field(default_factory=tuple)
    summary: ImpactSummary = field(default_factory=lambda: ImpactSummary(0, 0, 0, 0, 0, 0))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "plan_title": self.plan_title,
            "plan_file": self.plan_file,
            "graph_file": self.graph_file,
            "summary": self.summary.to_dict(),
            "impacted_entities": [e.to_dict() for e in self.impacted_entities],
            "ambiguous": [e.to_dict() for e in self.ambiguous_entities],
            "unresolved": [e.to_dict() for e in self.unresolved_entities],
            "not_code_entities": [e.to_dict() for e in self.not_code_entities],
        }


# -----------------------------------------------------------------------------
# Direct Relationship Extraction
# -----------------------------------------------------------------------------

def _extract_direct_relationships(
    graph: nx.DiGraph,
    node_id: str,
) -> tuple[ImpactRelationship, ...]:
    """Extract direct (1-hop) incoming and outgoing relationships for a graph node.

    Excludes documentation rationale nodes and edges. Sorts relationships deterministically.
    """
    if node_id not in graph:
        return ()

    collected: list[ImpactRelationship] = []
    seen: set[tuple[str, str, str]] = set()

    # 1. Outgoing edges (node_id -> target)
    for _, target_id, data in graph.out_edges(node_id, data=True):
        relation = str(data.get("relation", ""))
        if relation in EXCLUDED_RELATIONS:
            continue

        target_data = graph.nodes.get(target_id, {})
        target_file_type = str(target_data.get("file_type", "code"))
        if target_file_type in EXCLUDED_FILE_TYPES:
            continue

        target_label = str(target_data.get("label", target_id))
        if target_label.startswith("."):
            for p, _, pdata in graph.in_edges(target_id, data=True):
                if pdata.get("relation") == "method":
                    p_label = graph.nodes.get(p, {}).get("label", "")
                    if p_label:
                        target_label = f"{p_label}.{target_label.lstrip('.')}"
                        break

        target_file = str(target_data.get("source_file", ""))
        target_loc = target_data.get("source_location")
        if target_loc is not None:
            target_loc = str(target_loc)

        confidence = str(data.get("confidence", "EXTRACTED"))
        context = data.get("context")
        if context is not None:
            context = str(context)

        is_external = bool(
            target_data.get("external") is True
            or target_data.get("type") == "external"
            or not target_file
        )

        dedup_key = (relation, ImpactDirection.OUTGOING.value, target_id)
        if dedup_key not in seen:
            seen.add(dedup_key)
            collected.append(
                ImpactRelationship(
                    relation=relation,
                    direction=ImpactDirection.OUTGOING,
                    target_node_id=target_id,
                    target_label=target_label,
                    target_file=target_file,
                    target_location=target_loc,
                    target_file_type=target_file_type,
                    confidence=confidence,
                    context=context,
                    is_external=is_external,
                )
            )

    # 2. Incoming edges (source -> node_id)
    for source_id, _, data in graph.in_edges(node_id, data=True):
        relation = str(data.get("relation", ""))
        if relation in EXCLUDED_RELATIONS:
            continue

        source_data = graph.nodes.get(source_id, {})
        source_file_type = str(source_data.get("file_type", "code"))
        if source_file_type in EXCLUDED_FILE_TYPES:
            continue

        source_label = str(source_data.get("label", source_id))
        if source_label.startswith("."):
            for p, _, pdata in graph.in_edges(source_id, data=True):
                if pdata.get("relation") == "method":
                    p_label = graph.nodes.get(p, {}).get("label", "")
                    if p_label:
                        source_label = f"{p_label}.{source_label.lstrip('.')}"
                        break

        source_file = str(source_data.get("source_file", ""))
        source_loc = source_data.get("source_location")
        if source_loc is not None:
            source_loc = str(source_loc)

        confidence = str(data.get("confidence", "EXTRACTED"))
        context = data.get("context")
        if context is not None:
            context = str(context)

        is_external = bool(
            source_data.get("external") is True
            or source_data.get("type") == "external"
            or not source_file
        )

        dedup_key = (relation, ImpactDirection.INCOMING.value, source_id)
        if dedup_key not in seen:
            seen.add(dedup_key)
            collected.append(
                ImpactRelationship(
                    relation=relation,
                    direction=ImpactDirection.INCOMING,
                    target_node_id=source_id,
                    target_label=source_label,
                    target_file=source_file,
                    target_location=source_loc,
                    target_file_type=source_file_type,
                    confidence=confidence,
                    context=context,
                    is_external=is_external,
                )
            )

    # Sort deterministically: direction -> display_relation -> target_label -> target_file -> target_node_id
    collected.sort(
        key=lambda r: (
            r.direction.value,
            r.display_relation,
            r.target_label,
            r.target_file,
            r.target_node_id,
        )
    )
    return tuple(collected)


def _select_primary_node(resolved: ResolvedEntity) -> Node | None:
    """Select the primary graph node to analyze for a resolved entity.

    For methods: chooses the method node (second node in matched_nodes).
    For classes/functions/symbols: chooses the matched node.
    For files: chooses the file/module node if present, or first node.
    """
    if not resolved.matched_nodes:
        return None

    if resolved.candidate.kind == CandidateKind.METHOD and len(resolved.matched_nodes) >= 2:
        # (class_node, method_node)
        return resolved.matched_nodes[1]

    if resolved.candidate.kind == CandidateKind.FILE_PATH and len(resolved.matched_nodes) > 1:
        # Check if there is an explicit module/file node (e.g. source_location == 'L1' or label matches file)
        norm_text = resolved.candidate.text.replace("\\", "/").strip()
        base_name = norm_text.split("/")[-1]
        for node in resolved.matched_nodes:
            if node.label in (norm_text, base_name) or node.source_location == "L1":
                return node

    return resolved.matched_nodes[0]


def _derive_entity_display_label(
    node: Node,
    resolved_candidates: tuple[ResolvedEntity, ...] | list[ResolvedEntity],
    graph: nx.DiGraph,
) -> str:
    """Derive human-friendly qualified display label for an impacted entity.

    For methods: qualifies with the enclosing class name (e.g., 'Analyzer.process()'
    instead of '.process()'). For standalone functions and classes, preserves node.label.
    """
    for res in resolved_candidates:
        if res.candidate.kind == CandidateKind.METHOD and len(res.matched_nodes) >= 2:
            if res.matched_nodes[1].id == node.id:
                cls_label = res.matched_nodes[0].label
                clean_meth = node.label.lstrip(".")
                return f"{cls_label}.{clean_meth}"

    # Check incoming 'method' edges in the graph
    for u, _, data in graph.in_edges(node.id, data=True):
        if data.get("relation") == "method":
            cls_data = graph.nodes.get(u, {})
            cls_label = cls_data.get("label", "")
            if cls_label:
                clean_meth = node.label.lstrip(".")
                return f"{cls_label}.{clean_meth}"

    return node.label


# -----------------------------------------------------------------------------
# Main Analysis Entry Point
# -----------------------------------------------------------------------------

def analyze_impact(
    plan: ParsedPlan | ResolutionResult | str | Path,
    graph: nx.DiGraph | GraphIndex | str | Path,
) -> ImpactAnalysis:
    """Analyze the direct impact of a contributor plan against a code graph.

    Args:
        plan: A ParsedPlan, ResolutionResult, markdown string, or path to plan file.
        graph: A NetworkX DiGraph, GraphIndex, or path to graph.json.

    Returns:
        ImpactAnalysis: Structured impact analysis containing impacted entities,
                        relationships, unresolvable candidates, and summary counts.
    """
    plan_file_str: str | None = None
    graph_file_str: str | None = None

    if isinstance(graph, (str, Path)):
        graph_file_str = str(graph)
        di_graph = load_graph(graph)
    elif isinstance(graph, GraphIndex):
        di_graph = graph.graph
    elif isinstance(graph, nx.DiGraph):
        di_graph = graph
    else:
        raise PlanGraphError(f"Unsupported graph type: {type(graph).__name__}")

    # 1. Resolve candidates if not already a ResolutionResult
    if isinstance(plan, ResolutionResult):
        resolution = plan
        plan_title = plan.plan_title
        plan_file_str = plan.plan_file
        if plan.graph_file:
            graph_file_str = plan.graph_file
    else:
        if isinstance(plan, (str, Path)):
            if isinstance(plan, Path) or (isinstance(plan, str) and "\n" not in plan):
                plan_file_str = str(plan)
        resolution = resolve_plan(plan, di_graph)
        plan_title = resolution.plan_title
        if resolution.plan_file:
            plan_file_str = resolution.plan_file

    # 2. Group resolved entities by primary node to deduplicate
    # node_id -> {"node": Node, "candidates": list[CandidateEntity], "relationships": tuple}
    node_map: dict[str, dict[str, Any]] = {}

    for res_entity in resolution.resolved:
        primary_node = _select_primary_node(res_entity)
        if not primary_node:
            continue

        nid = primary_node.id
        if nid not in node_map:
            relationships = _extract_direct_relationships(di_graph, nid)
            node_map[nid] = {
                "node": primary_node,
                "candidates": [res_entity.candidate],
                "relationships": relationships,
            }
        else:
            # Deduplicate: append originating candidate
            if res_entity.candidate not in node_map[nid]["candidates"]:
                node_map[nid]["candidates"].append(res_entity.candidate)

    # 3. Build sorted ImpactedEntity instances
    impacted_list: list[ImpactedEntity] = []
    for nid, info in node_map.items():
        node: Node = info["node"]
        candidates = tuple(info["candidates"])
        relationships = info["relationships"]
        display_label = _derive_entity_display_label(node, resolution.resolved, di_graph)
        impacted_list.append(
            ImpactedEntity(
                node_id=nid,
                label=display_label,
                source_file=node.source_file,
                source_location=node.source_location,
                file_type=node.file_type,
                is_callable=node.is_callable,
                originating_candidates=candidates,
                relationships=relationships,
            )
        )

    # Sort impacted entities deterministically by (source_file, source_location, label, node_id)
    impacted_list.sort(
        key=lambda e: (
            e.source_file,
            e.source_location or "",
            e.label,
            e.node_id,
        )
    )

    # 4. Compute unique impacted target entities and relationship counts by type
    unique_target_node_ids: set[str] = set()
    relations_count: dict[str, int] = {}

    for entity in impacted_list:
        for rel in entity.relationships:
            unique_target_node_ids.add(rel.target_node_id)
            disp_rel = rel.display_relation
            relations_count[disp_rel] = relations_count.get(disp_rel, 0) + 1

    # Sort relationships by type
    sorted_relations_count = dict(sorted(relations_count.items()))

    summary = ImpactSummary(
        total_candidates=len(resolution.entities),
        resolved_candidates=len(resolution.resolved),
        ambiguous_candidates=len(resolution.ambiguous),
        unresolved_candidates=len(resolution.unresolved),
        not_code_candidates=len(resolution.not_code_entities),
        unique_impacted_entities=len(unique_target_node_ids),
        relationships_by_type=sorted_relations_count,
    )

    return ImpactAnalysis(
        plan_title=plan_title,
        plan_file=plan_file_str,
        graph_file=graph_file_str,
        resolution_result=resolution,
        impacted_entities=tuple(impacted_list),
        ambiguous_entities=resolution.ambiguous,
        unresolved_entities=resolution.unresolved,
        not_code_entities=resolution.not_code_entities,
        summary=summary,
    )


def format_impact_report(analysis: ImpactAnalysis, all_deps: bool = False) -> str:
    """Format an ImpactAnalysis result into a human-readable CLI report.

    Args:
        analysis: The ImpactAnalysis to format.
        all_deps: If True, include external and standard-library dependencies.
                 If False (default), hide external dependencies.

    Returns:
        Formatted string representation of the impact report.
    """
    lines: list[str] = []
    lines.append("PlanGraph Impact Analysis")
    if analysis.plan_title:
        lines.append(f"Plan: {analysis.plan_title}")
    lines.append("")
    lines.append("Candidates")
    lines.append(f"  Resolved:     {analysis.summary.resolved_candidates}")
    lines.append(f"  Ambiguous:    {analysis.summary.ambiguous_candidates}")
    lines.append(f"  Unresolved:   {analysis.summary.unresolved_candidates}")
    lines.append(f"  Context only: {analysis.summary.not_code_candidates}")
    lines.append("")
    lines.append(f"Impacted entities: {analysis.summary.unique_impacted_entities}")
    lines.append("")

    if analysis.impacted_entities:
        for entity in analysis.impacted_entities:
            loc_suffix = (
                f" ({entity.source_file}:{entity.source_location})"
                if entity.source_location
                else (f" ({entity.source_file})" if entity.source_file else "")
            )
            lines.append(f"{entity.label}{loc_suffix}")
            active_rels = [
                r for r in entity.relationships
                if all_deps or not r.is_external
            ]
            if active_rels:
                contains_rels = [r for r in active_rels if r.display_relation == "contains"]
                collapse_contains = len(contains_rels) > MAX_CONTAINMENT_DISPLAY
                seen_contains = False

                for rel in active_rels:
                    if rel.display_relation == "contains" and collapse_contains:
                        if not seen_contains:
                            seen_contains = True
                            show_count = 5
                            more_count = len(contains_rels) - show_count
                            lines.append(f"  contains → {len(contains_rels)} entities")
                            for cr in contains_rels[:show_count]:
                                target_file_ref = (
                                    f" ({cr.target_file})"
                                    if cr.target_file and cr.target_file != entity.source_file
                                    else ""
                                )
                                lines.append(f"    {cr.target_label}{target_file_ref}")
                            lines.append(f"    ... (+ {more_count} more)")
                        continue

                    target_file_ref = (
                        f" ({rel.target_file})"
                        if rel.target_file and rel.target_file != entity.source_file
                        else ""
                    )
                    lines.append(f"  {rel.display_relation} → {rel.target_label}{target_file_ref}")
            else:
                lines.append("  (no direct code relationships in graph)")
            lines.append("")

    if analysis.unresolved_entities:
        lines.append("Unresolved")
        for u in analysis.unresolved_entities:
            line_info = (
                f" (Line {u.candidate.location.line_number})"
                if u.candidate.location
                else ""
            )
            lines.append(f"  {u.candidate.text}{line_info}")
            if u.resolution_reason:
                lines.append(f"    → {u.resolution_reason}")
        lines.append("")

    if analysis.ambiguous_entities:
        lines.append("Ambiguous")
        for amb in analysis.ambiguous_entities:
            line_info = (
                f" (Line {amb.candidate.location.line_number})"
                if amb.candidate.location
                else ""
            )
            lines.append(f"  {amb.candidate.text}{line_info}")
            for node in amb.matched_nodes:
                loc = f":{node.source_location}" if node.source_location else ""
                lines.append(f"    - {node.source_file}{loc} ({node.label})")
        lines.append("")

    if analysis.not_code_entities:
        lines.append("Context (Not Code Entities)")
        for nc in analysis.not_code_entities:
            line_info = (
                f" (Line {nc.candidate.location.line_number})"
                if nc.candidate.location
                else ""
            )
            lines.append(f"  • {nc.candidate.text}{line_info}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
