"""Candidate entity resolution against Graphify code graphs.

Resolves extracted plan candidates (files, classes, functions, methods, symbols, concepts)
against Graphify's graph topology with deterministic indexing, ambiguity detection,
and explainable provenance.
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


# -----------------------------------------------------------------------------
# Phase 3 Domain Models
# -----------------------------------------------------------------------------

class ResolutionStatus(str, Enum):
    """Classification of how a plan candidate resolved against the graph."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    NOT_CODE_ENTITY = "not_code_entity"


@dataclass(frozen=True)
class ResolvedEntity:
    """The resolution of a single CandidateEntity against the repository graph."""

    candidate: CandidateEntity
    status: ResolutionStatus
    matched_nodes: tuple[Node, ...] = field(default_factory=tuple)
    resolution_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status.value,
            "matched_nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "source_file": n.source_file,
                    "source_location": n.source_location,
                    "file_type": n.file_type,
                    "is_callable": n.is_callable,
                }
                for n in self.matched_nodes
            ],
            "resolution_reason": self.resolution_reason,
        }


@dataclass(frozen=True)
class ResolutionResult:
    """The aggregate resolution of all candidates in a ParsedPlan."""

    plan_title: str
    plan_file: str | None
    graph_file: str | None
    entities: tuple[ResolvedEntity, ...] = field(default_factory=tuple)

    @property
    def resolved(self) -> tuple[ResolvedEntity, ...]:
        """Entities that resolved unambiguously to one or more expected nodes."""
        return tuple(e for e in self.entities if e.status == ResolutionStatus.RESOLVED)

    @property
    def ambiguous(self) -> tuple[ResolvedEntity, ...]:
        """Entities that matched multiple conflicting nodes without sufficient disambiguation."""
        return tuple(e for e in self.entities if e.status == ResolutionStatus.AMBIGUOUS)

    @property
    def unresolved(self) -> tuple[ResolvedEntity, ...]:
        """Entities that could not be found in the repository graph."""
        return tuple(e for e in self.entities if e.status == ResolutionStatus.UNRESOLVED)

    @property
    def not_code_entities(self) -> tuple[ResolvedEntity, ...]:
        """Entities (e.g. concepts) that do not represent concrete codebase entities."""
        return tuple(e for e in self.entities if e.status == ResolutionStatus.NOT_CODE_ENTITY)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "plan_title": self.plan_title,
            "plan_file": self.plan_file,
            "graph_file": self.graph_file,
            "total_candidates": len(self.entities),
            "summary": {
                "resolved": len(self.resolved),
                "ambiguous": len(self.ambiguous),
                "unresolved": len(self.unresolved),
                "not_code_entities": len(self.not_code_entities),
            },
            "entities": [e.to_dict() for e in self.entities],
        }


# -----------------------------------------------------------------------------
# Graph Indexing
# -----------------------------------------------------------------------------

def normalize_path(path_str: str) -> str:
    """Normalize file path for consistent repository-relative lookup."""
    p = path_str.replace("\\", "/").strip()
    if p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    while "//" in p:
        p = p.replace("//", "/")
    return p


class GraphIndex:
    """Pre-computed deterministic lookup index over a NetworkX code graph.

    Indexes nodes by normalized file path, class name, function name,
    qualified method signature, exact symbol, and concept.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

        # File lookups
        # normalized_path -> list of member Node objects
        self.nodes_by_source_file: dict[str, list[Node]] = {}
        # normalized_path -> list of explicit file Node objects
        self.explicit_file_nodes: dict[str, list[Node]] = {}
        # file_basename -> set of full normalized repository paths
        self.files_by_basename: dict[str, set[str]] = {}

        # Class lookups
        # class_name -> list of class Node objects
        self.classes_by_name: dict[str, list[Node]] = {}

        # Function lookups
        # normalized_func_name -> list of function Node objects
        self.functions_by_name: dict[str, list[Node]] = {}

        # Method lookups
        # (class_name, method_name) -> list of (class_node, method_node)
        self.methods_by_class_and_name: dict[tuple[str, str], list[tuple[Node, Node]]] = {}
        # method_name -> list of (class_node, method_node)
        self.methods_by_name: dict[str, list[tuple[Node, Node]]] = {}

        # General symbols
        # exact_label -> list of Node objects
        self.nodes_by_label: dict[str, list[Node]] = {}

        # Concept nodes
        # concept_label_lower -> list of concept Node objects
        self.concepts_by_name: dict[str, list[Node]] = {}

        self._build_index()

    def _build_index(self) -> None:
        # First pass: Index method edges between classes and methods
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "method":
                cls_node = get_node(self.graph, u)
                meth_node = get_node(self.graph, v)
                if cls_node and meth_node:
                    clean_m = meth_node.label.rstrip("()").lstrip(".")
                    key = (cls_node.label, clean_m)
                    self.methods_by_class_and_name.setdefault(key, []).append((cls_node, meth_node))
                    self.methods_by_name.setdefault(clean_m, []).append((cls_node, meth_node))

                    # Ensure cls_node is indexed as a class
                    self.classes_by_name.setdefault(cls_node.label, [])
                    if cls_node not in self.classes_by_name[cls_node.label]:
                        self.classes_by_name[cls_node.label].append(cls_node)

        # Second pass: Index all nodes
        for node_id, data in self.graph.nodes(data=True):
            node_dict = dict(data)
            node_dict["id"] = node_id
            node = Node.from_dict(node_dict)

            # 1. Source file index
            sf = normalize_path(node.source_file) if node.source_file else ""
            if sf:
                self.nodes_by_source_file.setdefault(sf, []).append(node)
                base = sf.split("/")[-1]
                self.files_by_basename.setdefault(base, set()).add(sf)

            # 2. Exact label index
            label = node.label.strip()
            if label:
                self.nodes_by_label.setdefault(label, []).append(node)

                # Check if this node is an explicit file representation
                if sf and (label == sf or data.get("node_kind") == "file" or data.get("file_type") == "document"):
                    self.explicit_file_nodes.setdefault(sf, []).append(node)

            # 3. Class index
            is_class = (
                data.get("_callable_class") is True
                or data.get("node_kind") == "class"
                or data.get("type") == "class"
                or (label and label[0].isupper() and not node.is_callable and node.file_type == "code")
            )
            if is_class and label:
                self.classes_by_name.setdefault(label, [])
                if node not in self.classes_by_name[label]:
                    self.classes_by_name[label].append(node)

            # 4. Function index
            if node.is_callable and not data.get("_callable_class"):
                clean_f = label.rstrip("()").lstrip(".")
                if clean_f:
                    self.functions_by_name.setdefault(clean_f, []).append(node)

            # 5. Concept index
            if node.file_type == "concept" and label:
                self.concepts_by_name.setdefault(label.lower(), []).append(node)

        # Sort all indexed lists deterministically by (source_file, source_location, id)
        key_fn = lambda n: (n.source_file or "", n.source_location or "", n.id)
        for k in self.nodes_by_source_file:
            self.nodes_by_source_file[k].sort(key=key_fn)
        for k in self.explicit_file_nodes:
            self.explicit_file_nodes[k].sort(key=key_fn)
        for k in self.classes_by_name:
            self.classes_by_name[k].sort(key=key_fn)
        for k in self.functions_by_name:
            self.functions_by_name[k].sort(key=key_fn)
        for k in self.nodes_by_label:
            self.nodes_by_label[k].sort(key=key_fn)
        for k in self.concepts_by_name:
            self.concepts_by_name[k].sort(key=key_fn)
        for k in self.methods_by_class_and_name:
            self.methods_by_class_and_name[k].sort(key=lambda pair: key_fn(pair[1]))
        for k in self.methods_by_name:
            self.methods_by_name[k].sort(key=lambda pair: key_fn(pair[1]))


# -----------------------------------------------------------------------------
# Entity Resolution Logic
# -----------------------------------------------------------------------------

def _resolve_file_path(candidate: CandidateEntity, index: GraphIndex) -> ResolvedEntity:
    norm_path = normalize_path(candidate.text)

    # 1. Exact match on repository path
    if norm_path in index.nodes_by_source_file:
        file_nodes = index.explicit_file_nodes.get(norm_path)
        if file_nodes:
            matched = tuple(file_nodes)
        else:
            # Member nodes in that source file
            matched = tuple(index.nodes_by_source_file[norm_path])
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=matched,
            resolution_reason=f"exact file path match in repository: {norm_path}",
        )

    # 2. Basename or suffix match
    if "/" not in norm_path:
        # Candidate provided a basename like 'user.py'
        matching_paths = sorted(index.files_by_basename.get(norm_path, set()))
        if len(matching_paths) == 1:
            resolved_path = matching_paths[0]
            matched = tuple(
                index.explicit_file_nodes.get(resolved_path)
                or index.nodes_by_source_file.get(resolved_path, [])
            )
            return ResolvedEntity(
                candidate=candidate,
                status=ResolutionStatus.RESOLVED,
                matched_nodes=matched,
                resolution_reason=f"resolved file by unique basename match to {resolved_path}",
            )
        elif len(matching_paths) > 1:
            # Multiple files match the basename
            all_matches: list[Node] = []
            for p in matching_paths:
                all_matches.extend(index.nodes_by_source_file.get(p, []))
            return ResolvedEntity(
                candidate=candidate,
                status=ResolutionStatus.AMBIGUOUS,
                matched_nodes=tuple(all_matches),
                resolution_reason=(
                    f"ambiguous file basename '{norm_path}' matches {len(matching_paths)} repository paths: "
                    + ", ".join(matching_paths)
                ),
            )
    else:
        # Candidate provided a partial suffix path like 'auth/service.py'
        matching_paths = [
            sf for sf in index.nodes_by_source_file.keys()
            if sf.endswith(f"/{norm_path}")
        ]
        matching_paths.sort()
        if len(matching_paths) == 1:
            resolved_path = matching_paths[0]
            matched = tuple(
                index.explicit_file_nodes.get(resolved_path)
                or index.nodes_by_source_file.get(resolved_path, [])
            )
            return ResolvedEntity(
                candidate=candidate,
                status=ResolutionStatus.RESOLVED,
                matched_nodes=matched,
                resolution_reason=f"resolved file by unique path suffix match to {resolved_path}",
            )
        elif len(matching_paths) > 1:
            all_matches = []
            for p in matching_paths:
                all_matches.extend(index.nodes_by_source_file.get(p, []))
            return ResolvedEntity(
                candidate=candidate,
                status=ResolutionStatus.AMBIGUOUS,
                matched_nodes=tuple(all_matches),
                resolution_reason=(
                    f"ambiguous path suffix '{norm_path}' matches {len(matching_paths)} repository paths: "
                    + ", ".join(matching_paths)
                ),
            )

    return ResolvedEntity(
        candidate=candidate,
        status=ResolutionStatus.UNRESOLVED,
        matched_nodes=(),
        resolution_reason=f"no Graphify file or member node matched path: {norm_path}",
    )


def _resolve_class(
    candidate: CandidateEntity,
    index: GraphIndex,
    file_context: str | None = None,
) -> ResolvedEntity:
    name = candidate.text.strip()
    matches = index.classes_by_name.get(name, [])

    if len(matches) == 1:
        node = matches[0]
        loc = f":{node.source_location}" if node.source_location else ""
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=tuple(matches),
            resolution_reason=f"exact class-name match in {node.source_file}{loc}",
        )
    elif len(matches) > 1:
        if file_context:
            scoped = [n for n in matches if n.source_file == file_context]
            if len(scoped) == 1:
                node = scoped[0]
                loc = f":{node.source_location}" if node.source_location else ""
                return ResolvedEntity(
                    candidate=candidate,
                    status=ResolutionStatus.RESOLVED,
                    matched_nodes=tuple(scoped),
                    resolution_reason=f"exact class-name match in plan-scoped file {node.source_file}{loc}",
                )
            elif len(scoped) > 1:
                matches = scoped
        locs = [f"{n.source_file}:{n.source_location or ''}" for n in matches]
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=tuple(matches),
            resolution_reason=(
                f"multiple class nodes ({len(matches)}) share the name '{name}': "
                + ", ".join(locs)
            ),
        )

    # Fallback: check general symbol lookup
    symbol_matches = index.nodes_by_label.get(name, [])
    code_matches = [n for n in symbol_matches if n.file_type == "code"]
    if len(code_matches) == 1:
        node = code_matches[0]
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=tuple(code_matches),
            resolution_reason=f"exact code symbol match for '{name}' in {node.source_file}",
        )
    elif len(code_matches) > 1:
        if file_context:
            scoped_code = [n for n in code_matches if n.source_file == file_context]
            if len(scoped_code) == 1:
                node = scoped_code[0]
                return ResolvedEntity(
                    candidate=candidate,
                    status=ResolutionStatus.RESOLVED,
                    matched_nodes=tuple(scoped_code),
                    resolution_reason=f"exact code symbol match for '{name}' in plan-scoped file {node.source_file}",
                )
            elif len(scoped_code) > 1:
                code_matches = scoped_code
        locs = [f"{n.source_file}:{n.source_location or ''}" for n in code_matches]
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=tuple(code_matches),
            resolution_reason=f"multiple code nodes ({len(code_matches)}) match symbol '{name}': " + ", ".join(locs),
        )

    return ResolvedEntity(
        candidate=candidate,
        status=ResolutionStatus.UNRESOLVED,
        matched_nodes=(),
        resolution_reason=f"no class or code node found with name '{name}' in repository",
    )


def _resolve_function(
    candidate: CandidateEntity,
    index: GraphIndex,
    file_context: str | None = None,
) -> ResolvedEntity:
    clean_name = candidate.text.rstrip("()").lstrip(".")
    matches = index.functions_by_name.get(clean_name, [])

    if len(matches) == 1:
        node = matches[0]
        loc = f":{node.source_location}" if node.source_location else ""
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=tuple(matches),
            resolution_reason=f"exact function match for '{clean_name}()' in {node.source_file}{loc}",
        )
    elif len(matches) > 1:
        if file_context:
            scoped = [n for n in matches if n.source_file == file_context]
            if len(scoped) == 1:
                node = scoped[0]
                loc = f":{node.source_location}" if node.source_location else ""
                return ResolvedEntity(
                    candidate=candidate,
                    status=ResolutionStatus.RESOLVED,
                    matched_nodes=tuple(scoped),
                    resolution_reason=f"exact function match for '{clean_name}()' in plan-scoped file {node.source_file}{loc}",
                )
            elif len(scoped) > 1:
                matches = scoped
        locs = [f"{n.source_file}:{n.source_location or ''}" for n in matches]
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=tuple(matches),
            resolution_reason=(
                f"multiple function nodes ({len(matches)}) share the name '{clean_name}()': "
                + ", ".join(locs)
            ),
        )

    # Check if this function name matches a method
    method_pairs = index.methods_by_name.get(clean_name, [])
    if len(method_pairs) == 1:
        cls_node, meth_node = method_pairs[0]
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=(meth_node,),
            resolution_reason=f"matched callable method '{clean_name}()' on {cls_node.label} in {meth_node.source_file}",
        )
    elif len(method_pairs) > 1:
        if file_context:
            scoped_pairs = [pair for pair in method_pairs if pair[1].source_file == file_context]
            if len(scoped_pairs) == 1:
                cls_node, meth_node = scoped_pairs[0]
                return ResolvedEntity(
                    candidate=candidate,
                    status=ResolutionStatus.RESOLVED,
                    matched_nodes=(meth_node,),
                    resolution_reason=f"matched callable method '{clean_name}()' on {cls_node.label} in plan-scoped file {meth_node.source_file}",
                )
            elif len(scoped_pairs) > 1:
                method_pairs = scoped_pairs
        meth_nodes = tuple(pair[1] for pair in method_pairs)
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=meth_nodes,
            resolution_reason=f"multiple methods ({len(method_pairs)}) share name '{clean_name}()'",
        )

    return ResolvedEntity(
        candidate=candidate,
        status=ResolutionStatus.UNRESOLVED,
        matched_nodes=(),
        resolution_reason=f"no callable function found with name '{clean_name}()' in repository",
    )


def _resolve_method(
    candidate: CandidateEntity,
    index: GraphIndex,
    file_context: str | None = None,
) -> ResolvedEntity:
    text = candidate.text.strip()
    if "." in text:
        cls_part, meth_part = text.split(".", 1)
        cls_name = cls_part.strip()
        meth_name = meth_part.rstrip("()").lstrip(".")

        pairs = index.methods_by_class_and_name.get((cls_name, meth_name), [])
        if len(pairs) == 1:
            cls_node, meth_node = pairs[0]
            loc = f":{meth_node.source_location}" if meth_node.source_location else ""
            return ResolvedEntity(
                candidate=candidate,
                status=ResolutionStatus.RESOLVED,
                matched_nodes=(cls_node, meth_node),
                resolution_reason=f"exact qualified method match for {cls_name}.{meth_name} in {meth_node.source_file}{loc}",
            )
        elif len(pairs) > 1:
            if file_context:
                scoped_pairs = [pair for pair in pairs if pair[1].source_file == file_context]
                if len(scoped_pairs) == 1:
                    cls_node, meth_node = scoped_pairs[0]
                    loc = f":{meth_node.source_location}" if meth_node.source_location else ""
                    return ResolvedEntity(
                        candidate=candidate,
                        status=ResolutionStatus.RESOLVED,
                        matched_nodes=(cls_node, meth_node),
                        resolution_reason=f"exact qualified method match for {cls_name}.{meth_name} in plan-scoped file {meth_node.source_file}{loc}",
                    )
                elif len(scoped_pairs) > 1:
                    pairs = scoped_pairs
            nodes = []
            for c, m in pairs:
                nodes.extend([c, m])
            return ResolvedEntity(
                candidate=candidate,
                status=ResolutionStatus.AMBIGUOUS,
                matched_nodes=tuple(nodes),
                resolution_reason=f"multiple methods ({len(pairs)}) match qualified name {cls_name}.{meth_name}",
            )

        # Disambiguate reason: did the class exist at all?
        class_nodes = index.classes_by_name.get(cls_name, [])
        if class_nodes:
            return ResolvedEntity(
                candidate=candidate,
                status=ResolutionStatus.UNRESOLVED,
                matched_nodes=(),
                resolution_reason=f"class '{cls_name}' exists, but method '{meth_name}' was not found on it",
            )
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.UNRESOLVED,
            matched_nodes=(),
            resolution_reason=f"neither class '{cls_name}' nor method '{cls_name}.{meth_name}' found in repository",
        )

    # Standalone method name
    clean_name = text.rstrip("()").lstrip(".")
    pairs = index.methods_by_name.get(clean_name, [])
    if len(pairs) == 1:
        cls_node, meth_node = pairs[0]
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=(cls_node, meth_node),
            resolution_reason=f"exact method match for '{clean_name}()' on {cls_node.label}",
        )
    elif len(pairs) > 1:
        if file_context:
            scoped_pairs = [pair for pair in pairs if pair[1].source_file == file_context]
            if len(scoped_pairs) == 1:
                cls_node, meth_node = scoped_pairs[0]
                return ResolvedEntity(
                    candidate=candidate,
                    status=ResolutionStatus.RESOLVED,
                    matched_nodes=(cls_node, meth_node),
                    resolution_reason=f"exact method match for '{clean_name}()' on {cls_node.label} in plan-scoped file {meth_node.source_file}",
                )
            elif len(scoped_pairs) > 1:
                pairs = scoped_pairs
        nodes = []
        for c, m in pairs:
            nodes.extend([c, m])
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=tuple(nodes),
            resolution_reason=f"multiple methods ({len(pairs)}) share name '{clean_name}()'",
        )

    return ResolvedEntity(
        candidate=candidate,
        status=ResolutionStatus.UNRESOLVED,
        matched_nodes=(),
        resolution_reason=f"no method found with name '{clean_name}()' in repository",
    )


def _resolve_symbol(
    candidate: CandidateEntity,
    index: GraphIndex,
    file_context: str | None = None,
) -> ResolvedEntity:
    name = candidate.text.strip()
    matches = index.nodes_by_label.get(name, [])

    if len(matches) == 1:
        node = matches[0]
        loc = f":{node.source_location}" if node.source_location else ""
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=tuple(matches),
            resolution_reason=f"exact symbol match for '{name}' in {node.source_file}{loc}",
        )
    elif len(matches) > 1:
        if file_context:
            scoped = [n for n in matches if n.source_file == file_context]
            if len(scoped) == 1:
                node = scoped[0]
                loc = f":{node.source_location}" if node.source_location else ""
                return ResolvedEntity(
                    candidate=candidate,
                    status=ResolutionStatus.RESOLVED,
                    matched_nodes=tuple(scoped),
                    resolution_reason=f"exact symbol match for '{name}' in plan-scoped file {node.source_file}{loc}",
                )
            elif len(scoped) > 1:
                matches = scoped
        locs = [f"{n.source_file}:{n.source_location or ''}" for n in matches]
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=tuple(matches),
            resolution_reason=f"multiple entities ({len(matches)}) match symbol '{name}': " + ", ".join(locs),
        )

    return ResolvedEntity(
        candidate=candidate,
        status=ResolutionStatus.UNRESOLVED,
        matched_nodes=(),
        resolution_reason=f"no code entity matches symbol '{name}' in repository",
    )


def _resolve_concept(candidate: CandidateEntity, index: GraphIndex) -> ResolvedEntity:
    name_lower = candidate.text.strip().lower()

    # Check if Graphify has an explicit concept node for this term
    concept_matches = index.concepts_by_name.get(name_lower, [])
    if len(concept_matches) == 1:
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=tuple(concept_matches),
            resolution_reason=f"matched explicit concept node '{candidate.text}' in graph",
        )
    elif len(concept_matches) > 1:
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=tuple(concept_matches),
            resolution_reason=f"multiple concept nodes match '{candidate.text}'",
        )

    # Check if a code node matches the concept term exactly (e.g. class Redis or module)
    code_matches = [
        n for n in index.nodes_by_label.get(candidate.text.strip(), [])
        if n.file_type == "code"
    ]
    if len(code_matches) == 1:
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.RESOLVED,
            matched_nodes=tuple(code_matches),
            resolution_reason=f"concept term '{candidate.text}' matches code entity in {code_matches[0].source_file}",
        )
    elif len(code_matches) > 1:
        return ResolvedEntity(
            candidate=candidate,
            status=ResolutionStatus.AMBIGUOUS,
            matched_nodes=tuple(code_matches),
            resolution_reason=f"concept term '{candidate.text}' matches {len(code_matches)} code entities",
        )

    # Standard case: concepts are architectural vocabulary, not code entities
    return ResolvedEntity(
        candidate=candidate,
        status=ResolutionStatus.NOT_CODE_ENTITY,
        matched_nodes=(),
        resolution_reason="architectural/technology concept; skipped from code-entity resolution",
    )


def _find_scoped_file_context(
    candidate: CandidateEntity,
    all_candidates: tuple[CandidateEntity, ...] | list[CandidateEntity],
    index: GraphIndex,
) -> str | None:
    """Find an explicit, unambiguous repository file path associated with this candidate in the plan.

    Checks:
    1. Exact same step (same section + step_number) or same line_number
    2. Same section

    Returns normalized repository file path if exactly one unique resolved file is found in scope,
    or None if 0 or >1 files are found.
    """
    if candidate.kind == CandidateKind.FILE_PATH:
        return None

    def get_repo_file(cand: CandidateEntity) -> str | None:
        norm = normalize_path(cand.text)
        if norm in index.nodes_by_source_file:
            return norm
        if "/" not in norm:
            matches = index.files_by_basename.get(norm, set())
            if len(matches) == 1:
                return next(iter(matches))
        else:
            suffix_matches = [sf for sf in index.nodes_by_source_file if sf.endswith(f"/{norm}")]
            if len(suffix_matches) == 1:
                return suffix_matches[0]
        return None

    # Scope 1: Exact same step (same section + step_number) or same line
    step_files: set[str] = set()
    for other in all_candidates:
        if other.kind == CandidateKind.FILE_PATH:
            same_step = (
                candidate.location.step_number is not None
                and other.location.section == candidate.location.section
                and other.location.step_number == candidate.location.step_number
            )
            same_line = other.location.line_number == candidate.location.line_number
            if same_step or same_line:
                repo_file = get_repo_file(other)
                if repo_file:
                    step_files.add(repo_file)

    if len(step_files) == 1:
        return next(iter(step_files))
    if len(step_files) > 1:
        # Multiple conflicting files in same step -> do not guess
        return None

    # Scope 2: Nearest preceding step declaring a file in the same section
    if candidate.location.step_number is not None:
        preceding_step_files: dict[int, set[str]] = {}
        for other in all_candidates:
            if (
                other.kind == CandidateKind.FILE_PATH
                and other.location.section == candidate.location.section
                and other.location.step_number is not None
                and other.location.step_number < candidate.location.step_number
            ):
                repo_file = get_repo_file(other)
                if repo_file:
                    preceding_step_files.setdefault(other.location.step_number, set()).add(repo_file)

        if preceding_step_files:
            latest_step = max(preceding_step_files.keys())
            latest_files = preceding_step_files[latest_step]
            if len(latest_files) == 1:
                return next(iter(latest_files))
            if len(latest_files) > 1:
                return None

    # Scope 3: Section prose files (outside of any numbered step)
    section_prose_files: set[str] = set()
    for other in all_candidates:
        if (
            other.kind == CandidateKind.FILE_PATH
            and other.location.section == candidate.location.section
            and other.location.step_number is None
        ):
            repo_file = get_repo_file(other)
            if repo_file:
                section_prose_files.add(repo_file)

    if len(section_prose_files) == 1:
        return next(iter(section_prose_files))
    if len(section_prose_files) > 1:
        return None

    # Scope 4: Unambiguous section-wide file (if exactly one unique file across the entire section)
    section_all_files: set[str] = set()
    for other in all_candidates:
        if (
            other.kind == CandidateKind.FILE_PATH
            and other.location.section == candidate.location.section
        ):
            repo_file = get_repo_file(other)
            if repo_file:
                section_all_files.add(repo_file)

    if len(section_all_files) == 1:
        return next(iter(section_all_files))

    return None


def resolve_candidate(
    candidate: CandidateEntity,
    index: GraphIndex,
    file_context: str | None = None,
) -> ResolvedEntity:
    """Resolve a single CandidateEntity against the repository graph index."""
    if candidate.kind == CandidateKind.FILE_PATH:
        return _resolve_file_path(candidate, index)
    elif candidate.kind == CandidateKind.CLASS:
        return _resolve_class(candidate, index, file_context=file_context)
    elif candidate.kind == CandidateKind.FUNCTION:
        return _resolve_function(candidate, index, file_context=file_context)
    elif candidate.kind == CandidateKind.METHOD:
        return _resolve_method(candidate, index, file_context=file_context)
    elif candidate.kind == CandidateKind.SYMBOL:
        return _resolve_symbol(candidate, index, file_context=file_context)
    elif candidate.kind == CandidateKind.CONCEPT:
        return _resolve_concept(candidate, index)

    return ResolvedEntity(
        candidate=candidate,
        status=ResolutionStatus.UNRESOLVED,
        matched_nodes=(),
        resolution_reason=f"unknown candidate kind: {candidate.kind}",
    )


# -----------------------------------------------------------------------------
# Plan Resolution Entry Point
# -----------------------------------------------------------------------------

def resolve_plan(
    plan: ParsedPlan | str | Path,
    graph: nx.DiGraph | GraphIndex | str | Path,
) -> ResolutionResult:
    """Resolve all candidate references in a contributor plan against a code graph.

    Args:
        plan: Either an already parsed ParsedPlan, or a path/string to a markdown plan.
        graph: Either a NetworkX DiGraph, a pre-computed GraphIndex, or a path to graph.json.

    Returns:
        ResolutionResult: Structured result categorizing each candidate as
                          RESOLVED, AMBIGUOUS, UNRESOLVED, or NOT_CODE_ENTITY.
    """
    plan_file_str: str | None = None
    if isinstance(plan, (str, Path)):
        if isinstance(plan, Path) or (isinstance(plan, str) and "\n" not in plan):
            plan_file_str = str(plan)
        parsed_plan = parse_plan(plan)
    elif isinstance(plan, ParsedPlan):
        parsed_plan = plan
        plan_file_str = str(plan.file_path) if plan.file_path else None
    else:
        raise PlanGraphError(f"Unsupported plan type: {type(plan).__name__}")

    graph_file_str: str | None = None
    if isinstance(graph, (str, Path)):
        graph_file_str = str(graph)
        di_graph = load_graph(graph)
        index = GraphIndex(di_graph)
    elif isinstance(graph, nx.DiGraph):
        index = GraphIndex(graph)
    elif isinstance(graph, GraphIndex):
        index = graph
    else:
        raise PlanGraphError(f"Unsupported graph type: {type(graph).__name__}")

    resolved_list: list[ResolvedEntity] = []
    for candidate in parsed_plan.all_candidates:
        file_ctx = _find_scoped_file_context(candidate, parsed_plan.all_candidates, index)
        resolved_entity = resolve_candidate(candidate, index, file_context=file_ctx)
        resolved_list.append(resolved_entity)

    return ResolutionResult(
        plan_title=parsed_plan.title,
        plan_file=plan_file_str,
        graph_file=graph_file_str,
        entities=tuple(resolved_list),
    )
