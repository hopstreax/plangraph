"""Graph loading and normalization for PlanGraph.

Loads Graphify graph.json artifacts into NetworkX DiGraph instances with
strict validation and preserved directed edge semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from plangraph.exceptions import (
    GraphFormatError,
    GraphNotFoundError,
    GraphValidationError,
)
from plangraph.models import Edge, Node


REQUIRED_NODE_FIELDS = ("id", "label")
REQUIRED_EDGE_FIELDS = ("source", "target", "relation")


def load_graph(path: str | Path) -> nx.DiGraph:
    """Load and validate a Graphify graph.json into a directed NetworkX graph.

    Args:
        path: Path to the graph.json file.

    Returns:
        nx.DiGraph: A directed graph where edges flow from source to target
                    (e.g., caller -> callee, importer -> imported).

    Raises:
        GraphNotFoundError: If the specified file does not exist.
        GraphFormatError: If the file is not valid JSON or root is not an object.
        GraphValidationError: If required fields or types are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise GraphNotFoundError(f"Graph file not found: {file_path}")

    if file_path.is_dir():
        raise GraphFormatError(f"Expected a graph JSON file, but path is a directory: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GraphFormatError(f"Unable to read graph file {file_path}: {exc}") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GraphFormatError(
            f"Invalid JSON in graph file {file_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(data, dict):
        raise GraphFormatError(
            f"Graph JSON root must be an object/dict, got {type(data).__name__}"
        )

    # Validate root keys
    if "nodes" not in data:
        raise GraphValidationError("Graph JSON missing required root key 'nodes'")

    raw_nodes = data["nodes"]
    if not isinstance(raw_nodes, list):
        raise GraphValidationError(
            f"Root key 'nodes' must be a list, got {type(raw_nodes).__name__}"
        )

    # Edge list can be keyed as 'links' (NetworkX node_link default) or 'edges' (raw extraction)
    if "links" in data:
        raw_edges = data["links"]
    elif "edges" in data:
        raw_edges = data["edges"]
    else:
        raise GraphValidationError("Graph JSON missing required edge key ('links' or 'edges')")

    if not isinstance(raw_edges, list):
        raise GraphValidationError(
            f"Edge list must be a list, got {type(raw_edges).__name__}"
        )

    # Build directed graph
    graph = nx.DiGraph()

    # Preserve top-level graph metadata (commit, hyperedges, etc.)
    top_level_meta = {
        k: v for k, v in data.items() if k not in ("nodes", "links", "edges")
    }
    graph.graph.update(top_level_meta)

    # Process and validate nodes
    for idx, node_data in enumerate(raw_nodes):
        if not isinstance(node_data, dict):
            raise GraphValidationError(
                f"Node at index {idx} must be a dictionary, got {type(node_data).__name__}"
            )

        for req_field in REQUIRED_NODE_FIELDS:
            if req_field not in node_data:
                node_identifier = node_data.get("id", f"index {idx}")
                raise GraphValidationError(
                    f"Node '{node_identifier}' missing required field '{req_field}'"
                )

        node_id = str(node_data["id"])
        if not node_id:
            raise GraphValidationError(f"Node at index {idx} has an empty 'id'")

        # Node attributes to store in graph
        attrs = {k: v for k, v in node_data.items() if k != "id"}
        graph.add_node(node_id, **attrs)

    # Process and validate edges
    for idx, edge_data in enumerate(raw_edges):
        if not isinstance(edge_data, dict):
            raise GraphValidationError(
                f"Edge at index {idx} must be a dictionary, got {type(edge_data).__name__}"
            )

        # Support 'from' / 'to' alias if present
        source = edge_data.get("source", edge_data.get("from"))
        target = edge_data.get("target", edge_data.get("to"))

        if source is None:
            raise GraphValidationError(f"Edge at index {idx} missing required field 'source'")
        if target is None:
            raise GraphValidationError(f"Edge at index {idx} missing required field 'target'")

        if "relation" not in edge_data:
            raise GraphValidationError(
                f"Edge from '{source}' to '{target}' missing required field 'relation'"
            )

        src_id = str(source)
        tgt_id = str(target)
        edge_attrs = {k: v for k, v in edge_data.items() if k not in ("source", "target", "from", "to")}

        # Ensure edge direction is strictly source -> target
        graph.add_edge(src_id, tgt_id, **edge_attrs)

    return graph


def get_node(graph: nx.DiGraph, node_id: str) -> Node | None:
    """Retrieve a strongly-typed Node object from the graph, or None if not found."""
    if node_id not in graph:
        return None
    data = dict(graph.nodes[node_id])
    data["id"] = node_id
    return Node.from_dict(data)


def get_edge(graph: nx.DiGraph, source: str, target: str) -> Edge | None:
    """Retrieve a strongly-typed Edge object between source and target, or None if not found."""
    if not graph.has_edge(source, target):
        return None
    data = dict(graph.edges[source, target])
    data["source"] = source
    data["target"] = target
    return Edge.from_dict(data)
