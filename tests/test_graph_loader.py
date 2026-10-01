"""Unit tests for the PlanGraph graph loader."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from plangraph.exceptions import (
    GraphFormatError,
    GraphNotFoundError,
    GraphValidationError,
)
from plangraph.graph import get_edge, get_node, load_graph
from plangraph.models import Edge, Node


def test_load_valid_graph_with_links(synthetic_graph_file: Path) -> None:
    """Test loading a standard Graphify graph.json using the 'links' key."""
    graph = load_graph(synthetic_graph_file)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3

    # Check top-level metadata preservation
    assert graph.graph.get("built_at_commit") == "a1b2c3d4"

    # Check node attributes
    service = graph.nodes["services_user_userservice"]
    assert service["label"] == "UserService"
    assert service["source_file"] == "services/user.py"
    assert service["source_location"] == "L25"
    assert service["community"] == 1


def test_load_valid_graph_with_edges_key(synthetic_graph_edges_file: Path) -> None:
    """Test loading a graph where the edge list is keyed as 'edges'."""
    graph = load_graph(synthetic_graph_edges_file)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3


def test_directed_edge_preservation(synthetic_graph_file: Path) -> None:
    """Test that caller -> callee directionality is strictly preserved."""
    graph = load_graph(synthetic_graph_file)

    # UserController -> calls -> UserService
    assert graph.has_edge("controllers_user_usercontroller", "services_user_userservice")
    # Reverse edge must NOT exist
    assert not graph.has_edge("services_user_userservice", "controllers_user_usercontroller")

    # Inbound callers of UserService
    callers = list(graph.predecessors("services_user_userservice"))
    assert set(callers) == {
        "controllers_user_usercontroller",
        "tests_test_user_service_test_user_service",
    }

    # Outbound dependencies of UserService
    dependencies = list(graph.successors("services_user_userservice"))
    assert dependencies == ["repositories_user_userrepository"]


def test_missing_file_raises_not_found(tmp_path: Path) -> None:
    """Loading a non-existent file should raise GraphNotFoundError."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(GraphNotFoundError, match="Graph file not found"):
        load_graph(missing)


def test_directory_raises_format_error(tmp_path: Path) -> None:
    """Providing a directory path instead of a file should raise GraphFormatError."""
    with pytest.raises(GraphFormatError, match="path is a directory"):
        load_graph(tmp_path)


def test_invalid_json_raises_format_error(tmp_path: Path) -> None:
    """Providing a malformed JSON file should raise GraphFormatError."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{ unquoted_key: 123 ", encoding="utf-8")

    with pytest.raises(GraphFormatError, match="Invalid JSON"):
        load_graph(bad_json)


def test_non_dict_root_raises_format_error(tmp_path: Path) -> None:
    """JSON root that is a list instead of an object should raise GraphFormatError."""
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    with pytest.raises(GraphFormatError, match="must be an object/dict"):
        load_graph(list_json)


def test_missing_nodes_key_raises_validation_error(tmp_path: Path) -> None:
    """Missing 'nodes' key should raise GraphValidationError."""
    invalid = tmp_path / "no_nodes.json"
    invalid.write_text(json.dumps({"links": []}), encoding="utf-8")

    with pytest.raises(GraphValidationError, match="missing required root key 'nodes'"):
        load_graph(invalid)


def test_missing_edge_keys_raises_validation_error(tmp_path: Path) -> None:
    """Missing both 'links' and 'edges' key should raise GraphValidationError."""
    invalid = tmp_path / "no_edges.json"
    invalid.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    with pytest.raises(GraphValidationError, match="missing required edge key"):
        load_graph(invalid)


def test_node_missing_id_raises_validation_error(tmp_path: Path) -> None:
    """Node missing 'id' field should raise GraphValidationError."""
    invalid = tmp_path / "bad_node.json"
    data = {
        "nodes": [{"label": "UserService"}],
        "links": [],
    }
    invalid.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(GraphValidationError, match="missing required field 'id'"):
        load_graph(invalid)


def test_node_missing_label_raises_validation_error(tmp_path: Path) -> None:
    """Node missing 'label' field should raise GraphValidationError."""
    invalid = tmp_path / "bad_node.json"
    data = {
        "nodes": [{"id": "user_service"}],
        "links": [],
    }
    invalid.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(GraphValidationError, match="missing required field 'label'"):
        load_graph(invalid)


def test_edge_missing_relation_raises_validation_error(tmp_path: Path) -> None:
    """Edge missing 'relation' field should raise GraphValidationError."""
    invalid = tmp_path / "bad_edge.json"
    data = {
        "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        "links": [{"source": "a", "target": "b"}],
    }
    invalid.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(GraphValidationError, match="missing required field 'relation'"):
        load_graph(invalid)


def test_edge_from_to_alias_supported(tmp_path: Path) -> None:
    """Edges using legacy 'from'/'to' keys should be accepted and normalized."""
    alias_json = tmp_path / "alias.json"
    data = {
        "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        "links": [{"from": "a", "to": "b", "relation": "calls"}],
    }
    alias_json.write_text(json.dumps(data), encoding="utf-8")

    graph = load_graph(alias_json)
    assert graph.has_edge("a", "b")


def test_get_node_and_get_edge_helpers(synthetic_graph_file: Path) -> None:
    """Test strongly-typed node and edge helper lookups."""
    graph = load_graph(synthetic_graph_file)

    # Node lookup
    node = get_node(graph, "services_user_userservice")
    assert isinstance(node, Node)
    assert node.id == "services_user_userservice"
    assert node.label == "UserService"
    assert node.source_file == "services/user.py"

    # Nonexistent node
    assert get_node(graph, "does_not_exist") is None

    # Edge lookup
    edge = get_edge(graph, "controllers_user_usercontroller", "services_user_userservice")
    assert isinstance(edge, Edge)
    assert edge.source == "controllers_user_usercontroller"
    assert edge.target == "services_user_userservice"
    assert edge.relation == "calls"
    assert edge.confidence == "EXTRACTED"

    # Nonexistent edge
    assert get_edge(graph, "services_user_userservice", "controllers_user_usercontroller") is None
