"""Deterministic unit tests for PlanGraph Phase 3 candidate entity resolver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from plangraph.graph import load_graph
from plangraph.models import Node
from plangraph.plan import (
    CandidateEntity,
    CandidateKind,
    Confidence,
    PlanLocation,
    parse_plan,
)
from plangraph.resolver import (
    GraphIndex,
    ResolutionResult,
    ResolutionStatus,
    ResolvedEntity,
    normalize_path,
    resolve_candidate,
    resolve_plan,
)


@pytest.fixture
def resolver_test_graph() -> nx.DiGraph:
    """A realistic graph topology with classes, methods, functions, and ambiguous entities."""
    data: dict[str, Any] = {
        "directed": True,
        "nodes": [
            # Files / Member nodes
            {
                "id": "services_user_userservice",
                "label": "UserService",
                "file_type": "code",
                "source_file": "services/user.py",
                "source_location": "L20",
                "_callable_class": True,
            },
            {
                "id": "services_user_userservice_get_user",
                "label": ".get_user()",
                "file_type": "code",
                "source_file": "services/user.py",
                "source_location": "L25",
                "_callable": True,
            },
            {
                "id": "controllers_user_usercontroller",
                "label": "UserController",
                "file_type": "code",
                "source_file": "controllers/user.py",
                "source_location": "L10",
                "_callable_class": True,
            },
            {
                "id": "controllers_user_usercontroller_get_user",
                "label": ".get_user()",
                "file_type": "code",
                "source_file": "controllers/user.py",
                "source_location": "L15",
                "_callable": True,
            },
            {
                "id": "models_user_userservice",
                "label": "UserService",
                "file_type": "code",
                "source_file": "models/user.py",
                "source_location": "L5",
                "_callable_class": True,
            },
            {
                "id": "auth_token_validate_token",
                "label": "validate_token()",
                "file_type": "code",
                "source_file": "auth/token.py",
                "source_location": "L30",
                "_callable": True,
            },
            {
                "id": "auth_token_hash_password",
                "label": "hash_password",
                "file_type": "code",
                "source_file": "auth/token.py",
                "source_location": "L45",
                "_callable": True,
            },
            {
                "id": "utils_helpers_format_name",
                "label": "format_name()",
                "file_type": "code",
                "source_file": "utils/helpers.py",
                "source_location": "L12",
                "_callable": True,
            },
            {
                "id": "legacy_helpers_format_name",
                "label": "format_name()",
                "file_type": "code",
                "source_file": "legacy/helpers.py",
                "source_location": "L12",
                "_callable": True,
            },
            {
                "id": "config_constants_max_retries",
                "label": "MAX_RETRIES",
                "file_type": "code",
                "source_file": "config/constants.py",
                "source_location": "L3",
            },
            {
                "id": "concept_email",
                "label": "Email",
                "file_type": "concept",
                "source_file": "docs/architecture.md",
            },
        ],
        "links": [
            # UserService.get_user method edge
            {
                "source": "services_user_userservice",
                "target": "services_user_userservice_get_user",
                "relation": "method",
            },
            # UserController.get_user method edge
            {
                "source": "controllers_user_usercontroller",
                "target": "controllers_user_usercontroller_get_user",
                "relation": "method",
            },
        ],
    }

    graph = nx.DiGraph()
    for n in data["nodes"]:
        attrs = {k: v for k, v in n.items() if k != "id"}
        graph.add_node(n["id"], **attrs)
    for e in data["links"]:
        attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
        graph.add_edge(e["source"], e["target"], **attrs)
    return graph


@pytest.fixture
def resolver_index(resolver_test_graph: nx.DiGraph) -> GraphIndex:
    return GraphIndex(resolver_test_graph)


# -----------------------------------------------------------------------------
# Path Normalization Tests
# -----------------------------------------------------------------------------

def test_normalize_path() -> None:
    """Test file path normalization across formats and separators."""
    assert normalize_path(r"services\user.py") == "services/user.py"
    assert normalize_path("./services/user.py") == "services/user.py"
    assert normalize_path("services//user.py") == "services/user.py"
    assert normalize_path("/services/user.py") == "services/user.py"
    assert normalize_path("  common/cache.py  ") == "common/cache.py"


# -----------------------------------------------------------------------------
# File Resolution Tests
# -----------------------------------------------------------------------------

def test_file_resolution_exact_relative(resolver_index: GraphIndex) -> None:
    """Test resolving an exact repository-relative path."""
    cand = CandidateEntity(
        text="services/user.py",
        raw_text="`services/user.py`",
        kind=CandidateKind.FILE_PATH,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Check services/user.py",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert len(res.matched_nodes) >= 1
    assert any(n.id == "services_user_userservice" for n in res.matched_nodes)
    assert "exact file path match" in res.resolution_reason


def test_file_resolution_windows_path_normalization(resolver_index: GraphIndex) -> None:
    """Test resolving a Windows-style path with backslashes."""
    cand = CandidateEntity(
        text="services\\user.py",
        raw_text="services\\user.py",
        kind=CandidateKind.FILE_PATH,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Check services\\user.py",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert any(n.source_file == "services/user.py" for n in res.matched_nodes)


def test_file_resolution_leading_dot_slash(resolver_index: GraphIndex) -> None:
    """Test resolving a path with leading ./."""
    cand = CandidateEntity(
        text="./services/user.py",
        raw_text="`./services/user.py`",
        kind=CandidateKind.FILE_PATH,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Check ./services/user.py",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED


def test_file_resolution_nonexistent(resolver_index: GraphIndex) -> None:
    """Test that a nonexistent file returns UNRESOLVED."""
    cand = CandidateEntity(
        text="nonexistent/file.py",
        raw_text="`nonexistent/file.py`",
        kind=CandidateKind.FILE_PATH,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Check nonexistent/file.py",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.UNRESOLVED
    assert len(res.matched_nodes) == 0
    assert "no Graphify file" in res.resolution_reason


def test_file_resolution_ambiguous_basename(resolver_index: GraphIndex) -> None:
    """Test that an unqualified basename matching multiple files returns AMBIGUOUS."""
    cand = CandidateEntity(
        text="helpers.py",
        raw_text="`helpers.py`",
        kind=CandidateKind.FILE_PATH,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Check helpers.py",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.AMBIGUOUS
    assert len(res.matched_nodes) == 2
    assert "matches 2 repository paths" in res.resolution_reason


# -----------------------------------------------------------------------------
# Class Resolution Tests
# -----------------------------------------------------------------------------

def test_class_resolution_exact(resolver_index: GraphIndex) -> None:
    """Test resolving an unambiguous class name."""
    cand = CandidateEntity(
        text="UserController",
        raw_text="`UserController`",
        kind=CandidateKind.CLASS,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Update UserController",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert len(res.matched_nodes) == 1
    assert res.matched_nodes[0].id == "controllers_user_usercontroller"


def test_class_resolution_ambiguous(resolver_index: GraphIndex) -> None:
    """Test that duplicate class names in different files return AMBIGUOUS with all matches preserved."""
    cand = CandidateEntity(
        text="UserService",
        raw_text="`UserService`",
        kind=CandidateKind.CLASS,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Update UserService",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.AMBIGUOUS
    assert len(res.matched_nodes) == 2
    matched_ids = {n.id for n in res.matched_nodes}
    assert "services_user_userservice" in matched_ids
    assert "models_user_userservice" in matched_ids
    assert "multiple class nodes (2)" in res.resolution_reason


def test_class_resolution_nonexistent(resolver_index: GraphIndex) -> None:
    """Test that a nonexistent class returns UNRESOLVED."""
    cand = CandidateEntity(
        text="MissingService",
        raw_text="`MissingService`",
        kind=CandidateKind.CLASS,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Add MissingService",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.UNRESOLVED
    assert len(res.matched_nodes) == 0


# -----------------------------------------------------------------------------
# Function Resolution Tests
# -----------------------------------------------------------------------------

def test_function_resolution_with_parentheses(resolver_index: GraphIndex) -> None:
    """Test resolving a function name written with parentheses."""
    cand = CandidateEntity(
        text="validate_token()",
        raw_text="`validate_token()`",
        kind=CandidateKind.FUNCTION,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Call validate_token()",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert len(res.matched_nodes) == 1
    assert res.matched_nodes[0].id == "auth_token_validate_token"


def test_function_resolution_without_parentheses(resolver_index: GraphIndex) -> None:
    """Test resolving a function name written without parentheses."""
    cand = CandidateEntity(
        text="hash_password",
        raw_text="`hash_password`",
        kind=CandidateKind.FUNCTION,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Call hash_password",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert res.matched_nodes[0].id == "auth_token_hash_password"


def test_function_resolution_ambiguous(resolver_index: GraphIndex) -> None:
    """Test that multiple functions sharing the same name return AMBIGUOUS."""
    cand = CandidateEntity(
        text="format_name()",
        raw_text="`format_name()`",
        kind=CandidateKind.FUNCTION,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Invoke format_name()",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.AMBIGUOUS
    assert len(res.matched_nodes) == 2


# -----------------------------------------------------------------------------
# Method Resolution Tests
# -----------------------------------------------------------------------------

def test_method_resolution_qualified(resolver_index: GraphIndex) -> None:
    """Test resolving an exact qualified method call Class.method()."""
    cand = CandidateEntity(
        text="UserService.get_user()",
        raw_text="`UserService.get_user()`",
        kind=CandidateKind.METHOD,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Call UserService.get_user()",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert len(res.matched_nodes) == 2
    class_node, meth_node = res.matched_nodes
    assert class_node.label == "UserService"
    assert meth_node.id == "services_user_userservice_get_user"


def test_method_resolution_unqualified_same_name_different_classes(resolver_index: GraphIndex) -> None:
    """Test that an unqualified method name matching multiple classes returns AMBIGUOUS."""
    cand = CandidateEntity(
        text="get_user()",
        raw_text="`get_user()`",
        kind=CandidateKind.METHOD,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Call get_user()",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.AMBIGUOUS
    assert len(res.matched_nodes) == 4  # (UserService, get_user) and (UserController, get_user)


def test_method_resolution_missing_method_on_existing_class(resolver_index: GraphIndex) -> None:
    """Test that an existing class with a missing method returns UNRESOLVED with clear reason."""
    cand = CandidateEntity(
        text="UserService.delete_user()",
        raw_text="`UserService.delete_user()`",
        kind=CandidateKind.METHOD,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Call UserService.delete_user()",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.UNRESOLVED
    assert "class 'UserService' exists, but method 'delete_user' was not found" in res.resolution_reason


# -----------------------------------------------------------------------------
# Symbol Resolution Tests
# -----------------------------------------------------------------------------

def test_symbol_resolution_exact(resolver_index: GraphIndex) -> None:
    """Test resolving an exact symbol like MAX_RETRIES."""
    cand = CandidateEntity(
        text="MAX_RETRIES",
        raw_text="`MAX_RETRIES`",
        kind=CandidateKind.SYMBOL,
        confidence=Confidence.EXPLICIT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Check MAX_RETRIES",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert res.matched_nodes[0].id == "config_constants_max_retries"


# -----------------------------------------------------------------------------
# Concept Resolution Tests
# -----------------------------------------------------------------------------

def test_concept_resolution_not_code_entity(resolver_index: GraphIndex) -> None:
    """Test that architectural concepts (e.g. Redis, caching) become NOT_CODE_ENTITY and not unresolved code."""
    cand = CandidateEntity(
        text="Redis",
        raw_text="Redis",
        kind=CandidateKind.CONCEPT,
        confidence=Confidence.CONCEPT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Add Redis caching",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.NOT_CODE_ENTITY
    assert len(res.matched_nodes) == 0
    assert "architectural/technology concept" in res.resolution_reason


def test_concept_resolution_explicit_graph_concept(resolver_index: GraphIndex) -> None:
    """Test that a concept explicitly present in Graphify as a concept node resolves."""
    cand = CandidateEntity(
        text="Email",
        raw_text="Email",
        kind=CandidateKind.CONCEPT,
        confidence=Confidence.CONCEPT,
        location=PlanLocation(line_number=1, section="Overview"),
        context_sentence="Send Email",
    )
    res = resolve_candidate(cand, resolver_index)
    assert res.status == ResolutionStatus.RESOLVED
    assert res.matched_nodes[0].id == "concept_email"


# -----------------------------------------------------------------------------
# End-to-End Plan Resolution Tests
# -----------------------------------------------------------------------------

def test_resolve_plan_mixed(resolver_test_graph: nx.DiGraph) -> None:
    """Test resolving a complete realistic plan against a code graph."""
    plan_md = """# Caching & User Service Plan

## Changes
1. Update `services/user.py` to add Redis caching.
2. Call `UserService.get_user()` in handler.
3. Update `UserController` endpoint.
4. Add `CacheManager` in `common/cache.py`.
5. Invoke `validate_token()`.
"""
    result = resolve_plan(plan_md, resolver_test_graph)

    assert isinstance(result, ResolutionResult)
    assert result.plan_title == "Caching & User Service Plan"

    # Verify counts
    assert len(result.resolved) >= 3  # services/user.py, UserService.get_user(), UserController, validate_token()
    assert len(result.unresolved) >= 2  # CacheManager, common/cache.py
    assert len(result.not_code_entities) >= 1  # Redis / caching

    # Verify serialization
    res_dict = result.to_dict()
    assert res_dict["summary"]["resolved"] == len(result.resolved)
    assert res_dict["summary"]["unresolved"] == len(result.unresolved)
    assert res_dict["summary"]["not_code_entities"] == len(result.not_code_entities)


def test_resolve_plan_determinism(resolver_test_graph: nx.DiGraph) -> None:
    """Test that running resolve_plan repeatedly produces identical candidate ordering and matches."""
    plan_md = """## Steps
1. Modify `services/user.py`.
2. Update `UserService`.
3. Check `UserController`.
4. Run `validate_token()`.
"""
    res1 = resolve_plan(plan_md, resolver_test_graph)
    res2 = resolve_plan(plan_md, resolver_test_graph)

    assert len(res1.entities) == len(res2.entities)
    for e1, e2 in zip(res1.entities, res2.entities):
        assert e1.candidate.text == e2.candidate.text
        assert e1.status == e2.status
        assert [n.id for n in e1.matched_nodes] == [n.id for n in e2.matched_nodes]
        assert e1.resolution_reason == e2.resolution_reason


# -----------------------------------------------------------------------------
# CLI Integration Tests
# -----------------------------------------------------------------------------

def test_cli_resolve_plan_human_readable(
    tmp_path: Path, resolver_test_graph: nx.DiGraph, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI resolve-plan subcommand with human-readable output."""
    from plangraph.__main__ import main

    # Prepare plan and graph files
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(
        "# Feature Plan\n\n1. Update `services/user.py`.\n2. Call `UserService.get_user()`.\n3. Add Redis caching.\n",
        encoding="utf-8",
    )

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [
            {
                "id": node_id,
                **data,
            }
            for node_id, data in resolver_test_graph.nodes(data=True)
        ],
        "links": [
            {
                "source": u,
                "target": v,
                **data,
            }
            for u, v, data in resolver_test_graph.edges(data=True)
        ],
    }
    graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

    exit_code = main(["resolve-plan", str(plan_file), "--graph", str(graph_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "PlanGraph: Plan Resolution" in captured.out
    assert "services/user.py" in captured.out
    assert "UserService.get_user()" in captured.out
    assert "Redis" in captured.out
    assert "Summary:" in captured.out
    assert "Resolved:" in captured.out


def test_cli_resolve_plan_json(
    tmp_path: Path, resolver_test_graph: nx.DiGraph, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI resolve-plan subcommand with --json output."""
    from plangraph.__main__ import main

    plan_file = tmp_path / "plan.md"
    plan_file.write_text(
        "## Steps\n1. Modify `services/user.py`.\n2. Add `UnknownClass`.\n",
        encoding="utf-8",
    )

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [
            {"id": node_id, **data}
            for node_id, data in resolver_test_graph.nodes(data=True)
        ],
        "links": [
            {"source": u, "target": v, **data}
            for u, v, data in resolver_test_graph.edges(data=True)
        ],
    }
    graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

    exit_code = main(["resolve-plan", str(plan_file), "-g", str(graph_file), "--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["total_candidates"] == 2
    assert payload["summary"]["resolved"] == 1
    assert payload["summary"]["unresolved"] == 1
    assert len(payload["entities"]) == 2


def test_cli_resolve_plan_missing_files_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI resolve-plan with missing files returns exit code 1."""
    from plangraph.__main__ import main

    plan_file = tmp_path / "plan.md"
    plan_file.write_text("Update `services/user.py`", encoding="utf-8")

    exit_code = main(["resolve-plan", str(plan_file), "-g", "nonexistent_graph.json"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error:" in captured.err


# -----------------------------------------------------------------------------
# Context-Aware Disambiguation Tests
# -----------------------------------------------------------------------------

def test_contextual_disambiguation_unique_in_file(resolver_test_graph: nx.DiGraph) -> None:
    """A globally ambiguous function is disambiguated when scoped to a file in the same plan step."""
    plan_md = """# Implementation Plan
## Step 1
1. In `utils/helpers.py`, update `format_name()`.
"""
    res = resolve_plan(plan_md, resolver_test_graph)
    file_cand = next(e for e in res.entities if e.candidate.kind == CandidateKind.FILE_PATH)
    assert file_cand.status == ResolutionStatus.RESOLVED

    fn_cand = next(e for e in res.entities if e.candidate.kind == CandidateKind.FUNCTION)
    assert fn_cand.status == ResolutionStatus.RESOLVED
    assert len(fn_cand.matched_nodes) == 1
    assert fn_cand.matched_nodes[0].source_file == "utils/helpers.py"
    assert "plan-scoped file" in fn_cand.resolution_reason


def test_contextual_disambiguation_multiple_in_file_remains_ambiguous(
    resolver_test_graph: nx.DiGraph,
) -> None:
    """If a file contains multiple entities matching the candidate, it remains AMBIGUOUS."""
    graph = resolver_test_graph.copy()
    graph.add_node(
        "utils_helpers_format_name_v2",
        label="format_name()",
        file_type="code",
        source_file="utils/helpers.py",
        source_location="L50",
        _callable=True,
    )
    plan_md = """# Plan
1. In `utils/helpers.py`, update `format_name()`.
"""
    res = resolve_plan(plan_md, graph)
    fn_cand = next(e for e in res.entities if e.candidate.kind == CandidateKind.FUNCTION)
    assert fn_cand.status == ResolutionStatus.AMBIGUOUS
    assert len(fn_cand.matched_nodes) == 2


def test_no_file_context_preserves_ambiguity(resolver_test_graph: nx.DiGraph) -> None:
    """When no file context is provided in the plan, ambiguous symbols remain AMBIGUOUS."""
    plan_md = """# Plan
1. Update `format_name()`.
"""
    res = resolve_plan(plan_md, resolver_test_graph)
    fn_cand = next(e for e in res.entities if e.candidate.kind == CandidateKind.FUNCTION)
    assert fn_cand.status == ResolutionStatus.AMBIGUOUS
    assert len(fn_cand.matched_nodes) == 2


def test_unrelated_file_context_does_not_affect_resolution(
    resolver_test_graph: nx.DiGraph,
) -> None:
    """When the file context in scope does not contain the candidate, it preserves global ambiguity."""
    plan_md = """# Plan
1. In `auth/token.py`, update `format_name()`.
"""
    res = resolve_plan(plan_md, resolver_test_graph)
    fn_cand = next(e for e in res.entities if e.candidate.kind == CandidateKind.FUNCTION)
    assert fn_cand.status == ResolutionStatus.AMBIGUOUS
    assert len(fn_cand.matched_nodes) == 2
