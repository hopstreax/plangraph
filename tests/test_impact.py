"""Focused unit and integration tests for PlanGraph Phase 4 Impact Analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from plangraph.__main__ import main
from plangraph.impact import (
    MAX_CONTAINMENT_DISPLAY,
    ImpactAnalysis,
    ImpactDirection,
    ImpactRelationship,
    ImpactSummary,
    ImpactedEntity,
    analyze_impact,
    format_impact_report,
)
from plangraph.models import Node
from plangraph.plan import (
    CandidateEntity,
    CandidateKind,
    Confidence,
    PlanLocation,
    parse_plan,
)
from plangraph.resolver import (
    ResolutionResult,
    ResolutionStatus,
    ResolvedEntity,
    resolve_plan,
)


@pytest.fixture
def impact_test_graph() -> nx.DiGraph:
    """A synthetic code graph with direct calls, imports, containment, and multi-hop chains."""
    graph = nx.DiGraph()

    # Nodes
    nodes: list[dict[str, Any]] = [
        # Services & Methods
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
        # Caller Controller
        {
            "id": "controllers_user_usercontroller",
            "label": "UserController",
            "file_type": "code",
            "source_file": "controllers/user.py",
            "source_location": "L10",
            "_callable_class": True,
        },
        {
            "id": "controllers_user_usercontroller_show",
            "label": ".show()",
            "file_type": "code",
            "source_file": "controllers/user.py",
            "source_location": "L15",
            "_callable": True,
        },
        # Downstream dependency 1 (direct 1-hop)
        {
            "id": "db_database_query",
            "label": "query()",
            "file_type": "code",
            "source_file": "db/database.py",
            "source_location": "L40",
            "_callable": True,
        },
        # Downstream dependency 2 (2-hop: must NOT be traversed)
        {
            "id": "db_pool_acquire",
            "label": "acquire()",
            "file_type": "code",
            "source_file": "db/pool.py",
            "source_location": "L80",
            "_callable": True,
        },
        # Downstream dependency 3 (3-hop: must NOT be traversed)
        {
            "id": "socket_connect",
            "label": "connect()",
            "file_type": "code",
            "source_file": "net/socket.py",
            "source_location": "L15",
            "_callable": True,
        },
        # Module file node
        {
            "id": "services_user",
            "label": "user.py",
            "file_type": "code",
            "source_file": "services/user.py",
            "source_location": "L1",
        },
        # Common cache dependency
        {
            "id": "common_cache",
            "label": "cache.py",
            "file_type": "code",
            "source_file": "common/cache.py",
            "source_location": "L1",
        },
        # Documentation / rationale node (noise to exclude)
        {
            "id": "rationale_userservice_doc",
            "label": "UserService manages user lifecycles.",
            "file_type": "rationale",
            "source_file": "services/user.py",
            "source_location": "L19",
        },
        # Ambiguous nodes
        {
            "id": "pkg_a_ambiguous",
            "label": "AmbiguousHelper",
            "file_type": "code",
            "source_file": "pkg_a/helper.py",
            "source_location": "L5",
            "_callable_class": True,
        },
        {
            "id": "pkg_b_ambiguous",
            "label": "AmbiguousHelper",
            "file_type": "code",
            "source_file": "pkg_b/helper.py",
            "source_location": "L5",
            "_callable_class": True,
        },
    ]

    for n in nodes:
        attrs = {k: v for k, v in n.items() if k != "id"}
        graph.add_node(n["id"], **attrs)

    # Edges
    links = [
        # UserService -> get_user (method)
        {"source": "services_user_userservice", "target": "services_user_userservice_get_user", "relation": "method"},
        # UserController -> show (method)
        {"source": "controllers_user_usercontroller", "target": "controllers_user_usercontroller_show", "relation": "method"},
        # UserController.show() -> UserService.get_user() (calls)
        {"source": "controllers_user_usercontroller_show", "target": "services_user_userservice_get_user", "relation": "calls"},
        # UserService.get_user() -> Database.query() (calls) [Direct 1-hop]
        {"source": "services_user_userservice_get_user", "target": "db_database_query", "relation": "calls"},
        # Database.query() -> Pool.acquire() (calls) [2-hop]
        {"source": "db_database_query", "target": "db_pool_acquire", "relation": "calls"},
        # Pool.acquire() -> Socket.connect() (calls) [3-hop]
        {"source": "db_pool_acquire", "target": "socket_connect", "relation": "calls"},
        # services/user.py -> common/cache.py (imports)
        {"source": "services_user", "target": "common_cache", "relation": "imports"},
        # Noise edge: rationale_for
        {"source": "rationale_userservice_doc", "target": "services_user_userservice", "relation": "rationale_for"},
    ]

    for e in links:
        attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
        graph.add_edge(e["source"], e["target"], **attrs)

    return graph


# -----------------------------------------------------------------------------
# 1. Direct Relationship Discovery & Direction Preservation
# -----------------------------------------------------------------------------

def test_direct_relationship_discovery_and_direction(impact_test_graph: nx.DiGraph) -> None:
    """Test that direct relationships are discovered and direction is accurately preserved."""
    plan_md = "1. Modify `UserService.get_user()`."
    analysis = analyze_impact(plan_md, impact_test_graph)

    assert len(analysis.impacted_entities) == 1
    entity = analysis.impacted_entities[0]
    assert entity.node_id == "services_user_userservice_get_user"
    assert entity.label == "UserService.get_user()"

    # Check relationships on get_user():
    # 1. OUTGOING: calls -> query()
    # 2. INCOMING: called_by -> UserController.show()
    # 3. INCOMING: defined_in -> UserService
    rel_map = {(r.display_relation, r.direction, r.target_label) for r in entity.relationships}
    assert ("calls", ImpactDirection.OUTGOING, "query()") in rel_map
    assert ("called_by", ImpactDirection.INCOMING, "UserController.show()") in rel_map
    assert ("defined_in", ImpactDirection.INCOMING, "UserService") in rel_map


# -----------------------------------------------------------------------------
# 2. Multiple Relationship Types & Noise Exclusion
# -----------------------------------------------------------------------------

def test_multiple_relationship_types_and_noise_exclusion(impact_test_graph: nx.DiGraph) -> None:
    """Test that call, import, and method relationships are included while rationale noise is excluded."""
    plan_md = """1. Update `UserService`.
2. Update `services/user.py`.
"""
    analysis = analyze_impact(plan_md, impact_test_graph)

    # UserService should have method -> get_user(), but NO rationale_for
    user_service = next(e for e in analysis.impacted_entities if e.node_id == "services_user_userservice")
    rels = {r.relation for r in user_service.relationships}
    assert "method" in rels
    assert "rationale_for" not in rels

    # Target node should not be a rationale node
    target_ids = {r.target_node_id for r in user_service.relationships}
    assert "rationale_userservice_doc" not in target_ids

    # services/user.py should have imports -> common/cache.py
    user_file = next(e for e in analysis.impacted_entities if e.node_id == "services_user")
    file_rels = {(r.display_relation, r.target_label) for r in user_file.relationships}
    assert ("imports", "cache.py") in file_rels


# -----------------------------------------------------------------------------
# 3. Multiple Candidates Resolving to the Same Entity (Deduplication)
# -----------------------------------------------------------------------------

def test_multiple_candidates_same_entity_deduplication(impact_test_graph: nx.DiGraph) -> None:
    """Test that when multiple plan candidates resolve to the same entity, it is deduplicated."""
    plan_md = """## Changes
1. Update `UserService.get_user()`.
2. Re-check `UserService.get_user()` in handler.
"""
    analysis = analyze_impact(plan_md, impact_test_graph)

    # Must be deduplicated: exactly 1 impacted entity
    assert len(analysis.impacted_entities) == 1
    entity = analysis.impacted_entities[0]
    assert entity.node_id == "services_user_userservice_get_user"

    # Both candidate occurrences should be preserved
    assert len(entity.originating_candidates) == 2
    assert entity.originating_candidates[0].location.line_number == 2
    assert entity.originating_candidates[1].location.line_number == 3


# -----------------------------------------------------------------------------
# 4. Strict 1-Hop Boundary (No Recursive Explosion)
# -----------------------------------------------------------------------------

def test_no_recursive_traversal(impact_test_graph: nx.DiGraph) -> None:
    """Verify that traversal does not recursively follow downstream dependencies."""
    plan_md = "1. Update `UserService.get_user()`."
    analysis = analyze_impact(plan_md, impact_test_graph)

    entity = analysis.impacted_entities[0]
    target_ids = {r.target_node_id for r in entity.relationships}

    # 1-hop dependency: Database.query() is included
    assert "db_database_query" in target_ids

    # 2-hop dependency: Pool.acquire() must NOT be included
    assert "db_pool_acquire" not in target_ids

    # 3-hop dependency: Socket.connect() must NOT be included
    assert "socket_connect" not in target_ids


# -----------------------------------------------------------------------------
# 5. Handling of Ambiguous, Unresolved, and NOT_CODE_ENTITY Candidates
# -----------------------------------------------------------------------------

def test_graceful_handling_of_all_candidate_statuses(impact_test_graph: nx.DiGraph) -> None:
    """Test that ambiguous, unresolved, and non-code candidates are preserved without failure."""
    plan_md = """# Mixed Plan
1. Update `UserService`.
2. Inspect `AmbiguousHelper`.
3. Add `NonExistentService`.
4. Configure Redis caching.
"""
    analysis = analyze_impact(plan_md, impact_test_graph)

    # Resolved
    assert len(analysis.impacted_entities) == 1
    assert analysis.impacted_entities[0].label == "UserService"

    # Ambiguous: AmbiguousHelper
    assert len(analysis.ambiguous_entities) == 1
    assert analysis.ambiguous_entities[0].candidate.text == "AmbiguousHelper"
    assert len(analysis.ambiguous_entities[0].matched_nodes) == 2

    # Unresolved: NonExistentService
    assert len(analysis.unresolved_entities) == 1
    assert analysis.unresolved_entities[0].candidate.text == "NonExistentService"

    # Not Code Entities: Redis, caching
    assert len(analysis.not_code_entities) >= 1
    nc_texts = {nc.candidate.text for nc in analysis.not_code_entities}
    assert "Redis" in nc_texts


# -----------------------------------------------------------------------------
# 6. Impact Summary Counts
# -----------------------------------------------------------------------------

def test_impact_summary_metrics(impact_test_graph: nx.DiGraph) -> None:
    """Test that the summary accurately tallies candidates, entities, and relationship types."""
    plan_md = """## Steps
1. Call `UserService.get_user()`.
2. Check `AmbiguousHelper`.
3. Add `MissingComponent`.
4. Use Redis.
"""
    analysis = analyze_impact(plan_md, impact_test_graph)
    summary = analysis.summary

    assert isinstance(summary, ImpactSummary)
    assert summary.total_candidates == 4
    assert summary.resolved_candidates == 1
    assert summary.ambiguous_candidates == 1
    assert summary.unresolved_candidates == 1
    assert summary.not_code_candidates == 1

    # Unique impacted entities connected to get_user(): query(), UserController.show(), UserService
    assert summary.unique_impacted_entities == 3

    # Relationship counts by type
    assert summary.relationships_by_type.get("calls") == 1
    assert summary.relationships_by_type.get("called_by") == 1
    assert summary.relationships_by_type.get("defined_in") == 1


# -----------------------------------------------------------------------------
# 7. Deterministic Ordering
# -----------------------------------------------------------------------------

def test_deterministic_ordering(impact_test_graph: nx.DiGraph) -> None:
    """Test that running analyze_impact repeatedly produces identical ordering."""
    plan_md = """## Changes
1. Modify `UserService`.
2. Call `UserService.get_user()`.
3. Import `services/user.py`.
"""
    res1 = analyze_impact(plan_md, impact_test_graph)
    res2 = analyze_impact(plan_md, impact_test_graph)

    assert len(res1.impacted_entities) == len(res2.impacted_entities)
    for e1, e2 in zip(res1.impacted_entities, res2.impacted_entities):
        assert e1.node_id == e2.node_id
        assert [r.target_node_id for r in e1.relationships] == [r.target_node_id for r in e2.relationships]
        assert [r.display_relation for r in e1.relationships] == [r.display_relation for r in e2.relationships]


# -----------------------------------------------------------------------------
# 8. JSON Serialization
# -----------------------------------------------------------------------------

def test_json_serialization(impact_test_graph: nx.DiGraph) -> None:
    """Test that ImpactAnalysis serializes to valid, complete JSON."""
    plan_md = "# Title\n1. Modify `UserService.get_user()`.\n2. Add `Missing`.\n"
    analysis = analyze_impact(plan_md, impact_test_graph)
    payload = analysis.to_dict()

    assert payload["plan_title"] == "Title"
    assert "summary" in payload
    assert payload["summary"]["resolved_candidates"] == 1
    assert payload["summary"]["unresolved_candidates"] == 1
    assert len(payload["impacted_entities"]) == 1
    assert len(payload["unresolved"]) == 1

    # Ensure json.dumps succeeds
    json_str = json.dumps(payload)
    assert "UserService.get_user()" in json_str


# -----------------------------------------------------------------------------
# 9. CLI Integration Tests
# -----------------------------------------------------------------------------

def test_cli_analyze_plan_human_readable(
    tmp_path: Path, impact_test_graph: nx.DiGraph, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI analyze-plan human-readable output."""
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# Feature Plan\n\n1. Call `UserService.get_user()`.\n2. Add `Missing`.\n", encoding="utf-8")

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [{"id": nid, **data} for nid, data in impact_test_graph.nodes(data=True)],
        "links": [{"source": u, "target": v, **data} for u, v, data in impact_test_graph.edges(data=True)],
    }
    graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

    exit_code = main(["analyze-plan", str(plan_file), "--graph", str(graph_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "PlanGraph Impact Analysis" in captured.out
    assert "Candidates" in captured.out
    assert "Resolved:     1" in captured.out
    assert "Unresolved:   1" in captured.out
    assert "Impacted entities: 3" in captured.out
    assert "UserService.get_user()" in captured.out
    assert "calls → query()" in captured.out
    assert "called_by → UserController.show()" in captured.out
    assert "Unresolved" in captured.out
    assert "Missing" in captured.out


def test_cli_impact_plan_alias_and_json(
    tmp_path: Path, impact_test_graph: nx.DiGraph, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI impact-plan alias with --json output."""
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("1. Call `UserService.get_user()`.\n", encoding="utf-8")

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [{"id": nid, **data} for nid, data in impact_test_graph.nodes(data=True)],
        "links": [{"source": u, "target": v, **data} for u, v, data in impact_test_graph.edges(data=True)],
    }
    graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

    exit_code = main(["impact-plan", str(plan_file), "-g", str(graph_file), "--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["summary"]["resolved_candidates"] == 1
    assert payload["summary"]["unique_impacted_entities"] == 3
    assert len(payload["impacted_entities"]) == 1
    assert payload["impacted_entities"][0]["label"] == "UserService.get_user()"


# -----------------------------------------------------------------------------
# 10. Integration Test with Real Graphify Graph
# -----------------------------------------------------------------------------

def test_integration_real_graphify_graph() -> None:
    """Integration test against the real Graphify graph if available."""
    graph_path = Path("../graphify/graphify-out/graph.json")
    if not graph_path.exists():
        pytest.skip("Real Graphify graph.json not found")

    plan_md = """# Analysis Plan
1. Update `Analyzer.process()`.
2. Inspect `graphify/serve.py`.
3. Add `NonExistentFeature`.
4. Configure Redis.
"""
    analysis = analyze_impact(plan_md, graph_path)
    assert analysis.summary.resolved_candidates >= 2
    assert analysis.summary.unresolved_candidates >= 1
    assert analysis.summary.not_code_candidates >= 1
    assert analysis.summary.unique_impacted_entities > 0

    # Verify that Analyzer.process() has direct relationships
    analyzer_entity = next((e for e in analysis.impacted_entities if "process" in e.label), None)
    assert analyzer_entity is not None
    assert len(analyzer_entity.relationships) > 0


# -----------------------------------------------------------------------------
# 11. Refinement 1: Method Display Name Qualification Tests
# -----------------------------------------------------------------------------

def test_method_display_name_qualification(impact_test_graph: nx.DiGraph) -> None:
    """When a method has an enclosing class, its display label is Class.method()."""
    plan_md = "1. Call `UserService.get_user()`."
    analysis = analyze_impact(plan_md, impact_test_graph)
    assert len(analysis.impacted_entities) == 1
    entity = analysis.impacted_entities[0]
    assert entity.label == "UserService.get_user()"

    rel_labels = [r.target_label for r in entity.relationships]
    assert "UserController.show()" in rel_labels


def test_standalone_function_display_name_preserved(impact_test_graph: nx.DiGraph) -> None:
    """Standalone functions preserve their existing label without qualification."""
    plan_md = "1. Call `query()`."
    analysis = analyze_impact(plan_md, impact_test_graph)
    assert len(analysis.impacted_entities) == 1
    entity = analysis.impacted_entities[0]
    assert entity.label == "query()"


# -----------------------------------------------------------------------------
# 12. Refinement 3A: External Dependency Classification and Filtering Tests
# -----------------------------------------------------------------------------

def test_external_dependency_classification() -> None:
    """Verify is_external classification for stdlib/external vs internal repo entities."""
    g = nx.DiGraph()
    # Internal function
    g.add_node("svc_fn", label="do_work()", source_file="src/service.py", file_type="code")
    # External node (stdlib: os)
    g.add_node("ext_os", label="os", source_file="", file_type="concept", external=True, type="external")
    # External node (third-party: requests)
    g.add_node("ext_requests", label="requests", source_file="", file_type="concept", external=True)
    # Internal dependency
    g.add_node("util_fn", label="helper()", source_file="src/utils.py", file_type="code")

    g.add_edge("svc_fn", "ext_os", relation="imports")
    g.add_edge("svc_fn", "ext_requests", relation="imports")
    g.add_edge("svc_fn", "util_fn", relation="calls")

    plan_md = "1. Update `do_work()` in `src/service.py`."
    analysis = analyze_impact(plan_md, g)
    assert len(analysis.impacted_entities) == 1
    entity = analysis.impacted_entities[0]

    rels = {r.target_label: r for r in entity.relationships}
    assert rels["os"].is_external is True
    assert rels["requests"].is_external is True
    assert rels["helper()"].is_external is False

    # In JSON, all relationships are preserved with is_external flag
    data = analysis.to_dict()
    ent_json = data["impacted_entities"][0]
    assert len(ent_json["relationships"]) == 3
    json_rels = {r["target_label"]: r["is_external"] for r in ent_json["relationships"]}
    assert json_rels["os"] is True
    assert json_rels["requests"] is True
    assert json_rels["helper()"] is False


def test_dependency_filtering_cli_default_and_all_deps() -> None:
    """CLI default hides external dependencies; --all-deps restores them."""
    g = nx.DiGraph()
    g.add_node("svc_fn", label="do_work()", source_file="src/service.py", file_type="code")
    g.add_node("ext_sys", label="sys", source_file="", file_type="concept", external=True)
    g.add_node("util_fn", label="helper()", source_file="src/utils.py", file_type="code")

    g.add_edge("svc_fn", "ext_sys", relation="imports")
    g.add_edge("svc_fn", "util_fn", relation="calls")

    plan_md = "1. Update `do_work()` in `src/service.py`."
    analysis = analyze_impact(plan_md, g)

    # Default output: hides external (sys)
    report_default = format_impact_report(analysis, all_deps=False)
    assert "calls → helper()" in report_default
    assert "sys" not in report_default

    # With all_deps=True: includes external (sys)
    report_all = format_impact_report(analysis, all_deps=True)
    assert "calls → helper()" in report_all
    assert "imports → sys" in report_all


# -----------------------------------------------------------------------------
# 13. Refinement 3B: Containment List Collapsing Tests
# -----------------------------------------------------------------------------

def test_containment_fewer_than_threshold() -> None:
    """When containment count is fewer than MAX_CONTAINMENT_DISPLAY, show all individually."""
    g = nx.DiGraph()
    g.add_node("file_node", label="mod.py", source_file="src/mod.py", file_type="code", source_location="L1")
    for i in range(5):
        nid = f"fn_{i}"
        g.add_node(nid, label=f"fn_{i}()", source_file="src/mod.py", file_type="code")
        g.add_edge("file_node", nid, relation="contains")

    plan_md = "1. Update `src/mod.py`."
    analysis = analyze_impact(plan_md, g)
    report = format_impact_report(analysis)

    assert "contains → 5 entities" not in report
    for i in range(5):
        assert f"contains → fn_{i}()" in report


def test_containment_exactly_threshold() -> None:
    """When containment count equals MAX_CONTAINMENT_DISPLAY, show all individually."""
    g = nx.DiGraph()
    g.add_node("file_node", label="mod.py", source_file="src/mod.py", file_type="code", source_location="L1")
    for i in range(MAX_CONTAINMENT_DISPLAY):
        nid = f"fn_{i:02d}"
        g.add_node(nid, label=f"fn_{i:02d}()", source_file="src/mod.py", file_type="code")
        g.add_edge("file_node", nid, relation="contains")

    plan_md = "1. Update `src/mod.py`."
    analysis = analyze_impact(plan_md, g)
    report = format_impact_report(analysis)

    assert f"contains → {MAX_CONTAINMENT_DISPLAY} entities" not in report
    for i in range(MAX_CONTAINMENT_DISPLAY):
        assert f"contains → fn_{i:02d}()" in report


def test_containment_greater_than_threshold_and_deterministic_order() -> None:
    """When containment count exceeds threshold, collapse with count and examples in deterministic order."""
    g = nx.DiGraph()
    g.add_node("file_node", label="mod.py", source_file="src/mod.py", file_type="code", source_location="L1")
    total_fns = 25
    for i in range(total_fns):
        nid = f"fn_{i:02d}"
        g.add_node(nid, label=f"fn_{i:02d}()", source_file="src/mod.py", file_type="code")
        g.add_edge("file_node", nid, relation="contains")

    plan_md = "1. Update `src/mod.py`."
    analysis = analyze_impact(plan_md, g)

    # 1. Report formatting
    report = format_impact_report(analysis)
    assert f"contains → {total_fns} entities" in report
    assert "fn_00()" in report
    assert "fn_01()" in report
    assert "fn_02()" in report
    assert "fn_03()" in report
    assert "fn_04()" in report
    # 5 shown, total 25 -> 20 more
    assert f"... (+ {total_fns - 5} more)" in report
    # 6th should not be printed as an example
    assert "fn_05()" not in report

    # 2. JSON preserves all 25 without any truncation
    data = analysis.to_dict()
    ent_json = data["impacted_entities"][0]
    assert len(ent_json["relationships"]) == 25

    # 3. Determinism: run report formatting twice and verify exact match
    report2 = format_impact_report(analysis)
    assert report == report2


def test_cli_all_deps_flag(tmp_path: Path) -> None:
    """Test CLI analyze-plan with and without --all-deps flag."""
    g = nx.DiGraph()
    g.add_node("svc_fn", label="do_work()", source_file="src/service.py", file_type="code")
    g.add_node("ext_sys", label="sys", source_file="", file_type="concept", external=True)
    g.add_edge("svc_fn", "ext_sys", relation="imports")

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [{"id": nid, **data} for nid, data in g.nodes(data=True)],
        "links": [{"source": u, "target": v, **data} for u, v, data in g.edges(data=True)],
    }
    graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

    plan_file = tmp_path / "plan.md"
    plan_file.write_text("1. Update `do_work()` in `src/service.py`.\n", encoding="utf-8")

    from io import StringIO
    import contextlib

    # Default: external hidden
    stdout = StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(["analyze-plan", str(plan_file), "-g", str(graph_file)])
    assert exit_code == 0
    assert "sys" not in stdout.getvalue()

    # With --all-deps: external visible
    stdout_all = StringIO()
    with contextlib.redirect_stdout(stdout_all):
        exit_code_all = main(["analyze-plan", str(plan_file), "-g", str(graph_file), "--all-deps"])
    assert exit_code_all == 0
    assert "imports → sys" in stdout_all.getvalue()
