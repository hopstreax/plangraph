"""Unit tests for Phase 6 codebase-grounded plan review layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from plangraph.impact import ImpactAnalysis, analyze_impact, format_impact_report
from plangraph.plan import parse_plan
from plangraph.resolver import resolve_plan
from plangraph.review import (
    ClassifiedFile,
    FileRole,
    PlanReview,
    ReviewItem,
    ReviewSummary,
    StepReview,
    format_review_report,
    is_actionable_step,
    is_test_path,
    review_plan,
)
from plangraph.validation import (
    FindingCode,
    ValidationConfig,
    validate_plan,
)


@pytest.fixture
def review_test_graph() -> nx.DiGraph:
    """Synthetic code graph topology for testing all plan review dimensions."""
    g = nx.DiGraph()

    # 1. Main service and method in services/user.py
    g.add_node("svc_user", label="UserService", source_file="services/user.py", source_location="L10", file_type="code", _callable_class=True)
    g.add_node("meth_get", label=".get_user()", source_file="services/user.py", source_location="L15", file_type="code", _callable=True)
    g.add_edge("svc_user", "meth_get", relation="method")
    g.add_edge("meth_get", "svc_user", relation="contained_in")

    # 2. Downstream database connection in db/database.py
    g.add_node("db_query", label="query()", source_file="db/database.py", source_location="L25", file_type="code", _callable=True)
    g.add_edge("meth_get", "db_query", relation="calls")
    g.add_edge("db_query", "meth_get", relation="called_by")

    # 3. Upstream controller in controllers/user.py
    g.add_node("ctrl_show", label="UserController.show()", source_file="controllers/user.py", source_location="L35", file_type="code", _callable=True)
    g.add_edge("ctrl_show", "meth_get", relation="calls")
    g.add_edge("meth_get", "ctrl_show", relation="called_by")

    # 4. Test file and test caller in tests/test_user.py
    g.add_node("file_test_user", label="tests/test_user.py", source_file="tests/test_user.py", source_location="L1", file_type="code")
    g.add_node("test_fn", label="test_get_user()", source_file="tests/test_user.py", source_location="L12", file_type="code", _callable=True)
    g.add_edge("file_test_user", "test_fn", relation="contains")
    g.add_edge("test_fn", "meth_get", relation="calls")
    g.add_edge("meth_get", "test_fn", relation="called_by")

    # 5. Ambiguous symbol in two different files
    g.add_node("fn_status_a", label="status()", source_file="pkg/alpha.py", source_location="L20", file_type="code", _callable=True)
    g.add_node("fn_status_b", label="status()", source_file="pkg/beta.py", source_location="L30", file_type="code", _callable=True)

    return g


# -----------------------------------------------------------------------------
# 1. Clean resolved plan produces a review & 2. Actionable step review contains resolved entities
# -----------------------------------------------------------------------------

def test_clean_resolved_plan_review(review_test_graph: nx.DiGraph) -> None:
    """A clean plan produces a structured PlanReview with grounded actionable steps."""
    plan_md = """# Implementation Plan
## Changes
1. Update `UserService.get_user()` in `services/user.py`.
2. Add regression tests in `tests/test_user.py`.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    rev = analysis.review

    assert rev.plan_title == "Implementation Plan"
    assert rev.summary.total_steps == 2
    assert rev.summary.actionable_steps == 2
    assert rev.summary.grounded_steps == 2
    assert rev.summary.resolved_entities >= 2
    assert rev.summary.ambiguous_entities == 0
    assert rev.summary.unresolved_entities == 0

    # Step reviews
    assert len(rev.step_reviews) == 2
    s1 = rev.step_reviews[0]
    assert s1.step_number == 1
    assert s1.is_actionable is True
    assert s1.has_grounding is True
    assert any("services/user.py" in r for r in s1.resolved_entities)

    s2 = rev.step_reviews[1]
    assert s2.step_number == 2
    assert s2.is_actionable is True
    assert s2.has_grounding is True
    assert any("tests/test_user.py" in r for r in s2.resolved_entities)


# -----------------------------------------------------------------------------
# 3. Ambiguous entity appears in the appropriate step review
# -----------------------------------------------------------------------------

def test_ambiguous_entity_in_step_review(review_test_graph: nx.DiGraph) -> None:
    """Ambiguous candidate references are linked to the specific step review."""
    plan_md = """# Implementation Plan
## Changes
1. Update `services/user.py`.
2. Modify `status()`.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    rev = analysis.review

    s2 = next(s for s in rev.step_reviews if s.step_number == 2)
    assert s2.has_grounding is False
    assert "status()" in s2.ambiguous_entities
    assert any(f.code == FindingCode.AMBIGUOUS_ENTITY for f in s2.findings)


# -----------------------------------------------------------------------------
# 4. Unresolved entity appears in the appropriate step review
# -----------------------------------------------------------------------------

def test_unresolved_entity_in_step_review(review_test_graph: nx.DiGraph) -> None:
    """Unresolved candidate references are linked to the specific step review."""
    plan_md = """# Implementation Plan
## Changes
1. Update `nonexistent_service()`.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    rev = analysis.review

    s1 = next(s for s in rev.step_reviews if s.step_number == 1)
    assert s1.has_grounding is False
    assert "nonexistent_service()" in s1.unresolved_entities
    assert any(f.code == FindingCode.UNRESOLVED_ENTITY for f in s1.findings)


# -----------------------------------------------------------------------------
# 5. Explicitly referenced files vs 6. Impacted files
# -----------------------------------------------------------------------------

def test_file_classification_explicit_vs_impacted(review_test_graph: nx.DiGraph) -> None:
    """Review cleanly distinguishes explicitly referenced files from 1-hop impacted files."""
    plan_md = """# Implementation Plan
## Changes
1. Update `UserService.get_user()` in `services/user.py`.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    rev = analysis.review

    # Explicit: services/user.py
    assert "services/user.py" in rev.explicit_files
    assert "db/database.py" not in rev.explicit_files

    # Impacted: db/database.py and controllers/user.py
    assert "db/database.py" in rev.impacted_files
    assert "controllers/user.py" in rev.impacted_files
    assert "services/user.py" not in rev.impacted_files


# -----------------------------------------------------------------------------
# 7. Test files identified from existing graph evidence
# -----------------------------------------------------------------------------

def test_affected_tests_identification(review_test_graph: nx.DiGraph) -> None:
    """Test files touched via direct callers/callees are surfaced in affected_tests."""
    plan_md = """# Implementation Plan
## Changes
1. Update `UserService.get_user()` in `services/user.py`.
"""
    # tests/test_user.py calls UserService.get_user() in the graph
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    rev = analysis.review

    assert "tests/test_user.py" in rev.affected_tests
    assert rev.summary.affected_tests_count >= 1


# -----------------------------------------------------------------------------
# 8. Validation findings surfaced in review items
# -----------------------------------------------------------------------------

def test_validation_findings_surfaced_in_review(review_test_graph: nx.DiGraph) -> None:
    """Validation findings from Phase 5 are translated into structured ReviewItems."""
    plan_md = """# Implementation Plan
## Changes
1. Update `nonexistent_symbol()`.
2. Refactor database connection retry logic.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    rev = analysis.review

    assert len(rev.review_items) >= 2
    item_codes = [item.code for item in rev.review_items]
    assert FindingCode.UNRESOLVED_ENTITY.value in item_codes
    assert FindingCode.STEP_WITHOUT_RESOLVED_ENTITY.value in item_codes


# -----------------------------------------------------------------------------
# 9. Impact relationships remain one-hop
# -----------------------------------------------------------------------------

def test_impact_relationships_remain_one_hop(review_test_graph: nx.DiGraph) -> None:
    """Review layer does not expand graph traversal beyond 1-hop direct relationships."""
    # Add a 2-hop node to graph (db/database.py -> driver/raw.py)
    g = review_test_graph.copy()
    g.add_node("raw_driver", label="raw_socket", source_file="driver/raw.py", source_location="L1", file_type="code")
    g.add_edge("db_query", "raw_driver", relation="calls")

    plan_md = """# Implementation Plan
## Changes
1. Update `UserService.get_user()` in `services/user.py`.
"""
    analysis = analyze_impact(plan_md, g)
    assert analysis.review is not None
    rev = analysis.review

    # db/database.py is 1-hop, so it is in impacted_files
    assert "db/database.py" in rev.impacted_files
    # driver/raw.py is 2-hop, so it must NOT be in impacted_files
    assert "driver/raw.py" not in rev.impacted_files


# -----------------------------------------------------------------------------
# 10. No duplicate files & 11. Deterministic file ordering
# -----------------------------------------------------------------------------

def test_deterministic_file_ordering_and_no_duplicates(review_test_graph: nx.DiGraph) -> None:
    """Files are deduplicated and deterministically sorted alphabetically."""
    plan_md = """# Implementation Plan
## Changes
1. Update `UserService` in `services/user.py`.
2. Modify `UserService.get_user()` in `services/user.py`.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    rev = analysis.review

    # No duplicates in lists
    assert len(rev.explicit_files) == len(set(rev.explicit_files))
    assert len(rev.impacted_files) == len(set(rev.impacted_files))
    assert len(rev.affected_tests) == len(set(rev.affected_tests))

    # Deterministic alphabetical ordering
    assert rev.explicit_files == tuple(sorted(rev.explicit_files))
    assert rev.impacted_files == tuple(sorted(rev.impacted_files))
    assert rev.affected_tests == tuple(sorted(rev.affected_tests))


# -----------------------------------------------------------------------------
# 12. Deterministic step ordering & 13. Deterministic JSON output
# -----------------------------------------------------------------------------

def test_deterministic_review_and_json_output(review_test_graph: nx.DiGraph) -> None:
    """Repeated review runs produce identical step ordering and identical JSON serialization."""
    plan_md = """# Implementation Plan
## Changes
1. Update `services/user.py`.
2. Modify `UserService.get_user()`.
3. Add tests in `tests/test_user.py`.
"""
    a1 = analyze_impact(plan_md, review_test_graph)
    a2 = analyze_impact(plan_md, review_test_graph)

    j1 = json.dumps(a1.to_dict(), indent=2, sort_keys=True)
    j2 = json.dumps(a2.to_dict(), indent=2, sort_keys=True)

    assert j1 == j2
    assert [s.step_number for s in a1.review.step_reviews] == [1, 2, 3]


# -----------------------------------------------------------------------------
# 14. Review JSON serialization
# -----------------------------------------------------------------------------

def test_review_json_serialization(review_test_graph: nx.DiGraph) -> None:
    """PlanReview.to_dict() is fully serializable and preserves all structured sections."""
    plan_md = """# Implementation Plan
## Changes
1. Update `UserService.get_user()` in `services/user.py`.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    d = analysis.to_dict()

    assert "review" in d
    rev_dict = d["review"]
    assert "summary" in rev_dict
    assert "explicit_files" in rev_dict
    assert "impacted_files" in rev_dict
    assert "affected_tests" in rev_dict
    assert "step_reviews" in rev_dict
    assert "review_items" in rev_dict

    # Valid round-trip
    dumped = json.dumps(rev_dict)
    reloaded = json.loads(dumped)
    assert reloaded["summary"]["total_steps"] == rev_dict["summary"]["total_steps"]


# -----------------------------------------------------------------------------
# 15. Clean plan produces understandable human-readable review report
# -----------------------------------------------------------------------------

def test_clean_plan_human_review_report(review_test_graph: nx.DiGraph) -> None:
    """format_review_report formats a factual terminal summary with coverage, files, and steps."""
    plan_md = """# Implementation Plan
## Changes
1. Update `UserService.get_user()` in `services/user.py`.
2. Add tests in `tests/test_user.py`.
"""
    analysis = analyze_impact(plan_md, review_test_graph)
    assert analysis.review is not None
    out = format_review_report(analysis.review)

    assert "Plan Review" in out
    assert "Coverage" in out
    assert "Steps: 2" in out
    assert "Actionable: 2" in out
    assert "Grounded: 2" in out
    assert "Explicit files" in out
    assert "services/user.py" in out
    assert "Impacted files" in out
    assert "db/database.py" in out
    assert "Affected tests" in out
    assert "tests/test_user.py" in out
    assert "Step review" in out
    assert "Review items" in out


# -----------------------------------------------------------------------------
# 16. CLI analyze-plan includes Plan Review in output
# -----------------------------------------------------------------------------

def test_cli_analyze_plan_includes_plan_review(tmp_path: Path, review_test_graph: nx.DiGraph) -> None:
    """CLI analyze-plan output contains the Plan Review section."""
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# Plan\n\n1. Update `services/user.py`.\n", encoding="utf-8")

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [{"id": nid, **data} for nid, data in review_test_graph.nodes(data=True)],
        "links": [{"source": u, "target": v, **data} for u, v, data in review_test_graph.edges(data=True)],
    }
    graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

    from io import StringIO
    import contextlib
    from plangraph.__main__ import main

    stdout = StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(["analyze-plan", str(plan_file), "-g", str(graph_file)])

    assert exit_code == 0
    out = stdout.getvalue()
    assert "Plan Review" in out
    assert "Coverage" in out
    assert "Explicit files" in out
    assert "services/user.py" in out


def test_cli_analyze_plan_review_json_strict_utf8(tmp_path: Path, review_test_graph: nx.DiGraph) -> None:
    """Test that CLI --json output contains review and is strictly decodable as UTF-8."""
    import os
    import subprocess
    import sys

    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# Plan\n\n1. Update `services/user.py`.\n", encoding="utf-8")

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [{"id": nid, **data} for nid, data in review_test_graph.nodes(data=True)],
        "links": [{"source": u, "target": v, **data} for u, v, data in review_test_graph.edges(data=True)],
    }
    graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

    out_file = tmp_path / "output.json"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    with open(out_file, "wb") as out_fp:
        subprocess.run(
            [sys.executable, "-m", "plangraph", "analyze-plan", str(plan_file), "-g", str(graph_file), "--json"],
            stdout=out_fp,
            stderr=subprocess.PIPE,
            env=env,
            check=True,
        )

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "review" in data
    assert "summary" in data["review"]
    assert "explicit_files" in data["review"]
    assert "services/user.py" in data["review"]["explicit_files"]
