"""Unit tests for Phase 5 plan validation layer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from plangraph.impact import ImpactAnalysis, ImpactSummary, ImpactedEntity, analyze_impact
from plangraph.plan import parse_plan
from plangraph.resolver import resolve_plan
from plangraph.validation import (
    DEFAULT_HIGH_IMPACT_THRESHOLD,
    FindingCode,
    ValidationConfig,
    ValidationFinding,
    ValidationResult,
    ValidationSeverity,
    format_validation_report,
    validate_plan,
)


@pytest.fixture
def validation_graph() -> nx.DiGraph:
    """Synthetic graph with clear topologies for testing all validation findings."""
    g = nx.DiGraph()

    # 1. Normal service and method
    g.add_node("svc_user", label="UserService", source_file="services/user.py", source_location="L10", file_type="code")
    g.add_node("meth_get", label=".get_user()", source_file="services/user.py", source_location="L15", file_type="code", _callable=True)
    g.add_edge("svc_user", "meth_get", relation="contains")
    g.add_edge("meth_get", "svc_user", relation="contained_in")

    # 2. High-impact hub node with 12 distinct callers
    g.add_node("hub_node", label="DatabasePool", source_file="db/pool.py", source_location="L5", file_type="code")
    for i in range(12):
        caller_id = f"client_{i}"
        g.add_node(caller_id, label=f"Client{i}", source_file=f"clients/{i}.py", source_location="L1", file_type="code")
        g.add_edge(caller_id, "hub_node", relation="calls")

    # 3. Ambiguous functions (status exists in two files)
    g.add_node("fn_status_1", label="status()", source_file="pkg/a.py", source_location="L20", file_type="code", _callable=True)
    g.add_node("fn_status_2", label="status()", source_file="pkg/b.py", source_location="L30", file_type="code", _callable=True)

    # 4. Non-code concept node
    g.add_node("concept_redis", label="Redis", file_type="concept")

    return g


# -----------------------------------------------------------------------------
# 1. UNRESOLVED_ENTITY
# -----------------------------------------------------------------------------

def test_validation_unresolved_entity(validation_graph: nx.DiGraph) -> None:
    """Referencing an entity not in the codebase emits an ERROR UNRESOLVED_ENTITY finding."""
    plan_md = """# Implementation Plan
## Changes
1. Update `nonexistent_function()`.
"""
    parsed = parse_plan(plan_md)
    res = resolve_plan(parsed, validation_graph)
    result = validate_plan(parsed, resolution=res)

    assert not result.is_valid
    assert result.summary.error_count == 1
    finding = next(f for f in result.findings if f.code == FindingCode.UNRESOLVED_ENTITY)
    assert finding.severity == ValidationSeverity.ERROR
    assert finding.target_entity == "nonexistent_function()"
    assert finding.line_number == 3
    assert "nonexistent_function()" in finding.message
    assert finding.evidence is not None
    assert "no callable function found" in finding.evidence


# -----------------------------------------------------------------------------
# 2. AMBIGUOUS_ENTITY
# -----------------------------------------------------------------------------

def test_validation_ambiguous_entity(validation_graph: nx.DiGraph) -> None:
    """Referencing an ambiguous entity without disambiguation emits an ERROR AMBIGUOUS_ENTITY finding."""
    plan_md = """# Implementation Plan
## Changes
1. Update `status()`.
"""
    parsed = parse_plan(plan_md)
    res = resolve_plan(parsed, validation_graph)
    result = validate_plan(parsed, resolution=res)

    assert not result.is_valid
    assert result.summary.error_count == 1
    finding = next(f for f in result.findings if f.code == FindingCode.AMBIGUOUS_ENTITY)
    assert finding.severity == ValidationSeverity.ERROR
    assert finding.target_entity == "status()"
    assert finding.line_number == 3
    assert "matches 2 entities" in finding.message
    assert "pkg/a.py:L20" in finding.message
    assert "pkg/b.py:L30" in finding.message


# -----------------------------------------------------------------------------
# 3. NO_CODE_ENTITY
# -----------------------------------------------------------------------------

def test_validation_not_code_entity(validation_graph: nx.DiGraph) -> None:
    """Referencing a non-code concept emits an INFO NO_CODE_ENTITY finding without failing validation."""
    plan_md = """# Implementation Plan
## Architecture
1. We will use Redis for session caching.
"""
    parsed = parse_plan(plan_md)
    res = resolve_plan(parsed, validation_graph)
    result = validate_plan(parsed, resolution=res)

    assert result.is_valid  # INFO does not fail validation
    assert result.summary.error_count == 0
    assert result.summary.info_count >= 1
    finding = next(f for f in result.findings if f.code == FindingCode.NO_CODE_ENTITY)
    assert finding.severity == ValidationSeverity.INFO
    assert finding.target_entity in ("Redis", "caching")


# -----------------------------------------------------------------------------
# 4. HIGH_IMPACT_ENTITY & 5. Below Threshold
# -----------------------------------------------------------------------------

def test_validation_high_impact_threshold(validation_graph: nx.DiGraph) -> None:
    """Entities crossing the high-impact threshold emit a WARNING HIGH_IMPACT_ENTITY finding."""
    plan_md = """# Implementation Plan
## Changes
1. Update `DatabasePool` in `db/pool.py`.
"""
    analysis = analyze_impact(plan_md, validation_graph)
    result = validate_plan(parse_plan(plan_md), impact=analysis, config=ValidationConfig(high_impact_threshold=10))

    assert result.is_valid  # Warnings do not block is_valid
    assert result.summary.warning_count >= 1
    high_impact = next(f for f in result.findings if f.code == FindingCode.HIGH_IMPACT_ENTITY)
    assert high_impact.severity == ValidationSeverity.WARNING
    assert high_impact.target_entity == "DatabasePool"
    assert high_impact.metadata["impact_count"] == 12
    assert high_impact.metadata["threshold"] == 10
    assert "12 direct relationships" in high_impact.message


def test_validation_below_high_impact_threshold(validation_graph: nx.DiGraph) -> None:
    """Entities below the high-impact threshold do not emit a high impact finding."""
    plan_md = """# Implementation Plan
## Changes
1. Update `UserService` in `services/user.py`.
"""
    # UserService has only 1-2 relationships in validation_graph
    analysis = analyze_impact(plan_md, validation_graph)
    result = validate_plan(parse_plan(plan_md), impact=analysis, config=ValidationConfig(high_impact_threshold=10))

    assert not any(f.code == FindingCode.HIGH_IMPACT_ENTITY for f in result.findings)


# -----------------------------------------------------------------------------
# 6. Actionable Step Without Resolved Entity
# -----------------------------------------------------------------------------

def test_validation_actionable_step_without_resolved_entity(validation_graph: nx.DiGraph) -> None:
    """An implementation step using action verbs without resolved code entities emits STEP_WITHOUT_RESOLVED_ENTITY."""
    plan_md = """# Implementation Plan
## Changes
1. Update `services/user.py`.
2. Refactor database connection retry logic.
"""
    parsed = parse_plan(plan_md)
    res = resolve_plan(parsed, validation_graph)
    result = validate_plan(parsed, resolution=res)

    step_finding = next((f for f in result.findings if f.code == FindingCode.STEP_WITHOUT_RESOLVED_ENTITY), None)
    assert step_finding is not None
    assert step_finding.severity == ValidationSeverity.WARNING
    assert step_finding.step_number == 2
    assert "Refactor database connection retry logic" in step_finding.message


# -----------------------------------------------------------------------------
# 7. Rationale / Concept / Setup Steps Do Not Create False Positives
# -----------------------------------------------------------------------------

def test_validation_non_implementation_steps_no_false_positive(validation_graph: nx.DiGraph) -> None:
    """Explanatory, setup, architecture, and concept steps do not emit STEP_WITHOUT_RESOLVED_ENTITY."""
    plan_md = """# Implementation Plan

## Background
1. Note: This refactor improves maintainability.
2. Architecture discussion: Event-driven pub/sub model is selected.

## Setup
1. Setup local environment using docker compose.

## Implementation
1. Update `services/user.py`.
2. Note that backward compatibility is preserved.
3. Use Redis for fast lookup.
"""
    parsed = parse_plan(plan_md)
    res = resolve_plan(parsed, validation_graph)
    result = validate_plan(parsed, resolution=res)

    # None of the steps should trigger STEP_WITHOUT_RESOLVED_ENTITY
    step_findings = [f for f in result.findings if f.code == FindingCode.STEP_WITHOUT_RESOLVED_ENTITY]
    assert len(step_findings) == 0


# -----------------------------------------------------------------------------
# 8. Empty or Low-Evidence Plan
# -----------------------------------------------------------------------------

def test_validation_empty_or_low_evidence_plan(validation_graph: nx.DiGraph) -> None:
    """Plans containing zero actionable code references emit LOW_PLAN_EVIDENCE."""
    plan_md = """# Implementation Plan
We plan to make general architectural improvements to the codebase.
"""
    parsed = parse_plan(plan_md)
    res = resolve_plan(parsed, validation_graph)
    result = validate_plan(parsed, resolution=res)

    assert any(f.code == FindingCode.LOW_PLAN_EVIDENCE for f in result.findings)
    low_ev = next(f for f in result.findings if f.code == FindingCode.LOW_PLAN_EVIDENCE)
    assert low_ev.severity == ValidationSeverity.WARNING


# -----------------------------------------------------------------------------
# 9. Deduplication of Findings
# -----------------------------------------------------------------------------

def test_validation_deduplication(validation_graph: nx.DiGraph) -> None:
    """Identical conditions on the same line/step do not produce duplicate findings."""
    plan_md = """# Implementation Plan
## Changes
1. Update `nonexistent()` and verify `nonexistent()`.
"""
    parsed = parse_plan(plan_md)
    res = resolve_plan(parsed, validation_graph)
    result = validate_plan(parsed, resolution=res)

    unresolved_findings = [
        f for f in result.findings
        if f.code == FindingCode.UNRESOLVED_ENTITY and f.target_entity == "nonexistent()"
    ]
    assert len(unresolved_findings) == 1


# -----------------------------------------------------------------------------
# 10. Deterministic Finding Ordering
# -----------------------------------------------------------------------------

def test_validation_deterministic_ordering(validation_graph: nx.DiGraph) -> None:
    """Findings are deterministically ordered with errors first, then warnings, then info, sorted by line."""
    plan_md = """# Implementation Plan
## Changes
1. We use Redis for caching.
2. Update `nonexistent_function()`.
3. Update `DatabasePool` in `db/pool.py`.
4. Refactor general helper logic.
"""
    parsed = parse_plan(plan_md)
    analysis = analyze_impact(parsed, validation_graph)
    res1 = validate_plan(parsed, impact=analysis)
    res2 = validate_plan(parsed, impact=analysis)

    # Identical across multiple runs
    assert [f.to_dict() for f in res1.findings] == [f.to_dict() for f in res2.findings]

    # Verify severity ordering: ERRORs come before WARNINGs, and WARNINGs come before INFOs
    seen_severities = [f.severity for f in res1.findings]
    sev_values = [
        0 if s == ValidationSeverity.ERROR else (1 if s == ValidationSeverity.WARNING else 2)
        for s in seen_severities
    ]
    assert sev_values == sorted(sev_values)


# -----------------------------------------------------------------------------
# 11. JSON Serialization
# -----------------------------------------------------------------------------

def test_validation_json_serialization(validation_graph: nx.DiGraph) -> None:
    """ValidationResult and ImpactAnalysis with validation cleanly serialize to JSON."""
    plan_md = """# Implementation Plan
## Changes
1. Update `DatabasePool` in `db/pool.py`.
2. Update `nonexistent()`.
"""
    analysis = analyze_impact(plan_md, validation_graph)
    d = analysis.to_dict()

    assert "validation" in d
    val_data = d["validation"]
    assert "is_valid" in val_data
    assert "summary" in val_data
    assert "findings" in val_data
    assert val_data["is_valid"] is False
    assert val_data["summary"]["error_count"] >= 1

    # Roundtrip JSON dumps
    dumped = json.dumps(d)
    reloaded = json.loads(dumped)
    assert reloaded["validation"]["summary"]["error_count"] == val_data["summary"]["error_count"]


# -----------------------------------------------------------------------------
# 12. CLI analyze-plan includes Validation Section
# -----------------------------------------------------------------------------

def test_cli_analyze_plan_human_readable_validation(tmp_path: Path, validation_graph: nx.DiGraph) -> None:
    """CLI analyze-plan human-readable output includes the Validation findings section."""
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# Plan\n\n1. Update `nonexistent_func()`.\n", encoding="utf-8")

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [{"id": nid, **data} for nid, data in validation_graph.nodes(data=True)],
        "links": [{"source": u, "target": v, **data} for u, v, data in validation_graph.edges(data=True)],
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
    assert "Validation" in out
    assert "ERROR" in out
    assert "UNRESOLVED_ENTITY" in out
    assert "nonexistent_func()" in out


# -----------------------------------------------------------------------------
# 13. Clean Plan Validation Success Output
# -----------------------------------------------------------------------------

def test_cli_analyze_plan_clean_validation(tmp_path: Path, validation_graph: nx.DiGraph) -> None:
    """When a plan has no validation issues, the validation section reports clean status."""
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# Plan\n\n1. Update `services/user.py`.\n", encoding="utf-8")

    graph_file = tmp_path / "graph.json"
    graph_data = {
        "directed": True,
        "nodes": [{"id": nid, **data} for nid, data in validation_graph.nodes(data=True)],
        "links": [{"source": u, "target": v, **data} for u, v, data in validation_graph.edges(data=True)],
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
    assert "Validation" in out
    assert "No validation issues detected" in out
