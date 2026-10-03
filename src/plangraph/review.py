"""Codebase-grounded plan review layer for PlanGraph.

Composes parsing, resolution, direct impact analysis, and validation evidence
into a structured, step-level review that answers what the plan currently covers,
what the codebase evidence reveals, and what parts need clarification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plangraph.impact import ImpactAnalysis, ImpactedEntity, ImpactRelationship

from plangraph.plan import CandidateEntity, CandidateKind, ParsedPlan, PlanStep
from plangraph.resolver import ResolutionResult, ResolutionStatus, ResolvedEntity
from plangraph.validation import (
    FindingCode,
    ValidationFinding,
    ValidationResult,
    ValidationSeverity,
)


# -----------------------------------------------------------------------------
# File Role Classification
# -----------------------------------------------------------------------------

class FileRole(str, Enum):
    """Role classification of a file touched or referenced by the plan."""

    EXPLICIT = "explicit"
    IMPACTED = "impacted"
    TEST = "test"


@dataclass(frozen=True)
class ClassifiedFile:
    """A unique source file associated with the plan, classified by its evidence roles."""

    file_path: str
    roles: tuple[FileRole, ...]
    reference_count: int = 1

    @property
    def is_explicit(self) -> bool:
        return FileRole.EXPLICIT in self.roles

    @property
    def is_impacted(self) -> bool:
        return FileRole.IMPACTED in self.roles

    @property
    def is_test(self) -> bool:
        return FileRole.TEST in self.roles

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "roles": [r.value for r in self.roles],
            "is_explicit": self.is_explicit,
            "is_impacted": self.is_impacted,
            "is_test": self.is_test,
            "reference_count": self.reference_count,
        }


def is_test_path(file_path: str) -> bool:
    """Determine if a file path represents test code based on path conventions."""
    norm = file_path.replace("\\", "/").lower()
    parts = norm.split("/")
    if any(p in ("tests", "test") for p in parts):
        return True
    filename = parts[-1] if parts else ""
    return (
        filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith("_test.go")
        or filename.endswith(".test.ts")
        or filename.endswith(".test.js")
        or filename.endswith(".spec.ts")
        or filename.endswith(".spec.js")
    )


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewItem:
    """A concrete plan review item derived from validation findings."""

    code: str
    severity: str
    title: str
    description: str
    step_number: int | None = None
    line_number: int | None = None
    target_entity: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "step_number": self.step_number,
            "line_number": self.line_number,
            "target_entity": self.target_entity,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class StepReview:
    """Step-level traceability review grounded in codebase evidence."""

    step_number: int | None
    step_index: int
    section: str
    line_number: int
    text: str
    is_actionable: bool
    has_grounding: bool
    referenced_entities: tuple[str, ...] = field(default_factory=tuple)
    resolved_entities: tuple[str, ...] = field(default_factory=tuple)
    unresolved_entities: tuple[str, ...] = field(default_factory=tuple)
    ambiguous_entities: tuple[str, ...] = field(default_factory=tuple)
    impacted_entities: tuple[str, ...] = field(default_factory=tuple)
    affected_files: tuple[str, ...] = field(default_factory=tuple)
    direct_impact_count: int = 0
    findings: tuple[ValidationFinding, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_index": self.step_index,
            "section": self.section,
            "line_number": self.line_number,
            "text": self.text,
            "is_actionable": self.is_actionable,
            "has_grounding": self.has_grounding,
            "referenced_entities": list(self.referenced_entities),
            "resolved_entities": list(self.resolved_entities),
            "unresolved_entities": list(self.unresolved_entities),
            "ambiguous_entities": list(self.ambiguous_entities),
            "impacted_entities": list(self.impacted_entities),
            "affected_files": list(self.affected_files),
            "direct_impact_count": self.direct_impact_count,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass(frozen=True)
class ReviewSummary:
    """Deterministic summary metrics for the plan review."""

    total_steps: int
    actionable_steps: int
    grounded_steps: int
    resolved_entities: int
    ambiguous_entities: int
    unresolved_entities: int
    not_code_concepts: int
    explicit_files_count: int
    impacted_files_count: int
    affected_tests_count: int
    total_direct_relationships: int
    validation_errors: int
    validation_warnings: int
    validation_infos: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "actionable_steps": self.actionable_steps,
            "grounded_steps": self.grounded_steps,
            "resolved_entities": self.resolved_entities,
            "ambiguous_entities": self.ambiguous_entities,
            "unresolved_entities": self.unresolved_entities,
            "not_code_concepts": self.not_code_concepts,
            "explicit_files_count": self.explicit_files_count,
            "impacted_files_count": self.impacted_files_count,
            "affected_tests_count": self.affected_tests_count,
            "total_direct_relationships": self.total_direct_relationships,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "validation_infos": self.validation_infos,
        }


@dataclass(frozen=True)
class PlanReview:
    """The aggregate structured review of a contributor implementation plan."""

    plan_title: str
    plan_file: str | None
    graph_file: str | None
    summary: ReviewSummary
    step_reviews: tuple[StepReview, ...] = field(default_factory=tuple)
    explicit_files: tuple[str, ...] = field(default_factory=tuple)
    impacted_files: tuple[str, ...] = field(default_factory=tuple)
    affected_tests: tuple[str, ...] = field(default_factory=tuple)
    classified_files: tuple[ClassifiedFile, ...] = field(default_factory=tuple)
    review_items: tuple[ReviewItem, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_title": self.plan_title,
            "plan_file": self.plan_file,
            "graph_file": self.graph_file,
            "summary": self.summary.to_dict(),
            "explicit_files": list(self.explicit_files),
            "impacted_files": list(self.impacted_files),
            "affected_tests": list(self.affected_tests),
            "classified_files": [f.to_dict() for f in self.classified_files],
            "step_reviews": [s.to_dict() for s in self.step_reviews],
            "review_items": [r.to_dict() for r in self.review_items],
        }


# -----------------------------------------------------------------------------
# Actionable Step Detection
# -----------------------------------------------------------------------------

_RATIONALE_PREFIX_RE = re.compile(
    r"^(?:note(?:\s+that)?|rationale|explanation|why|because|context|background|purpose|motivation|tip|warning|caution)\b",
    re.IGNORECASE,
)

_SETUP_PREFIX_RE = re.compile(
    r"^(?:setup|set\s+up|install|prerequisites?|environment|prepare|run\s+(?:migrations?|tests?|server|build|script|docker|uv|npm|pip|compose))\b",
    re.IGNORECASE,
)

_ARCHITECTURE_PREFIX_RE = re.compile(
    r"^(?:architecture|design|discussion|trade[\s\-]?offs?|considerations?|overview|approach|strategy)\b",
    re.IGNORECASE,
)

_NON_IMPL_SECTION_RE = re.compile(
    r"^(?:context|background|rationale|overview|notes?|considerations?|discussion|trade[\s\-]?offs?)$",
    re.IGNORECASE,
)

_IMPL_ACTION_RE = re.compile(
    r"^(?:add|create|implement|modify|update|refactor|remove|delete|change|replace|integrate|support|extend|rewrite|fix|migrate|wire\s+up|hook\s+up)\b",
    re.IGNORECASE,
)


def is_actionable_step(step: PlanStep) -> bool:
    """Determine if a plan step represents an actionable implementation step."""
    clean_text = step.text.strip()
    # If the step already contains code candidates, it is an actionable reference
    has_code_candidate = any(c.kind != CandidateKind.CONCEPT for c in step.candidates)
    if has_code_candidate:
        return True

    # Exclude non-implementation sections
    if _NON_IMPL_SECTION_RE.match(step.section.strip()):
        return False

    # Exclude rationale, setup, architecture discussions
    if _RATIONALE_PREFIX_RE.match(clean_text):
        return False
    if _SETUP_PREFIX_RE.match(clean_text):
        return False
    if _ARCHITECTURE_PREFIX_RE.match(clean_text):
        return False

    # Check for implementation action verbs
    return bool(_IMPL_ACTION_RE.match(clean_text))


# -----------------------------------------------------------------------------
# Review Engine
# -----------------------------------------------------------------------------

def review_plan(
    plan: ParsedPlan,
    impact: ImpactAnalysis,
    validation: ValidationResult | None = None,
) -> PlanReview:
    """Compose parsing, resolution, impact, and validation into a structured PlanReview.

    Args:
        plan: The structured ParsedPlan.
        impact: The direct ImpactAnalysis result.
        validation: Optional ValidationResult (extracted from impact if not provided).

    Returns:
        Deterministic PlanReview with step-level grounding, classified files, and review items.
    """
    res: ResolutionResult = impact.resolution_result
    val: ValidationResult = validation or impact.validation or ValidationResult(plan.title, None, None)

    # 1. File Classification & Affected Tests
    # A. Explicit files: directly referenced as file_path or enclosing file of resolved code entities
    explicit_file_counts: dict[str, int] = {}
    for entity in res.resolved:
        if entity.candidate.kind == CandidateKind.FILE_PATH and entity.matched_nodes:
            sf = entity.matched_nodes[0].source_file or entity.candidate.text
            if sf:
                explicit_file_counts[sf] = explicit_file_counts.get(sf, 0) + 1
        elif entity.matched_nodes:
            for node in entity.matched_nodes:
                if node.source_file:
                    explicit_file_counts[node.source_file] = explicit_file_counts.get(node.source_file, 0) + 1

    # B. Impacted files: files holding entities with direct relationships to resolved entities
    impacted_file_counts: dict[str, int] = {}
    for imp_entity in impact.impacted_entities:
        for rel in imp_entity.relationships:
            if rel.is_external:
                continue
            tf = rel.target_file
            if tf:
                impacted_file_counts[tf] = impacted_file_counts.get(tf, 0) + 1

    # Combine all unique files
    all_files = sorted(set(explicit_file_counts.keys()) | set(impacted_file_counts.keys()))
    classified_list: list[ClassifiedFile] = []
    explicit_files_list: list[str] = sorted(explicit_file_counts.keys())
    # Impacted files: files with direct relationships that were not explicitly referenced
    impacted_files_list: list[str] = sorted(f for f in impacted_file_counts if f not in explicit_file_counts)
    # Affected tests: all test files present in either explicit or impacted files
    affected_tests_set: set[str] = set()

    for fp in all_files:
        roles: list[FileRole] = []
        if fp in explicit_file_counts:
            roles.append(FileRole.EXPLICIT)
        if fp in impacted_file_counts:
            roles.append(FileRole.IMPACTED)
        if is_test_path(fp):
            roles.append(FileRole.TEST)
            affected_tests_set.add(fp)

        total_refs = explicit_file_counts.get(fp, 0) + impacted_file_counts.get(fp, 0)
        classified_list.append(
            ClassifiedFile(
                file_path=fp,
                roles=tuple(roles),
                reference_count=total_refs,
            )
        )

    # Check for test entities in impact relationships (e.g. test callers)
    for imp_entity in impact.impacted_entities:
        for rel in imp_entity.relationships:
            if is_test_path(rel.target_file) or rel.target_label.startswith("test_"):
                if rel.target_file:
                    affected_tests_set.add(rel.target_file)

    affected_tests_list = sorted(affected_tests_set)

    # 2. Map candidates and findings to step/line locations
    # Candidate lookup by location
    candidates_by_step: dict[tuple[str, int | None], list[CandidateEntity]] = {}
    for cand in plan.all_candidates:
        key = (cand.location.section, cand.location.step_number)
        candidates_by_step.setdefault(key, []).append(cand)

    # Resolution entity lookup by candidate
    resolved_by_cand = {e.candidate: e for e in res.resolved}
    unresolved_by_cand = {e.candidate: e for e in res.unresolved}
    ambiguous_by_cand = {e.candidate: e for e in res.ambiguous}

    # Impacted entity lookup by node id
    impact_by_node = {ie.node_id: ie for ie in impact.impacted_entities}

    # Findings mapped by step and line
    findings_by_step: dict[tuple[str, int | None], list[ValidationFinding]] = {}
    findings_by_line: dict[int, list[ValidationFinding]] = {}
    for f in val.findings:
        if f.line_number is not None:
            findings_by_line.setdefault(f.line_number, []).append(f)
        if f.section is not None:
            key = (f.section, f.step_number)
            findings_by_step.setdefault(key, []).append(f)

    # 3. Build Step Reviews
    step_reviews_list: list[StepReview] = []
    grounded_steps_count = 0
    actionable_steps_count = 0
    total_steps_count = 0

    for section in plan.sections:
        for step in section.steps:
            total_steps_count += 1
            actionable = is_actionable_step(step)
            if actionable:
                actionable_steps_count += 1

            # Match candidates for this step
            step_key = (step.section, step.step_number)
            cands = list(step.candidates)
            # Also include any candidates matching step line number or step number
            for extra in candidates_by_step.get(step_key, []):
                if extra not in cands:
                    cands.append(extra)

            referenced: list[str] = [c.text for c in cands]
            resolved: list[str] = []
            unresolved: list[str] = []
            ambiguous: list[str] = []
            impacted_labels: list[str] = []
            step_affected_files: set[str] = set()
            step_direct_impact_count = 0

            for c in cands:
                if c in resolved_by_cand:
                    rent = resolved_by_cand[c]
                    for node in rent.matched_nodes:
                        if c.kind == CandidateKind.FILE_PATH:
                            label = node.source_file or c.text
                        else:
                            loc = f":{node.source_location}" if node.source_location else ""
                            label = f"{node.source_file}{loc}::{node.label}"
                        if label not in resolved:
                            resolved.append(label)
                        if node.source_file:
                            step_affected_files.add(node.source_file)

                        # Impact data for this node
                        if node.id in impact_by_node:
                            imp = impact_by_node[node.id]
                            step_direct_impact_count += len(imp.relationships)
                            for r in imp.relationships:
                                if not r.is_external:
                                    if r.target_label not in impacted_labels:
                                        impacted_labels.append(r.target_label)
                                    if r.target_file:
                                        step_affected_files.add(r.target_file)

                elif c in unresolved_by_cand:
                    if c.text not in unresolved:
                        unresolved.append(c.text)
                elif c in ambiguous_by_cand:
                    if c.text not in ambiguous:
                        ambiguous.append(c.text)

            has_grounding = len(resolved) > 0
            if has_grounding:
                grounded_steps_count += 1

            # Findings for this step
            matched_findings: list[ValidationFinding] = []
            seen_finding_keys: set[tuple[str, str, str | None]] = set()

            for f in findings_by_step.get(step_key, []):
                fkey = (f.code.value, f.severity.value, f.target_entity)
                if fkey not in seen_finding_keys:
                    seen_finding_keys.add(fkey)
                    matched_findings.append(f)

            for f in findings_by_line.get(step.line_number, []):
                fkey = (f.code.value, f.severity.value, f.target_entity)
                if fkey not in seen_finding_keys:
                    seen_finding_keys.add(fkey)
                    matched_findings.append(f)

            step_reviews_list.append(
                StepReview(
                    step_number=step.step_number,
                    step_index=step.index,
                    section=step.section,
                    line_number=step.line_number,
                    text=step.text,
                    is_actionable=actionable,
                    has_grounding=has_grounding,
                    referenced_entities=tuple(referenced),
                    resolved_entities=tuple(resolved),
                    unresolved_entities=tuple(unresolved),
                    ambiguous_entities=tuple(ambiguous),
                    impacted_entities=tuple(impacted_labels),
                    affected_files=tuple(sorted(step_affected_files)),
                    direct_impact_count=step_direct_impact_count,
                    findings=tuple(matched_findings),
                )
            )

    # 4. Review Items (derived from validation findings)
    review_items_list: list[ReviewItem] = []
    for f in val.findings:
        title = f.target_entity or f.code.value
        review_items_list.append(
            ReviewItem(
                code=f.code.value,
                severity=f.severity.value,
                title=title,
                description=f.message,
                step_number=f.step_number,
                line_number=f.line_number,
                target_entity=f.target_entity,
                evidence=f.evidence,
            )
        )

    # 5. Build Summary
    summary = ReviewSummary(
        total_steps=total_steps_count,
        actionable_steps=actionable_steps_count,
        grounded_steps=grounded_steps_count,
        resolved_entities=len(res.resolved),
        ambiguous_entities=len(res.ambiguous),
        unresolved_entities=len(res.unresolved),
        not_code_concepts=len(res.not_code_entities),
        explicit_files_count=len(explicit_files_list),
        impacted_files_count=len(impacted_files_list),
        affected_tests_count=len(affected_tests_list),
        total_direct_relationships=sum(len(e.relationships) for e in impact.impacted_entities),
        validation_errors=val.summary.error_count,
        validation_warnings=val.summary.warning_count,
        validation_infos=val.summary.info_count,
    )

    return PlanReview(
        plan_title=plan.title or impact.plan_title,
        plan_file=str(plan.file_path) if plan.file_path else impact.plan_file,
        graph_file=impact.graph_file,
        summary=summary,
        step_reviews=tuple(step_reviews_list),
        explicit_files=tuple(explicit_files_list),
        impacted_files=tuple(impacted_files_list),
        affected_tests=tuple(affected_tests_list),
        classified_files=tuple(classified_list),
        review_items=tuple(review_items_list),
    )


# -----------------------------------------------------------------------------
# Human-Readable Formatting
# -----------------------------------------------------------------------------

def format_review_report(review: PlanReview) -> str:
    """Format a PlanReview into a concise, structured human-readable terminal report."""
    lines: list[str] = []
    lines.append("Plan Review")
    lines.append("-----------")
    lines.append("")

    # 1. Coverage
    lines.append("Coverage")
    lines.append(f"  Steps: {review.summary.total_steps}")
    lines.append(f"  Actionable: {review.summary.actionable_steps}")
    lines.append(f"  Grounded: {review.summary.grounded_steps}")
    lines.append(f"  Resolved entities: {review.summary.resolved_entities}")
    lines.append(f"  Ambiguous: {review.summary.ambiguous_entities}")
    lines.append(f"  Unresolved: {review.summary.unresolved_entities}")
    lines.append("")

    # 2. Explicit Files
    lines.append("Explicit files")
    if review.explicit_files:
        for f in review.explicit_files:
            lines.append(f"  {f}")
    else:
        lines.append("  (none)")
    lines.append("")

    # 3. Impacted Files
    lines.append("Impacted files")
    if review.impacted_files:
        for f in review.impacted_files:
            lines.append(f"  {f}")
    else:
        lines.append("  (none)")
    lines.append("")

    # 4. Affected Tests
    lines.append("Affected tests")
    if review.affected_tests:
        for f in review.affected_tests:
            lines.append(f"  {f}")
    else:
        lines.append("  (no test files or test entities appear in direct impact graph)")
    lines.append("")

    # 5. Step Review
    if review.step_reviews:
        lines.append("Step review")
        for s in review.step_reviews:
            if not s.is_actionable and not s.referenced_entities:
                continue

            num_str = f"{s.step_number}." if s.step_number is not None else "•"
            lines.append(f"  {num_str} {s.text}")

            if s.resolved_entities:
                for r in s.resolved_entities:
                    lines.append(f"     ✓ {r}")
            elif s.ambiguous_entities:
                for a in s.ambiguous_entities:
                    lines.append(f"     ? ambiguous: {a}")
            elif s.unresolved_entities:
                for u in s.unresolved_entities:
                    lines.append(f"     ✗ unresolved: {u}")
            elif s.is_actionable and not s.has_grounding:
                lines.append("     ⚠ no resolved codebase entities")

            if s.direct_impact_count > 0:
                lines.append(f"     → {s.direct_impact_count} direct relationships")

            lines.append("")

    # 6. Review Items
    lines.append("Review items")
    if review.review_items:
        for item in review.review_items:
            prefix = "✗" if item.severity == "error" else ("⚠" if item.severity == "warning" else "ℹ")
            line_ref = f" (Line {item.line_number})" if item.line_number else ""
            lines.append(f"  {prefix} {item.code}: {item.title}{line_ref}")
            lines.append(f"    {item.description}")
            if item.evidence and item.evidence != item.description and not item.description.endswith(item.evidence):
                lines.append(f"    → {item.evidence}")
            lines.append("")
    else:
        lines.append("  ✓ No review warnings or errors detected")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
