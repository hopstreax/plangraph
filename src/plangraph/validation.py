"""Deterministic plan validation layer for PlanGraph.

Examines parsed plans, entity resolutions, and direct impact analysis results
to identify concrete problems, ambiguities, and potential risks in a contributor's
implementation plan based on codebase evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plangraph.impact import ImpactAnalysis, ImpactedEntity

from plangraph.plan import CandidateEntity, CandidateKind, ParsedPlan, PlanStep
from plangraph.resolver import ResolutionResult, ResolutionStatus, ResolvedEntity


# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------

DEFAULT_HIGH_IMPACT_THRESHOLD: int = 10


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration options for plan validation."""

    high_impact_threshold: int = DEFAULT_HIGH_IMPACT_THRESHOLD


# -----------------------------------------------------------------------------
# Severity and Finding Codes
# -----------------------------------------------------------------------------

class ValidationSeverity(str, Enum):
    """Severity classification of a validation finding.

    - ERROR: Concrete blockers preventing reliable codebase validation (unresolved or ambiguous entities).
    - WARNING: Potential risks, high direct impact, or actionable steps lacking codebase references.
    - INFO: Contextual, non-code informational notices that do not indicate a problem.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class FindingCode(str, Enum):
    """Stable identification codes for validation findings."""

    UNRESOLVED_ENTITY = "UNRESOLVED_ENTITY"
    AMBIGUOUS_ENTITY = "AMBIGUOUS_ENTITY"
    NO_CODE_ENTITY = "NO_CODE_ENTITY"
    HIGH_IMPACT_ENTITY = "HIGH_IMPACT_ENTITY"
    STEP_WITHOUT_RESOLVED_ENTITY = "STEP_WITHOUT_RESOLVED_ENTITY"
    LOW_PLAN_EVIDENCE = "LOW_PLAN_EVIDENCE"


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationFinding:
    """A single deterministic validation finding with provenance and evidence."""

    code: FindingCode
    severity: ValidationSeverity
    message: str
    target_entity: str | None = None
    line_number: int | None = None
    section: str | None = None
    step_number: int | None = None
    evidence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert finding to a JSON-serializable dictionary."""
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "target_entity": self.target_entity,
            "line_number": self.line_number,
            "section": self.section,
            "step_number": self.step_number,
            "evidence": self.evidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationSummary:
    """Summary counts of validation findings."""

    total_findings: int
    error_count: int
    warning_count: int
    info_count: int
    findings_by_code: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to a JSON-serializable dictionary."""
        return {
            "total_findings": self.total_findings,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "findings_by_code": dict(self.findings_by_code),
        }


@dataclass(frozen=True)
class ValidationResult:
    """The aggregate validation result for an implementation plan."""

    plan_title: str
    plan_file: str | None
    graph_file: str | None
    findings: tuple[ValidationFinding, ...] = field(default_factory=tuple)
    summary: ValidationSummary = field(default_factory=lambda: ValidationSummary(0, 0, 0, 0, {}))

    @property
    def is_valid(self) -> bool:
        """True if there are no ERROR-severity findings."""
        return self.summary.error_count == 0

    @property
    def errors(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity == ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity == ValidationSeverity.WARNING)

    @property
    def infos(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity == ValidationSeverity.INFO)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to a JSON-serializable dictionary."""
        return {
            "plan_title": self.plan_title,
            "plan_file": self.plan_file,
            "graph_file": self.graph_file,
            "is_valid": self.is_valid,
            "summary": self.summary.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


# -----------------------------------------------------------------------------
# Conservative Pattern Definitions for Step Analysis
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


# -----------------------------------------------------------------------------
# Validation Implementation
# -----------------------------------------------------------------------------

def validate_plan(
    plan: ParsedPlan,
    impact: ImpactAnalysis | None = None,
    resolution: ResolutionResult | None = None,
    *,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """Validate an implementation plan against resolution and impact evidence.

    Args:
        plan: The structured ParsedPlan to validate.
        impact: Optional ImpactAnalysis containing direct relationships and resolution.
        resolution: Optional ResolutionResult (inferred from impact if not provided).
        config: Optional ValidationConfig controlling thresholds.

    Returns:
        ValidationResult containing sorted, deduplicated findings and summary metrics.
    """
    cfg = config or ValidationConfig()

    res = resolution
    if res is None and impact is not None:
        res = impact.resolution_result

    if res is None:
        raise ValueError("Either impact or resolution must be provided to validate_plan.")

    impacted_entities: tuple[ImpactedEntity, ...] = impact.impacted_entities if impact is not None else ()

    seen_keys: set[tuple[str, str, str | None, int | None, int | None]] = set()
    findings_list: list[ValidationFinding] = []

    def add_finding(f: ValidationFinding) -> None:
        key = (f.code.value, f.severity.value, f.target_entity, f.line_number, f.step_number)
        if key in seen_keys:
            return
        seen_keys.add(key)
        findings_list.append(f)

    # 1. UNRESOLVED_ENTITY (ERROR)
    for entity in res.unresolved:
        loc = entity.candidate.location
        reason = entity.resolution_reason or f"no entity found matching '{entity.candidate.text}'"
        add_finding(
            ValidationFinding(
                code=FindingCode.UNRESOLVED_ENTITY,
                severity=ValidationSeverity.ERROR,
                message=f"Unresolved entity '{entity.candidate.text}': {reason}",
                target_entity=entity.candidate.text,
                line_number=loc.line_number if loc else None,
                section=loc.section if loc else None,
                step_number=loc.step_number if loc else None,
                evidence=reason,
                metadata={"candidate_kind": entity.candidate.kind.value},
            )
        )

    # 2. AMBIGUOUS_ENTITY (ERROR)
    for entity in res.ambiguous:
        loc = entity.candidate.location
        matched_locs = [
            f"{n.source_file}:{n.source_location or ''} ({n.label})"
            for n in entity.matched_nodes
        ]
        locs_str = ", ".join(matched_locs) if matched_locs else "multiple conflicting nodes"
        add_finding(
            ValidationFinding(
                code=FindingCode.AMBIGUOUS_ENTITY,
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Ambiguous reference '{entity.candidate.text}' matches {len(entity.matched_nodes)} "
                    f"entities without sufficient disambiguation: {locs_str}"
                ),
                target_entity=entity.candidate.text,
                line_number=loc.line_number if loc else None,
                section=loc.section if loc else None,
                step_number=loc.step_number if loc else None,
                evidence=entity.resolution_reason,
                metadata={
                    "candidate_kind": entity.candidate.kind.value,
                    "match_count": len(entity.matched_nodes),
                    "matches": matched_locs,
                },
            )
        )

    # 3. NO_CODE_ENTITY (INFO)
    for entity in res.not_code_entities:
        loc = entity.candidate.location
        add_finding(
            ValidationFinding(
                code=FindingCode.NO_CODE_ENTITY,
                severity=ValidationSeverity.INFO,
                message=f"Concept reference '{entity.candidate.text}' is a non-code entity",
                target_entity=entity.candidate.text,
                line_number=loc.line_number if loc else None,
                section=loc.section if loc else None,
                step_number=loc.step_number if loc else None,
                evidence=entity.candidate.context_sentence or "concept reference",
                metadata={"candidate_kind": entity.candidate.kind.value},
            )
        )

    # 4. HIGH_IMPACT_ENTITY (WARNING)
    for entity in impacted_entities:
        impact_count = len(entity.relationships)
        if impact_count >= cfg.high_impact_threshold:
            orig = entity.originating_candidates[0] if entity.originating_candidates else None
            loc = orig.location if orig else None
            add_finding(
                ValidationFinding(
                    code=FindingCode.HIGH_IMPACT_ENTITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"High-impact entity '{entity.label}' has {impact_count} direct relationships in codebase",
                    target_entity=entity.label,
                    line_number=loc.line_number if loc else None,
                    section=loc.section if loc else None,
                    step_number=loc.step_number if loc else None,
                    evidence=f"{impact_count} direct graph relationships (threshold: {cfg.high_impact_threshold})",
                    metadata={
                        "impact_count": impact_count,
                        "threshold": cfg.high_impact_threshold,
                        "node_id": entity.node_id,
                        "source_file": entity.source_file,
                    },
                )
            )

    # 5. STEP_WITHOUT_RESOLVED_ENTITY (WARNING)
    resolved_code_candidates = {
        e.candidate for e in res.resolved
        if e.candidate.kind != CandidateKind.CONCEPT
    }
    unresolved_or_ambig_code_candidates = {
        e.candidate for e in (*res.unresolved, *res.ambiguous)
        if e.candidate.kind != CandidateKind.CONCEPT
    }
    not_code_candidates = {
        e.candidate for e in res.not_code_entities
    }

    for section in plan.sections:
        if _NON_IMPL_SECTION_RE.match(section.heading.strip()):
            continue

        for step in section.steps:
            step_cands = set(step.candidates)

            # Do not flag if step has at least one resolved code entity
            if any(c in resolved_code_candidates for c in step_cands):
                continue

            # Do not flag if step intentionally references non-code concepts
            if any(c.kind == CandidateKind.CONCEPT or c in not_code_candidates for c in step_cands):
                continue

            # Do not flag if step has explicit unresolved or ambiguous candidates (already reported)
            if any(c in unresolved_or_ambig_code_candidates for c in step_cands):
                continue

            clean_text = step.text.strip()

            # Conservative exclusions
            if _RATIONALE_PREFIX_RE.match(clean_text):
                continue
            if _SETUP_PREFIX_RE.match(clean_text):
                continue
            if _ARCHITECTURE_PREFIX_RE.match(clean_text):
                continue

            # Only flag if step starts with an implementation-oriented action verb
            if _IMPL_ACTION_RE.match(clean_text):
                add_finding(
                    ValidationFinding(
                        code=FindingCode.STEP_WITHOUT_RESOLVED_ENTITY,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Step {step.step_number or step.index} ('{clean_text}') contains actionable "
                            f"implementation language but references no resolved code entities"
                        ),
                        target_entity=None,
                        line_number=step.line_number,
                        section=step.section,
                        step_number=step.step_number,
                        evidence=f"Actionable step in section '{step.section}' lacks codebase references",
                        metadata={"step_text": clean_text},
                    )
                )

    # 6. LOW_PLAN_EVIDENCE (WARNING)
    has_code_evidence = bool(res.resolved or res.unresolved or res.ambiguous)
    if not has_code_evidence:
        add_finding(
            ValidationFinding(
                code=FindingCode.LOW_PLAN_EVIDENCE,
                severity=ValidationSeverity.WARNING,
                message="Plan contains no actionable codebase evidence or resolved code references",
                target_entity=None,
                line_number=1,
                section=None,
                step_number=None,
                evidence="Zero codebase entities were resolved or referenced in the plan",
                metadata={
                    "total_candidates": len(plan.all_candidates),
                    "resolved_candidates": len(res.resolved),
                },
            )
        )

    # Deterministic sorting
    severity_order = {
        ValidationSeverity.ERROR: 0,
        ValidationSeverity.WARNING: 1,
        ValidationSeverity.INFO: 2,
    }
    findings_list.sort(
        key=lambda f: (
            severity_order.get(f.severity, 99),
            f.line_number if f.line_number is not None else 999999,
            f.step_number if f.step_number is not None else 999999,
            f.code.value,
            f.target_entity or "",
            f.message,
        )
    )

    # Compute summary
    error_count = sum(1 for f in findings_list if f.severity == ValidationSeverity.ERROR)
    warning_count = sum(1 for f in findings_list if f.severity == ValidationSeverity.WARNING)
    info_count = sum(1 for f in findings_list if f.severity == ValidationSeverity.INFO)
    findings_by_code: dict[str, int] = {}
    for f in findings_list:
        findings_by_code[f.code.value] = findings_by_code.get(f.code.value, 0) + 1

    summary = ValidationSummary(
        total_findings=len(findings_list),
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
        findings_by_code=dict(sorted(findings_by_code.items())),
    )

    return ValidationResult(
        plan_title=plan.title or res.plan_title,
        plan_file=str(plan.file_path) if plan.file_path else res.plan_file,
        graph_file=res.graph_file,
        findings=tuple(findings_list),
        summary=summary,
    )


# -----------------------------------------------------------------------------
# Human-Readable Formatting
# -----------------------------------------------------------------------------

def format_validation_report(validation: ValidationResult) -> str:
    """Format a ValidationResult into a human-readable CLI report section."""
    lines: list[str] = ["Validation"]

    if not validation.findings:
        lines.append("  ✓ No validation issues detected")
        lines.append("")
        return "\n".join(lines)

    for f in validation.findings:
        sev_tag = f.severity.value.upper()
        line_ref = f" (Line {f.line_number})" if f.line_number else ""
        entity_ref = f": {f.target_entity}" if f.target_entity else ""
        lines.append(f"  {sev_tag:<7} {f.code.value}{entity_ref}{line_ref}")
        lines.append(f"          {f.message}")
        if f.evidence and f.evidence != f.message:
            lines.append(f"          → {f.evidence}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
