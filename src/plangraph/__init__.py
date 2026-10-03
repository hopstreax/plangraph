"""PlanGraph - Validate your implementation plan against the real codebase before you code."""

from __future__ import annotations

from plangraph.exceptions import (
    GraphFormatError,
    GraphNotFoundError,
    GraphValidationError,
    PlanGraphError,
    PlanParseError,
)
from plangraph.graph import get_edge, get_node, load_graph
from plangraph.models import Edge, ImpactEvidence, Node
from plangraph.plan import (
    CandidateEntity,
    CandidateKind,
    Confidence,
    ParsedPlan,
    PlanLocation,
    PlanSection,
    PlanStep,
    parse_plan,
)
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
from plangraph.resolver import (
    GraphIndex,
    ResolutionResult,
    ResolutionStatus,
    ResolvedEntity,
    resolve_plan,
)
from plangraph.validation import (
    DEFAULT_HIGH_IMPACT_THRESHOLD,
    FindingCode,
    ValidationConfig,
    ValidationFinding,
    ValidationResult,
    ValidationSeverity,
    ValidationSummary,
    format_validation_report,
    validate_plan,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_graph",
    "get_node",
    "get_edge",
    "Node",
    "Edge",
    "ImpactEvidence",
    "PlanGraphError",
    "GraphNotFoundError",
    "GraphFormatError",
    "GraphValidationError",
    "PlanParseError",
    "CandidateEntity",
    "CandidateKind",
    "Confidence",
    "ParsedPlan",
    "PlanLocation",
    "PlanSection",
    "PlanStep",
    "parse_plan",
    "ResolutionStatus",
    "ResolvedEntity",
    "ResolutionResult",
    "GraphIndex",
    "resolve_plan",
    "ImpactDirection",
    "ImpactRelationship",
    "ImpactedEntity",
    "ImpactSummary",
    "ImpactAnalysis",
    "analyze_impact",
    "format_impact_report",
    "MAX_CONTAINMENT_DISPLAY",
    "DEFAULT_HIGH_IMPACT_THRESHOLD",
    "FindingCode",
    "ValidationConfig",
    "ValidationFinding",
    "ValidationResult",
    "ValidationSeverity",
    "ValidationSummary",
    "validate_plan",
    "format_validation_report",
]
