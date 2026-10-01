"""Custom exceptions for PlanGraph."""

from __future__ import annotations


class PlanGraphError(Exception):
    """Base exception for all PlanGraph errors."""


class GraphNotFoundError(PlanGraphError):
    """Raised when the specified graph file cannot be found."""


class GraphFormatError(PlanGraphError):
    """Raised when the graph file is malformed or invalid JSON."""


class GraphValidationError(PlanGraphError):
    """Raised when the graph fails structural or schema validation."""


class PlanParseError(PlanGraphError):
    """Raised when a plan document cannot be read or parsed."""
