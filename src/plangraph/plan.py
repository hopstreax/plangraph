"""Contributor implementation plan models and deterministic Markdown parser.

Extracts structured sections, steps, and candidate code/concept entities
with exact line-level provenance from a contributor's plan.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from plangraph.exceptions import PlanParseError


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

class CandidateKind(str, Enum):
    """Classification of an extracted candidate reference."""

    FILE_PATH = "file_path"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    SYMBOL = "symbol"
    CONCEPT = "concept"


class Confidence(str, Enum):
    """Certainty level of a candidate reference."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    CONCEPT = "concept"


@dataclass(frozen=True)
class PlanLocation:
    """Exact location and structural context of a reference in the plan."""

    line_number: int
    section: str
    step_number: int | None = None
    is_list_item: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "section": self.section,
            "step_number": self.step_number,
            "is_list_item": self.is_list_item,
        }


@dataclass(frozen=True)
class CandidateEntity:
    """A candidate entity extracted from the plan prose or code references."""

    text: str
    raw_text: str
    kind: CandidateKind
    confidence: Confidence
    location: PlanLocation
    context_sentence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "kind": self.kind.value,
            "confidence": self.confidence.value,
            "location": self.location.to_dict(),
            "context_sentence": self.context_sentence,
        }


@dataclass(frozen=True)
class PlanStep:
    """A structured list item (step or bullet) within a plan section."""

    index: int
    step_number: int | None
    text: str
    section: str
    line_number: int
    candidates: tuple[CandidateEntity, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "step_number": self.step_number,
            "text": self.text,
            "section": self.section,
            "line_number": self.line_number,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass(frozen=True)
class PlanSection:
    """A markdown section introduced by a heading."""

    heading: str
    level: int
    line_number: int
    content: str
    steps: tuple[PlanStep, ...] = field(default_factory=tuple)
    candidates: tuple[CandidateEntity, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "level": self.level,
            "line_number": self.line_number,
            "content": self.content,
            "steps": [s.to_dict() for s in self.steps],
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass(frozen=True)
class ParsedPlan:
    """The complete structured representation of a contributor plan."""

    file_path: Path | None
    title: str
    sections: tuple[PlanSection, ...] = field(default_factory=tuple)
    all_candidates: tuple[CandidateEntity, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path) if self.file_path else None,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "all_candidates": [c.to_dict() for c in self.all_candidates],
        }


# -----------------------------------------------------------------------------
# Vocabulary and Regex Patterns
# -----------------------------------------------------------------------------

# File extensions recognized as source, test, configuration, or documentation
RECOGNIZED_EXTENSIONS = (
    "py", "ts", "tsx", "js", "jsx", "go", "rs", "java", "cpp", "c", "h",
    "cs", "rb", "json", "yaml", "yml", "sql", "md", "toml"
)

_EXT_PATTERN = "|".join(RECOGNIZED_EXTENSIONS)

# Regex matching file paths (POSIX and Windows separators)
# Trailing punctuation like . , ; : ) ] is handled by stripping afterwards
_FILE_PATH_RE = re.compile(
    rf"(?:(?<=[\s`\"'(\[])|^)"
    rf"([a-zA-Z0-9_\.\-]+(?:[/\\][a-zA-Z0-9_\.\-]+)*\.(?:{_EXT_PATTERN}))"
    rf"(?=[`\"'\s\),;:\]\.!?]|$)",
    re.IGNORECASE,
)

# Explicit backtick spans: `identifier`
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")

# Method call pattern: ClassName.methodName(...) or ClassName.methodName
_METHOD_CALL_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)(?:\(\))?\b")

# Function call pattern: function_name()
_FUNCTION_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(\)")

# PascalCase identifier pattern: Must contain at least two uppercase humps (e.g. UserService, CacheManager)
_PASCAL_CASE_RE = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")

# Known infrastructure / concept keywords (case-insensitive)
CONCEPT_VOCABULARY: dict[str, str] = {
    "redis": "Redis",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "kafka": "Kafka",
    "docker": "Docker",
    "jwt": "JWT",
    "caching": "caching",
    "cache": "cache",
    "middleware": "middleware",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "oauth": "OAuth",
    "rabbitmq": "RabbitMQ",
    "celery": "Celery",
    "elasticsearch": "Elasticsearch",
}

# Words that commonly start sentences or actions in plans and must not be treated as class symbols
_SENTENCE_STARTER_VERBS = frozenset({
    "add", "create", "update", "modify", "delete", "remove", "implement",
    "refactor", "fix", "verify", "test", "check", "ensure", "use", "make",
    "run", "start", "stop", "build", "set", "get", "change", "write", "note",
    "consider", "allow", "support", "provide", "define", "the", "when", "first",
    "finally", "next", "after", "before", "in", "for", "with", "this", "that"
})


# -----------------------------------------------------------------------------
# Candidate Extraction Engine
# -----------------------------------------------------------------------------

def _spans_overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    """Check if two character ranges in a string overlap."""
    return max(span_a[0], span_b[0]) < min(span_a[1], span_b[1])


def _clean_trailing_punctuation(token: str) -> tuple[str, str]:
    """Strip trailing sentence punctuation from a token, returning (cleaned, stripped_punctuation)."""
    cleaned = token
    stripped = ""
    while cleaned and cleaned[-1] in ".,;:!?)]}":
        stripped = cleaned[-1] + stripped
        cleaned = cleaned[:-1]
    return cleaned, stripped


def _classify_backtick_span(span_text: str) -> list[tuple[str, CandidateKind]]:
    """Determine CandidateKinds for an explicit backtick reference."""
    trimmed = span_text.strip()
    if not trimmed:
        return []

    # If the backtick span contains whitespace, it is a multi-token expression
    # (e.g. a shell command like `pytest tests/test_user.py` or flag `--config config.yaml`).
    # It must NOT be treated as a single file path or symbol.
    if any(ch.isspace() for ch in trimmed):
        results: list[tuple[str, CandidateKind]] = []
        tokens = trimmed.split()
        for tok in tokens:
            cleaned, _ = _clean_trailing_punctuation(tok)
            # Skip flags or options starting with '-'
            if not cleaned or cleaned.startswith("-"):
                continue
            norm_posix = cleaned.replace("\\", "/")
            # Legitimate file path in a command must end with a recognized extension
            if norm_posix.lower().endswith(tuple(f".{ext}" for ext in RECOGNIZED_EXTENSIONS)):
                results.append((norm_posix, CandidateKind.FILE_PATH))
        return results

    # Single-token backtick handling:
    norm_posix = trimmed.replace("\\", "/")

    # File path if it contains slashes or ends with recognized extension
    if "/" in norm_posix or norm_posix.lower().endswith(tuple(f".{ext}" for ext in RECOGNIZED_EXTENSIONS)):
        return [(norm_posix, CandidateKind.FILE_PATH)]

    # Method call (e.g. UserService.get_user or UserService.get_user())
    if "." in trimmed and not trimmed.endswith(tuple(f".{ext}" for ext in RECOGNIZED_EXTENSIONS)):
        return [(trimmed, CandidateKind.METHOD)]

    # Function call with parentheses (e.g. validate_token())
    if trimmed.endswith("()"):
        return [(trimmed, CandidateKind.FUNCTION)]

    # PascalCase class (e.g. UserService)
    if _PASCAL_CASE_RE.fullmatch(trimmed):
        return [(trimmed, CandidateKind.CLASS)]

    # Otherwise treat as general code symbol
    return [(trimmed, CandidateKind.SYMBOL)]


def extract_candidates(line: str, location: PlanLocation) -> list[CandidateEntity]:
    """Extract candidate entities from a single line of text with occurrence-aware deduplication."""
    # Track candidate entries with exact match start offset for deterministic document ordering:
    # (match_start_offset, insertion_order, candidate)
    entries: list[tuple[int, int, CandidateEntity]] = []
    # Track occupied spans (start, end) to prevent multiple rules from claiming the exact same occurrence
    occupied_spans: list[tuple[int, int]] = []

    context_sentence = line.strip()

    # Stage 1: Explicit backtick spans
    for match in _BACKTICK_RE.finditer(line):
        span = match.span()
        inner_text = match.group(1).strip()
        if not inner_text:
            continue

        occupied_spans.append(span)
        extracted = _classify_backtick_span(inner_text)
        for normalized_text, kind in extracted:
            candidate = CandidateEntity(
                text=normalized_text,
                raw_text=match.group(0),
                kind=kind,
                confidence=Confidence.EXPLICIT,
                location=location,
                context_sentence=context_sentence,
            )
            entries.append((span[0], len(entries), candidate))

    # Stage 2: Explicit file paths outside backticks
    for match in _FILE_PATH_RE.finditer(line):
        span = match.span(1)
        if any(_spans_overlap(span, occ) for occ in occupied_spans):
            continue

        raw_token = match.group(1)
        cleaned_path, _ = _clean_trailing_punctuation(raw_token)
        if not cleaned_path:
            continue

        # Reject URLs (http://, https://, www.)
        prefix_start = max(0, span[0] - 10)
        prefix = line[prefix_start:span[0]].lower()
        if "http://" in prefix or "https://" in prefix or "www." in prefix:
            continue

        normalized_path = cleaned_path.replace("\\", "/")
        occupied_spans.append(span)

        candidate = CandidateEntity(
            text=normalized_path,
            raw_text=raw_token,
            kind=CandidateKind.FILE_PATH,
            confidence=Confidence.EXPLICIT,
            location=location,
            context_sentence=context_sentence,
        )
        entries.append((span[0], len(entries), candidate))

    # Stage 3A: Method calls in prose (outside backticks)
    for match in _METHOD_CALL_RE.finditer(line):
        span = match.span()
        if any(_spans_overlap(span, occ) for occ in occupied_spans):
            continue

        raw_token = match.group(0)
        occupied_spans.append(span)

        candidate = CandidateEntity(
            text=raw_token,
            raw_text=raw_token,
            kind=CandidateKind.METHOD,
            confidence=Confidence.INFERRED,
            location=location,
            context_sentence=context_sentence,
        )
        entries.append((span[0], len(entries), candidate))

    # Stage 3B: Function calls in prose (outside backticks, e.g. validate_token())
    for match in _FUNCTION_CALL_RE.finditer(line):
        span = match.span()
        if any(_spans_overlap(span, occ) for occ in occupied_spans):
            continue

        raw_token = match.group(0)
        occupied_spans.append(span)

        candidate = CandidateEntity(
            text=raw_token,
            raw_text=raw_token,
            kind=CandidateKind.FUNCTION,
            confidence=Confidence.EXPLICIT,
            location=location,
            context_sentence=context_sentence,
        )
        entries.append((span[0], len(entries), candidate))

    # Stage 3C: Inferred PascalCase classes in prose
    for match in _PASCAL_CASE_RE.finditer(line):
        span = match.span(1)
        if any(_spans_overlap(span, occ) for occ in occupied_spans):
            continue

        raw_token = match.group(1)

        # Do not classify single-word capitalized sentence starters as symbols
        if raw_token.lower() in _SENTENCE_STARTER_VERBS:
            continue

        occupied_spans.append(span)

        candidate = CandidateEntity(
            text=raw_token,
            raw_text=raw_token,
            kind=CandidateKind.CLASS,
            confidence=Confidence.INFERRED,
            location=location,
            context_sentence=context_sentence,
        )
        entries.append((span[0], len(entries), candidate))

    # Stage 4: Concepts
    # Word boundary matching for technology vocabulary
    for word_lower, canonical_name in CONCEPT_VOCABULARY.items():
        pattern = re.compile(rf"\b{re.escape(word_lower)}\b", re.IGNORECASE)
        for match in pattern.finditer(line):
            span = match.span()
            if any(_spans_overlap(span, occ) for occ in occupied_spans):
                continue

            raw_token = match.group(0)
            occupied_spans.append(span)

            candidate = CandidateEntity(
                text=canonical_name,
                raw_text=raw_token,
                kind=CandidateKind.CONCEPT,
                confidence=Confidence.CONCEPT,
                location=location,
                context_sentence=context_sentence,
            )
            entries.append((span[0], len(entries), candidate))

    # Sort candidates in document appearance order by actual matched start offset,
    # with insertion order as secondary tie-breaker.
    entries.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in entries]


# -----------------------------------------------------------------------------
# Markdown Parser State Machine
# -----------------------------------------------------------------------------

def _is_path_like(source: str) -> bool:
    """Determine whether a string source is intended as a file path rather than raw Markdown prose."""
    if "\n" in source or "\r" in source:
        return False
    stripped = source.strip()
    if not stripped:
        return False
    # Common Markdown indicators mean this is prose, not a file path
    if (
        stripped.startswith(("#", ">", "- ", "* ", "+ ", "`", "*", "_", "["))
        or stripped.endswith("`")
        or re.match(r"^\d+\.\s+", stripped)
    ):
        return False
    # Existing file on disk
    if Path(stripped).is_file():
        return True
    # Explicit Markdown file extension
    if stripped.lower().endswith((".md", ".markdown")):
        return True
    # Path containing directory separators without spaces
    if ("/" in stripped or "\\" in stripped) and not any(ch.isspace() for ch in stripped):
        return True
    return False


def parse_plan(source: str | Path) -> ParsedPlan:
    """Parse a contributor implementation plan into a deterministic ParsedPlan model.

    Args:
        source: Either a string containing markdown text, or a Path to a plan.md file.

    Returns:
        ParsedPlan: Immutable structured representation with sections, steps, and candidates.

    Raises:
        PlanParseError: If the source path cannot be read.
    """
    file_path: Path | None = None
    if isinstance(source, Path):
        file_path = source
    elif isinstance(source, str) and _is_path_like(source):
        file_path = Path(source)

    if file_path is not None:
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PlanParseError(f"Failed to read plan file {file_path}: {exc}") from exc
    else:
        content = str(source)

    # Normalize CRLF to LF
    content = content.replace("\r\n", "\n")
    lines = content.split("\n")

    title = ""
    current_section_heading = "Overview"
    current_section_level = 1
    current_section_line = 1
    current_section_content: list[str] = []
    current_section_steps: list[PlanStep] = []
    current_section_candidates: list[CandidateEntity] = []

    sections: list[PlanSection] = []
    all_candidates: list[CandidateEntity] = []

    in_code_block = False
    step_index = 0

    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    numbered_item_re = re.compile(r"^\s*(\d+)\.\s+(.*)$")
    bullet_item_re = re.compile(r"^\s*[-*+]\s+(.*)$")

    def _flush_section() -> None:
        nonlocal current_section_heading, current_section_level, current_section_line
        nonlocal current_section_content, current_section_steps, current_section_candidates
        nonlocal step_index

        has_content = any(line.strip() for line in current_section_content)
        if has_content or current_section_steps or current_section_candidates:
            section = PlanSection(
                heading=current_section_heading,
                level=current_section_level,
                line_number=current_section_line,
                content="\n".join(current_section_content).strip(),
                steps=tuple(current_section_steps),
                candidates=tuple(current_section_candidates),
            )
            sections.append(section)

        current_section_content = []
        current_section_steps = []
        current_section_candidates = []
        step_index = 0

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Handle fenced code blocks (``` or ~~~)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            current_section_content.append(line)
            continue

        # Ignore candidate extraction while inside code blocks
        if in_code_block:
            current_section_content.append(line)
            continue

        # Handle headings
        heading_match = heading_re.match(line)
        if heading_match:
            hashes, heading_text = heading_match.groups()
            level = len(hashes)
            heading_title = heading_text.strip()

            if level == 1 and not title:
                title = heading_title

            _flush_section()

            current_section_heading = heading_title
            current_section_level = level
            current_section_line = line_idx
            continue

        current_section_content.append(line)

        if not stripped:
            continue

        # Check for numbered list item (e.g. 1. Step text)
        numbered_match = numbered_item_re.match(line)
        if numbered_match:
            num_str, item_text = numbered_match.groups()
            step_num = int(num_str)
            step_index += 1

            location = PlanLocation(
                line_number=line_idx,
                section=current_section_heading,
                step_number=step_num,
                is_list_item=True,
            )
            step_candidates = extract_candidates(item_text, location)

            step = PlanStep(
                index=step_index,
                step_number=step_num,
                text=item_text.strip(),
                section=current_section_heading,
                line_number=line_idx,
                candidates=tuple(step_candidates),
            )
            current_section_steps.append(step)
            current_section_candidates.extend(step_candidates)
            all_candidates.extend(step_candidates)
            continue

        # Check for bullet list item (e.g. - Bullet text)
        bullet_match = bullet_item_re.match(line)
        if bullet_match:
            item_text = bullet_match.group(1)
            step_index += 1

            location = PlanLocation(
                line_number=line_idx,
                section=current_section_heading,
                step_number=None,
                is_list_item=True,
            )
            step_candidates = extract_candidates(item_text, location)

            step = PlanStep(
                index=step_index,
                step_number=None,
                text=item_text.strip(),
                section=current_section_heading,
                line_number=line_idx,
                candidates=tuple(step_candidates),
            )
            current_section_steps.append(step)
            current_section_candidates.extend(step_candidates)
            all_candidates.extend(step_candidates)
            continue

        # Regular prose paragraph line
        location = PlanLocation(
            line_number=line_idx,
            section=current_section_heading,
            step_number=None,
            is_list_item=False,
        )
        line_candidates = extract_candidates(line, location)
        current_section_candidates.extend(line_candidates)
        all_candidates.extend(line_candidates)

    _flush_section()

    # Fallback title if document had no H1 heading
    if not title:
        if file_path:
            title = file_path.stem.replace("_", " ").title()
        elif sections:
            title = sections[0].heading
        else:
            title = "Untitled Plan"

    return ParsedPlan(
        file_path=file_path,
        title=title,
        sections=tuple(sections),
        all_candidates=tuple(all_candidates),
    )
