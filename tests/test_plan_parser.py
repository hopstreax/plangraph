"""Deterministic unit tests for PlanGraph Phase 2 plan parser and candidate extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from plangraph.__main__ import main
from plangraph.exceptions import PlanParseError
from plangraph.plan import (
    CandidateKind,
    Confidence,
    ParsedPlan,
    PlanLocation,
    extract_candidates,
    parse_plan,
)


def test_simple_heading_and_paragraph() -> None:
    """Test parsing a single heading and a basic prose paragraph."""
    md = """# Caching Plan
This document outlines caching improvements for the system.
"""
    plan = parse_plan(md)

    assert plan.title == "Caching Plan"
    assert len(plan.sections) == 1
    assert plan.sections[0].heading == "Caching Plan"
    assert plan.sections[0].level == 1
    assert plan.sections[0].line_number == 1
    assert "caching improvements" in plan.sections[0].content


def test_numbered_steps() -> None:
    """Test extracting numbered list items with exact step numbers."""
    md = """# Plan
## Implementation Steps
1. Modify `UserService` to add caching.
2. Update `UserController` endpoint.
"""
    plan = parse_plan(md)

    assert len(plan.sections) == 1
    section = plan.sections[0]
    assert len(section.steps) == 2

    step1 = section.steps[0]
    assert step1.step_number == 1
    assert step1.index == 1
    assert "UserService" in step1.text
    assert len(step1.candidates) >= 1
    assert any(c.text == "UserService" for c in step1.candidates)

    step2 = section.steps[1]
    assert step2.step_number == 2
    assert step2.index == 2
    assert "UserController" in step2.text


def test_bullet_lists() -> None:
    """Test extracting bullet list items (- and *)."""
    md = """## Tasks
- Add `common/cache.py`
* Update `services/user.py`
+ Configure Redis settings
"""
    plan = parse_plan(md)

    section = plan.sections[0]
    assert len(section.steps) == 3
    # Step numbers are None for unnumbered bullets
    assert all(s.step_number is None for s in section.steps)
    assert section.steps[0].candidates[0].text == "common/cache.py"
    assert section.steps[1].candidates[0].text == "services/user.py"
    assert any(c.text == "Redis" for c in section.steps[2].candidates)


def test_backtick_symbols() -> None:
    """Test that backticked identifiers receive EXPLICIT confidence and correct kinds."""
    md = "Refactor `UserService` and update constant `MAX_RETRIES`."
    plan = parse_plan(md)

    cands = {c.text: c for c in plan.all_candidates}
    assert "UserService" in cands
    assert cands["UserService"].kind == CandidateKind.CLASS
    assert cands["UserService"].confidence == Confidence.EXPLICIT
    assert cands["UserService"].raw_text == "`UserService`"

    assert "MAX_RETRIES" in cands
    assert cands["MAX_RETRIES"].kind == CandidateKind.SYMBOL
    assert cands["MAX_RETRIES"].confidence == Confidence.EXPLICIT


def test_backtick_file_paths() -> None:
    """Test that backticked file paths receive FILE_PATH kind and EXPLICIT confidence."""
    md = "Check `services/user.py` and `config/settings.json`."
    plan = parse_plan(md)

    cands = {c.text: c for c in plan.all_candidates}
    assert "services/user.py" in cands
    assert cands["services/user.py"].kind == CandidateKind.FILE_PATH
    assert cands["services/user.py"].confidence == Confidence.EXPLICIT

    assert "config/settings.json" in cands
    assert cands["config/settings.json"].kind == CandidateKind.FILE_PATH


def test_inferred_pascal_case_symbols() -> None:
    """Test that unquoted PascalCase words in prose are extracted as CLASS with INFERRED confidence."""
    md = "We should update UserService and introduce CacheManager."
    plan = parse_plan(md)

    cands = {c.text: c for c in plan.all_candidates}
    assert "UserService" in cands
    assert cands["UserService"].kind == CandidateKind.CLASS
    assert cands["UserService"].confidence == Confidence.INFERRED

    assert "CacheManager" in cands
    assert cands["CacheManager"].kind == CandidateKind.CLASS
    assert cands["CacheManager"].confidence == Confidence.INFERRED


def test_method_calls() -> None:
    """Test extraction of method calls both in prose and in backticks."""
    md = """
1. Call `UserService.get_user()` in handler.
2. Ensure UserRepository.find_by_id returns a model.
"""
    plan = parse_plan(md)

    cands = [c for c in plan.all_candidates if c.kind == CandidateKind.METHOD]
    assert len(cands) == 2
    assert any(c.text == "UserService.get_user()" for c in cands)
    assert any(c.text == "UserRepository.find_by_id" for c in cands)


def test_function_calls() -> None:
    """Test extraction of function calls with parentheses."""
    md = "We must invoke `validate_token()` and check hash_password()."
    plan = parse_plan(md)

    cands = [c for c in plan.all_candidates if c.kind == CandidateKind.FUNCTION]
    assert len(cands) == 2
    texts = {c.text for c in cands}
    assert "validate_token()" in texts
    assert "hash_password()" in texts


def test_posix_paths() -> None:
    """Test detection of unquoted POSIX file paths."""
    md = "Inspect services/user.py and tests/test_user.py."
    plan = parse_plan(md)

    paths = [c.text for c in plan.all_candidates if c.kind == CandidateKind.FILE_PATH]
    assert "services/user.py" in paths
    assert "tests/test_user.py" in paths


def test_windows_paths_normalized() -> None:
    """Test that Windows backslash paths are normalized to POSIX while preserving raw_text."""
    md = r"Examine services\user.py and common\cache.py for details."
    plan = parse_plan(md)

    cands = {c.text: c for c in plan.all_candidates if c.kind == CandidateKind.FILE_PATH}
    assert "services/user.py" in cands
    assert cands["services/user.py"].raw_text == r"services\user.py"

    assert "common/cache.py" in cands
    assert cands["common/cache.py"].raw_text == r"common\cache.py"


def test_punctuation_trimming() -> None:
    """Test that trailing sentence punctuation is stripped from file paths."""
    md = "Look at services/user.py, common/cache.py; also check tests/test_user.py!"
    plan = parse_plan(md)

    paths = [c.text for c in plan.all_candidates if c.kind == CandidateKind.FILE_PATH]
    assert "services/user.py" in paths
    assert "common/cache.py" in paths
    assert "tests/test_user.py" in paths
    # Verify no stray punctuation was included
    assert not any(p.endswith((".", ",", ";", "!")) for p in paths)


def test_sentence_starter_suppression() -> None:
    """Test that ordinary English action verbs starting a sentence are not treated as classes,
    while valid PascalCase symbols at sentence start ARE preserved."""
    md = """
Add caching to the profile endpoint.
Create a new store.
Update the database schema.
UserService should now validate user credentials.
"""
    plan = parse_plan(md)

    names = {c.text for c in plan.all_candidates}
    # Verbs must be suppressed
    assert "Add" not in names
    assert "Create" not in names
    assert "Update" not in names

    # True PascalCase symbol starting a sentence must be extracted
    assert "UserService" in names


def test_concept_extraction() -> None:
    """Test extraction of architectural / technology concept terms."""
    md = "Implement Redis caching with Docker and PostgreSQL, protected by JWT middleware."
    plan = parse_plan(md)

    concepts = {c.text: c for c in plan.all_candidates if c.kind == CandidateKind.CONCEPT}
    assert "Redis" in concepts
    assert "caching" in concepts
    assert "Docker" in concepts
    assert "PostgreSQL" in concepts
    assert "JWT" in concepts
    assert "middleware" in concepts
    assert all(c.confidence == Confidence.CONCEPT for c in concepts.values())


def test_location_and_context_tracking() -> None:
    """Test that candidates preserve accurate 1-indexed line numbers and step indices."""
    md = """# Title
## Changes
1. Update `UserService` in services/user.py.
"""
    plan = parse_plan(md)

    cand = next(c for c in plan.all_candidates if c.text == "UserService")
    assert cand.location.line_number == 3
    assert cand.location.section == "Changes"
    assert cand.location.step_number == 1
    assert cand.location.is_list_item is True
    assert "Update `UserService`" in cand.context_sentence


def test_fenced_code_block_exclusion() -> None:
    """Test that code inside fenced blocks (```) does NOT produce candidate entities."""
    md = """# Plan
## Overview
Modify `UserService`.

```python
# This should NOT be extracted
class FakeService:
    def fake_func(self):
        pass
```

Finally update `UserController`.
"""
    plan = parse_plan(md)

    cands = [c.text for c in plan.all_candidates]
    assert "UserService" in cands
    assert "UserController" in cands
    # Code block contents must be omitted
    assert "FakeService" not in cands
    assert "fake_func()" not in cands


def test_empty_and_whitespace_markdown() -> None:
    """Test that empty or whitespace-only input parses safely with zero candidates."""
    plan_empty = parse_plan("")
    assert plan_empty.title == "Untitled Plan"
    assert len(plan_empty.sections) == 0
    assert len(plan_empty.all_candidates) == 0

    plan_ws = parse_plan("   \n\n\t   \n")
    assert len(plan_ws.sections) == 0
    assert len(plan_ws.all_candidates) == 0


def test_duplicate_textual_references_retain_separate_provenance() -> None:
    """Test that multiple mentions of the same symbol across different lines/steps
    retain their distinct provenance and are NOT collapsed."""
    md = """## Changes
1. First modify `UserService`.
2. Later re-test `UserService` with mock storage.
"""
    plan = parse_plan(md)

    user_service_cands = [c for c in plan.all_candidates if c.text == "UserService"]
    assert len(user_service_cands) == 2

    # Step 1 mention
    assert user_service_cands[0].location.step_number == 1
    assert user_service_cands[0].location.line_number == 2

    # Step 2 mention
    assert user_service_cands[1].location.step_number == 2
    assert user_service_cands[1].location.line_number == 3


def test_realistic_plan_fixture() -> None:
    """End-to-end integration test on a realistic contributor plan."""
    md = """# Add Redis caching to UserService

## Summary
Add Redis caching to reduce latency on profile lookups.

## Changes
1. Update `UserService` in `services/user.py` to add Redis-backed caching.
2. Add a `CacheManager` in `common/cache.py`.
3. Update `UserController` to use the cached service.
4. Add unit tests in `tests/test_user_service.py`.
"""
    plan = parse_plan(md)

    assert plan.title == "Add Redis caching to UserService"
    assert len(plan.sections) == 2

    changes_sec = next(s for s in plan.sections if s.heading == "Changes")
    assert len(changes_sec.steps) == 4

    # Verify extracted entities
    all_texts = {c.text for c in plan.all_candidates}
    assert "UserService" in all_texts
    assert "services/user.py" in all_texts
    assert "CacheManager" in all_texts
    assert "common/cache.py" in all_texts
    assert "UserController" in all_texts
    assert "tests/test_user_service.py" in all_texts
    assert "Redis" in all_texts
    assert "caching" in all_texts


def test_url_rejection() -> None:
    """Test that web URLs are not accidentally parsed as file paths."""
    md = "Refer to https://github.com/org/repo/blob/main/docs/api.py for specifications."
    plan = parse_plan(md)

    paths = [c.text for c in plan.all_candidates if c.kind == CandidateKind.FILE_PATH]
    assert len(paths) == 0


def test_parse_plan_file_input(tmp_path: Path) -> None:
    """Test parse_plan loading from a file path."""
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# My Plan\n\n1. Modify `AuthService`.\n", encoding="utf-8")

    plan = parse_plan(plan_file)
    assert plan.title == "My Plan"
    assert plan.file_path == plan_file
    assert len(plan.all_candidates) == 1
    assert plan.all_candidates[0].text == "AuthService"


def test_parse_plan_missing_file_raises_error(tmp_path: Path) -> None:
    """Test that a non-existent file path raises PlanParseError."""
    missing = tmp_path / "nonexistent_plan.md"
    with pytest.raises(PlanParseError, match="Failed to read plan file"):
        parse_plan(missing)


def test_multi_token_shell_command_in_backticks() -> None:
    """Test that multi-token shell commands in backticks extract the file path without fake paths."""
    md = "Run `pytest tests/test_user.py`."
    plan = parse_plan(md)

    assert len(plan.all_candidates) == 1
    cand = plan.all_candidates[0]
    assert cand.text == "tests/test_user.py"
    assert cand.raw_text == "`pytest tests/test_user.py`"
    assert cand.kind == CandidateKind.FILE_PATH
    assert cand.confidence == Confidence.EXPLICIT


def test_multi_token_python_command_in_backticks() -> None:
    """Test that python command in backticks extracts script.py as FILE_PATH."""
    md = "Execute `python script.py` to seed data."
    plan = parse_plan(md)

    assert len(plan.all_candidates) == 1
    cand = plan.all_candidates[0]
    assert cand.text == "script.py"
    assert cand.raw_text == "`python script.py`"
    assert cand.kind == CandidateKind.FILE_PATH


def test_cli_flag_config_command_in_backticks() -> None:
    """Test that CLI flag with config in backticks extracts config.yaml as FILE_PATH."""
    md = "Use `--config config.yaml` to specify settings."
    plan = parse_plan(md)

    assert len(plan.all_candidates) == 1
    cand = plan.all_candidates[0]
    assert cand.text == "config.yaml"
    assert cand.raw_text == "`--config config.yaml`"
    assert cand.kind == CandidateKind.FILE_PATH


def test_multi_token_backticks_without_file_paths_ignored() -> None:
    """Test that commands without file paths produce no spurious candidates."""
    md = "Run `npm run build` and then check `git status`."
    plan = parse_plan(md)

    assert len(plan.all_candidates) == 0


def test_candidate_ordering_repeated_symbols_on_same_line() -> None:
    """Test that multiple occurrences of the same token on one line preserve document order."""
    md = "Call `foo` on `bar` and then call `foo` again."
    plan = parse_plan(md)

    assert [c.text for c in plan.all_candidates] == ["foo", "bar", "foo"]


def test_candidate_ordering_repeated_prose_classes_on_same_line() -> None:
    """Test that repeated unquoted PascalCase symbols preserve exact document order."""
    md = "UserService interacts with CacheManager and returns UserService instance."
    plan = parse_plan(md)

    assert [c.text for c in plan.all_candidates] == [
        "UserService",
        "CacheManager",
        "UserService",
    ]


def test_parse_plan_missing_string_path_raises_error() -> None:
    """Test that a non-existent .md path string raises PlanParseError instead of parsing as prose."""
    with pytest.raises(PlanParseError, match="Failed to read plan file"):
        parse_plan("missing_plan.md")


def test_parse_plan_one_line_markdown_prose() -> None:
    """Test that single-line Markdown prose strings parse successfully without being treated as file paths."""
    plan1 = parse_plan("Update UserService.")
    assert plan1.title == "Overview"
    assert len(plan1.all_candidates) == 1
    assert plan1.all_candidates[0].text == "UserService"

    plan2 = parse_plan("# Add Redis caching")
    assert plan2.title == "Add Redis caching"

    plan3 = parse_plan("1. Modify services/user.py")
    assert len(plan3.all_candidates) == 1
    assert plan3.all_candidates[0].text == "services/user.py"


def test_parse_plan_existing_valid_string_path(tmp_path: Path) -> None:
    """Test that an existing valid file path passed as a string loads correctly."""
    plan_file = tmp_path / "valid_plan.md"
    plan_file.write_text("# Valid Plan\n1. Modify `AuthService`.\n", encoding="utf-8")

    plan = parse_plan(str(plan_file))
    assert plan.title == "Valid Plan"
    assert plan.file_path == plan_file
    assert len(plan.all_candidates) == 1
    assert plan.all_candidates[0].text == "AuthService"


def test_cli_mixed_section_renders_all_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that mixed sections containing both prose candidates and steps display all candidates."""
    plan_file = tmp_path / "mixed_plan.md"
    plan_file.write_text(
        "## Changes\n\nThe BaseService coordinates the operation.\n\n"
        "1. Update UserService.\n2. Add CacheManager.\n",
        encoding="utf-8",
    )

    exit_code = main(["parse-plan", str(plan_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "BaseService" in captured.out
    assert "UserService" in captured.out
    assert "CacheManager" in captured.out
    assert "Total Candidates: 3" in captured.out


def test_parse_plan_single_inline_code_span() -> None:
    """Test that a single-line string with backtick code is parsed as prose rather than a file path."""
    plan = parse_plan("`services/user.py`")
    assert len(plan.all_candidates) == 1
    assert plan.all_candidates[0].text == "services/user.py"
    assert plan.all_candidates[0].kind == CandidateKind.FILE_PATH
