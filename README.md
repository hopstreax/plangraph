# PlanGraph

Validate your implementation plan against the real codebase before you code.

## Overview

PlanGraph is an offline, deterministic developer CLI tool that validates a contributor's implementation plan (`plan.md`) against a repository's code graph (generated via Graphify). It identifies referenced entities, direct codebase impact, surrounding dependencies, and ambiguities before a single line of code is written.

### Core Philosophy

The contributor owns the implementation plan and engineering decisions. PlanGraph does **not** automatically generate implementation plans, rewrite code, or replace the contributor's reasoning. Instead, it provides codebase-aware evidence to inform the contributor's judgment.

---

## Workflow

```text
GitHub Issue
     ↓
Contributor writes implementation plan (plan.md)
     ↓
PlanGraph parses plan (Candidate extraction)
     ↓
PlanGraph resolves references against Graphify code graph
     ↓
PlanGraph analyzes direct codebase impact
     ↓
Structured Impact Report (Human-readable or JSON)
```

---

## Usage

### 1. Check Graph

Verify that a Graphify `graph.json` artifact loads correctly:

```bash
plangraph check-graph path/to/graph.json
```

### 2. Parse Implementation Plan

Extract candidate entities (files, classes, methods, functions, symbols, concepts) from Markdown:

```bash
plangraph parse-plan plan.md
plangraph parse-plan plan.md --json
```

### 3. Resolve Plan References

Resolve extracted candidate entities against the Graphify code graph:

```bash
plangraph resolve-plan plan.md --graph path/to/graph.json
plangraph resolve-plan plan.md --graph path/to/graph.json --json
```

### 4. Analyze Direct Codebase Impact

Analyze direct (1-hop) relationships for all resolved entities in the plan:

```bash
plangraph analyze-plan plan.md --graph path/to/graph.json
# or alias
plangraph impact-plan plan.md --graph path/to/graph.json
```

Include standard-library and third-party external dependencies (hidden by default):

```bash
plangraph analyze-plan plan.md --graph path/to/graph.json --all-deps
```

For machine-readable CI/tooling output:

```bash
plangraph analyze-plan plan.md --graph path/to/graph.json --json
```

#### Example Output:

```text
PlanGraph Impact Analysis
Plan: User Service Caching

Candidates
  Resolved:     3
  Ambiguous:    1
  Unresolved:   1
  Context only: 1

Impacted entities: 8

UserService.get_user() (services/user.py:L25)
  defined_in → UserService
  calls → query() (db/database.py)
  called_by → UserController.show() (controllers/user.py)

user.py (services/user.py:L1)
  contains → 15 entities
    _init_cache()
    clear_cache()
    get_user()
    set_user()
    validate_user()
    ... (+ 10 more)
  imports → cache.py (common/cache.py)

Unresolved
  MissingCacheManager (Line 6)
    → no class or code node found with name 'MissingCacheManager' in repository

Ambiguous
  UserService (Line 2)
    - services/user.py:L20 (UserService)
    - models/user.py:L5 (UserService)

Context (Not Code Entities)
  • Redis (Line 3)
```

---

## Impact Analysis Principles

### What Impact Analysis Does
- **Discovers Direct Relationships**: Identifies direct (1-hop) outgoing dependencies (`calls`, `imports`, `contains`) and incoming dependents (`called_by`, `imported_by`, `defined_in`).
- **Qualifies Method Names**: Methods with an enclosing class are displayed as `ClassName.method()` (e.g. `Analyzer.process()`) rather than `.process()`.
- **Context-Aware Disambiguation**: Resolves symbols uniquely when an explicit, unambiguous file context is specified in the same plan step or section, while preserving ambiguity if multiple matches or no file context exist.
- **Default Dependency Filtering**: External third-party packages and standard library imports are hidden by default in terminal output to emphasize repository code. Pass `--all-deps` to inspect all external dependencies.
- **Containment Summarization**: Large containment listings (exceeding 10 child entities) are summarized in terminal output to avoid flooding, while `--json` retains 100% of relationship data.
- **Deduplicates Multi-Referenced Entities**: If multiple plan candidates reference the same codebase entity, it is analyzed once while preserving all originating candidate occurrences.
- **Preserves Uncertainty**: Ambiguous candidates and unresolved candidates are explicitly surfaced with line provenance rather than guessed.
- **Filters Noise**: Excludes internal documentation and rationale metadata (`rationale_for` edges) to keep reports focused on code impact.

### What Impact Analysis Does NOT Do
- **Does NOT Recursively Traverse**: Does not perform multi-hop graph expansion (e.g. `A → B → C → D`) that could lead to recursive noise.
- **Does NOT Score Risk**: Does not assign arbitrary risk scores or claim that an impact is a defect.
- **Does NOT Generate Code**: Does not suggest implementations or generate code for the contributor.
