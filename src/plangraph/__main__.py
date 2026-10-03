"""Minimal CLI entry point for PlanGraph foundation verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plangraph import __version__
from plangraph.exceptions import PlanGraphError
from plangraph.graph import load_graph
from plangraph.plan import CandidateKind, parse_plan
from plangraph.impact import ImpactAnalysis, analyze_impact, format_impact_report
from plangraph.resolver import (
    ResolutionResult,
    ResolutionStatus,
    ResolvedEntity,
    resolve_plan,
)


def _status_icon(status: ResolutionStatus) -> str:
    icons = {
        ResolutionStatus.RESOLVED: "✓",
        ResolutionStatus.AMBIGUOUS: "?",
        ResolutionStatus.UNRESOLVED: "✗",
        ResolutionStatus.NOT_CODE_ENTITY: "•",
    }
    return icons.get(status, " ")


def _format_entity_resolution(entity: ResolvedEntity) -> str:
    if entity.status == ResolutionStatus.RESOLVED:
        if entity.candidate.kind == CandidateKind.FILE_PATH and entity.matched_nodes:
            sf = entity.matched_nodes[0].source_file or entity.candidate.text
            return f"resolved to {sf}"
        elif entity.matched_nodes:
            if len(entity.matched_nodes) == 1:
                n = entity.matched_nodes[0]
                loc = f":{n.source_location}" if n.source_location else ""
                return f"resolved to {n.source_file}{loc}::{n.label}"
            elif len(entity.matched_nodes) == 2 and entity.candidate.kind == CandidateKind.METHOD:
                cls_node, meth_node = entity.matched_nodes
                loc = f":{meth_node.source_location}" if meth_node.source_location else ""
                meth_clean = meth_node.label.lstrip(".")
                return f"resolved to {meth_node.source_file}{loc}::{cls_node.label}.{meth_clean}"
            else:
                targets = [f"{n.source_file}::{n.label}" for n in entity.matched_nodes]
                return f"resolved to {', '.join(targets)}"
        return f"resolved: {entity.resolution_reason}"
    elif entity.status == ResolutionStatus.AMBIGUOUS:
        return f"ambiguous: {entity.resolution_reason}"
    elif entity.status == ResolutionStatus.UNRESOLVED:
        return f"unresolved: {entity.resolution_reason}"
    elif entity.status == ResolutionStatus.NOT_CODE_ENTITY:
        return f"not a code entity; {entity.resolution_reason}"
    return entity.resolution_reason


def _emit_json(data: dict[str, Any]) -> None:
    """Emit JSON to stdout ensuring strict UTF-8 / ASCII compatibility across platforms."""
    payload = json.dumps(data, indent=2, ensure_ascii=True) + "\n"
    try:
        sys.stdout.buffer.write(payload.encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception:
        sys.stdout.write(payload)
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the plangraph command-line interface."""
    # Ensure stdout/stderr handle UTF-8 symbols gracefully on platforms like Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="plangraph",
        description="Validate your implementation plan against the real codebase before you code.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"plangraph {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Foundation check subcommand
    check_parser = subparsers.add_parser(
        "check-graph",
        help="Verify and inspect a Graphify graph.json artifact.",
    )
    check_parser.add_argument(
        "graph_path",
        type=str,
        help="Path to the graph.json file to validate.",
    )

    # Parse plan subcommand
    parse_parser = subparsers.add_parser(
        "parse-plan",
        help="Parse a contributor implementation plan and extract candidate entities.",
    )
    parse_parser.add_argument(
        "plan_path",
        type=str,
        help="Path to the implementation plan markdown file.",
    )
    parse_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit output as machine-readable JSON.",
    )

    # Resolve plan subcommand
    resolve_parser = subparsers.add_parser(
        "resolve-plan",
        help="Resolve plan candidates against a Graphify code graph.",
    )
    resolve_parser.add_argument(
        "plan_path",
        type=str,
        help="Path to the implementation plan markdown file.",
    )
    resolve_parser.add_argument(
        "-g",
        "--graph",
        dest="graph_path",
        required=True,
        type=str,
        help="Path to the Graphify graph.json artifact.",
    )
    resolve_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit output as machine-readable JSON.",
    )

    # Analyze plan impact subcommand
    analyze_parser = subparsers.add_parser(
        "analyze-plan",
        aliases=["impact-plan"],
        help="Analyze the direct codebase impact of a contributor implementation plan.",
    )
    analyze_parser.add_argument(
        "plan_path",
        type=str,
        help="Path to the implementation plan markdown file.",
    )
    analyze_parser.add_argument(
        "-g",
        "--graph",
        dest="graph_path",
        required=True,
        type=str,
        help="Path to the Graphify graph.json artifact.",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit output as machine-readable JSON.",
    )
    analyze_parser.add_argument(
        "--all-deps",
        action="store_true",
        help="Include standard-library and external dependency relationships in terminal output.",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "check-graph":
        target = Path(args.graph_path)
        try:
            graph = load_graph(target)
            print(f"Successfully loaded graph from: {target}")
            print(f"  • Total nodes: {graph.number_of_nodes()}")
            print(f"  • Total edges: {graph.number_of_edges()}")
            print(f"  • Graph type:  {type(graph).__name__}")
            return 0
        except PlanGraphError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.command == "parse-plan":
        target = Path(args.plan_path)
        try:
            plan = parse_plan(target)
            if args.json:
                _emit_json(plan.to_dict())
                return 0

            print(f"PlanGraph: Parsed Plan from {target}")
            print(f"Title: {plan.title}")
            print()
            print(f"Sections ({len(plan.sections)}):")
            for section in plan.sections:
                header_prefix = "#" * section.level
                print(f"  [{header_prefix} {section.heading}] (Line {section.line_number})")
                prose_candidates = [c for c in section.candidates if not c.location.is_list_item]
                for candidate in prose_candidates:
                    print(
                        f"    • {candidate.text} [{candidate.kind.value} | {candidate.confidence.value}] (Line {candidate.location.line_number})"
                    )
                if section.steps:
                    for step in section.steps:
                        step_label = (
                            f"Step {step.step_number}"
                            if step.step_number is not None
                            else f"Item {step.index}"
                        )
                        print(f"    {step_label} (Line {step.line_number}): {step.text}")
                        for candidate in step.candidates:
                            print(
                                f"      • {candidate.text} [{candidate.kind.value} | {candidate.confidence.value}]"
                            )
            print()
            print("Summary:")
            print(f"  Total Candidates: {len(plan.all_candidates)}")
            by_kind: dict[str, int] = {}
            for c in plan.all_candidates:
                by_kind[c.kind.value] = by_kind.get(c.kind.value, 0) + 1
            if by_kind:
                breakdown = ", ".join(f"{cnt} {k}" for k, cnt in sorted(by_kind.items()))
                print(f"  Breakdown: {breakdown}")
            return 0
        except PlanGraphError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.command == "resolve-plan":
        target_plan = Path(args.plan_path)
        target_graph = Path(args.graph_path)
        try:
            result = resolve_plan(target_plan, target_graph)
            if args.json:
                _emit_json(result.to_dict())
                return 0

            print("PlanGraph: Plan Resolution")
            print()
            for entity in result.entities:
                icon = _status_icon(entity.status)
                kind_str = entity.candidate.kind.value
                line_info = (
                    f" (Line {entity.candidate.location.line_number})"
                    if entity.candidate.location
                    else ""
                )
                print(f"{icon} {entity.candidate.text}")
                print(f"  {kind_str}{line_info}")
                detail = _format_entity_resolution(entity)
                print(f"  → {detail}")
                print()

            print("Summary:")
            print(f"  Total Candidates:    {len(result.entities)}")
            print(f"  • Resolved:          {len(result.resolved)}")
            print(f"  • Ambiguous:         {len(result.ambiguous)}")
            print(f"  • Unresolved:        {len(result.unresolved)}")
            print(f"  • Not Code Entities: {len(result.not_code_entities)}")
            return 0
        except PlanGraphError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.command in ("analyze-plan", "impact-plan"):
        target_plan = Path(args.plan_path)
        target_graph = Path(args.graph_path)
        try:
            analysis = analyze_impact(target_plan, target_graph)
            if args.json:
                _emit_json(analysis.to_dict())
                return 0

            print(format_impact_report(analysis, all_deps=args.all_deps), end="")
            return 0
        except PlanGraphError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
