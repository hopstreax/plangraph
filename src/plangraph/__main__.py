"""Minimal CLI entry point for PlanGraph foundation verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plangraph import __version__
from plangraph.exceptions import PlanGraphError
from plangraph.graph import load_graph
from plangraph.plan import parse_plan


def main(argv: list[str] | None = None) -> int:
    """Entry point for the plangraph command-line interface."""
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
                print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
