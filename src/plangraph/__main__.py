"""Minimal CLI entry point for PlanGraph foundation verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plangraph import __version__
from plangraph.exceptions import PlanGraphError
from plangraph.graph import load_graph


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

    return 0


if __name__ == "__main__":
    sys.exit(main())
