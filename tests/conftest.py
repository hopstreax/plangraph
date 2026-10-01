"""Pytest fixtures for PlanGraph unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def synthetic_graph_dict() -> dict[str, Any]:
    """Return a dictionary representing a realistic code graph topology.

    Topology:
        UserController (controllers/user.py)
            │ [calls]
            ▼
        UserService (services/user.py)
            │ [calls]
            ▼
        UserRepository (repositories/user.py)

    And test coverage:
        test_user_service (tests/test_user_service.py)
            │ [calls]
            ▼
        UserService (services/user.py)
    """
    return {
        "directed": True,
        "multigraph": False,
        "built_at_commit": "a1b2c3d4",
        "nodes": [
            {
                "id": "controllers_user_usercontroller",
                "label": "UserController",
                "file_type": "code",
                "source_file": "controllers/user.py",
                "source_location": "L15",
                "community": 1,
                "community_name": "user_api",
                "_callable": False,
            },
            {
                "id": "services_user_userservice",
                "label": "UserService",
                "file_type": "code",
                "source_file": "services/user.py",
                "source_location": "L25",
                "community": 1,
                "community_name": "user_api",
                "_callable": False,
            },
            {
                "id": "repositories_user_userrepository",
                "label": "UserRepository",
                "file_type": "code",
                "source_file": "repositories/user.py",
                "source_location": "L10",
                "community": 2,
                "community_name": "database",
                "_callable": False,
            },
            {
                "id": "tests_test_user_service_test_user_service",
                "label": "test_user_service",
                "file_type": "code",
                "source_file": "tests/test_user_service.py",
                "source_location": "L8",
                "community": 3,
                "community_name": "tests",
                "_callable": True,
            },
        ],
        "links": [
            {
                "source": "controllers_user_usercontroller",
                "target": "services_user_userservice",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "controllers/user.py",
                "source_location": "L30",
            },
            {
                "source": "services_user_userservice",
                "target": "repositories_user_userrepository",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "services/user.py",
                "source_location": "L45",
            },
            {
                "source": "tests_test_user_service_test_user_service",
                "target": "services_user_userservice",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "tests/test_user_service.py",
                "source_location": "L12",
            },
        ],
    }


@pytest.fixture
def synthetic_graph_file(tmp_path: Path, synthetic_graph_dict: dict[str, Any]) -> Path:
    """Write the synthetic graph dictionary using the 'links' key to a temporary file."""
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(synthetic_graph_dict), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_graph_edges_file(tmp_path: Path, synthetic_graph_dict: dict[str, Any]) -> Path:
    """Write the synthetic graph dictionary using the 'edges' key to a temporary file."""
    graph_copy = dict(synthetic_graph_dict)
    graph_copy["edges"] = graph_copy.pop("links")
    path = tmp_path / "graph_edges.json"
    path.write_text(json.dumps(graph_copy), encoding="utf-8")
    return path
