"""Unit tests for PlanGraph internal models."""

from __future__ import annotations

from plangraph.models import Edge, ImpactEvidence, Node


def test_node_from_dict() -> None:
    """Test Node instantiation from a dictionary."""
    data = {
        "id": "services_user_userservice",
        "label": "UserService",
        "file_type": "code",
        "source_file": "services/user.py",
        "source_location": "L42",
        "community": "5",
        "community_name": "user_module",
        "_callable": True,
        "custom_attr": "extra",
    }
    node = Node.from_dict(data)

    assert node.id == "services_user_userservice"
    assert node.label == "UserService"
    assert node.file_type == "code"
    assert node.source_file == "services/user.py"
    assert node.source_location == "L42"
    assert node.community == 5
    assert node.community_name == "user_module"
    assert node.is_callable is True
    assert node.metadata.get("custom_attr") == "extra"


def test_edge_from_dict() -> None:
    """Test Edge instantiation from a dictionary."""
    data = {
        "source": "controller",
        "target": "service",
        "relation": "calls",
        "confidence": "EXTRACTED",
        "source_file": "controller.py",
        "source_location": "L10",
        "context": "call",
    }
    edge = Edge.from_dict(data)

    assert edge.source == "controller"
    assert edge.target == "service"
    assert edge.relation == "calls"
    assert edge.confidence == "EXTRACTED"
    assert edge.source_file == "controller.py"
    assert edge.source_location == "L10"
    assert edge.context == "call"


def test_impact_evidence_format() -> None:
    """Test formatting of ImpactEvidence."""
    evidence = ImpactEvidence(
        source_id="UserController",
        target_id="UserService",
        relation="calls",
        source_file="controllers/user.py",
        source_location="L35",
        description="Direct invocation in login handler",
    )
    formatted = evidence.format()

    assert "UserController --[calls]--> UserService" in formatted
    assert "[controllers/user.py:L35]" in formatted
    assert "(Direct invocation in login handler)" in formatted
