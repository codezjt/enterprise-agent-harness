import pytest

from enterprise_harness.policy import (
    DEFAULT_ROLES,
    Principal,
    RBAC,
)


def create_rbac() -> RBAC:
    return RBAC(DEFAULT_ROLES)


def test_viewer_can_read_order():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="viewer",
    )

    assert rbac.has_permission(
        principal,
        "order:read",
    )


def test_viewer_cannot_update_order():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="viewer",
    )

    assert not rbac.has_permission(
        principal,
        "order:update",
    )


def test_operator_can_update_order():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="operator",
    )

    assert rbac.has_permission(
        principal,
        "order:read",
    )

    assert rbac.has_permission(
        principal,
        "order:update",
    )


def test_operator_cannot_cancel_order():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="operator",
    )

    assert not rbac.has_permission(
        principal,
        "order:cancel",
    )


def test_manager_can_cancel_order():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="manager",
    )

    assert rbac.has_permission(
        principal,
        "order:read",
    )

    assert rbac.has_permission(
        principal,
        "order:update",
    )

    assert rbac.has_permission(
        principal,
        "order:cancel",
    )


def test_manager_cannot_refund_order():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="manager",
    )

    assert not rbac.has_permission(
        principal,
        "order:refund",
    )


def test_admin_can_all_permissions():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="admin",
    )

    permissions = [
        "order:read",
        "order:update",
        "order:cancel",
        "order:refund",
    ]

    for permission in permissions:
        assert rbac.has_permission(
            principal,
            permission,
        )


def test_unknown_role_has_no_permission():
    rbac = create_rbac()

    principal = Principal(
        principal_id="user-001",
        role="unknown",
    )

    assert not rbac.has_permission(
        principal,
        "order:read",
    )


def test_get_role():
    rbac = create_rbac()

    role = rbac.get_role("manager")

    assert role.name == "manager"
    assert "order:read" in role.permissions


def test_get_unknown_role():
    rbac = create_rbac()

    with pytest.raises(KeyError):
        rbac.get_role("unknown")