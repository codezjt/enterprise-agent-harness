from dataclasses import dataclass, field


@dataclass(frozen=True)
class Role:
    """RBAC 角色。"""

    name: str
    permissions: frozenset[str] = field(
        default_factory=frozenset
    )


@dataclass(frozen=True)
class Principal:
    """执行 Tool 的主体。"""

    principal_id: str
    role: str


class RBAC:
    """第一版简单 RBAC 权限控制。"""

    def __init__(
        self,
        roles: list[Role] | None = None,
    ):
        self._roles = {
            role.name: role
            for role in (roles or [])
        }

    def has_permission(
        self,
        principal: Principal,
        permission: str,
    ) -> bool:
        role = self._roles.get(principal.role)

        if role is None:
            return False

        return permission in role.permissions

    def get_role(self, role_name: str) -> Role:
        role = self._roles.get(role_name)

        if role is None:
            raise KeyError(
                f"Role not found: {role_name}"
            )

        return role
DEFAULT_ROLES = [
    Role(
        name="viewer",
        permissions=frozenset(
            {
                "order:read",
            }
        ),
    ),
    Role(
        name="operator",
        permissions=frozenset(
            {
                "order:read",
                "order:update",
            }
        ),
    ),
    Role(
        name="manager",
        permissions=frozenset(
            {
                "order:read",
                "order:update",
                "order:cancel",
            }
        ),
    ),
    Role(
        name="admin",
        permissions=frozenset(
            {
                "order:read",
                "order:update",
                "order:cancel",
                "order:refund",
            }
        ),
    ),
]