from datetime import datetime, timezone

from enterprise_harness.observability.metrics import MetricCollector
from enterprise_harness.repositories import (
    ApprovalRepository,
    InMemoryApprovalRepository,
)

from .approval import ApprovalRequest, ApprovalStatus


class ApprovalManager:
    """负责审批请求的创建、查询和处理。"""

    def __init__(
        self,
        approval_repository: ApprovalRepository | None = None,
        metric_collector: MetricCollector | None = None,
    ):
        self._requests: dict[str, ApprovalRequest] = {}
        self._repository = (
            approval_repository
            or InMemoryApprovalRepository()
        )
        self._metric_collector = metric_collector

    async def create_request(
        self,
        run_id: str,
        tool_name: str,
        arguments: dict,
        requester_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ApprovalRequest:
        request = ApprovalRequest(
            run_id=run_id,
            tool_name=tool_name,
            arguments=arguments,
            requester_id=requester_id,
            tenant_id=tenant_id,
        )

        self._requests[request.approval_id] = request

        await self._repository.save(request)
        return request

    async def get_request(
        self,
        approval_id: str,
    ) -> ApprovalRequest:
        request = self._requests.get(approval_id)

        if request is None:
            request = await self._repository.get(approval_id)

        if request is None:
            raise KeyError(
                f"Approval request not found: {approval_id}"
            )

        return request

    async def approve(
        self,
        approval_id: str,
        comment: str | None = None,
    ) -> ApprovalRequest:
        request = await self.get_request(
            approval_id
        )

        self._ensure_pending(request)

        request.status = ApprovalStatus.APPROVED
        request.comment = comment
        request.resolved_at = datetime.now(timezone.utc)

        if self._metric_collector is not None:
            self._metric_collector.record_human_approval()

        await self._repository.save(request)
        return request

    async def reject(
        self,
        approval_id: str,
        comment: str | None = None,
    ) -> ApprovalRequest:
        request = await self.get_request(
            approval_id
        )

        self._ensure_pending(request)

        request.status = ApprovalStatus.REJECTED
        request.comment = comment
        request.resolved_at = datetime.now(timezone.utc)

        await self._repository.save(request)
        return request

    async def validate_approval(
        self,
        approval_id: str,
        run_id: str,
        tool_name: str,
        arguments: dict,
    ) -> ApprovalRequest:
        request = await self.get_request(
            approval_id
        )

        if request.status != ApprovalStatus.APPROVED:
            raise PermissionError(
                "Approval request is not approved"
            )

        if request.run_id != run_id:
            raise PermissionError(
                "Approval does not belong to this run"
            )

        if request.tool_name != tool_name:
            raise PermissionError(
                "Approval does not match tool"
            )

        if request.arguments != arguments:
            raise PermissionError(
                "Approval does not match arguments"
            )

        return request
    async def list_pending(
        self,
        tenant_id: str | None = None,
    ) -> list[ApprovalRequest]:
        requests = [
            request
            for request in self._requests.values()
            if request.status == ApprovalStatus.PENDING
        ]
        if tenant_id is not None:
            requests = [
                r for r in requests
                if r.tenant_id == tenant_id
            ]
        return requests

    @staticmethod
    def _ensure_pending(
        request: ApprovalRequest,
    ) -> None:
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                "Approval request has already been resolved"
            )