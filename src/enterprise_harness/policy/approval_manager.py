from datetime import datetime, timezone

from .approval import ApprovalRequest, ApprovalStatus


class ApprovalManager:
    """负责审批请求的创建、查询和处理。"""

    def __init__(self):
        self._requests: dict[str, ApprovalRequest] = {}

    async def create_request(
        self,
        run_id: str,
        tool_name: str,
        arguments: dict,
        requester_id: str | None = None,
    ) -> ApprovalRequest:
        request = ApprovalRequest(
            run_id=run_id,
            tool_name=tool_name,
            arguments=arguments,
            requester_id=requester_id,
        )

        self._requests[request.approval_id] = request

        return request

    async def get_request(
        self,
        approval_id: str,
    ) -> ApprovalRequest:
        request = self._requests.get(approval_id)

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

        return request

    async def list_pending(
        self,
    ) -> list[ApprovalRequest]:
        return [
            request
            for request in self._requests.values()
            if request.status == ApprovalStatus.PENDING
        ]

    @staticmethod
    def _ensure_pending(
        request: ApprovalRequest,
    ) -> None:
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                "Approval request has already been resolved"
            )