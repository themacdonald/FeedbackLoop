from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class WorkEvent:
    event_type: str
    work_id: str
    actor_id: str
    from_state: str
    to_state: str
    occurred_at: str
    details: dict[str, Any]

    @classmethod
    def create(cls, *, event_type: str, work_id: str, actor_id: str,
               from_state: str, to_state: str,
               details: dict[str, Any] | None = None) -> "WorkEvent":
        return cls(
            event_type=event_type, work_id=work_id, actor_id=actor_id,
            from_state=from_state, to_state=to_state,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            details=dict(details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type, "work_id": self.work_id,
            "actor_id": self.actor_id, "from_state": self.from_state,
            "to_state": self.to_state, "occurred_at": self.occurred_at,
            "details": dict(self.details),
        }
