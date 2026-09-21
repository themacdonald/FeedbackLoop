from enum import StrEnum


class WorkState(StrEnum):
    PROPOSED = "proposed"
    AUTHORIZED = "authorized"
    RUNNING = "running"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
