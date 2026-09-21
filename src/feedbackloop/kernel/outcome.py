from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Outcome:
    """The observed result of executing work."""

    status: str
    values: dict[str, Any]

    def __post_init__(self) -> None:
        allowed = {"success", "failure", "partial"}
        if self.status not in allowed:
            raise ValueError(f"Outcome status must be one of {sorted(allowed)}.")

    @property
    def successful(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "values": dict(self.values)}
