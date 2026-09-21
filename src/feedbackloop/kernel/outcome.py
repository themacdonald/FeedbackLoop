from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Outcome:
    status: str
    values: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"success", "failure", "partial"}:
            raise ValueError("Outcome status must be one of ['failure', 'partial', 'success'].")

    @property
    def successful(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "values": dict(self.values)}
