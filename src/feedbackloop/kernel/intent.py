from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Intent:
    objective: str
    success_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("Intent objective cannot be empty.")
        if any(not x.strip() for x in self.success_conditions):
            raise ValueError("Success conditions cannot be empty strings.")

    def to_dict(self) -> dict[str, Any]:
        return {"objective": self.objective, "success_conditions": list(self.success_conditions)}
