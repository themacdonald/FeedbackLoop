from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Intent:
    """The desired result of a piece of work."""

    objective: str
    success_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("Intent objective cannot be empty.")
        if any(not condition.strip() for condition in self.success_conditions):
            raise ValueError("Success conditions cannot be empty strings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "success_conditions": list(self.success_conditions),
        }
