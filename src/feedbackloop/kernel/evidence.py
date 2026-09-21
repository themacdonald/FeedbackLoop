from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """An immutable artifact supporting a work claim or outcome."""

    evidence_id: str
    evidence_type: str
    source: str
    value: Any

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("evidence_type", self.evidence_type),
            ("source", self.source),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "value": self.value,
        }
