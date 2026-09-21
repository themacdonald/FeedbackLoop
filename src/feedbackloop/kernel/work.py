from dataclasses import dataclass, field
from typing import Any

from .authority import Authority
from .capability import Capability
from .evidence import Evidence
from .event import WorkEvent
from .intent import Intent
from .outcome import Outcome
from .state import WorkState

@dataclass
class Work:
    work_id: str
    intent: Intent
    capability: Capability
    authority: Authority
    required_evidence: frozenset[str] = frozenset()
    state: WorkState = WorkState.PROPOSED
    evidence: list[Evidence] = field(default_factory=list)
    outcome: Outcome | None = None
    events: list[WorkEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.work_id.strip():
            raise ValueError("work_id cannot be empty.")
        if not self.authority.permits(self.authority.actor_id, self.capability.name):
            raise ValueError("Authority does not grant the declared capability.")

    def evidence_types(self) -> frozenset[str]:
        return frozenset(x.evidence_type for x in self.evidence)

    def has_required_evidence(self) -> bool:
        return self.required_evidence.issubset(self.evidence_types())

    def add_evidence(self, evidence: Evidence) -> None:
        if self.state in {WorkState.COMPLETED, WorkState.FAILED, WorkState.CANCELLED}:
            raise ValueError(f"Cannot add evidence in terminal state: {self.state.value}")
        if any(x.evidence_id == evidence.evidence_id for x in self.evidence):
            raise ValueError(f"Duplicate evidence id: {evidence.evidence_id}")
        self.evidence.append(evidence)

    def record_event(self, event: WorkEvent) -> None:
        if self.events and event.work_id != self.work_id:
            raise ValueError("Event work_id does not match Work.")
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "intent": self.intent.to_dict(),
            "capability": self.capability.name,
            "authority": {"actor_id": self.authority.actor_id,
                          "capabilities": sorted(self.authority.capabilities)},
            "required_evidence": sorted(self.required_evidence),
            "state": self.state.value,
            "evidence": [x.to_dict() for x in self.evidence],
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "events": [x.to_dict() for x in self.events],
        }
