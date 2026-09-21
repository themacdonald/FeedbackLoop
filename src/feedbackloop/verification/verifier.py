from dataclasses import dataclass
from feedbackloop.kernel.state import WorkState
from feedbackloop.kernel.work import Work
from feedbackloop.runtime.runtime import WorkRuntime

@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"valid": self.valid, "reasons": list(self.reasons)}

class WorkVerifier:
    def verify(self, work: Work) -> VerificationResult:
        reasons: list[str] = []
        if not work.intent.objective.strip():
            reasons.append("intent_missing")
        if not work.authority.permits(work.authority.actor_id, work.capability.name):
            reasons.append("capability_not_authorized")
        missing = work.required_evidence - work.evidence_types()
        if missing:
            reasons.append(f"missing_evidence:{','.join(sorted(missing))}")
        if work.outcome is None:
            reasons.append("outcome_missing")
        elif not work.outcome.successful:
            reasons.append("outcome_not_successful")
        if work.state != WorkState.COMPLETED:
            reasons.append(f"work_not_completed:{work.state.value}")
        reasons.extend(self._verify_events(work))
        return VerificationResult(valid=not reasons, reasons=tuple(reasons))

    def _verify_events(self, work: Work) -> list[str]:
        reasons: list[str] = []
        expected = WorkState.PROPOSED
        transitions = WorkRuntime._TRANSITIONS
        for index, event in enumerate(work.events):
            if event.work_id != work.work_id:
                reasons.append(f"event_work_id_mismatch:{index}")
                continue
            if event.from_state != expected.value:
                reasons.append(f"event_chain_broken:{index}")
            try:
                from_state = WorkState(event.from_state)
                to_state = WorkState(event.to_state)
            except ValueError:
                reasons.append(f"event_unknown_state:{index}")
                continue
            if to_state not in transitions[from_state]:
                reasons.append(f"illegal_event_transition:{index}")
            expected = to_state
        if work.state.value != expected.value:
            reasons.append("event_state_mismatch")
        return reasons
