from dataclasses import dataclass

from feedbackloop.kernel.work import Work


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"valid": self.valid, "reasons": list(self.reasons)}


class WorkVerifier:
    """Independently verifies whether a Work object satisfies its contract."""

    def verify(self, work: Work) -> VerificationResult:
        reasons: list[str] = []

        if not work.intent.objective.strip():
            reasons.append("intent_missing")

        if not work.authority.permits(work.authority.actor_id, work.capability):
            reasons.append("capability_not_authorized")

        missing = work.required_evidence - work.evidence_types()
        if missing:
            reasons.append(f"missing_evidence:{','.join(sorted(missing))}")

        if work.outcome is None:
            reasons.append("outcome_missing")
        elif not work.outcome.successful:
            reasons.append("outcome_not_successful")

        if work.state.value != "completed":
            reasons.append(f"work_not_completed:{work.state.value}")

        if work.events:
            if work.events[0].from_state != "proposed":
                reasons.append("event_history_invalid:initial_state")
            previous = work.events[0].from_state
            for event in work.events:
                if event.work_id != work.work_id:
                    reasons.append("event_history_invalid:work_id")
                    break
                if event.from_state != previous:
                    reasons.append("event_history_invalid:state_chain")
                    break
                previous = event.to_state
            if previous != work.state.value:
                reasons.append("event_history_invalid:terminal_state")
        elif work.state.value != "proposed":
            reasons.append("event_history_missing")

        return VerificationResult(valid=not reasons, reasons=tuple(reasons))
