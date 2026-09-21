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

        return VerificationResult(valid=not reasons, reasons=tuple(reasons))
