from feedbackloop.kernel.authority import Authority
from feedbackloop.kernel.capability import Capability
from feedbackloop.kernel.event import WorkEvent
from feedbackloop.kernel.outcome import Outcome
from feedbackloop.kernel.state import WorkState
from feedbackloop.kernel.work import Work


class WorkRuntime:
    """Executes the controlled lifecycle of a Work object."""

    _TRANSITIONS = {
        WorkState.PROPOSED: {WorkState.AUTHORIZED, WorkState.CANCELLED},
        WorkState.AUTHORIZED: {WorkState.RUNNING, WorkState.CANCELLED},
        WorkState.RUNNING: {WorkState.VERIFYING, WorkState.BLOCKED, WorkState.FAILED},
        WorkState.VERIFYING: {WorkState.COMPLETED, WorkState.BLOCKED, WorkState.FAILED},
        WorkState.BLOCKED: {WorkState.RUNNING, WorkState.CANCELLED},
        WorkState.FAILED: set(),
        WorkState.COMPLETED: set(),
        WorkState.CANCELLED: set(),
    }

    def authorize(self, work: Work, actor_id: str) -> None:
        self._require_transition(work, WorkState.AUTHORIZED)
        if not work.authority.permits(actor_id, work.capability):
            raise PermissionError(
                f"Actor '{actor_id}' is not authorized for '{work.capability.name}'."
            )
        self._transition(work, WorkState.AUTHORIZED, actor_id, "authorized")

    def start(self, work: Work, actor_id: str) -> None:
        self._require_transition(work, WorkState.RUNNING)
        if work.authority.actor_id != actor_id:
            raise PermissionError(f"Actor '{actor_id}' does not own this work authority.")
        self._transition(work, WorkState.RUNNING, actor_id, "started")

    def submit_for_verification(self, work: Work, actor_id: str, outcome: Outcome) -> None:
        self._require_transition(work, WorkState.VERIFYING)
        work.outcome = outcome
        self._transition(work, WorkState.VERIFYING, actor_id, "submitted_for_verification")

    def complete(self, work: Work, actor_id: str) -> None:
        self._require_transition(work, WorkState.COMPLETED)
        if not work.has_required_evidence():
            missing = sorted(work.required_evidence - work.evidence_types())
            raise ValueError(f"Cannot complete work. Missing evidence: {missing}")
        if work.outcome is None or not work.outcome.successful:
            raise ValueError("Cannot complete work without a successful outcome.")
        self._transition(work, WorkState.COMPLETED, actor_id, "completed")

    def block(self, work: Work, actor_id: str, reason: str) -> None:
        self._require_transition(work, WorkState.BLOCKED)
        if not reason.strip():
            raise ValueError("A block reason is required.")
        self._transition(work, WorkState.BLOCKED, actor_id, "blocked", {"reason": reason})

    def _require_transition(self, work: Work, target: WorkState) -> None:
        if target not in self._TRANSITIONS[work.state]:
            raise ValueError(
                f"Invalid state transition: {work.state.value} -> {target.value}"
            )

    def _transition(
        self,
        work: Work,
        target: WorkState,
        actor_id: str,
        event_type: str,
        details: dict | None = None,
    ) -> None:
        previous = work.state
        work.state = target
        work.record_event(
            WorkEvent.create(
                event_type=event_type,
                work_id=work.work_id,
                actor_id=actor_id,
                from_state=previous.value,
                to_state=target.value,
                details=details,
            )
        )
