import pytest

from feedbackloop import (
    Authority,
    Capability,
    Evidence,
    Intent,
    Outcome,
    Work,
    WorkState,
)
from feedbackloop.runtime.runtime import WorkRuntime
from feedbackloop.verification.verifier import WorkVerifier


def make_work(required_evidence=frozenset({"source"})):
    capability = Capability("data.transform")
    authority = Authority("worker-1", frozenset({"data.transform"}))
    return Work(
        work_id="work-1",
        intent=Intent("Transform the dataset", ("transformed",)),
        capability=capability,
        authority=authority,
        required_evidence=required_evidence,
    )


def authorize_and_start(work):
    runtime = WorkRuntime()
    runtime.authorize(work, "worker-1")
    runtime.start(work, "worker-1")
    return runtime


def test_work_starts_in_proposed_state():
    assert make_work().state == WorkState.PROPOSED


def test_unauthorized_actor_cannot_authorize_work():
    with pytest.raises(PermissionError):
        WorkRuntime().authorize(make_work(), "intruder")


def test_duplicate_evidence_is_rejected():
    work = make_work()
    evidence = Evidence("e1", "source", "test", {"rows": 10})
    work.add_evidence(evidence)
    with pytest.raises(ValueError, match="Duplicate evidence"):
        work.add_evidence(evidence)


def test_work_cannot_complete_without_required_evidence():
    work = make_work()
    runtime = authorize_and_start(work)
    runtime.submit_for_verification(work, "worker-1", Outcome("success", {"transformed": True}))
    with pytest.raises(ValueError, match="Missing evidence"):
        runtime.complete(work, "worker-1")


def test_work_cannot_complete_with_failed_outcome():
    work = make_work()
    runtime = authorize_and_start(work)
    work.add_evidence(Evidence("e1", "source", "test", {"rows": 10}))
    runtime.submit_for_verification(work, "worker-1", Outcome("failure", {"transformed": False}))
    with pytest.raises(ValueError, match="successful outcome"):
        runtime.complete(work, "worker-1")


def test_successful_work_completes_and_records_events():
    work = make_work()
    runtime = authorize_and_start(work)
    work.add_evidence(Evidence("e1", "source", "test", {"rows": 10}))
    runtime.submit_for_verification(work, "worker-1", Outcome("success", {"transformed": True}))
    runtime.complete(work, "worker-1")

    assert work.state == WorkState.COMPLETED
    assert [event.event_type for event in work.events] == [
        "authorized",
        "started",
        "submitted_for_verification",
        "completed",
    ]


def test_invalid_state_transition_is_rejected():
    work = make_work()
    with pytest.raises(ValueError, match="Invalid state transition"):
        WorkRuntime().complete(work, "worker-1")


def test_verifier_independently_rejects_incomplete_work():
    result = WorkVerifier().verify(make_work())
    assert result.valid is False
    assert "outcome_missing" in result.reasons
    assert "work_not_completed:proposed" in result.reasons


def test_verifier_accepts_a_completed_work():
    work = make_work()
    runtime = authorize_and_start(work)
    work.add_evidence(Evidence("e1", "source", "test", {"rows": 10}))
    runtime.submit_for_verification(work, "worker-1", Outcome("success", {"transformed": True}))
    runtime.complete(work, "worker-1")

    result = WorkVerifier().verify(work)
    assert result.valid is True
    assert result.reasons == ()


def test_event_history_is_append_only_from_the_work_api():
    work = make_work()
    runtime = authorize_and_start(work)
    assert len(work.events) == 2
    assert work.events[0].from_state == "proposed"
    assert work.events[1].to_state == "running"
