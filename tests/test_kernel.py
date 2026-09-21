import pytest

from feedbackloop import Authority, Capability, Evidence, Intent, Outcome, Work, WorkState
from feedbackloop.runtime.runtime import WorkRuntime
from feedbackloop.verification.verifier import WorkVerifier

def make_work():
    return Work(
        "w1", Intent("transform"), Capability("data.transform"),
        Authority("worker", frozenset({"data.transform"})),
        frozenset({"source"}),
    )

def running_work():
    w = make_work()
    r = WorkRuntime()
    r.authorize(w, "worker")
    r.start(w, "worker")
    return w, r

def finish(w, r):
    w.add_evidence(Evidence("e1", "source", "test", {"rows": 10}))
    r.submit_for_verification(w, "worker", Outcome("success", {"ok": True}))
    r.complete(w, "worker")

def test_starts_proposed():
    assert make_work().state == WorkState.PROPOSED

def test_invalid_authority_rejected_at_construction():
    with pytest.raises(ValueError):
        Work("w", Intent("x"), Capability("x"), Authority("a", frozenset()), frozenset())

def test_every_mutating_runtime_operation_requires_authority():
    w, r = running_work()
    for fn in (
        lambda: r.submit_for_verification(w, "intruder", Outcome("success", {})),
        lambda: r.block(w, "intruder", "reason"),
    ):
        with pytest.raises(PermissionError):
            fn()

def test_failed_and_cancelled_are_terminal():
    w, r = running_work()
    r.fail(w, "worker", "boom")
    with pytest.raises(ValueError):
        r.start(w, "worker")

def test_cancel_from_proposed_is_recorded():
    w = make_work()
    WorkRuntime().cancel(w, "worker", "no longer needed")
    assert w.state == WorkState.CANCELLED
    assert w.events[-1].event_type == "cancelled"

def test_evidence_cannot_change_terminal_work():
    w, r = running_work()
    finish(w, r)
    with pytest.raises(ValueError):
        w.add_evidence(Evidence("e2", "source", "test", {}))

def test_verifier_detects_corrupted_event_chain():
    w, r = running_work()
    finish(w, r)
    w.events[1] = type(w.events[1])(
        event_type=w.events[1].event_type, work_id=w.work_id, actor_id=w.events[1].actor_id,
        from_state="proposed", to_state=w.events[1].to_state,
        occurred_at=w.events[1].occurred_at, details=w.events[1].details,
    )
    result = WorkVerifier().verify(w)
    assert result.valid is False
    assert any(x.startswith("event_chain_broken") for x in result.reasons)

def test_verifier_rejects_illegal_transition():
    w = make_work()
    w.events.append(type("E", (), {
        "work_id": w.work_id, "from_state": "proposed", "to_state": "completed"
    })())
    result = WorkVerifier().verify(w)
    assert result.valid is False

def test_successful_work_is_verifiable():
    w, r = running_work()
    finish(w, r)
    result = WorkVerifier().verify(w)
    assert result.valid is True
    assert result.reasons == ()

def test_events_are_ordered():
    w, r = running_work()
    assert [e.event_type for e in w.events] == ["authorized", "started"]
