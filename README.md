# FeedbackLoop

**FeedbackLoop is an experimental typed runtime for representing, executing, verifying, and learning from work.**

## North Star

> Make work a first-class primitive of computing.

v0.4 focuses on **trust-boundary hardening**. The kernel now treats authority, lifecycle transitions, evidence mutation, and event provenance as explicit invariants.

### Kernel primitives

- Intent
- Capability
- Authority
- Evidence
- Outcome
- WorkState
- WorkEvent
- WorkRuntime
- WorkVerifier

### Security invariants

1. Every runtime mutation requires the work authority.
2. A declared capability must be granted by its authority.
3. Terminal work cannot be mutated with new evidence.
4. Only legal lifecycle transitions can be emitted by the runtime.
5. Verification independently checks the event chain and final state.
6. Corrupted provenance is surfaced as verification failure.

The project intentionally remains free of LLMs, agents, databases, and domain-specific policy until the kernel survives adversarial testing.

## Run

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
feedbackloop --demo
```
