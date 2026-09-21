# FeedbackLoop

**FeedbackLoop is an experimental typed runtime for representing, executing, verifying, and learning from work.**

## North Star

> **Make work a first-class primitive of computing.**

The project starts from a simple hypothesis:

If computing has primitives such as types, functions, processes, transactions, permissions, resources, and events, then work should also have explicit computational primitives.

FeedbackLoop explores that hypothesis through a small, dependency-light Work Kernel.

## v0.2: Work Kernel

The first implementation deliberately avoids LLMs, agents, RLHF, dashboards, databases, and domain-specific workflows.

It currently provides:

- **Intent**: what the work is trying to accomplish.
- **Capability**: what operation can be performed.
- **Authority**: which actor may perform that capability.
- **Evidence**: immutable artifacts supporting execution or claims.
- **Outcome**: the observed result of execution.
- **WorkState**: an explicit lifecycle.
- **WorkEvent**: immutable lifecycle history.
- **WorkRuntime**: controlled execution and state transitions.
- **WorkVerifier**: deterministic verification independent of execution.

### Example lifecycle

```text
PROPOSED
   ↓
AUTHORIZED
   ↓
RUNNING
   ↓
VERIFYING
   ↓
COMPLETED
```

Failure and blocking states are explicit rather than silently swallowed.

## Design principles

1. **Typed over implicit**: important work concepts are explicit objects.
2. **Authority before action**: execution requires explicit capability authority.
3. **Evidence before completion**: required evidence must exist before work can complete.
4. **Verification is independent**: the verifier does not trust the executor.
5. **Failure is data**: blocked, failed, cancelled, and invalid states are first-class.
6. **Domain neutrality**: the kernel should not be designed around recruiting, O&M, or any other single domain.

## Running

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
feedbackloop --demo
```

The demo produces `artifacts/work_demo.json`.

## Roadmap

The immediate research loop is:

```text
Build
  ↓
Stress
  ↓
Find missing primitive
  ↓
Revise kernel
  ↓
Repeat
  ↓
Adapt to real domains
```

O&M Agency, BiasGuard, AI agents, and RLHF are future validation/adaptation layers, not assumptions baked into the kernel.

## License

MIT


**v0.3.0:** trust-boundary hardening, terminal-state evidence protection, and event-chain verification.
