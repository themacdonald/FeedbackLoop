# Technical Specification: FeedbackLoop Work Kernel v0.2

## Architecture

```text
Intent
  ↓
Work
  ├── Capability
  ├── Authority
  ├── Required Evidence
  └── Lifecycle State
        ↓
     Runtime
        ↓
      Events
        ↓
     Outcome
        ↓
   Verification
```

## Trust boundaries

### Work declaration
Defines intent and required conditions.

### Authority
Determines whether a named actor may execute the declared capability.

### Runtime
Controls legal state transitions.

### Evidence
Provides artifacts supporting work execution and claims.

### Outcome
Records what happened.

### Verification
Independently checks whether the completed work satisfies its basic contract.

The verifier is intentionally separate from the runtime so successful execution does not automatically imply valid work.

## Current limitations

The v0.2 kernel does not yet model:

- complex dependencies
- resource accounting
- delegated authority
- human feedback
- temporal constraints
- distributed execution
- persistent storage
- policy engines
- probabilistic or semantic verification

These are intentionally deferred. They should be introduced only when experiments demonstrate that the primitive is missing them.

## Design constraint

No domain-specific abstraction should enter the kernel merely because a single application needs it.


## v0.3 Trust-Boundary Hardening

The runtime now treats lifecycle mutation as an authority-bound operation. The authority owner must perform submission, completion, and blocking actions. Evidence cannot be appended after terminal states. The independent verifier also checks that the event history forms a coherent state chain ending at the current work state.

These checks are deliberately deterministic and do not depend on an LLM. They make unauthorized mutation and corrupted provenance visible before FeedbackLoop is extended with persistence, delegation, or human feedback.
