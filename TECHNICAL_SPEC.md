# FeedbackLoop Work Kernel v0.4

## Trust model

FeedbackLoop separates four concerns:

```text
Declaration → Runtime Authority → Evidence/Outcome → Independent Verification
```

The runtime is allowed to mutate work state. The verifier is not trusted with execution and independently reconstructs whether the recorded lifecycle is legal.

## Invariants

### I1: Capability authority
`Authority(actor, capabilities)` must grant the Work capability.

### I2: Actor authorization
All runtime operations that mutate lifecycle state require the authorized actor.

### I3: Transition legality
A runtime transition must belong to the explicit state-transition graph.

### I4: Terminal immutability
Completed, failed, and cancelled work cannot receive new evidence.

### I5: Evidence completeness
Completion requires every declared evidence type.

### I6: Outcome validity
Completion requires a successful outcome.

### I7: Provenance consistency
The event chain must start from `proposed`, use legal transitions, preserve work identity, and terminate at the current state.

## Known next attack surfaces

- dependency graphs and cycle detection
- delegated authority and least privilege
- resource budgets and exhaustion
- idempotency and replay
- concurrency/race conditions
- durable persistence and crash recovery
- cryptographic event integrity
- policy versioning
- semantic verification
