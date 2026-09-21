from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kernel.authority import Authority
from .kernel.capability import Capability
from .kernel.evidence import Evidence
from .kernel.intent import Intent
from .kernel.outcome import Outcome
from .kernel.work import Work
from .runtime.runtime import WorkRuntime
from .verification.verifier import WorkVerifier


def demo() -> Path:
    """Run a deterministic Work Kernel demonstration and write its proof artifact."""
    capability = Capability("filesystem.sort")
    authority = Authority("demo-worker", frozenset({capability.name}))
    work = Work(
        work_id="demo-sort-001",
        intent=Intent(
            objective="Sort a collection by file size",
            success_conditions=("output_is_size_sorted",),
        ),
        capability=capability,
        authority=authority,
        required_evidence=frozenset({"input_manifest", "output_manifest"}),
    )

    runtime = WorkRuntime()
    runtime.authorize(work, "demo-worker")
    runtime.start(work, "demo-worker")
    work.add_evidence(
        Evidence("e1", "input_manifest", "demo", {"files": 3})
    )
    work.add_evidence(
        Evidence("e2", "output_manifest", "demo", {"sorted_by": "size"})
    )
    runtime.submit_for_verification(
        work,
        "demo-worker",
        Outcome("success", {"output_is_size_sorted": True}),
    )
    runtime.complete(work, "demo-worker")

    verification = WorkVerifier().verify(work)
    artifact = {
        "work": work.to_dict(),
        "verification": verification.to_dict(),
    }

    output = Path("artifacts/work_demo.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="feedbackloop",
        description="FeedbackLoop Work Kernel CLI",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the deterministic Work Kernel demonstration.",
    )
    args = parser.parse_args()

    if args.demo:
        print(f"Wrote: {demo()}")
        return

    parser.print_help()
