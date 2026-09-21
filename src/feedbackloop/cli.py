import argparse
import json
from pathlib import Path

from feedbackloop.kernel.authority import Authority
from feedbackloop.kernel.capability import Capability
from feedbackloop.kernel.evidence import Evidence
from feedbackloop.kernel.intent import Intent
from feedbackloop.kernel.outcome import Outcome
from feedbackloop.kernel.work import Work
from feedbackloop.runtime.runtime import WorkRuntime
from feedbackloop.verification.verifier import WorkVerifier

def main() -> None:
    parser = argparse.ArgumentParser(prog="feedbackloop")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if not args.demo:
        raise SystemExit("Use --demo to run the Work Kernel demo.")

    work = Work(
        work_id="demo-sort",
        intent=Intent("Sort a collection by file size", ("sorted",)),
        capability=Capability("filesystem.sort"),
        authority=Authority("demo-worker", frozenset({"filesystem.sort"})),
        required_evidence=frozenset({"input_manifest", "output_manifest"}),
    )
    runtime = WorkRuntime()
    runtime.authorize(work, "demo-worker")
    runtime.start(work, "demo-worker")
    work.add_evidence(Evidence("e1", "input_manifest", "demo", {"files": 3}))
    work.add_evidence(Evidence("e2", "output_manifest", "demo", {"sorted": True}))
    runtime.submit_for_verification(work, "demo-worker", Outcome("success", {"sorted": True}))
    runtime.complete(work, "demo-worker")

    result = WorkVerifier().verify(work)
    if not result.valid:
        raise RuntimeError(result.reasons)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "work_demo.json"
    path.write_text(json.dumps(work.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote: {path}")
