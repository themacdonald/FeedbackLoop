"""FeedbackLoop: a typed runtime for representing, executing, and verifying work."""

__version__ = "0.2.0"

from .kernel.work import Work
from .kernel.intent import Intent
from .kernel.capability import Capability
from .kernel.authority import Authority
from .kernel.evidence import Evidence
from .kernel.outcome import Outcome
from .kernel.event import WorkEvent
from .kernel.state import WorkState
from .runtime.runtime import WorkRuntime
from .verification.verifier import WorkVerifier

__all__ = [
    "Authority",
    "Capability",
    "Evidence",
    "Intent",
    "Outcome",
    "Work",
    "WorkEvent",
    "WorkRuntime",
    "WorkState",
    "WorkVerifier",
]
