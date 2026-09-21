from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capability import Capability


@dataclass(frozen=True)
class Authority:
    """Explicit authority granted to an actor for named capabilities."""

    actor_id: str
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not self.actor_id.strip():
            raise ValueError("Authority actor_id cannot be empty.")

    def permits(self, actor_id: str, capability: Capability) -> bool:
        return self.actor_id == actor_id and capability.name in self.capabilities
