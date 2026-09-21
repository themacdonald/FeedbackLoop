from dataclasses import dataclass

@dataclass(frozen=True)
class Authority:
    actor_id: str
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not self.actor_id.strip():
            raise ValueError("Authority actor_id cannot be empty.")

    def permits(self, actor_id: str, capability_name: str) -> bool:
        return self.actor_id == actor_id and capability_name in self.capabilities
