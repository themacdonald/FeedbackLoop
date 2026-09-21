from dataclasses import dataclass

@dataclass(frozen=True)
class Capability:
    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Capability name cannot be empty.")
