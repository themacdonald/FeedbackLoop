from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    """A specific operation a worker may request."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Capability name cannot be empty.")
