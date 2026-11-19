"""Execution context shared by YAML actions and procedural tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GenerationContext:
    """Hold recipe variables, deterministic randomness and generated IDs."""

    values: dict[str, Any]
    random: np.random.Generator
    identifier_counts: dict[str, int] = field(default_factory=dict)

    def new_identifier(self, prefix: str) -> str:
        """Create stable, readable IDs for generated tree components."""
        next_number = self.identifier_counts.get(prefix, 0) + 1
        self.identifier_counts[prefix] = next_number
        return f"{prefix}_{next_number:04d}"
