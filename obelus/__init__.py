"""obelus — offline-first architecture verification & non-inferiority gating."""

from __future__ import annotations

from obelus.api import AblationResult, AutoAblate, AutoInsert, run_verification

__version__ = "0.1.0"

__all__ = [
    "AutoAblate",
    "AutoInsert",
    "run_verification",
    "AblationResult",
    "__version__",
]
