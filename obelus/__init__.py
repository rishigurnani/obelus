"""obelus — offline-first architecture verification & non-inferiority gating."""

from __future__ import annotations

from obelus.api import AblationResult, AutoAblate, run_ablation

__version__ = "0.1.0"

__all__ = ["AutoAblate", "run_ablation", "AblationResult", "__version__"]
