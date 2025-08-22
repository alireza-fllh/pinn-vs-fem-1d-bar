"""
Training and CLI entry-points for running individual models.
Intended for scripts.
"""
# Black-box model helpers
from .train_blackbox import BBNet, train_blackbox

__all__ = ["BBNet", "train_blackbox"]
