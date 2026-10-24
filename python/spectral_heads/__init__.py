"""
spectral-heads: zero-training attention head pruning.

Measure per-head key-gram spectra, rank by lambda_min, and get your
pruning list — no calibration data, no gradients, one forward pass.

Usage:
    from spectral_heads import measure, recommend_pruning

    # Step 1: measure (one forward pass on any context)
    measure("gpt2", "keys.json")

    # Step 2: get the pruning list (use the CLI or pure Python)
    heads = recommend_pruning("keys.json", ratio=0.10)
"""
from .core import measure, recommend_pruning, HeadScore

__version__ = "0.1.0"
__all__ = ["measure", "recommend_pruning", "HeadScore"]
