"""
Visualization utilities for PINN vs FEM comparison.

Provides plotting and animation capabilities for analyzing training dynamics,
prediction accuracy, and comparative performance between physics-informed
neural networks and finite element methods.

Author: Alireza Fallahnejad

Key Exports:
    animate_losses: Training loss evolution animations
    animate_preds: Prediction evolution animations
"""

from .animate_losses import animate_losses
from .animate_predictions import animate_preds

__all__ = ["animate_losses", "animate_preds"]
