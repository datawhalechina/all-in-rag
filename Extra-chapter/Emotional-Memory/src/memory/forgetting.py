"""Compatibility import for Reverie's non-destructive memory decay system.

The former deletion/confabulation implementation was removed deliberately.
Forgetting now means retrieval inhibition; canonical memory rows are never
deleted or rewritten by a maintenance cycle.
"""

from .cognitive_decay import CognitiveDecaySystem

ForgettingSystem = CognitiveDecaySystem

__all__ = ["CognitiveDecaySystem", "ForgettingSystem"]
