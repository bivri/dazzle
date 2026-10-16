"""Dazzle: protective patterns against unauthorized style mimicry."""

from dazzle.generator import DazzleGenerator, levels_to_epsilon
from dazzle.protect import protect_image

__all__ = ["DazzleGenerator", "levels_to_epsilon", "protect_image"]
__version__ = "0.1.0"
