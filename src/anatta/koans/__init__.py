"""Koan environments. Each module exposes an env class and a `suite(...)` of episodes."""

from . import commons, impossible, pushback, shutdown
from .base import FRAMINGS, Env, Policy, Step

SUITES = {
    "impossible": impossible.suite,
    "shutdown": shutdown.suite,
    "commons": commons.suite,
    "pushback": pushback.suite,
}

__all__ = ["FRAMINGS", "SUITES", "Env", "Policy", "Step", "commons", "impossible", "pushback", "shutdown"]
