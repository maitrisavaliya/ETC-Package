"""
etc.problems.registry
=====================
Global catalogue of ETC problems.

Usage
-----
    from etc.problems import registry

    registry.list()                  # print all problems
    p = registry.get("sign_of_pi")   # fetch a problem by name
    result = p.solve()               # Certified answer
"""

from __future__ import annotations
from typing import Dict, List, Optional, Type

from etc.problems.base import Problem


class ProblemRegistry:
    """A simple name-keyed catalogue of Problem instances."""

    def __init__(self) -> None:
        self._problems: Dict[str, Problem] = {}

    def register(self, problem: Problem) -> None:
        """Add a Problem instance to the registry."""
        self._problems[problem.name] = problem

    def get(self, name: str) -> Optional[Problem]:
        """Retrieve a problem by name, or None if not found."""
        return self._problems.get(name)

    def list(self) -> List[str]:
        """Return all registered problem names."""
        names = sorted(self._problems)
        for n in names:
            print(f"  {n:30s}  {self._problems[n].description[:60]}")
        return names

    def __len__(self) -> int:
        return len(self._problems)

    def __repr__(self) -> str:
        return f"ProblemRegistry({len(self)} problems)"


# Global singleton registry
registry = ProblemRegistry()
