"""
etc.problems.base
=================
Base class for all ETC benchmark / research problems.

A Problem encodes:
  - A mathematical question with precise input/output types.
  - A `solve()` method that returns a Certified result.
  - A `verify(answer)` method that re-checks any claimed answer.

Subclass this to add new problems to the ETC catalogue.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from etc.certified import Certified


class Problem(ABC):
    """
    Abstract base class for ETC problems.

    Subclasses must implement:
      - name        : str            – short identifier
      - description : str            – mathematical statement
      - solve()     : Certified[Any] – compute and certify the answer
      - verify(x)   : bool           – re-verify a claimed answer
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def solve(self) -> Certified[Any]:
        """Compute the answer and return it wrapped in a Certified."""
        ...

    @abstractmethod
    def verify(self, answer: Any) -> bool:
        """Re-verify a claimed answer independently."""
        ...

    def __repr__(self) -> str:
        return f"Problem('{self.name}')"
