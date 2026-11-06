"""Build one locomotion and expression order."""

from __future__ import annotations

from sandbox.observation_types import ThreeBranchesAction

from ._model import RULES

EMOTES = tuple(RULES["emotes"])
_EXPRESSIONS = {"none", "use", *EMOTES}


def walk(heading: float, speed: float = 1.0, expression: str = "none") -> ThreeBranchesAction:
    """Walk at a clamped relative speed while showing an optional expression."""
    if expression not in _EXPRESSIONS:
        raise ValueError(f"unknown expression {expression!r}")
    order: ThreeBranchesAction = {
        "heading": float(heading) % 360.0,
        "speed": min(1.0, max(0.0, float(speed))),
        "action": _action(expression),
    }
    return order


def stand(heading: float, expression: str = "none") -> ThreeBranchesAction:
    """Hold position while showing an optional expression or using the selected prop."""
    return walk(heading, 0.0, expression)


def _action(expression: str) -> int:
    return 0 if expression == "none" else 1 if expression == "use" else EMOTES.index(expression) + 2
