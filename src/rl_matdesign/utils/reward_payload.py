"""Pass canonical designs directly when the reward adapter supports them."""
from __future__ import annotations


def call_terminal_reward(reward_fn, candidate, formula):
    """Use structured scientific input, retaining plain callable compatibility.

    Generic experiment adapters implement ``evaluate_candidate``. Lower-level
    users may still supply an ordinary formula callback; no exception-based
    guessing or scenario-specific conversion is performed.
    """
    evaluate = getattr(reward_fn, "evaluate_candidate", None)
    return evaluate(candidate) if callable(evaluate) else reward_fn(formula)
