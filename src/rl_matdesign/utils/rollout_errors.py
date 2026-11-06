"""A rejected construction prefix is not a failed property prediction."""


class EvaluationProtocolError(ValueError):
    """The oracle contract is invalid; retrying another candidate cannot fix it."""


class InfeasibleEpisode(Exception):
    """No legal continuation exists; discard this prefix without oracle work.

    Configuration errors, malformed actions and plugin contract violations must
    not be wrapped in this exception. Training retries only genuine empty legal
    action sets, with a bounded consecutive rejection limit.
    """
