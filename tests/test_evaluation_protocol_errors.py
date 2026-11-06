"""Broken plugin protocols cannot become endlessly retried candidate failures."""
import pytest

from rl_matdesign.configuration import _TerminalReward
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
from rl_matdesign.utils.episode_batch import CandidateEvaluationFailed


class Broken:
    def predict(self, candidate):
        raise EvaluationProtocolError("missing sample protocol")


@pytest.mark.parametrize("batch", [False, True])
def test_protocol_errors_escape_scalar_and_legacy_batch(batch):
    reward = _TerminalReward(Broken(), "fraction")
    with pytest.raises(EvaluationProtocolError, match="sample protocol"):
        if batch:
            reward.evaluate_many([{"Ni": 1}])
        else:
            reward({"Ni": 1})


@pytest.mark.parametrize("returned", [False, True])
def test_partial_batch_protocol_errors_escape(returned):
    class Partial(Broken):
        def batch_predict(self, candidates, *, return_exceptions=False):
            error = EvaluationProtocolError("broken batch protocol")
            if returned:
                return [error] * len(candidates)
            raise error
    reward = _TerminalReward(Partial(), "fraction")
    with pytest.raises(EvaluationProtocolError, match="broken batch"):
        reward.evaluate_many([{"Ni": 1}])


def test_candidate_failure_still_uses_retry_contract():
    class Failed:
        def predict(self, candidate):
            raise RuntimeError("this geometry failed")
    reward = _TerminalReward(Failed(), "fraction")
    with pytest.raises(CandidateEvaluationFailed):
        reward({"Ni": 1})
    assert isinstance(reward.evaluate_many([{"Ni": 1}])[0], CandidateEvaluationFailed)
