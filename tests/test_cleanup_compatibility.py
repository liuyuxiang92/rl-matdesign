"""Compatibility contracts for shared helpers and legacy entry points."""
import numpy as np
import pytest

from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.composition import composition_key
from rl_matdesign.utils.timing import candidate_key


@pytest.mark.parametrize("candidate,expected", [
    ({"Ni": 0.75, "Fe": "0.25"}, (("Fe", 0.25), ("Ni", 0.75))),
    ({"B": {"element": "Ti"}, "A": {"kind": "oxide", "amount": "0.5"}},
     (("A", (("amount", 0.5), ("kind", "oxide"))),
      ("B", (("element", "Ti"),)))),
    ({}, ()),
])
def test_cache_key_contracts(candidate, expected):
    assert composition_key(candidate) == expected
    assert candidate_key(candidate) == expected
    assert StructureScorePredictor._key(candidate) == expected
    assert composition_key(dict(reversed(list(candidate.items())))) == expected


def test_structure_precision_and_cost_rounding_remain_distinct():
    a, b = {"Fe": 0.12345671}, {"Fe": 0.12345672}
    assert StructureScorePredictor._key(a) != StructureScorePredictor._key(b)
    assert candidate_key(a) == candidate_key(b) == (("Fe", 0.123457),)


def test_timer_non_mapping_fallback_is_preserved():
    assert candidate_key(["custom", 1]) == ("__repr__", "['custom', 1]")


def test_legacy_encoding_imports_and_validation():
    from abcde_ooh.encoding import decode_one_hot, encode_choice, one_hot_index

    np.testing.assert_array_equal(encode_choice("Ni", ["Fe", "Ni"]), [0., 1.])
    assert decode_one_hot(one_hot_index(1, 2), ["Fe", "Ni"]) == "Ni"
    with pytest.raises(ValueError, match="Unknown choice"):
        encode_choice("Co", ["Fe", "Ni"])
    with pytest.raises(ValueError, match="Bad one-hot shape"):
        decode_one_hot([1], ["Fe", "Ni"])


def test_legacy_runner_config_imports():
    import run_experiment
    from rl_matdesign import configuration

    for name in ("load_config", "build_env", "build_predictor",
                 "build_constraint_filter", "_apply_flag_aliases"):
        assert getattr(run_experiment, name) is getattr(configuration, name)
