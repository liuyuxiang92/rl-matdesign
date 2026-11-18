"""Exercise the shipped peptide configs through public loading and evaluation."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pytest

from rl_matdesign.configuration import build_env, load_config
from rl_matdesign.peptide_design import compile_groups
from rl_matdesign.utils.composition import composition_key
from rl_matdesign.utils.episode_batch import EpisodeBatchEvaluator, snapshot_episode


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
THIOLS = {"D-HCys", "C", "c", "HCys", "BB103", "BB137", "b-HCys"}
PARENT = ["A", "HCys", "F", "K", "V", "W", "W", "A", "P", "R", "b-HCys"]


class MockPeptidePredictor:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.seen, self.metadata_seen, self.batch_payloads = [], [], []
        self.cache = {}

    def evaluation_batch_size(self):
        return self.batch_size

    def validate(self, candidate):
        assert set(candidate) == {"n_terminal", "peptide"}
        assert len(candidate["peptide"]) == 11
        assert sum(token in THIOLS for token in candidate["peptide"]) == 2
        assert candidate["n_terminal"]["treatment"] in ("H", "Ac")

    def predict(self, candidate):
        self.validate(candidate)
        self.seen.append(deepcopy(candidate))
        key = composition_key(candidate)
        self.cache.setdefault(key, (float(len(self.cache) + 1), 0.1))
        return self.cache[key]

    def batch_predict(self, candidates, *, return_exceptions=False):
        self.batch_payloads.append(deepcopy(candidates))
        return [self.predict(candidate) for candidate in candidates]

    def candidate_metadata(self, candidate):
        self.validate(candidate)
        self.metadata_seen.append(deepcopy(candidate))
        return {"monomers": "|".join(candidate["peptide"]), "smiles": "mock"}


def build(name, oracle=None):
    cfg = load_config(str(CONFIGS / f"{name}.yaml"))
    predictor = oracle or MockPeptidePredictor()
    env, kind, _ = build_env(cfg, predictor)
    assert kind == "multi_group"
    assert {group["kind"] for group in cfg["groups"]} == {"categorical"}
    assert all("parent" not in group and "tokens" not in group for group in cfg["groups"])
    return env, predictor


def choose(env, target, replacement):
    inner = env._active()
    li = inner.species_set.index(target)
    code = next(code for value, code in inner._slot_codes[li] if value == replacement)
    local = (tuple(np.eye(len(inner.species_set))[li]),
             tuple(np.eye(len(inner.action_values))[inner.action_values.index(code)]))
    env.step(env._to_union(local, inner))


def parent_episode(env, last_position=7):
    env.initialize()
    choose(env, "treatment", "H")
    for target in (1, 3, 4, 5, last_position):
        choose(env, target, "BB2")
    choose(env, 2, "HCys")
    choose(env, 11, "b-HCys")


def novel_episode(env, pair):
    env.initialize()
    choose(env, "treatment", "Ac")
    for position in range(1, 12):
        if position not in pair:
            choose(env, position, "BB2")
    assert {inner_label for inner_label in env._active()._eligible} == set(pair)
    for position in reversed(pair):
        choose(env, position, "C")


@pytest.mark.parametrize("name", ["peptide_parent_gp", "peptide_parent_ecfp_gp"])
def test_public_parent_configs_make_five_real_edits_and_keep_original_thiol_positions(name):
    env, oracle = build(name)
    thiols = env._inners[2]
    assert all("b-HCys" not in values for target, values in thiols.slots if target != 11)
    assert "b-HCys" in next(values for target, values in thiols.slots if target == 11)
    initial = env.current_state_features()
    parent_episode(env)
    final = env.terminal_cation_fractions()
    assert env.counter == 8
    assert sum(a != b for a, b in zip(PARENT, final["peptide"])) == 5
    assert [i + 1 for i, token in enumerate(final["peptide"]) if token in THIOLS] == [2, 11]
    assert final["peptide"][1] == "HCys" and final["peptide"][10] == "b-HCys"
    assert oracle.seen == [final]
    assert not np.array_equal(initial, env.current_state_features())
    first_key, first_state = env.terminal_comp_key(), env.current_state_features()
    parent_episode(env, last_position=6)
    assert first_key != env.terminal_comp_key()
    assert not np.array_equal(first_state, env.current_state_features())
    assert len(oracle.cache) == 2


@pytest.mark.parametrize("name", ["peptide_novel_gp", "peptide_novel_ecfp_gp"])
def test_public_novel_configs_accept_every_pair_of_thiol_positions(name):
    env, oracle = build(name)
    keys = set()
    for pair in combinations(range(1, 12), 2):
        novel_episode(env, pair)
        sequence = env.terminal_cation_fractions()["peptide"]
        assert tuple(i + 1 for i, token in enumerate(sequence) if token in THIOLS) == pair
        assert env.counter == 12
        keys.add(env.terminal_comp_key())
    assert len(keys) == len(oracle.cache) == len(oracle.seen) == 55


def test_batch_reward_and_episode_snapshot_receive_only_the_combined_peptide():
    oracle = MockPeptidePredictor(batch_size=2)
    env, _ = build("peptide_parent_gp", oracle)
    evaluator = EpisodeBatchEvaluator(env, capture_predictions=True)
    plans = iter([7, 6])
    def rollout():
        parent_episode(env, next(plans))
    first = evaluator.next(rollout, limit=2)
    second = evaluator.next(rollout, limit=1)
    assert len(oracle.batch_payloads) == 1 and len(oracle.batch_payloads[0]) == 2
    for episode, payload in zip((first, second), oracle.batch_payloads[0]):
        assert episode.composition == payload
        assert json.loads(episode.formula) == payload
        assert json.loads(episode.prediction_record["candidate_json"]) == payload
        assert set(payload) == {"n_terminal", "peptide"}
    assert first.terminal_key != second.terminal_key
    assert snapshot_episode(env).composition == second.composition


def test_generation_exports_combined_candidate_and_sends_it_to_metadata(monkeypatch):
    from rl_matdesign import training
    env, oracle = build("peptide_parent_gp")
    monkeypatch.setattr(training, "_rollout_policy_episode", lambda **kwargs: parent_episode(kwargs["env"]))
    class Model:
        def eval(self):
            pass
    rows = training.generate_candidates(
        env=env, predictor=oracle, scaler=None, device=None,
        elem_feats_scaled=np.empty((0, 0)), action_values=env.action_values,
        qnet=Model(), num_gen_eps=1, max_attempts=1)
    assert len(rows) == 1
    candidate = json.loads(rows[0]["candidate_json"])
    assert set(candidate) == {"n_terminal", "peptide"}
    assert len(candidate["peptide"]) == 11
    assert oracle.metadata_seen == [candidate]
    assert rows[0]["meta_monomers"].split("|") == candidate["peptide"]


class MockChemistryPlugin:
    def sequence_constraints(self, group_name, tokens):
        assert group_name == "peptide"
        return [{"tokens": [token for token in tokens if token in THIOLS], "min": 2, "max": 2}]


@pytest.mark.parametrize("name", ["peptide_parent_gp", "peptide_parent_ecfp_gp",
                                  "peptide_novel_gp", "peptide_novel_ecfp_gp"])
def test_plugin_chemistry_startup_validation_accepts_public_catalogs(name):
    cfg = load_config(str(CONFIGS / f"{name}.yaml"))
    _, adapter = compile_groups(cfg)
    if "parent" in cfg["structure"]:
        assert "b-HCys" in adapter.tokens
        assert "b-HCys" not in cfg["groups"][2]["values"]
    bounds = MockChemistryPlugin().sequence_constraints("peptide", adapter.tokens)
    adapter.validate_constraints(bounds)
    assert adapter.design_schema()["intrinsic_count_constraints"] == bounds


@pytest.mark.parametrize("name", ["peptide_parent_gp", "peptide_novel_gp"])
@pytest.mark.parametrize("group,token", [(1, "C"), (2, "BB2")])
def test_plugin_rejects_catalogs_that_can_violate_its_chemical_count(name, group, token):
    cfg = load_config(str(CONFIGS / f"{name}.yaml"))
    cfg["groups"][group]["values"].append(token)
    # Generic compilation has no chemical assumptions: only the plugin's
    # explicit membership set identifies the invalid catalog partition.
    _, adapter = compile_groups(cfg)
    assert adapter.constraints == []
    bounds = MockChemistryPlugin().sequence_constraints("peptide", adapter.tokens)
    with pytest.raises(ValueError, match="catalogs can produce"):
        adapter.validate_constraints(bounds)


@pytest.mark.parametrize("bounds", [None, {}, [None], [{}],
    [{"tokens": ["C"], "where": "thiol"}], [{"tokens": ["C", "C"]}],
    [{"tokens": ["unknown"]}], [{"tokens": ["C"], "min": True}],
    [{"tokens": ["C"], "min": 3, "max": 2}], [{"tokens": ["C"], "max": 12}],
])
def test_malformed_plugin_count_constraints_are_rejected(bounds):
    _, adapter = compile_groups(load_config(str(CONFIGS / "peptide_parent_gp.yaml")))
    with pytest.raises(ValueError):
        adapter.validate_constraints(bounds)
