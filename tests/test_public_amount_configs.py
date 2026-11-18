"""Public configuration regressions for chemical amounts and builder chemistry.

Model and POSCAR assets are deliberately not loaded: these tests exercise the
same environment construction, payload, descriptor and filter path as training.
Private scenario files are validated when supplied locally; a clean checkout may
omit only the explicitly listed private configs under the existing ignore rules.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Composition
import yaml

from rl_matdesign.amounts import ChemicalAmountsGroup
from rl_matdesign.configuration import build_env, load_config
from rl_matdesign.featurization import featurize_formula
from rl_matdesign.utils.composition import composition_key
from rl_matdesign.utils.episode_batch import snapshot_episode


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PRIVATE_CONFIGS = {
    "ooh", "ti_alloy", "perovskite",
    "cs2agbicl6_level1", "cs2agbicl6_level2", "cs2agbicl6_level3",
}
PUBLIC_CONFIGS = [
    "ooh", "hea", "ti_alloy", "cs2agbicl6_level1", "cs2agbicl6_level2", "cs2agbicl6_level3",
    "perovskite", "lips_sse", "oxides_calcine", "oxides_sinter", "oxides_sinter_calcine", "actmof",
    "peptide_parent_gp", "peptide_novel_gp", "peptide_parent_ecfp_gp", "peptide_novel_ecfp_gp",
    *[f"glassnet_{grid}_{method}" for grid in ("coarse_2pct", "fine_0p5pct", "1pct")
      for method in ("a2c", "dqn_bootstrap", "dqn_mc")],
]


class RecordingPredictor:
    def __init__(self):
        self.seen = []
        self.cache = {}

    def predict(self, candidate):
        self.seen.append(deepcopy(candidate))
        key = composition_key(candidate)
        self.cache.setdefault(key, (float(len(self.cache) + 1), 0.0))
        return self.cache[key]


def build(name):
    path = CONFIGS / f"{name}.yaml"
    if name in PRIVATE_CONFIGS and not path.is_file():
        pytest.skip(f"Private local configuration is not supplied: {name}")
    cfg = load_config(str(path))
    oracle = RecordingPredictor()
    env, kind, _ = build_env(cfg, oracle)
    assert kind == "multi_group"
    return env, oracle, cfg


def build_authored(tmp_path, groups, **settings):
    path = tmp_path / "design.yaml"
    path.write_text(yaml.safe_dump({"groups": groups,
        "properties": [{"name": "quality", "predictor": "dummy", "direction": "max"}], **settings}))
    cfg = load_config(str(path))
    oracle = RecordingPredictor()
    env, kind, _ = build_env(cfg, oracle)
    assert kind == "multi_group"
    return env, oracle


def choose(env, label, amount):
    inner = env._active()
    label_index = inner.species_set.index(label)
    value_index = next(i for i, value in enumerate(inner.action_values)
                       if Fraction(value) == Fraction(str(amount)))
    local = (tuple(np.eye(len(inner.species_set))[label_index]),
             tuple(np.eye(len(inner.action_values))[value_index]))
    env.step(env._to_union(local, inner))


def allowed(env):
    inner = env._active()
    return {(inner.species_set[int(np.argmax(label))], Fraction(inner.action_values[int(np.argmax(value))]))
            for label, value in (env._from_union(action) for action in env.allowed_actions())}


def rollout(env, actions):
    env.initialize()
    for label, amount in actions:
        choose(env, label, amount)
    assert env.is_terminal


@pytest.mark.parametrize("name", PUBLIC_CONFIGS)
def test_all_public_scenarios_load_and_build_without_models_or_templates(name):
    env, _, cfg = build(name)
    assert set(group["kind"] for group in cfg["groups"]) <= {"composition", "categorical"}
    assert "application" not in cfg and "structure_sources" not in cfg and "source" not in cfg
    assert env.allowed_actions()
    initial = env.current_state_features()
    assert initial.ndim == 1 and np.isfinite(initial).all()
    assert all(isinstance(inner, ChemicalAmountsGroup)
               for group, inner in zip(cfg["groups"], env._inners) if group["kind"] == "composition")
    assert all("token_requirements" not in group and "sites" not in group for group in cfg["groups"])


@pytest.mark.parametrize("name", ["ooh", "hea"])
def test_public_positive_unique_menus_preserve_exactly_five_species(name):
    env, oracle, _ = build(name)
    inner = env._inners[0]
    assert inner.n_steps == 5 and inner.unique
    assert min(float(value) for value in inner.action_values) > 0
    selected = set()
    while not env.is_terminal:
        action = env.allowed_actions()[0]
        local = env._from_union(action)
        label = inner.species_set[int(np.argmax(local[0]))]
        assert label not in selected
        selected.add(label)
        env.step(action)
    assert len(inner.terminal_cation_fractions()) == len(selected) == 5
    assert sum(inner.terminal_cation_fractions().values()) == pytest.approx(1)
    assert len(oracle.seen) == 1


def test_public_oxide_oxygen_anywhere_raw_bounds_and_empirical_cache():
    env, oracle, cfg = build("oxides_calcine")
    assert [item["type"] for item in cfg["groups"][0]["filters"]] == ["smact_charge", "pauling_en"]
    # Author the same amount domain without optional chemistry screens to isolate
    # its intrinsic bounds and raw-to-empirical candidate identity.
    cfg["groups"][0].pop("filters")
    env, _, _ = build_env(cfg, oracle)
    assert env._inners[0].filters == []
    paths = [
        [("O", 2), ("Fe", 2), ("Fe", 0), ("Fe", 0), ("Fe", 0)],
        [("Fe", 2), ("Fe", 0), ("O", 2), ("Fe", 0), ("Fe", 0)],
        [("Fe", 0), ("Fe", 0), ("Fe", 0), ("Fe", 1), ("O", 1)],
    ]
    keys, states = [], []
    for path in paths:
        rollout(env, path)
        assert env.terminal_cation_fractions() == {"Fe": 1, "O": 1}
        keys.append(env.terminal_comp_key())
        states.append(env.current_state_features())
    assert keys[0] == keys[1] == keys[2]
    np.testing.assert_array_equal(states[0], states[1])
    assert not np.array_equal(states[1], states[2])  # Raw action quantities remain in the training state.
    assert len(oracle.cache) == 1
    env.initialize()
    for _ in range(4):
        choose(env, "Fe", 1)
    assert allowed(env) == {("O", Fraction(value)) for value in range(1, 10)}
    env.initialize()
    choose(env, "O", 9)
    assert ("O", Fraction(1)) not in allowed(env)  # Final O max is raw 9, before GCD reduction.


def test_public_per_species_amount_menus_and_zero_ban(tmp_path):
    env, oracle = build_authored(tmp_path, [{
        "name": "oxide", "kind": "composition", "species_set": ["Fe", "NiO", "Ni2O3"],
        "amount_set": {"Fe": [0, 2], "NiO": [0, 1], "Ni2O3": [0, 1]}, "n_steps": 2,
        "element_bounds": {"NiO": [0, 0], "Ni2O3": [1, 1]},
    }])
    assert ("NiO", Fraction(1)) not in allowed(env)
    assert ("Ni2O3", Fraction(1)) in allowed(env)
    rollout(env, [("Ni2O3", 1), ("Fe", 2)])
    assert oracle.seen == [{"Ni2O3": 1, "Fe": 2}]


def test_public_glass_uses_formula_unit_amounts_and_preserves_recipe_labels():
    env, oracle, _ = build("glassnet_1pct_a2c")
    rollout(env, [("SiO2", .5), ("SiO2", .1), ("B2O3", .2), ("Al2O3", .1),
                  ("Na2O", .1), ("K2O", 0), ("MgO", 0)])
    inner = env._inners[0]
    assert inner.composition_resolver is None
    assert inner.cation_fractions()["SiO2"] == .6
    expected = {"Si": .6, "B": .4, "Al": .2, "Na": .2, "O": 2.2}
    formula = "Si0.6B0.4Al0.2Na0.2O2.2"
    np.testing.assert_allclose(inner._descriptor_features(), featurize_formula(formula))
    assert Composition(env.terminal_formula).get_el_amt_dict() == pytest.approx(expected)
    assert oracle.seen[0]["SiO2"] == .6 and "Si" not in oracle.seen[0]
    assert snapshot_episode(env).composition == oracle.seen[0]


def test_sse_intrinsic_form_exclusion_stays_active_without_optional_filters():
    env, oracle, cfg = build("lips_sse")
    assert all(not group.get("filters") for group in cfg["groups"])
    assert all(inner.filters == [] for inner in env._inners)
    choose(env, "P", .92)
    choose(env, "Ni2O3", .04)
    assert ("Ni", Fraction(".04")) not in allowed(env)
    assert ("Ni2O3", Fraction(".04")) in allowed(env)  # Repeating the same form is allowed.
    choose(env, "Ni2O3", .04)
    inner = env._inners[0]
    assert inner.cation_fractions() == {"P": .92, "Ni2O3": .08}
    np.testing.assert_allclose(inner._descriptor_features(), featurize_formula("P0.92Ni0.08O0.12"))
    assert inner.composition_resolver(inner.cation_fractions()) == pytest.approx({"P": .92, "Ni": .08, "O": .12})
    choose(env, "Cl", .8)
    assert oracle.seen[0] == {"P_site": {"P": .92, "Ni2O3": .08}, "halide": {"Cl": .8}}


def test_signed_defect_descriptor_uses_host_and_retains_signed_edits(tmp_path):
    groups = [
        {"name": "A_dopant", "kind": "composition", "species_set": ["Sr", "Ca", "Ba"],
         "amount_set": [0, .5, 1], "n_steps": 3, "total": 1},
        {"name": "A_defect", "kind": "composition", "species_set": ["Sr", "Ca", "Ba"],
         "amount_set": [-.03125, 0], "n_steps": 1},
        {"name": "B_dopant", "kind": "composition", "species_set": ["Fe", "Mn", "Co"],
         "amount_set": [0, .5, 1], "n_steps": 3, "total": 1},
    ]
    env, _ = build_authored(tmp_path, groups, structure={"builder": "pipeline", "base_poscar": "unused",
        "steps": [{"type": "substitute", "a_site_symbol": "Sr", "a_dopant_group": "A_dopant",
                   "a_defect_group": "A_defect", "b_site_symbol": "Fe", "b_dopant_group": "B_dopant"}]})
    for action in [("Sr", .5), ("Ca", .5), ("Ba", 0)]:
        choose(env, *action)
    defects = env._active()
    assert defects.composition_resolver({"Sr": -.03125}) == pytest.approx({"Sr": .46875, "Ca": .5, "Ba": 0})
    choose(env, "Sr", -.03125)
    assert defects.cation_fractions() == {"Sr": -.03125}
    np.testing.assert_allclose(defects._descriptor_features(), featurize_formula("Sr0.46875Ca0.5"))
    raw = defects.current_state_features()[defects._descriptor_dim:][:len(defects.state_labels)]
    assert raw[defects.state_labels.index("Sr")] == -.03125
    assert defects.state_feature_schema()["descriptor_mode"] == "resolved_composition"
    for action in [("Fe", 1), ("Co", 0), ("Mn", 0)]:
        choose(env, *action)
    assert env.terminal_cation_fractions()["A_defect"] == {"Sr": -.03125}


def test_charge_filter_receives_prior_groups_and_raw_additions_in_menu_order(tmp_path, monkeypatch):
    from rl_matdesign.constraints import charge
    observed = []
    def neutral(composition, **_):
        observed.append(dict(composition))
        return composition == {"O": 3, "Sr": 2, "Fe": 3}
    monkeypatch.setattr(charge, "charge_neutral", neutral)
    env, _ = build_authored(tmp_path, [
        {"name": "A", "kind": "composition", "species_set": ["Sr"], "amount_set": [2], "n_steps": 1},
        {"name": "B", "kind": "composition", "species_set": ["Fe"], "amount_set": [2, 1], "n_steps": 2,
         "filters": [{"type": "smact_charge", "scaffold_formula": "O3"}]},
    ])
    choose(env, "Sr", 2)
    choose(env, "Fe", 1)
    assert allowed(env) == {("Fe", Fraction(2))}
    assert {"O": 3, "Sr": 2, "Fe": 3} in observed
    assert {"O": 3, "Sr": 2, "Fe": 2} in observed
    choose(env, "Fe", 2)
    assert env.terminal_cation_fractions() == {"A": {"Sr": 2}, "B": {"Fe": 3}}


def test_same_formula_on_different_groups_has_distinct_candidate_identity(tmp_path):
    env, oracle = build_authored(tmp_path, [
        {"name": name, "kind": "composition", "species_set": ["Fe", "Ni"],
         "amount_set": [1], "n_steps": 1, "total": 1} for name in ("A", "B")
    ])
    rollout(env, [("Fe", 1), ("Ni", 1)])
    key, state = env.terminal_comp_key(), env.current_state_features()
    rollout(env, [("Ni", 1), ("Fe", 1)])
    assert env.terminal_comp_key() != key
    assert not np.array_equal(state, env.current_state_features())
    assert len(oracle.cache) == 2


def test_zero_valued_chemical_choices_do_not_change_final_recipe_identity():
    env, oracle, _ = build("glassnet_1pct_a2c")
    prefix = [("SiO2", .5), ("SiO2", .1), ("B2O3", .2), ("Al2O3", .1), ("Na2O", .1)]
    rollout(env, prefix + [("K2O", 0), ("K2O", 0)])
    key = env.terminal_comp_key()
    rollout(env, prefix + [("MgO", 0), ("MgO", 0)])
    assert env.terminal_comp_key() == key
    assert oracle.seen[0] == oracle.seen[1]
    assert len(oracle.cache) == 1
    assert all(value > 0 for value in oracle.seen[0].values())



def test_zero_choices_do_not_change_nested_structure_recipe_identity():
    env, oracle, _ = build("perovskite")
    rollout(env, [("Sr", 1), ("Ca", 0), ("Ba", 0),
                  ("Fe", 1), ("Co", 0), ("Mn", 0)])
    key = env.terminal_comp_key()
    rollout(env, [("Ba", 0), ("Sr", 1), ("Ca", 0),
                  ("Mn", 0), ("Co", 0), ("Fe", 1)])
    assert env.terminal_comp_key() == key
    assert oracle.seen[0] == oracle.seen[1] == {
        "A_dopant": {"Sr": 1}, "B_dopant": {"Fe": 1}}
    assert len(oracle.cache) == 1



def test_only_missing_private_configs_are_optional_in_clean_checkouts(tmp_path, monkeypatch):
    monkeypatch.setitem(build.__globals__, "CONFIGS", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="Private local configuration"):
        build("perovskite")
    with pytest.raises(FileNotFoundError):
        build("glassnet_1pct_a2c")


@pytest.mark.parametrize("name", PUBLIC_CONFIGS)
def test_public_configs_use_one_property_schema_and_section_order(name):
    path = CONFIGS / f"{name}.yaml"
    if name in PRIVATE_CONFIGS and not path.is_file():
        pytest.skip(f"Private local configuration is not supplied: {name}")
    cfg = yaml.safe_load(path.read_text())
    assert not {"predictor", "models", "model", "dp_models"}.intersection(cfg)
    sections = [key for key in ("groups", "structure", "properties", "reward") if key in cfg]
    assert list(cfg)[:len(sections)] == sections
    assert cfg["properties"]
    for prop in cfg["properties"]:
        assert "predictor" in prop
        assert not {"model_ref", "model", "dp_models"}.intersection(prop)
        if "models" in prop:
            assert isinstance(prop["models"], list) and prop["models"]


def test_requested_scenario_inventory_is_consolidated():
    assert len(PUBLIC_CONFIGS) == 25
    removed = ["ooh_dqn", "ooh_fine", "ooh_design_space_29", "ooh_design_space_34", "perovskite_level1"]
    removed += [f"oxides_{kind}_{variant}" for kind in ("calcine", "sinter", "sinter_calcine")
                for variant in ("a2c", "no_train_filter")]
    assert not any((CONFIGS / f"{name}.yaml").exists() for name in removed)


def test_ooh_retains_29_species_and_mc_training():
    _, _, cfg = build("ooh")
    assert len(cfg["groups"][0]["species_set"]) == 29
    assert cfg["method"] == "dqn" and cfg["dqn_target_mode"] == "mc"
    prop, = cfg["properties"]
    assert prop["predictor"] == "ooh_frames" and len(prop["models"]) == 5
    assert prop["direction"] == "min" and prop["objective"] == "mean_plus_kstd"


def test_perovskite_retains_supplied_gate_models_conditions_and_two_md_samples():
    _, _, cfg = build("perovskite")
    assert [group["name"] for group in cfg["groups"]] == ["A_dopant", "B_dopant"]
    assert all(group["unique"] and group["n_steps"] == 3 for group in cfg["groups"])
    assert cfg["structure"]["n_random_configs"] == 2
    substitute, md, opt = cfg["structure"]["steps"]
    assert [step["type"] for step in (substitute, md, opt)] == ["substitute", "md_average", "geo_opt"]
    assert md["model"] == "./compressed_dpa4c.pt2" and md["nsteps"] == 50000
    assert opt["model"] == "./dpa4.ckpt.pt" and opt["steps"] == 500
    delta = cfg["properties"][:3]
    assert [prop["fparam"] for prop in delta] == [[.2, .005, temperature] for temperature in (673.15, 773.15, 873.15)]
    assert all(len(prop["models"]) == 5 and prop["head"] == "delta" for prop in delta)
    classifier = cfg["properties"][3]
    assert classifier["models"] == ["./synthesizability_binary.pt"] and classifier["output_index"] == 1
    assert cfg["reward"]["type"] == "gated"
    assert cfg["reward"]["gates"][0]["min"] == .8
    assert cfg["reward"]["failure"]["reward"] == -10
    assert cfg["method"] == "dqn" and cfg["dqn_num_train_eps"] == 1600
