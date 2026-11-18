"""End-to-end (no-GPU) integration for the LiPS config on the general multi_group.

Builds the public chemical amount groups and their operation-owned masks,
then plans the shared SSE pipeline against a synthetic template. No model or
geometry optimization runs; the planned atom counts must be charge neutral.
"""
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.configuration import build_env, load_config  # noqa: E402
from rl_matdesign.predictors.builders.pipeline import PipelineBuilder  # noqa: E402
from pymatgen.core import Composition  # noqa: E402

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "lips_sse.yaml")


def _load_cfg():
    return load_config(CFG_PATH)


def _build_env(cfg):
    class Predictor:
        def predict(self, candidate):
            return 0.0, 0.0
    return build_env(cfg, Predictor())[0]


def _val(valences, el, sc):
    v = valences[el]
    return v[sc] if isinstance(v, dict) else v


def test_lips_config_is_friendly():
    """The config carries no internal-encoding knobs (checks parsed keys)."""
    cfg = _load_cfg()
    assert "env_type" not in cfg                       # removed public key
    assert "constraint_filter" not in cfg
    # Amount menus are public; lattice units and legacy O-form flags are not.
    banned = {"cl_map", "o_off", "o_on", "formula_units", "total_units"}
    keys = set(cfg) | {k for g in cfg["groups"] for k in g}
    assert not (keys & banned), f"internal-encoding knobs leaked back: {keys & banned}"
    p_site = next(g for g in cfg["groups"] if g["name"] == "P_site")
    assert "P" in p_site["species_set"]                 # host listed explicitly


def test_lips_masking_and_charge_neutrality(tmp_path):
    cfg = _load_cfg()
    # The configured recipe infers 500 formula units from the original P block.
    # Positions are irrelevant to count planning; no structure is relaxed here.
    template = tmp_path / "POSCAR"
    total_atoms = 3000 + 500 + 3000
    lines = ["SSE test", "1", "100 0 0", "0 100 0", "0 0 100",
             "Li P S", "3000 500 3000", "Direct"]
    lines.extend(f"{i / total_atoms} 0 0" for i in range(total_atoms))
    template.write_text("\n".join(lines) + "\n")
    cfg = deepcopy(cfg)
    cfg["structure"]["base_poscar"] = str(template)
    env = _build_env(cfg)

    from rl_matdesign.structure_sources import resolve_structure_plan
    properties = [prop["name"] for prop in cfg["properties"]]
    plan = resolve_structure_plan(cfg, needs_structure=properties)
    source_ids = {plan.property_sources[name] for name in properties}
    assert len(source_ids) == 1
    builder_cfg = plan.structure.effective_config
    builder = PipelineBuilder(builder_cfg)
    balance = next(step for step in builder_cfg["steps"] if step["type"] == "sse_charge_balance")
    valences = balance["valences"]
    p_group = next(g for g in cfg["groups"] if g["name"] == "P_site")
    halide = next(g for g in cfg["groups"] if g["name"] == "halide")
    assert p_group["kind"] == halide["kind"] == "composition"
    assert "S_site" not in env.group_names

    def recipe(label):
        counts = Composition(label).get_el_amt_dict()
        metal = next(element for element in counts if element != "O")
        return metal, counts.get("O", 0) / counts[metal]

    # Explicitly verify that a selected metal form excludes only its alternate
    # oxide form, while selecting more of the same form remains legal.
    def choose(label, amount):
        inner = env._active()
        selected = next(a for a in env.allowed_actions()
                        if inner.species_set[int(np.argmax(env._from_union(a)[0]))] == label
                        and float(inner.action_values[int(np.argmax(env._from_union(a)[1]))]) == amount)
        env.step(selected)
    choose("P", .92)
    choose("Ni", .04)
    inner = env._active()
    remaining = {(inner.species_set[int(np.argmax(env._from_union(a)[0]))],
                  float(inner.action_values[int(np.argmax(env._from_union(a)[1]))]))
                 for a in env.allowed_actions()}
    assert ("Ni2O3", .04) not in remaining and ("Ni", .04) in remaining

    rng = np.random.default_rng(0)
    seen_metals = set()
    for _ in range(300):
        env.initialize()
        while not env.is_terminal:
            actions = env.allowed_actions()
            assert actions, "agent stranded with no legal action"
            env.step(actions[int(rng.integers(len(actions)))])
        term = env.terminal_cation_fractions()
        p_site, halide_amounts = term["P_site"], term["halide"]
        active_forms = {label: value for label, value in p_site.items() if label != "P" and value > 0}
        metals = [recipe(label)[0] for label in active_forms]
        assert 1 <= len(metals) <= 2
        assert len(metals) == len(set(metals))  # One positive form per base metal.
        seen_metals.update(metals)
        dopant_total = sum(active_forms.values())
        assert 0.02 - 1e-9 <= dopant_total <= 0.08 + 1e-9
        assert 0.92 - 1e-9 <= p_site["P"] <= 0.98 + 1e-9
        assert abs(p_site["P"] + dopant_total - 1.0) < 1e-9
        assert halide_amounts["Cl"] in halide["amount_set"]

        planned = builder.plan_candidate(term)
        counts = planned["sse"]["P_site"]
        assert counts["formula_units"] == 500
        assert counts["metals"] == {recipe(label)[0]: round(fraction * 500)
                                    for label, fraction in active_forms.items()}
        assert counts["O"] == sum(round(recipe(label)[1] * round(fraction * 500))
                                  for label, fraction in active_forms.items())
        positive = counts["Li"] + counts["P"] * 5
        for metal, number in counts["metals"].items():
            scenario = "oxide" if counts["o_forms"][metal] else "sulfide"
            positive += number * _val(valences, metal, scenario)
        negative = 2 * counts["S"] + 2 * counts["O"] + counts["Cl"] + counts["Br"]
        assert positive == negative, (active_forms, counts)
        assert counts["Br"] == round(1.7 * 500) - counts["Cl"]
        assert 0 <= counts["Li_delete"] <= 3000
        expected_counts = {**counts["metals"], **{element: counts[element]
                           for element in ("Li", "P", "S", "O", "Cl", "Br")}}
        assert planned["counts"] == {element: number for element, number in expected_counts.items() if number}
    assert len(seen_metals) > 10
