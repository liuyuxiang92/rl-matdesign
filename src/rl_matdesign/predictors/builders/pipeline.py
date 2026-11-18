"""Composable structure construction with stable original POSCAR block identities.

Planning is lazy and independent of models. A build executes each operation on
its predecessor; neither substitution nor later simulation reloads the template.
OOH frames are attached to the parent sample, never counted as new samples.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
import string
from typing import Any

import numpy as np

from ...utils.structure_samples import (
    StructureSample, attach_sample_identity, canonical_sample_candidate,
    execute_many_seeded_samples, execute_seeded_samples, positive_sample_count,
    sample_plan as make_sample_plan,
)

ORIGINAL_IDS = "_rl_original_id"
PIPELINE_INFO = "_rl_matdesign_pipeline"
FRAMES_INFO = "_rl_matdesign_frames"
_MODES = ("O*", "OH*", "OOH*")
_STEP_FIELDS = {
    "substitute": {"amount_basis", "interstitial_min_dist", "vacancy_policy"},
    "sse_charge_balance": {"p_site_group", "halide_group", "eligible_region", "eligible_region_fallback", "host", "halide_total", "valences", "p_site_per_fu", "s_site_per_fu", "li_per_fu"},
    "ooh_adsorbates": {"adsorbates", "ads_height", "ads_dz"},
    "geo_opt": {"model", "head", "fmax", "steps", "relax_cell"},
    "md_average": {"model", "head", "lmp_bin", "lmp_args", "env", "atom_style", "average_last_ps", "avg_rms_max", "dump_freq", "nsteps", "pair_style", "plugin", "pressure_bar", "tau_p", "tau_t", "temperature_k", "thermo_freq", "timestep_ps", "min_dist", "velocity_seed", "scratch_dir", "keep_scratch"},
}


def _selectors(step):
    return [(letter, step[letter + "_site_symbol"], step[letter + "_dopant_group"])
            for letter in string.ascii_lowercase if letter + "_site_symbol" in step]


def validate_pipeline_config(cfg: Mapping) -> None:
    """Validate the operation language without reading files or loading models."""
    if not isinstance(cfg, Mapping):
        raise TypeError("structure must be a mapping")
    unknown = set(cfg) - {"builder", "base_poscar", "n_random_configs", "steps", "structure_parallel"}
    if unknown:
        raise ValueError(f"Unsupported pipeline structure fields: {sorted(unknown)}")
    if "sites" in cfg or "site_map" in cfg:
        raise ValueError("pipeline selects original POSCAR symbol blocks; sites/site_map are unsupported")
    _template_specs(cfg.get("base_poscar"))
    steps = cfg.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("pipeline requires a nonempty steps list")
    seen_groups = set()
    group_bases = {}
    balanced_groups = set()
    adsorbates_seen = False
    for i, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("type") not in _STEP_FIELDS:
            raise ValueError(f"Unknown pipeline step {i}: {step!r}")
        kind = step["type"]
        allowed = _STEP_FIELDS[kind] | {"type"}
        if kind == "substitute":
            if adsorbates_seen:
                raise ValueError("substitute must precede ooh_adsorbates")
            allowed |= {f"{letter}_{suffix}" for letter in string.ascii_lowercase
                        for suffix in ("site_symbol", "dopant_group", "defect_group")}
            for letter in string.ascii_lowercase:
                present = [letter + "_" + key in step for key in ("site_symbol", "dopant_group")]
                if any(present) and not all(present):
                    raise ValueError(f"substitute requires paired {letter}_site_symbol/{letter}_dopant_group")
                if letter + "_defect_group" in step and not all(present):
                    raise ValueError(f"{letter}_defect_group requires its symbol/group pair")
            if not _selectors(step):
                raise ValueError("substitute requires at least one a..z symbol/group pair")
            for letter, symbol, group in _selectors(step):
                if not isinstance(symbol, str) or not symbol or not isinstance(group, str) or not group:
                    raise ValueError("site symbols and dopant group names must be nonempty strings")
                if group in seen_groups:
                    raise ValueError(f"Group {group!r} is bound more than once")
                seen_groups.add(group)
                group_bases[group] = step.get("amount_basis", "fraction")
            if step.get("amount_basis", "fraction") not in ("fraction", "cation_fraction"):
                raise ValueError("substitute amount_basis must be fraction or cation_fraction")
            if step.get("amount_basis") == "cation_fraction" and any(k.endswith("_defect_group") for k in step):
                raise ValueError("SSE cation_fraction substitution cannot also carry signed defects")
            if step.get("vacancy_policy", "clamp") not in ("clamp", "error"):
                raise ValueError("vacancy_policy must be clamp or error")
        elif kind == "ooh_adsorbates":
            if adsorbates_seen:
                raise ValueError("Only one ooh_adsorbates stage is supported")
            adsorbates_seen = True
            modes = step.get("adsorbates", list(_MODES))
            if not isinstance(modes, (list, tuple)) or any(x not in _MODES for x in modes) or len(set(modes)) != len(modes):
                raise ValueError("adsorbates must be a unique list of O*, OH*, OOH* (or [] for bare)")
        elif kind == "sse_charge_balance":
            if adsorbates_seen:
                raise ValueError("sse_charge_balance must precede ooh_adsorbates")
            group = step.get("p_site_group")
            if group not in seen_groups or group_bases.get(group) != "cation_fraction":
                raise ValueError("sse_charge_balance requires an earlier cation_fraction substitution")
            if group in balanced_groups:
                raise ValueError("A P group may be charge-balanced only once")
            balanced_groups.add(group)
            host = step.get("host", {})
            if not isinstance(host, Mapping) or set(host) - {"P", "S", "Li"}:
                raise ValueError("SSE host keys must be P, S, Li")
            if not isinstance(step.get("valences"), Mapping):
                raise ValueError("SSE valences must be a mapping")
            for key in ("p_site_group", "halide_group", "valences"):
                if not step.get(key):
                    raise ValueError(f"sse_charge_balance requires {key}")
        elif kind in ("geo_opt", "md_average"):
            if not step.get("model"):
                raise ValueError(f"{kind} requires model")
            if kind == "md_average" and adsorbates_seen:
                raise ValueError("md_average must precede ooh_adsorbates")
        unknown = set(step) - allowed
        if unknown:
            raise ValueError(f"Unsupported {kind} fields: {sorted(unknown)}")
    if {g for g, basis in group_bases.items() if basis == "cation_fraction"} != balanced_groups:
        raise ValueError("Every cation_fraction substitution requires its own sse_charge_balance stage")


@dataclass(frozen=True)
class _TemplateSpec:
    path: str
    n_configs: int | None = None


def _template_specs(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError("pipeline requires at least one base_poscar")
    result = []
    for entry in values:
        if isinstance(entry, (str, Path)) and str(entry):
            result.append(_TemplateSpec(str(entry)))
        elif isinstance(entry, Mapping) and entry.get("path"):
            unknown = set(entry) - {"path", "n_configs"}
            if unknown:
                raise ValueError(f"Unsupported pipeline template fields: {sorted(unknown)}")
            n = entry.get("n_configs")
            result.append(_TemplateSpec(str(entry["path"]), None if n is None else positive_sample_count(n)))
        else:
            raise ValueError("pipeline base_poscar must be a path or list of paths/{path,n_configs}")
    return result


def _recipe(label):
    """Return (cation, oxygen per cation); binary oxides are explicit recipes."""
    match = re.fullmatch(r"([A-Z][a-z]?)(\d*)(?:O(\d*))?", str(label))
    if not match or match.group(1) == "O":
        raise ValueError(f"cation_fraction requires an element or binary metal oxide, got {label!r}")
    from ase.data import atomic_numbers
    metal, count, oxygen = match.groups()
    if metal not in atomic_numbers:
        raise ValueError(f"Unknown element {metal!r}")
    n_metal = int(count or 1)
    if n_metal <= 0 or (oxygen is None and n_metal != 1):
        raise ValueError(f"Invalid cation formula {label!r}")
    n_oxygen = 0 if "O" not in str(label)[len(metal):] else int(oxygen or 1)
    if n_oxygen < 0:
        raise ValueError(f"Invalid oxide formula {label!r}")
    return metal, n_oxygen / n_metal


def pipeline_cation_amounts(amounts: Mapping[str, float]) -> dict[str, float]:
    """Expand cation fractions without mistaking oxide labels for formula units."""
    result = Counter()
    for label, amount in amounts.items():
        metal, oxygen = _recipe(label)
        amount = float(amount)
        result[metal] += amount
        if oxygen:
            result["O"] += oxygen * amount
    return dict(result)


def pipeline_group_constraints(structure_cfg: Mapping, groups=None):
    """Intrinsic partial-state predicates; never depend on template/model assets."""
    result = {}
    for step in structure_cfg.get("steps", []):
        if step.get("type") == "sse_charge_balance":
            def one_form(amounts):
                seen = set()
                for label, value in amounts.items():
                    if float(value) <= 0:
                        continue
                    metal, _ = _recipe(label)
                    if metal in seen:
                        return False
                    seen.add(metal)
                return True
            result[step["p_site_group"]] = one_form
    return result


def _amounts(candidate, group):
    value = candidate.get(group)
    if value is None and all(not isinstance(x, Mapping) for x in candidate.values()):
        value = candidate
    if not isinstance(value, Mapping):
        raise ValueError(f"Candidate group {group!r} must contain chemical amounts")
    result = {str(k): float(v) for k, v in sorted(value.items()) if float(v) != 0}
    if any(not np.isfinite(v) for v in result.values()):
        raise ValueError(f"Non-finite amounts in {group}")
    return result


def pipeline_frames(atoms):
    """Decode (label, frame, mask_indices) for the parent sample's OOH batch."""
    from ase import Atoms
    result = []
    for record in atoms.info.get(FRAMES_INFO, []):
        frame = Atoms(record["symbols"], positions=record["positions"], cell=record["cell"], pbc=record["pbc"])
        frame.set_array(ORIGINAL_IDS, np.asarray(record["original_ids"], dtype=int))
        frame.info = deepcopy(atoms.info)
        frame.info.pop(FRAMES_INFO, None)
        frame.info["_rl_matdesign_adsorbate"] = {k: deepcopy(v) for k, v in record.items()
                                                 if k not in ("symbols", "positions", "cell", "pbc", "original_ids")}
        result.append((record["label"], frame, list(record["mask_indices"])))
    return result


def _frame_record(label, atoms, masks, **metadata):
    return dict(label=label, symbols=atoms.get_chemical_symbols(), positions=atoms.positions.tolist(),
                cell=atoms.cell.array.tolist(), pbc=atoms.pbc.tolist(), mask_indices=list(masks),
                original_ids=atoms.arrays[ORIGINAL_IDS].tolist(), **metadata)


class PipelineBuilder:
    """One original-template namespace and an ordered sequence of operations."""

    def __init__(self, cfg: Mapping[str, Any], *, seed=None):
        validate_pipeline_config(cfg)
        self.cfg = deepcopy(dict(cfg))
        self.steps = self.cfg["steps"]
        # Timing consumers inspect these capabilities without executing stages.
        self.md = next((s for s in self.steps if s["type"] == "md_average"), None)
        self.opt = next((s for s in self.steps if s["type"] == "geo_opt"), None)
        self.timing_stages = tuple(name for name, stage in (("md", self.md), ("opt", self.opt)) if stage)
        self.specs = _template_specs(cfg["base_poscar"])
        self._seed = seed
        self._templates = {}
        self._calculators = {}
        self.last_sample_records = []

    def close(self):
        """Release lazily loaded model and template references."""
        for calculator in self._calculators.values():
            close = getattr(calculator, "close", None)
            if callable(close):
                close()
        self._calculators.clear()
        self._templates.clear()

    def _template(self, index):
        """Read a VASP5 template once, preserving repeated element header blocks."""
        path = self.specs[index].path
        if path not in self._templates:
            from io import StringIO
            from ase.io import read
            text = Path(path).read_text()
            lines = text.splitlines()
            if len(lines) < 8:
                raise ValueError(f"Incomplete POSCAR: {path}")
            symbols = lines[5].split()
            try:
                counts = [int(x) for x in lines[6].split()]
            except ValueError as exc:
                raise ValueError("pipeline requires a VASP5 POSCAR with explicit element header blocks") from exc
            if len(symbols) != len(counts) or any(n < 1 for n in counts):
                raise ValueError("POSCAR element/count headers must have matching nonempty blocks")
            atoms = read(StringIO(text), format="vasp")
            if sum(counts) != len(atoms):
                raise ValueError("POSCAR header population does not match atoms")
            blocks, start = [], 0
            for block_index, (symbol, count) in enumerate(zip(symbols, counts)):
                ids = list(range(start, start + count))
                if any(atoms[i].symbol != symbol for i in ids):
                    raise ValueError("POSCAR parsed species disagree with original header block")
                blocks.append(dict(index=block_index, symbol=symbol, population=count, original_ids=ids))
                start += count
            atoms.set_array(ORIGINAL_IDS, np.arange(len(atoms), dtype=int))
            self._templates[path] = (atoms, blocks)
        return self._templates[path]

    def _bindings(self, blocks):
        consumed, result = set(), {}
        for step_index, step in enumerate(self.steps):
            if step["type"] != "substitute":
                continue
            for letter, symbol, group in _selectors(step):
                block = next((b for b in blocks if b["symbol"] == symbol and b["index"] not in consumed), None)
                if block is None:
                    raise ValueError(f"No unused original POSCAR block for {letter}_site_symbol={symbol!r}")
                consumed.add(block["index"])
                result[group] = dict(block, letter=letter, group=group, step_index=step_index,
                                     amount_basis=step.get("amount_basis", "fraction"))
        return result

    def plan_candidate(self, candidate, *, template_index=0):
        """Plan counts/identities only; no placement, relaxation, or model loading."""
        from ...utils.structure import _fractions_to_counts, resolve_region
        template, blocks = self._template(template_index)
        bindings = self._bindings(blocks)
        total = Counter(template.get_chemical_symbols())
        plans, sse = {}, {}
        balance = {s["p_site_group"]: s for s in self.steps if s["type"] == "sse_charge_balance"}
        for group, binding in bindings.items():
            amount = _amounts(candidate, group)
            if any(v < 0 for v in amount.values()) or not np.isclose(sum(amount.values()), 1, atol=1e-8, rtol=0):
                raise ValueError(f"Substitution group {group!r} requires nonnegative fractions summing to 1")
            n = binding["population"]
            if binding["amount_basis"] == "cation_fraction":
                if group not in balance:
                    raise ValueError("cation_fraction requires a matching sse_charge_balance stage")
                c = self._sse_counts(candidate, binding, balance[group], template)
                counts = {**c["metals"], c["host"]["P"]: c["P"]}
                sse[group] = c
            else:
                from ase.data import atomic_numbers
                if any(label not in atomic_numbers for label in amount):
                    raise ValueError("fraction substitution species must be elements; use cation_fraction for SSE oxides")
                counts = _fractions_to_counts(amount, n)
            total[binding["symbol"]] -= n
            total.update(counts)
            step = self.steps[binding["step_index"]]
            defects = {}
            defect_group = step.get(binding["letter"] + "_defect_group")
            if defect_group:
                for element, fraction in _amounts(candidate, defect_group).items():
                    from ase.data import atomic_numbers
                    if element not in atomic_numbers:
                        raise ValueError(f"Defect species must be an element, got {element!r}")
                    available = counts.get(element, 0)
                    delta = int(round(fraction * n))
                    if delta < -available:
                        if step.get("vacancy_policy", "clamp") == "error":
                            raise ValueError(f"Vacancy request exceeds available {element} sites")
                        delta = -available
                    defects[element] = delta
                    total[element] += delta
            plans[group] = dict(binding, counts=counts, defects=defects)
        for group, c in sse.items():
            step = balance[group]
            subcount = c["O"] + c["Cl"] + c["Br"]
            region = resolve_region(template, step.get("eligible_region", c["host"]["S"]))
            fallback = step.get("eligible_region_fallback")
            if subcount > len(region) and fallback is not None:
                region = resolve_region(template, fallback)
            if len(set(region)) != len(region) or any(i < 0 or i >= len(template) for i in region):
                raise ValueError("SSE eligible region contains duplicate or out-of-range original sites")
            if subcount > len(region) or any(template[i].symbol != c["host"]["S"] for i in region):
                raise ValueError("SSE eligible region must contain enough original host S sites")
            c["anion_original_ids"] = [int(i) for i in region]
            c["li_original_ids"] = [i for i, atom in enumerate(template) if atom.symbol == c["host"]["Li"]]
            controlled = {i for name, b in bindings.items() if name != group for i in b["original_ids"]}
            if controlled.intersection(c["anion_original_ids"] + c["li_original_ids"]):
                raise ValueError("SSE anion/Li sites overlap another substitution group")
            total[c["host"]["S"]] -= subcount
            total.update({"O": c["O"], "Cl": c["Cl"], "Br": c["Br"]})
            total[c["host"]["Li"]] -= c["Li_delete"]
        return dict(template_index=template_index, blocks=deepcopy(blocks), bindings=plans,
                    sse=sse, counts={k: int(v) for k, v in sorted(total.items()) if v})

    @staticmethod
    def _sse_counts(candidate, binding, step, template):
        host = {"P": "P", "S": "S", "Li": "Li", **step.get("host", {})}
        amount = _amounts(candidate, binding["group"])
        predicate = pipeline_group_constraints({"steps": [step]})[binding["group"]]
        if not predicate(amount):
            raise ValueError("SSE requires at most one positive form per cation")
        def valence(element, scenario="sulfide"):
            value = step["valences"].get(element)
            if isinstance(value, Mapping):
                value = value.get(scenario)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"Missing {scenario} valence for {element}")
            return float(value)
        n_p = binding["population"]
        p_fu = int(step.get("p_site_per_fu", 1))
        if p_fu < 1 or n_p % p_fu:
            raise ValueError("P block population must be a multiple of p_site_per_fu")
        fu = n_p // p_fu
        n_s = int(step.get("s_site_per_fu", 6)) * fu
        n_li = int(step.get("li_per_fu", 6)) * fu
        original = Counter(template.get_chemical_symbols())
        if binding["symbol"] != host["P"] or original[host["P"]] != n_p or original[host["S"]] != n_s or original[host["Li"]] != n_li:
            raise ValueError("SSE requires the original P/S/Li populations to match configured formula-unit counts")
        metals, forms, n_o = {}, {}, 0
        for label, fraction in amount.items():
            metal, oxygen = _recipe(label)
            if metal == host["P"]:
                if oxygen:
                    raise ValueError("The SSE host P fraction must use its elemental label")
                continue
            count = int(round(fraction * n_p))  # independent rounding preserves legacy SSE semantics
            metals[metal] = count
            forms[metal] = bool(oxygen)
            if oxygen:
                v = valence(metal, "oxide")
                if not np.isclose(oxygen * 2, v):
                    raise ValueError(f"Oxide recipe {label} disagrees with configured oxide valence {v}")
                n_o += int(round(0.5 * v * count))
        residual_p = n_p - sum(metals.values())
        halide = _amounts(candidate, step["halide_group"])
        if set(halide) - {"Cl"} or any(v < 0 for v in halide.values()):
            raise ValueError("SSE halide group must contain only nonnegative Cl amount per formula unit")
        n_cl = int(round(halide.get("Cl", 0) * fu))
        n_br = int(round(float(step.get("halide_total", 1.7)) * fu)) - n_cl
        residual_s = n_s - n_o - n_cl - n_br
        if min(residual_p, residual_s, n_br) < 0:
            raise ValueError("Infeasible SSE P/S/Br allocation")
        anion = residual_s * abs(valence(host["S"])) + n_o * abs(valence("O")) + n_cl * abs(valence("Cl")) + n_br * abs(valence("Br"))
        non_li = residual_p * valence(host["P"]) + sum(n * valence(m, "oxide" if forms[m] else "sulfide") for m, n in metals.items())
        v_li = valence(host["Li"])
        if v_li <= 0:
            raise ValueError("Li valence must be positive")
        neutral_li = int(round((anion - non_li) / v_li))
        if not 0 <= neutral_li <= n_li:
            raise ValueError(f"Charge-neutral Li count {neutral_li} is outside original population [0,{n_li}]")
        return dict(metals=metals, o_forms=forms, P=residual_p, O=n_o, Cl=n_cl, Br=n_br,
                    S=residual_s, Li=neutral_li, Li_delete=n_li-neutral_li, host=host, formula_units=fu)

    def composition_formula(self, candidate):
        plan = self.plan_candidate(candidate)
        fu = next(iter(plan["sse"].values()))["formula_units"] if plan["sse"] else 1
        if any(step["type"] == "ooh_adsorbates" for step in self.steps):
            fu = sum(b["population"] for b in plan["bindings"].values()) or 1
        return "".join(f"{element}{count/fu:.12g}" for element, count in sorted(plan["counts"].items(), key=lambda item: (-item[1], item[0])))

    def sample_plan(self, candidate, *, n_configs, source_spec, eval_seed):
        return make_sample_plan(candidate, n_configs=n_configs, source_spec=source_spec, eval_seed=eval_seed,
                                templates=[dict(path=s.path, site_symbol="pipeline", n_configs=s.n_configs) for s in self.specs])

    def build_sample(self, candidate, *, sample: StructureSample):
        i = sample.template_index
        if i is None or not 0 <= i < len(self.specs) or sample.template_path != self.specs[i].path or sample.site_symbol != "pipeline":
            raise ValueError("Pipeline sample does not match a configured template")
        atoms = self._build_one(canonical_sample_candidate(candidate), i, np.random.default_rng(sample.seed))
        attach_sample_identity(atoms, sample)
        return atoms

    def build(self, candidate, *, n_configs=1, rng=None):
        n_configs = positive_sample_count(n_configs)
        rng = np.random.default_rng(self._seed) if rng is None else rng
        return [self._build_one(canonical_sample_candidate(candidate), i, rng)
                for i, spec in enumerate(self.specs) for _ in range(spec.n_configs or n_configs)]

    def build_seeded(self, candidate, *, n_configs, source_spec, eval_seed):
        return execute_seeded_samples(self, candidate, n_configs=n_configs, source_spec=source_spec, eval_seed=eval_seed)

    def build_many_seeded(self, candidates, *, n_configs, source_spec, eval_seed):
        return execute_many_seeded_samples(self, candidates, n_configs=n_configs, source_spec=source_spec, eval_seed=eval_seed)

    @staticmethod
    def _indices(atoms, original_ids):
        requested = set(original_ids)
        return [i for i, identity in enumerate(atoms.arrays[ORIGINAL_IDS]) if int(identity) in requested]

    @staticmethod
    def _apply_ops(atoms, operations, rng):
        from ...utils.structure import build_substituted_structure
        return build_substituted_structure(atoms, operations, n_configs=1, rng=rng)[0]

    def _build_one(self, candidate, index, rng):
        from ...utils.structure import SublatticeOp, _insert_interstitials
        plan = self.plan_candidate(candidate, template_index=index)
        atoms = self._template(index)[0].copy()
        atoms.info[PIPELINE_INFO] = deepcopy(plan)
        for step_index, step in enumerate(self.steps):
            kind = step["type"]
            if kind == "substitute":
                for binding in plan["bindings"].values():
                    if binding["step_index"] != step_index:
                        continue
                    indices = self._indices(atoms, binding["original_ids"])
                    if len(indices) != binding["population"]:
                        raise ValueError("An earlier operation removed sites needed by a substitution block")
                    atoms = self._apply_ops(atoms, [SublatticeOp(sites=indices, put=binding["counts"])], rng)
                    for element, delta in binding["defects"].items():
                        if delta < 0:
                            eligible = [i for i in self._indices(atoms, binding["original_ids"]) if atoms[i].symbol == element]
                            atoms = self._apply_ops(atoms, [SublatticeOp(sites=eligible, remove=-delta)], rng)
                        elif delta > 0:
                            old_len = len(atoms)
                            _insert_interstitials(atoms, [(element, delta)], rng=rng,
                                                  min_dist=float(step.get("interstitial_min_dist", 1.7)))
                            atoms.arrays[ORIGINAL_IDS][old_len:] = -1
            elif kind == "sse_charge_balance":
                c = plan["sse"].get(step["p_site_group"])
                if c is None:
                    raise ValueError("SSE substitution must specify amount_basis: cation_fraction")
                atoms = self._apply_ops(atoms, [
                    SublatticeOp(sites=self._indices(atoms, c["anion_original_ids"]), put={"O": c["O"], "Cl": c["Cl"], "Br": c["Br"]}),
                    SublatticeOp(sites=self._indices(atoms, c["li_original_ids"]), remove=c["Li_delete"]),
                ], rng)
            elif kind == "ooh_adsorbates":
                self._adsorbates(atoms, plan, step, rng)
            elif kind == "geo_opt":
                frames = pipeline_frames(atoms)
                if frames:
                    records = []
                    for label, frame, masks in frames:
                        metadata = frame.info["_rl_matdesign_adsorbate"]
                        result = self._relax(frame, step_index, step, masks)
                        records.append(_frame_record(label, result, masks, **{k:v for k,v in metadata.items() if k not in ("label", "mask_indices")}))
                    atoms.info[FRAMES_INFO] = records
                else:
                    atoms = self._relax(atoms, step_index, step, [])
            elif kind == "md_average":
                from .pipeline_md import md_average
                atoms = md_average(atoms, step, rng=rng)
        return atoms

    def _relax(self, atoms, index, step, masks):
        from ...utils import structure
        if index not in self._calculators:
            from deepmd.calculator import DP
            kwargs = {"head": step["head"]} if step.get("head") else {}
            self._calculators[index] = DP(model=step["model"], **kwargs)
        with structure.strict_relaxation_errors():
            return structure.relax_structure(atoms, calc=self._calculators[index],
                fmax=float(step.get("fmax", .001)), steps=int(step.get("steps", 1000)),
                relax_cell=bool(step.get("relax_cell", True)), mask_indices=masks)

    def _adsorbates(self, atoms, plan, step, rng):
        from ase import Atoms
        modes = step.get("adsorbates", list(_MODES))
        target_ids = sorted({i for b in plan["bindings"].values() for i in b["original_ids"]})
        eligible = self._indices(atoms, target_ids)
        if modes and not eligible:
            raise ValueError("ooh_adsorbates requires surviving original substitute target sites")
        candidates = sorted(eligible, key=lambda i: atoms.positions[i, 2], reverse=True)[:20]
        normal = atoms.cell.array[2].copy()
        length = float(np.linalg.norm(normal))
        if modes and length <= 0:
            raise ValueError("OOH slab requires a nonzero third lattice vector")
        if length:
            normal /= length
        records = []
        for mode in modes:
            anchor = int(rng.choice(candidates))
            frame = atoms.copy()
            first = atoms.positions[anchor] + float(step.get("ads_height", 1.9)) * normal
            second = first + float(step.get("ads_dz", 1.0)) * normal
            third = second + float(step.get("ads_dz", 1.0)) * normal
            n = len(frame)
            frame += Atoms("OOH", positions=[first, second, third])
            frame.arrays[ORIGINAL_IDS][n:] = -1
            masks = {"O*": [n+1, n+2], "OH*": [n+1], "OOH*": []}[mode]
            records.append(_frame_record(mode, frame, masks, anchor_index=anchor,
                anchor_original_id=int(atoms.arrays[ORIGINAL_IDS][anchor]), original_target_ids=target_ids,
                eligible_anchor_indices=eligible, eligible_anchor_original_ids=[int(atoms.arrays[ORIGINAL_IDS][i]) for i in eligible]))
        if not modes:
            records.append(_frame_record("bare", atoms, [], original_target_ids=target_ids,
                                         eligible_anchor_indices=eligible, eligible_anchor_original_ids=[int(atoms.arrays[ORIGINAL_IDS][i]) for i in eligible]))
        atoms.info[FRAMES_INFO] = records
