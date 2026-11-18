"""Bind operation-owned chemistry to generic groups before action selection."""
from __future__ import annotations


def bind_pipeline_domains(structure, groups, env):
    from .amounts import ChemicalAmountsGroup
    from .predictors.builders.pipeline import (
        pipeline_cation_amounts, pipeline_group_constraints, validate_pipeline_config,
    )

    validate_pipeline_config(structure)
    domains = dict(zip(env.group_names, env._inners))
    for name, predicate in pipeline_group_constraints(structure, groups).items():
        if name not in domains or not isinstance(domains[name], ChemicalAmountsGroup):
            raise ValueError(f"Pipeline chemistry needs chemical composition group {name!r}")
        domains[name].candidate_validator = predicate
    for step in structure["steps"]:
        if step["type"] != "substitute":
            continue
        for prefix in "abcdefghijklmnopqrstuvwxyz":
            key = f"{prefix}_dopant_group"
            if key not in step:
                continue
            name = step[key]
            if name not in domains or not isinstance(domains[name], ChemicalAmountsGroup):
                raise ValueError(f"Substitution needs chemical composition group {name!r}")
            if step.get("amount_basis") == "cation_fraction":
                domains[name].composition_resolver = pipeline_cation_amounts
            defect_name = step.get(f"{prefix}_defect_group")
            if defect_name is not None:
                if defect_name not in domains or not isinstance(domains[defect_name], ChemicalAmountsGroup):
                    raise ValueError(f"Defects need chemical composition group {defect_name!r}")
                host = domains[name]
                def effective_composition(edits, host=host):
                    result = host.cation_fractions()
                    for symbol, delta in edits.items():
                        result[symbol] = max(0.0, result.get(symbol, 0.0) + delta)
                    return result
                domains[defect_name].composition_resolver = effective_composition
