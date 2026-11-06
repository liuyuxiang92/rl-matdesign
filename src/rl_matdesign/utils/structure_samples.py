"""Explicit, deterministic structure samples shared by serial and worker paths.

Only builders with an explicit template/sample contract can promise bitwise
serial/parallel parity. An arbitrary external builder's native ensemble cannot
be inferred from repeated calls with ``n_configs=1``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .rollout_errors import EvaluationProtocolError


SAMPLE_SCHEMA_VERSION = 1
SAMPLE_INFO_KEY = "_rl_matdesign_sample"


class SamplingProtocolError(EvaluationProtocolError):
    """A builder cannot honor the configured deterministic sample protocol.

    This is a configuration error, not a failed scientific candidate. Callers
    must propagate it instead of retrying it as a terminal-reward failure.
    """


def require_sample_capability(builder: Any) -> None:
    if not callable(getattr(builder, "build_sample", None)):
        raise SamplingProtocolError(
            "Seeded sample execution requires explicit build_sample(candidate, *, sample) "
            "capability; an opaque native build ensemble cannot be split reproducibly."
        )


def sample_plan_failure_record(source_spec: Any, error: Exception) -> dict:
    """A failed plan has a source identity, but no fabricated sample identity."""
    return {"schema_version": SAMPLE_SCHEMA_VERSION, "source_id": source_spec.source_id,
            "source_fingerprint": source_spec.protocol_fingerprint, "status": "plan_failed",
            "error": f"{type(error).__name__}: {error}"}


def positive_sample_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("n_configs must be a positive integer")
    return int(value)


def canonical_sample_candidate(candidate: Any) -> Any:
    """Keep typed values and sequence positions, but fix mapping iteration order."""
    if isinstance(candidate, Mapping):
        if any(not isinstance(key, str) for key in candidate):
            raise ValueError("Structure candidate mapping keys must be strings")
        return {key: canonical_sample_candidate(candidate[key]) for key in sorted(candidate)}
    if isinstance(candidate, list):
        return [canonical_sample_candidate(value) for value in candidate]
    if isinstance(candidate, tuple):
        return tuple(canonical_sample_candidate(value) for value in candidate)
    return candidate


@dataclass(frozen=True)
class StructureSample:
    source_id: str
    source_fingerprint: str
    template_index: int | None
    template_path: str | None
    site_symbol: str | None
    sample_index: int
    seed: int

    @property
    def template_id(self) -> dict | None:
        if self.template_index is None:
            return None
        return {"index": self.template_index, "path": self.template_path,
                "site_symbol": self.site_symbol}

    def record(self, *, status: str | None = None, error: str | None = None) -> dict:
        result = {"schema_version": SAMPLE_SCHEMA_VERSION, "source_id": self.source_id,
                  "source_fingerprint": self.source_fingerprint,
                  "template_id": self.template_id, "sample_index": self.sample_index,
                  "seed": self.seed}
        if status is not None:
            result["status"] = status
        if error is not None:
            result["error"] = error
        return result


def sample_plan(candidate: Any, *, n_configs: int, source_spec: Any, eval_seed: int,
                templates: Sequence[Mapping] | None = None) -> tuple[StructureSample, ...]:
    """Expand template-local sample counts before execution or scheduling.

    A template's explicit count takes precedence over the source's fallback.
    Generic external workers use no template ID and one task per sample. All
    seeds come from the source's versioned seed protocol, never worker/order IDs.
    """
    n_configs = positive_sample_count(n_configs)
    entries = [(None, None, None, n_configs)] if templates is None else [
        (index, str(spec["path"]), str(spec["site_symbol"]),
         n_configs if spec.get("n_configs") is None else positive_sample_count(spec["n_configs"]))
        for index, spec in enumerate(templates)
    ]
    if not entries:
        raise ValueError("A structure sample plan needs at least one template")
    result = []
    for index, path, symbol, count in entries:
        template_id = None if index is None else {"index": index, "path": path, "site_symbol": symbol}
        for sample_index in range(count):
            seed = source_spec.seed_for(candidate, eval_seed=eval_seed, purpose="build",
                                        template_id=template_id, sample_index=sample_index)
            result.append(StructureSample(source_spec.source_id, source_spec.protocol_fingerprint,
                                          index, path, symbol, sample_index, seed))
    return tuple(result)


def attach_sample_identity(atoms: Any, sample: StructureSample) -> None:
    record = sample.record()
    if SAMPLE_INFO_KEY in atoms.info and atoms.info[SAMPLE_INFO_KEY] != record:
        raise ValueError(f"Builder returned conflicting {SAMPLE_INFO_KEY} metadata")
    atoms.info[SAMPLE_INFO_KEY] = record


def resolve_sample_plan(builder: Any, candidate: Any, *, n_configs: int,
                        source_spec: Any, eval_seed: int) -> tuple[StructureSample, ...]:
    """Require an explicit sample capability and validate its complete plan.

    A native ensemble ``build`` is deliberately not interpreted as a sample
    protocol. A custom planner must return descriptors using this source's seed
    protocol; serial execution and workers call this same resolver.
    """
    require_sample_capability(builder)
    n_configs = positive_sample_count(n_configs)
    canonical = canonical_sample_candidate(candidate)
    planner = getattr(builder, 'sample_plan', None)
    if planner is not None and not callable(planner):
        raise SamplingProtocolError('Builder sample_plan must be callable')
    plan = (planner(canonical_sample_candidate(canonical), n_configs=n_configs,
                    source_spec=source_spec, eval_seed=eval_seed) if callable(planner)
            else sample_plan(canonical, n_configs=n_configs, source_spec=source_spec, eval_seed=eval_seed))
    if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence) or not plan:
        raise SamplingProtocolError('sample_plan must return a nonempty sequence of StructureSample descriptors')
    plan = tuple(plan)
    identities = set()
    for sample in plan:
        if not isinstance(sample, StructureSample):
            raise SamplingProtocolError('sample_plan returned an invalid StructureSample descriptor')
        if (sample.source_id != source_spec.source_id
                or sample.source_fingerprint != source_spec.protocol_fingerprint):
            raise SamplingProtocolError('sample_plan returned a descriptor for another source')
        if isinstance(sample.sample_index, bool) or not isinstance(sample.sample_index, Integral) or sample.sample_index < 0:
            raise SamplingProtocolError('sample_plan sample_index must be a nonnegative integer')
        if sample.template_index is None:
            if sample.template_path is not None or sample.site_symbol is not None:
                raise SamplingProtocolError('A generic sample cannot declare partial template metadata')
        elif (isinstance(sample.template_index, bool) or not isinstance(sample.template_index, Integral)
              or sample.template_index < 0 or not isinstance(sample.template_path, str)
              or not sample.template_path or not isinstance(sample.site_symbol, str) or not sample.site_symbol):
            raise SamplingProtocolError('sample_plan returned invalid template metadata')
        expected_seed = source_spec.seed_for(canonical, eval_seed=eval_seed, purpose='build',
                                             template_id=sample.template_id, sample_index=sample.sample_index)
        if isinstance(sample.seed, bool) or not isinstance(sample.seed, Integral) or sample.seed != expected_seed:
            raise SamplingProtocolError('sample_plan seed does not match the source seed protocol')
        identity = (sample.template_index, sample.template_path, sample.site_symbol, sample.sample_index)
        if identity in identities:
            raise SamplingProtocolError('sample_plan must not duplicate sample identities')
        identities.add(identity)
    return plan


def execute_structure_sample(builder: Any, candidate: Any, *, sample: StructureSample):
    """Execute exactly one declared sample, in either the parent or a worker."""
    from ase import Atoms

    build_sample = getattr(builder, 'build_sample', None)
    require_sample_capability(builder)
    atoms = build_sample(canonical_sample_candidate(candidate), sample=sample)
    if not isinstance(atoms, Atoms):
        raise SamplingProtocolError('build_sample must return exactly one ASE Atoms, not an expanded ensemble')
    attach_sample_identity(atoms, sample)
    return atoms


def execute_many_seeded_samples(builder: Any, candidates: Any, *, n_configs: int,
                                source_spec: Any, eval_seed: int):
    """Execute the explicit protocol serially with complete sample diagnostics."""
    positive_sample_count(n_configs)
    builder.last_sample_records = []
    require_sample_capability(builder)
    results = []
    for candidate in candidates:
        try:
            plan = resolve_sample_plan(builder, candidate, n_configs=n_configs,
                                       source_spec=source_spec, eval_seed=eval_seed)
        except EvaluationProtocolError:
            raise
        except Exception as exc:
            results.append(exc)
            builder.last_sample_records.append([sample_plan_failure_record(source_spec, exc)])
            continue
        records, structures = [], []
        for sample in plan:
            try:
                structures.append(execute_structure_sample(builder, candidate, sample=sample))
            except EvaluationProtocolError:
                raise
            except Exception as exc:
                records.append(sample.record(status='failed', error=f'{type(exc).__name__}: {exc}'))
            else:
                records.append(sample.record(status='succeeded'))
        builder.last_sample_records.append(records)
        results.append(structures if structures else RuntimeError(
            f"No structure samples succeeded. First failure: {records[0]['error']}"))
    return results


def execute_seeded_samples(builder: Any, candidate: Any, *, n_configs: int,
                           source_spec: Any, eval_seed: int):
    """Scalar form of execute_many_seeded_samples, raising a failed candidate."""
    result = execute_many_seeded_samples(builder, [candidate], n_configs=n_configs,
                                         source_spec=source_spec, eval_seed=eval_seed)[0]
    if isinstance(result, Exception):
        raise result
    return result
