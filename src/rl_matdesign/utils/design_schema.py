"""Guard checkpoint reuse when ordered token/action semantics change."""
from __future__ import annotations

import json
from pathlib import Path


def save_or_check_design_schema(schema: dict, output_dir: str, *, load_paths=()) -> None:
    """Validate every loaded model/scaler's neighboring schema before writing.

    Matching tensor dimensions alone cannot detect a reordered monomer catalog.
    Keep sequence_schema.json beside exported weights and scalers when moving them.
    """
    expected = json.loads(json.dumps(schema, ensure_ascii=False))
    # Python considers True == 1 == 1.0; categorical token codes must not.
    def signature(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)

    expected_signature = signature(expected)
    for parent in {Path(path).resolve().parent for path in load_paths}:
        manifest = parent / "sequence_schema.json"
        if not manifest.is_file():
            raise ValueError(f"Sequence checkpoint requires {manifest}; token semantics cannot be verified")
        with manifest.open(encoding="utf-8") as stream:
            actual = json.load(stream)
        if signature(actual) != expected_signature:
            raise ValueError(f"Sequence design differs from {manifest}; retrain after changing tokens, order, choices or constraints")
    destination = Path(output_dir) / "sequence_schema.json"
    # A new training attempt can fail before replacing weights. Never relabel
    # old artifacts with a new vocabulary merely because a run was started.
    artifacts = ("qnet.pt", "policy.pt", "value_net.pt", "checkpoint.pt", "std_scaler.bin")
    if any((destination.parent / name).exists() for name in artifacts):
        if not destination.is_file():
            raise ValueError(f"Existing model artifacts lack {destination}; use a new output directory")
        with destination.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if signature(previous) != expected_signature:
            raise ValueError(f"Sequence design differs from existing artifacts in {output_dir}; use a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
