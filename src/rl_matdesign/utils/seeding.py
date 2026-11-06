"""Global seed utilities for reproducible experiments."""
from __future__ import annotations

import random
from contextlib import contextmanager


def capture_rng_state() -> dict:
    """Capture the position of training/generation streams, including CUDA."""
    import numpy as np
    import torch
    return {"schema_version": 1, "python": random.getstate(),
            "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []}


def restore_rng_state(state: dict) -> None:
    import numpy as np
    import torch
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Checkpoint lacks a supported RNG state; exact resume is unavailable")
    cuda = state["torch_cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("Exact resume requires the same CUDA device count")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda])


@contextmanager
def isolated_rng(seed: int | None = None):
    """Keep model loading and inference from consuming RL random streams.

    This synchronous context owns process-global RNGs. Threaded adapters must
    use their own generators; it is not a substitute for thread isolation.
    """
    state = capture_rng_state()
    try:
        if seed is not None:
            set_global_seed(seed)
        yield
    finally:
        restore_rng_state(state)


def candidate_seed(seed: int, candidate) -> int:
    """Stable evaluation substream, independent of query and dictionary order."""
    import hashlib
    from .composition import composition_key
    payload = repr((int(seed), composition_key(candidate))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU + GPU) for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.
    deterministic:
        If True, enable full GPU determinism (cudnn.deterministic,
        use_deterministic_algorithms).  Slower but fully reproducible on GPU.
        Mirror of feat/classical-dqn's behaviour when --train-seed is explicit.
    """
    import numpy as np

    if deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
    except ImportError:
        pass
