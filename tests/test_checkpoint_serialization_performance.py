"""Exact checkpoint values and smaller NumPy payloads, without timing thresholds."""
from __future__ import annotations

import copy
import pickletools
import random
import zipfile

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign import training
from rl_matdesign.utils.checkpoint_serialization import save_checkpoint_payload


def _assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, StandardScaler):
        _assert_exact(vars(left), vars(right))
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_exact(a, b)
    else:
        assert left == right


def _payload(rows=12):
    rng = np.random.default_rng(417)
    model = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    optimizer.zero_grad()
    model(torch.ones(2, 7)).square().sum().backward()
    optimizer.step()
    replay = []
    for i in range(rows):
        state = rng.standard_normal(64).astype(np.float64)
        replay.append({
            "s_raw": state,
            "s_step": np.eye(7)[i % 7],
            "a_feat": rng.standard_normal(9).astype(np.float32),
            "reward": float(i) / 17.,
            "next_s_raw": rng.standard_normal(64).astype(np.float64),
            "next_allowed_idx": [(i % 7, .1), ((i + 1) % 7, .2)],
            "done": bool(i % 7 == 6), "mc_return": float(i) / 19.,
        })
    # Include dtype and special-value cases that a lossy packing optimization
    # could accidentally coerce, although the ordinary replay is float data.
    replay[0]["special"] = np.array([0., -0., np.inf, -np.inf, np.nan], dtype='>f8')
    replay[0]["empty"] = np.array([], dtype=np.int64)
    replay[0]["noncontiguous"] = np.arange(24, dtype=np.int16).reshape(4, 6)[:, ::2]
    return {
        "type": "dqn", "buffer": replay,
        "qnet_state": copy.deepcopy(model.state_dict()),
        "target_net_state": copy.deepcopy(model.state_dict()),
        "opt_state": copy.deepcopy(optimizer.state_dict()),
        "scaler": StandardScaler().fit(np.arange(35).reshape(5, 7)),
        "rng_state": {"python": random.Random(124).getstate(),
                      "numpy": np.random.RandomState(143).get_state(),
                      "torch": torch.Generator().manual_seed(491).get_state()},
        "training_budget_state": {"seen": ["a", "b"], "attempts": 4},
        "hpo_state": {"seen": ["a", "b"], "records_sha256": "saved-exactly"},
        "episodes_completed": 9, "training_episodes_completed": 12,
        "eps": .21000000000000002,
    }


def test_protocol5_checkpoint_roundtrips_complete_original_payload(tmp_path):
    payload = _payload()
    snapshot = copy.deepcopy(payload)
    path = tmp_path / 'checkpoint.pt'
    measurements = save_checkpoint_payload(str(path), payload)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    _assert_exact(payload, restored)
    _assert_exact(payload, snapshot)
    assert measurements['checkpoint_serialize_s'] >= 0
    assert measurements['checkpoint_bytes'] == path.stat().st_size
    with zipfile.ZipFile(path) as archive:
        data = archive.read(next(name for name in archive.namelist() if name.endswith('/data.pkl')))
    opcode, protocol, _ = next(pickletools.genops(data))
    assert opcode.name == 'PROTO'
    assert protocol == 5


def test_numpy_replay_is_smaller_than_protocol2_without_changing_values(tmp_path):
    payload = _payload(rows=512)
    old, new = tmp_path / 'old.pt', tmp_path / 'new.pt'
    torch.save(payload, old, pickle_protocol=2)
    save_checkpoint_payload(str(new), payload)
    assert new.stat().st_size < old.stat().st_size
    _assert_exact(payload, torch.load(new, weights_only=False))


def test_atomic_checkpoint_captures_rng_and_reports_write_measurements(tmp_path, monkeypatch):
    payload = _payload()
    saved_rng = payload.pop('rng_state')
    monkeypatch.setattr(training, 'capture_rng_state', lambda: saved_rng)
    path = tmp_path / 'checkpoint.pt'
    measurements = training._save_checkpoint(str(path), payload)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    _assert_exact({**payload, 'rng_state': saved_rng}, restored)
    assert 'rng_state' not in payload
    assert measurements['checkpoint_serialize_s'] >= 0
    assert measurements['checkpoint_replace_s'] >= 0
    assert measurements['checkpoint_bytes'] == path.stat().st_size
    assert not (tmp_path / 'checkpoint.pt.tmp').exists()


def test_ragged_actions_optional_legacy_fields_and_types_are_lossless(tmp_path):
    rows = []
    for i in range(8):
        row = {
            's_mat_raw': np.array([i, -0., np.nan], dtype=np.float64),
            'scalar_array': np.array(i, dtype=np.int16),
            'zero_length': np.empty((0, 3), dtype=np.float32),
            'next_allowed_idx': [(j, j + 3) for j in range(i)],
            'done': i == 7, 'reward': float(i),
            'huge_integer': 1 << (70 + i),
            'a_comp_val': None if i % 2 else float(i) / 10.,
            'extra': {'tag': ('legacy', i), 'value': np.int32(i)},
        }
        if i % 2:
            row['a_comp_features'] = np.array([.2, .4], dtype=np.float32)
        if i % 3 == 0:
            row = dict(reversed(list(row.items())))
        rows.append(row)
    payload = {'buffer': rows}
    before = copy.deepcopy(payload)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint_payload(str(path), payload)
    restored = torch.load(path, weights_only=False)
    _assert_exact(before, restored)
    _assert_exact(before, payload)
    assert type(restored['buffer']) is list
    assert all(type(row) is dict for row in restored['buffer'])
    assert all(type(row['scalar_array']) is np.ndarray for row in restored['buffer'])


@pytest.mark.parametrize('buffer', [[], [1, 2], [{'empty': None}, {}],
    [{'next_allowed_idx': [[1, 2], [3, 4]]}], [{'value': np.array(['a', 'b'])}]])
def test_unsupported_or_empty_buffers_retain_values(tmp_path, buffer):
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint_payload(str(path), {'buffer': buffer})
    _assert_exact(buffer, torch.load(path, weights_only=False)['buffer'])


def test_compressed_columns_restore_exact_arrays_scalars_and_ragged_actions(tmp_path, monkeypatch):
    from rl_matdesign.utils import checkpoint_serialization as codec

    monkeypatch.setattr(codec, '_MIN_COMPRESS_BYTES', 32)
    rows = [{
        's_mat_raw': np.array([0., -0., 1., np.inf, -np.inf, np.nan] * 16),
        's_step': np.eye(7)[i % 7],
        'reward': float(i % 3), 'done': bool(i % 7 == 6),
        'a_elem_idx': i % 7,
        'next_allowed_idx': [(j, 2 * j) for j in range(i % 12)],
    } for i in range(256)]
    packed = codec.PackedReplaySnapshot(rows)
    array_column = next(column for column in packed.columns if column[0] == 's_mat_raw')
    assert array_column[2] == 'array'
    assert array_column[3][0] == 'zlib'
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint_payload(str(path), {'buffer': rows})
    _assert_exact(rows, torch.load(path, weights_only=False)['buffer'])


def test_incompressible_column_uses_plain_tensor_storage(monkeypatch):
    from rl_matdesign.utils import checkpoint_serialization as codec

    monkeypatch.setattr(codec, '_MIN_COMPRESS_BYTES', 32)
    array = np.random.default_rng(351).integers(0, 256, size=65536, dtype=np.uint8)
    packed = codec._pack_storage(array)
    assert packed[0] == 'tensor'
    _assert_exact(array, codec._unpack_storage(packed))
