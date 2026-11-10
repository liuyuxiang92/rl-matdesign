"""Exact normalization identity is stronger than equal feature dimensions."""
from copy import deepcopy

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign.utils.model_contract import (
    bind_scaler, load_artifact, require_scaler_fingerprint,
    save_weights, scaler_fingerprint, validate_scaler,
)
from tests.test_model_contract import make_env


def fitted(values=((0., 1.), (2., 5.)), **options):
    return bind_scaler(StandardScaler(**options).fit(values), make_env())


def test_same_environment_does_not_permit_different_fitted_values():
    first, second = fitted(), fitted(((0., 1.), (4., 9.)))
    validate_scaler(first, make_env())
    validate_scaler(second, make_env())
    assert scaler_fingerprint(first) != scaler_fingerprint(second)
    with pytest.raises(ValueError, match='fitted scaler fingerprint'):
        require_scaler_fingerprint(second, scaler_fingerprint(first))


@pytest.mark.parametrize('field,change', [
    ('mean_', lambda x: x + .1), ('scale_', lambda x: x * 2),
    ('var_', lambda x: x + 1), ('n_samples_seen_', lambda x: x + 1),
    ('n_features_in_', lambda x: x + 1),
    ('mean_', lambda x: x.astype(np.float32)),
    ('scale_', lambda x: x.reshape(1, -1)),
    ('with_mean', lambda x: not x), ('with_std', lambda x: not x),
])
def test_numeric_state_dtype_shape_and_normalization_options_are_bound(field, change):
    original = fitted()
    changed = deepcopy(original)
    setattr(changed, field, change(getattr(changed, field)))
    assert scaler_fingerprint(changed) != scaler_fingerprint(original)


def test_contract_is_bound_but_memory_layout_and_copy_option_are_not():
    original = fitted()
    changed = deepcopy(original)
    changed.mean_ = np.asarray([changed.mean_[0], 999., changed.mean_[1], 999.])[::2]
    changed.copy = not changed.copy
    assert scaler_fingerprint(changed) == scaler_fingerprint(original)
    changed.rl_matdesign_contract_['anion_formula'] = 'O2'
    assert scaler_fingerprint(changed) != scaler_fingerprint(original)


@pytest.mark.parametrize('with_mean,with_std', [(True, True), (False, True), (True, False), (False, False)])
def test_fingerprint_survives_joblib_roundtrip_and_supports_disabled_centering(tmp_path, with_mean, with_std):
    scaler = fitted(with_mean=with_mean, with_std=with_std)
    path = tmp_path / 'scaler.bin'
    joblib.dump(scaler, path)
    restored = joblib.load(path)
    assert scaler_fingerprint(restored) == scaler_fingerprint(scaler)
    require_scaler_fingerprint(restored, scaler_fingerprint(scaler))


def test_missing_fingerprint_or_unfitted_scaler_is_rejected():
    with pytest.raises(ValueError, match='unfitted'):
        scaler_fingerprint(StandardScaler())
    for missing in (None, '', {}, 'outdated'):
        with pytest.raises(ValueError, match='fingerprint'):
            require_scaler_fingerprint(fitted(), missing)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), 'object'])
def test_invalid_fitted_numeric_state_cannot_receive_a_valid_fingerprint(invalid):
    scaler = fitted()
    scaler.scale_ = np.asarray([invalid, invalid])
    with pytest.raises(ValueError, match='finite real numeric'):
        scaler_fingerprint(scaler)


def test_model_weights_require_the_original_fitted_scaler_when_supplied(tmp_path):
    scaler, different = fitted(), fitted(((2., 5.), (8., 12.)))
    path = tmp_path / 'policy.pt'
    save_weights(path, torch.nn.Linear(2, 1), make_env(), 'policy', scaler=scaler)
    raw = load_artifact(path, make_env(), scaler=scaler)
    assert raw['scaler_fingerprint'] == scaler_fingerprint(scaler)
    with pytest.raises(ValueError, match='fitted scaler fingerprint'):
        load_artifact(path, make_env(), scaler=different)
    save_weights(path, torch.nn.Linear(2, 1), make_env(), 'policy')
    with pytest.raises(ValueError, match='missing'):
        load_artifact(path, make_env(), scaler=scaler)
    load_artifact(path, make_env())  # Isolated lower-level contract test compatibility.


def test_checkpoint_loading_checks_same_scaler_contract(tmp_path):
    from rl_matdesign.utils.model_contract import environment_contract
    scaler = fitted()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'environment_contract': environment_contract(make_env()),
                'scaler_fingerprint': scaler_fingerprint(scaler)}, path)
    load_artifact(path, make_env(), checkpoint=True, scaler=scaler)
    changed = deepcopy(scaler)
    changed.mean_[0] += 1
    with pytest.raises(ValueError, match='fingerprint'):
        load_artifact(path, make_env(), checkpoint=True, scaler=changed)
