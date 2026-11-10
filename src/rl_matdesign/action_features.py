"""Keep action value identities alongside their shared network feature table."""
from __future__ import annotations

import numpy as np

from .action_encoding import ValueEncoding


class ValueAlphabet(tuple):
    """Immutable environment codes with numeric or categorical value features.

    Codes remain compatible with the environment's one-hot action vocabulary.
    Training consumes indices into ``features``, never the textual codes as
    artificial numerical distances for categorical choices.
    """

    def __new__(cls, codes, encoding):
        instance = super().__new__(cls, codes)
        if len(instance) != encoding.features.shape[0]:
            raise ValueError("Action value codes and features have different lengths")
        instance.encoding = encoding
        return instance

    @property
    def features(self):
        return self.encoding.features

    @property
    def feature_dim(self):
        return int(self.features.shape[1])

    def to_schema(self):
        return {"codes": list(self), "values": self.encoding.to_schema()}


def value_alphabet(env):
    specs = getattr(env, "value_specs", None)
    encoding = (ValueEncoding.from_specs(specs) if specs is not None
                else ValueEncoding([float(code) for code in env.fraction_set], kind="numeric"))
    return ValueAlphabet(env.fraction_set, encoding)


def value_features(alphabet, indices):
    """Gather the same rows for DQN, PG, generation and replay targets."""
    if hasattr(alphabet, "features"):
        table = alphabet.features
    else:
        # Public training functions retain their numeric-list calling convention.
        table = np.asarray([float(code) for code in alphabet], dtype=float).reshape(-1, 1)
    return np.asarray(table[np.asarray(indices, dtype=int)], dtype=float)


def value_feature_dim(alphabet):
    return int(alphabet.feature_dim) if hasattr(alphabet, "feature_dim") else 1
