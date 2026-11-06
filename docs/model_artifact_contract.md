# State and action compatibility of RL artifacts

The explicit composition state changes the inputs seen by all algorithms. It
contains material descriptors, absolute values on fixed species labels, a
selected mask when zero-valued assignments are legal, and remaining amount and
slot resources. Equivalent assignments in different execution orders have the
same state. Different absolute budgets remain distinguishable even when their
normalized material descriptors coincide.

This is an intentional model-input change. Previously trained networks and
scalers must be used with their archived implementation; they cannot be silently
loaded into this environment. The new runner rejects missing or different
contracts before model inference. Repeating the new version with the same seeds
is a separate requirement from reproducing old-version trajectories.

New weight files contain `environment_contract` and a role-specific state dict:
`qnet_state`, `policy_state`, or `value_net_state`. Scalers carry
`rl_matdesign_contract_`. Mid-training checkpoints record the same contract
alongside the optimizer and replay data. Contract version 3 records typed numeric/categorical action value features and the state
schema, action vocabularies in their actual index order, environment kind,
episode length, declared design constraints, group labels/sites, bounds and
anion formula. Numeric scientific values use their magnitudes; categorical tokens
use identity channels, never artificial ordinal numbers. Boolean/string/number
identities and group/position labels remain explicit. Contracts compare serialized
types strictly (Python `True == 1` is insufficient). Equal tensor shapes alone are not evidence of compatibility.

Minimal environments used through the lower-level training API may omit action
vocabularies. Their saved contract is explicitly incomplete; the strict artifact
loader cannot certify compatibility for them. Custom descriptors must have a
fixed dimension. Two custom callables with identical qualified names but
different hidden closure state are not distinguishable automatically; plugins
must expose their actual configuration rather than hide semantic parameters in
closures. Full plugin/model asset identity is handled separately from this
state/action contract.

For direct inspection:

```python
import torch
artifact = torch.load("runs/example/qnet.pt", map_location="cpu", weights_only=True)
print(artifact["environment_contract"])
weights = artifact["qnet_state"]
```

The runner and `scripts/check_invariance.py` validate contracts when loading.
The replay summarizer may still inspect historical data without claiming it is
safe to resume. Validation of this format does not by itself establish exact
interrupted-training continuation: complete RNG/optimizer/predictor state and
separate continuation tests are also required.
