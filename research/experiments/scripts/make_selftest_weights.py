"""Save a random 4 -> 16 -> 2 tanh network as weights/_selftest.npz + .json, the format load_policy reads
(rl_reference's self-check loads it). Run from the series folder: python3 research/experiments/scripts/make_selftest_weights.py"""
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
WEIGHTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
np.savez(WEIGHTS / "_selftest.npz", W0=rng.normal(0, 0.5, (4, 16)), b0=rng.normal(0, 0.1, 16),
         W1=rng.normal(0, 0.5, (16, 2)), b1=np.zeros(2))
(WEIGHTS / "_selftest.json").write_text(json.dumps(
    {"activation": "tanh", "output": "logits", "obs": "CartPole-v1 observation (x, x_dot, theta, theta_dot)",
     "notes": "random untrained network for rl_reference's load_policy self-check"}, indent=1))
print("wrote", sorted(p.name for p in WEIGHTS.glob("_selftest.*")))
