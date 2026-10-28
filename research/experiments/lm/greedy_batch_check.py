"""Is greedy accuracy batch-invariant? Decode the 500 held-out problems greedily with the base model and
with the GRPO adapter at several generation batch sizes (bf16 numerics change with the batch shape).
Usage: python -I greedy_batch_check.py lm_grpo
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

import common as C  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "lm_grpo"
cfg = json.loads((C.LOGS / f"{run}_config.json").read_text())["args"]
_, test = C.load_gsm8k()
test = test[:cfg["eval_n"]]
model, tok = C.load_model(cfg["model"])
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": cfg["lora_rank"], "scale": cfg["lora_scale"], "dropout": 0.0})
prompts = [C.encode(tok, C.build_prompt(tok, it["question"], cfg["model"])) for it in test]
res = {"batch_sizes": [16, 32, 64, 128, 256], "base": {}, "trained": {}}
correct = {}
for name in ("base", "trained"):
    if name == "trained":
        model.load_weights(str(Path(__file__).resolve().parent / "ckpt" / run / "adapters.safetensors"), strict=False)
    for b in res["batch_sizes"]:
        g, _, _ = C.generate(model, tok, prompts, cfg["max_tokens"], temp=0.0, seed=1234, batch_size=b)
        c = [C.reward_correct(C.decode(tok, x), it["ref"]) for x, it in zip(g, test)]
        correct[(name, b)] = c
        res[name][str(b)] = sum(c) / len(c)
        print(name, b, res[name][str(b)], flush=True)
for name in ("base", "trained"):
    v = list(res[name].values())
    res[name + "_mean"], res[name + "_min"], res[name + "_max"] = sum(v) / len(v), min(v), max(v)
    # fraction of questions whose greedy correctness is identical at every batch size
    res[name + "_stable_fraction"] = sum(len({correct[(name, b)][i] for b in res["batch_sizes"]}) == 1
                                         for i in range(len(test))) / len(test)
res["gain_mean"] = res["trained_mean"] - res["base_mean"]
(C.LOGS / f"{run}_greedy_batch_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)
