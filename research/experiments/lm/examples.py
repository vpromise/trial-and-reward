"""Greedy answers of the base model and of a trained adapter on the 500 held-out problems, verbatim,
for the before/after scenes. Same batch and settings as the evaluation in grpo.py, so the
accuracies reproduce the eval log. Usage: python -I examples.py lm_grpo
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

import common as C  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "lm_grpo"
BATCH = 64   # the eval used 256; smaller batches survive a shared GPU. Agreement with the eval log is recorded.
cfg = json.loads((C.LOGS / f"{run}_config.json").read_text())["args"]
_, test = C.load_gsm8k()
test = test[:cfg["eval_n"]]
model, tok = C.load_model(cfg["model"])
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": cfg["lora_rank"], "scale": cfg["lora_scale"], "dropout": 0.0})
prompts = [C.encode(tok, C.build_prompt(tok, it["question"], cfg["model"])) for it in test]
out = {}
for name in ("base", "trained"):
    if name == "trained":
        model.load_weights(str(Path(__file__).resolve().parent / "ckpt" / run / "adapters.safetensors"), strict=False)
    g, fin, st = C.generate(model, tok, prompts, cfg["max_tokens"], temp=0.0, seed=1234, batch_size=BATCH)
    texts = [C.decode(tok, x) for x in g]
    out[name] = [{"id": it["id"], "text": t, "tokens": len(x), "finish": f,
                  "answer": C.extract_answer(t), "correct": C.reward_correct(t, it["ref"])}
                 for it, t, x, f in zip(test, texts, g, fin)]
    ev = [json.loads(l) for l in (C.LOGS / f"{run}_eval.jsonl").read_text().splitlines() if l]
    ev = [e for e in ev if e["step"] == (0 if name == "base" else cfg["steps"])][-1]
    agree = sum(q["greedy_correct"] == r["correct"] for q, r in zip(ev["per_question"], out[name])) / len(test)
    out[f"{name}_agreement_with_eval_log"] = agree
    print(name, "greedy acc", sum(r["correct"] for r in out[name]) / len(test), "eval log", ev["greedy_acc"],
          "per-question agreement", agree, flush=True)
out["questions"] = [{"id": it["id"], "question": it["question"], "ref": it["ref"]} for it in test]
(C.LOGS / f"{run}_greedy_texts.json").write_text(json.dumps(out))
