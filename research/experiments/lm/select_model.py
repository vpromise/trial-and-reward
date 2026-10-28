"""Model selection: greedy accuracy of candidate models on the first 100 of the 500 held-out
GSM8K problems, plus generation throughput. Usage: python -I select_model.py qwen2.5-0.5b ..."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402
from common import (LOGS, build_prompt, decode, encode, extract_answer, generate, last_number,  # noqa: E402
                    load_gsm8k, load_model, reward_correct, same)

_, test = load_gsm8k()
items = test[:100]
res = {}
for key in sys.argv[1:]:
    model, tok = load_model(key)
    prompts = [encode(tok, build_prompt(tok, it["question"], key)) for it in items]
    mx.reset_peak_memory()
    out, fin, st = generate(model, tok, prompts, max_tokens=512, temp=0.0, batch_size=128)
    texts = [decode(tok, o) for o in out]
    r = {"greedy_acc": sum(reward_correct(t, it["ref"]) for t, it in zip(texts, items)) / len(items),
         "boxed_rate": sum(extract_answer(t) is not None for t in texts) / len(items),
         "lastnum_acc": sum(same(last_number(t), it["ref"]) for t, it in zip(texts, items)) / len(items),
         "mean_len": sum(map(len, out)) / len(out), "max_len": max(map(len, out)),
         "truncated": sum(f == "length" for f in fin) / len(fin),
         "mean_prompt_len": sum(map(len, prompts)) / len(prompts),
         "tokens_per_s": st["tokens_per_s"], "seconds": st["seconds"],
         "peak_mem_gb": mx.get_peak_memory() / 1e9, "eos": list(tok.eos_token_ids),
         "example": texts[0]}
    res[key] = r
    print(key, json.dumps({k: v for k, v in r.items() if k != "example"}), flush=True)
    del model
    mx.clear_cache()
(LOGS / "lm_model_selection.json").write_text(json.dumps(res, indent=1))
