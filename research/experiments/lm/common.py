"""Shared pieces for the language-model RL runs of "Trial and Reward" (Module 6).

Data (GSM8K), the prompt template, the answer checker (reward), batched sampling with
mlx-lm, per-token log-probabilities, and the unbiased pass@k estimator.
"""
import json
import math
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import pyarrow.parquet as pq
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

EXP = Path(__file__).resolve().parents[1]          # research/experiments
DATA = EXP / "data" / "gsm8k" / "main"
LOGS = EXP / "logs"

MODELS = {
    "qwen2.5-0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    "qwen2.5-1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "qwen3-0.6b": "mlx-community/Qwen3-0.6B-bf16",
}

# ---------------------------------------------------------------- data
N_TEST = 500          # fixed held-out subset of the GSM8K test split
SPLIT_SEED = 0


def _read(split):
    rows = pq.read_table(DATA / f"{split}-00000-of-00001.parquet").to_pylist()
    out = []
    for i, r in enumerate(rows):
        ref = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"{split}-{i}", "question": r["question"], "solution": r["answer"],
                    "ref": float(ref)})
    return out


def load_gsm8k():
    """Returns (train, test): train = all 7,473 GSM8K train problems in a fixed shuffled
    order (seed 0); test = a fixed random subset of 500 of the 1,319 test problems (seed 0)."""
    train, test = _read("train"), _read("test")
    random.Random(SPLIT_SEED).shuffle(train)
    test = sorted(random.Random(SPLIT_SEED).sample(test, N_TEST), key=lambda r: int(r["id"].split("-")[1]))
    return train, test


# ---------------------------------------------------------------- prompt
# Qwen2.5-Math's chain-of-thought instruction, used as the system message.
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(tok, question, model_key="qwen2.5-0.5b"):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    kw = {"enable_thinking": False} if model_key.startswith("qwen3") else {}
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------- reward
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")   # \boxed{...}, one level of nested braces
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")                 # 1,234.5  -7  18


def _num(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_answer(text):
    """The checker: the last number inside the last \\boxed{...}; None if there is no box."""
    boxes = BOXED.findall(text)
    if not boxes:
        return None
    nums = NUMBER.findall(boxes[-1].replace("\\!", "").replace("{,}", ""))
    return _num(nums[-1]) if nums else None


def last_number(text):
    """Diagnostic only (not a reward): the last number anywhere in the text."""
    nums = NUMBER.findall(text)
    return _num(nums[-1]) if nums else None


def same(a, b):
    return a is not None and abs(a - b) < 1e-4


def reward_correct(text, ref):
    """The true, verifiable reward: 1 if the boxed final answer equals the reference, else 0."""
    return 1.0 if same(extract_answer(text), ref) else 0.0


def reward_anywhere(text, ref):
    """A deliberately FLAWED reward (episode 25): 1 if the reference number appears anywhere."""
    return 1.0 if any(same(_num(n), ref) for n in NUMBER.findall(text)) else 0.0


# ---------------------------------------------------------------- sampling
def retry_oom(fn, tries=40, wait=15):
    """The GPU is shared with other jobs on this Mac; if Metal runs out of memory, wait and retry.
    Every call wrapped here is seeded, so a retry reproduces the same result."""
    for i in range(tries):
        try:
            return fn()
        except RuntimeError as e:
            if "Insufficient Memory" not in str(e) or i == tries - 1:
                raise
            print(f"[retry_oom] Metal out of memory, retry {i + 1} in {wait}s", flush=True)
            mx.clear_cache()
            time.sleep(wait)


def generate(model, tok, prompts, max_tokens=512, temp=1.0, seed=0, batch_size=128, chunk=None):
    """`chunk`: generate in independent pieces of this many prompts (seed + piece index), each retried
    on its own; used by the extras, which ran while another job shared the GPU."""
    if not chunk or len(prompts) <= chunk:
        return retry_oom(lambda: _generate(model, tok, prompts, max_tokens, temp, seed, batch_size))
    outs, fins, secs, ntok = [], [], 0.0, 0
    for i in range(0, len(prompts), chunk):
        o, f, st = retry_oom(lambda: _generate(model, tok, prompts[i:i + chunk], max_tokens, temp,
                                               seed + i // chunk, min(batch_size, chunk)))
        outs += o; fins += f; secs += st["seconds"]; ntok += st["tokens"]
    return outs, fins, {"seconds": secs, "tokens": ntok, "tokens_per_s": ntok / secs}


def _generate(model, tok, prompts, max_tokens, temp, seed, batch_size):
    """Sample one completion per prompt (lists of token ids). Pure temperature sampling
    (no top-p / top-k), so the sampling distribution is exactly softmax(logits / temp).
    Returns (completion token lists incl. the end-of-turn token if emitted, finish reasons, stats)."""
    mx.clear_cache()                        # hand freed training buffers back before sampling
    mx.random.seed(seed)
    sampler = make_sampler(temp=temp) if temp > 0 else None
    gen = BatchGenerator(model, stop_tokens=[[t] for t in tok.eos_token_ids], sampler=sampler,
                         completion_batch_size=batch_size, prefill_batch_size=min(batch_size, 32))
    t0 = time.perf_counter()
    uids = gen.insert(prompts, [max_tokens] * len(prompts))
    toks, fin = {u: [] for u in uids}, {}
    while responses := gen.next_generated():
        for r in responses:
            toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                fin[r.uid] = r.finish_reason
    gen.close()
    dt = time.perf_counter() - t0
    out = [toks[u] for u in uids]
    n_gen = sum(len(o) for o in out)
    return out, [fin.get(u, "length") for u in uids], {"seconds": dt, "tokens": n_gen,
                                                        "tokens_per_s": n_gen / dt}


def decode(tok, ids):
    eos = set(tok.eos_token_ids)
    return tok.decode([t for t in ids if t not in eos])


# ---------------------------------------------------------------- log-probs
def pad_batch(seqs, pad_id):
    L = max(len(s) for s in seqs)
    return mx.array([s + [pad_id] * (L - len(s)) for s in seqs])


def token_logprobs(model, tokens):
    """log pi(token_t | tokens_<t) for t = 1..L-1, shape [B, L-1], float32."""
    logits = model(tokens[:, :-1]).astype(mx.float32)
    tgt = tokens[:, 1:]
    return mx.take_along_axis(logits, tgt[..., None], axis=-1).squeeze(-1) - mx.logsumexp(logits, axis=-1)


# ---------------------------------------------------------------- pass@k
def pass_at_k(n, c, k):
    """Unbiased estimator of pass@k from n samples with c correct (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------- evaluation
def evaluate(model, tok, items, model_key, n_samples=8, temp=0.8, max_tokens=512, seed=1234,
             keep_text_ids=()):
    """Greedy pass@1 plus pass@k (k = 1, 2, 4, 8) from n_samples samples at `temp`."""
    prompts = [encode(tok, build_prompt(tok, it["question"], model_key)) for it in items]
    g_out, g_fin, g_st = generate(model, tok, prompts, max_tokens, temp=0.0, seed=seed, batch_size=256)
    rep = [p for p in prompts for _ in range(n_samples)]
    if n_samples:
        s_out, s_fin, s_st = generate(model, tok, rep, max_tokens, temp=temp, seed=seed, batch_size=256)
    else:                                   # greedy-only evaluation
        s_out, s_st = [], {"seconds": 0.0, "tokens_per_s": 0.0}
    per_q, keep = [], {}
    for i, it in enumerate(items):
        gt = decode(tok, g_out[i])
        samples = s_out[i * n_samples:(i + 1) * n_samples]
        stexts = [decode(tok, s) for s in samples]
        c = sum(reward_correct(t, it["ref"]) for t in stexts)
        per_q.append({"id": it["id"], "greedy_correct": reward_correct(gt, it["ref"]),
                      "greedy_boxed": extract_answer(gt) is not None,
                      "greedy_lastnum_correct": float(same(last_number(gt), it["ref"])),
                      "greedy_len": len(g_out[i]), "greedy_truncated": g_fin[i] == "length",
                      "n": n_samples, "c": int(c), "sample_lens": [len(s) for s in samples],
                      "sample_boxed": sum(extract_answer(t) is not None for t in stexts)})
        if it["id"] in keep_text_ids:
            keep[it["id"]] = {"question": it["question"], "ref": it["ref"], "greedy_text": gt,
                              "greedy_tokens": len(g_out[i]), "greedy_correct": per_q[-1]["greedy_correct"]}
    N = len(items)
    summ = {
        "n_questions": N, "n_samples": n_samples, "temperature": temp, "max_tokens": max_tokens,
        "greedy_acc": sum(q["greedy_correct"] for q in per_q) / N,
        "greedy_boxed_rate": sum(q["greedy_boxed"] for q in per_q) / N,
        "greedy_lastnum_acc": sum(q["greedy_lastnum_correct"] for q in per_q) / N,
        "greedy_mean_len": sum(q["greedy_len"] for q in per_q) / N,
        "greedy_truncated_rate": sum(q["greedy_truncated"] for q in per_q) / N,
        "pass_at_k": {str(k): sum(pass_at_k(q["n"], q["c"], k) for q in per_q) / N
                      for k in (1, 2, 4, 8) if k <= n_samples},
        "sample_mean_len": sum(sum(q["sample_lens"]) for q in per_q) / max(N * n_samples, 1),
        "sample_boxed_rate": sum(q["sample_boxed"] for q in per_q) / max(N * n_samples, 1),
        "greedy_seconds": g_st["seconds"], "greedy_tokens_per_s": g_st["tokens_per_s"],
        "sample_seconds": s_st["seconds"], "sample_tokens_per_s": s_st["tokens_per_s"],
    }
    return summ, per_q, keep


def load_model(model_key):
    model, tok = load(MODELS[model_key])
    return model, tok


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))
