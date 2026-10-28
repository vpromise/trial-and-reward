"""Collect the LM-RL logs into research/experiments/lm_rl.json (stable keys; see lm_rl.md).
Usage: python -I summarize.py
"""
import json
import platform
import re
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

L = C.LOGS
OUT = C.EXP / "lm_rl.json"


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []


def ma(xs, w=10):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def down(xs, n=200):
    if len(xs) <= n:
        return xs
    idx = [round(i * (len(xs) - 1) / (n - 1)) for i in range(n)]
    return [xs[i] for i in idx]


def r4(x):
    return round(x, 4) if isinstance(x, float) else x


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def short(t, head=420, tail=260):
    return t if len(t) <= head + tail + 20 else t[:head] + " [...] " + t[-tail:]


def evals_of(run):
    ev = {}
    for e in jl(L / f"{run}_eval.jsonl"):
        ev[e["step"]] = e                                  # last write wins (re-runs after a crash)
    return [ev[k] for k in sorted(ev)]


def eval_row(e):
    return {"step": e["step"], "greedy_acc": r4(e["greedy_acc"]),
            "pass_at_k": {k: r4(v) for k, v in e["pass_at_k"].items()},
            "greedy_boxed_rate": r4(e["greedy_boxed_rate"]), "greedy_lastnum_acc": r4(e["greedy_lastnum_acc"]),
            "greedy_mean_len": round(e["greedy_mean_len"], 1), "greedy_truncated_rate": r4(e["greedy_truncated_rate"]),
            "sample_mean_len": round(e["sample_mean_len"], 1), "sample_boxed_rate": r4(e["sample_boxed_rate"]),
            "eval_seconds": round(e["seconds"], 1)}


def curves(steps):
    keys = ["reward", "accuracy", "kl", "mean_len", "zero_adv_groups", "clip_frac", "boxed_rate",
            "truncated_rate", "gen_tokens_per_s", "train_tokens_per_s", "sample_s", "logprob_s", "update_s",
            "peak_mem_gb", "loss"]
    c = {"step": down([s["step"] for s in steps])}
    for k in keys:
        c[k] = [r4(x) for x in down([s[k] for s in steps])]
    for k in ("reward", "accuracy", "mean_len", "kl"):
        c[k + "_ma10"] = [r4(x) for x in down(ma([s[k] for s in steps]))]
    return c


def pick_group(rows, lo, hi):
    """A group with mixed rewards (some right, some wrong) and the shortest longest answer."""
    best = None
    for row in rows:
        if not lo <= row["step"] <= hi:
            continue
        for g in row["groups"]:
            k = sum(g["correct"])
            if 2 <= k <= 6 and max(g["tokens"]) < 512:
                key = (abs(k - 4) > 1, max(g["tokens"]), row["step"])   # prefer 3-5 of 8 right, then short
                if best is None or key < best[0]:
                    best = (key, row["step"], g)
    if best is None:
        return None
    _, step, g = best
    return {"step": step, "id": g["id"], "question": g["question"], "ref": g["ref"],
            "rewards": g["rewards"], "advantages": g["advantages"], "tokens": g["tokens"],
            "mean_reward": mean(g["rewards"]), "completions": g["completions"],
            "completions_short": [short(t) for t in g["completions"]]}


def paired_se(e0, e1):
    """Standard error of the before->after gain, paired over the same held-out questions."""
    a = {q["id"]: q for q in e0["per_question"]}
    out = {}
    for name, f in (("greedy_acc_gain", lambda q: q["greedy_correct"]), ("pass1_gain", lambda q: q["c"] / q["n"]),
                    ("pass8_gain", lambda q: C.pass_at_k(q["n"], q["c"], 8))):
        d = [f(q) - f(a[q["id"]]) for q in e1["per_question"]]
        m = mean(d)
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        out[name] = {"gain": r4(m), "se": r4(sd / len(d) ** 0.5)}
    return out


RECHECK = re.compile(r"re-?evaluat|re-?check|verif|double[- ]check|mistake", re.I)


def recheck_rate(rows):
    ts = [t for r in rows for g in r["groups"] for t in g["completions"]]
    return sum(bool(RECHECK.search(t)) for t in ts) / len(ts) if ts else None


def fmt(rows):
    ts = [(t, g["ref"]) for r in rows for g in r["groups"] for t in g["completions"]]
    return {"n": len(ts), "strict_acc": r4(mean([C.reward_correct(t, ref) for t, ref in ts])),
            "last_number_acc": r4(mean([float(C.same(C.last_number(t), ref)) for t, ref in ts])),
            "boxed_rate": r4(mean([float(C.extract_answer(t) is not None) for t, _ in ts]))}


def numbers_per_answer(rows):
    return [mean([len(C.NUMBER.findall(t)) for g in r["groups"] for t in g["completions"]]) for r in rows]


def main():
    cfg = json.loads((L / "lm_grpo_config.json").read_text())
    a = cfg["args"]
    steps = jl(L / "lm_grpo.jsonl")
    groups = jl(L / "lm_grpo_groups.jsonl")
    evs = evals_of("lm_grpo")
    sel = json.loads((L / "lm_model_selection.json").read_text())
    first, last = evs[0], evs[-1]
    n = len(steps)
    w = min(10, n)
    train_s = sum(s["sample_s"] + s["logprob_s"] + s["update_s"] for s in steps)
    eval_s = sum(e["seconds"] for e in evs)
    gen_tok = sum(s["gen_tokens"] for s in steps)
    gen_s = sum(s["sample_s"] for s in steps)

    # ---- before / after examples (greedy, verbatim)
    gt = json.loads((L / "lm_grpo_greedy_texts.json").read_text())
    qs = {q["id"]: q for q in gt["questions"]}
    base = {r["id"]: r for r in gt["base"]}
    tr = {r["id"]: r for r in gt["trained"]}
    ids = [q["id"] for q in gt["questions"]]
    eb = {q["id"]: q["greedy_correct"] for q in first["per_question"]}   # the eval log (batch 256)
    et = {q["id"]: q["greedy_correct"] for q in last["per_question"]}
    table = {"both_right": 0, "wrong_to_right": 0, "right_to_wrong": 0, "both_wrong": 0}
    for i in ids:
        b, t = eb[i], et[i]
        table[("both_right" if t else "right_to_wrong") if b else ("wrong_to_right" if t else "both_wrong")] += 1

    def ex(i, why):
        return {"id": i, "why_chosen": why, "question": qs[i]["question"], "ref": qs[i]["ref"],
                "base": {k: base[i][k] for k in ("text", "tokens", "answer", "correct", "finish")},
                "trained": {k: tr[i][k] for k in ("text", "tokens", "answer", "correct", "finish")}}

    def shortest(cond):   # the flip must also hold in the eval log (robust to the decode batch size)
        c = [i for i in ids if cond(base[i], tr[i]) and base[i]["tokens"] < 512 and tr[i]["tokens"] < 512
             and eb[i] == base[i]["correct"] and et[i] == tr[i]["correct"]]
        return sorted(c, key=lambda i: base[i]["tokens"] + tr[i]["tokens"] + len(qs[i]["question"]) // 4)

    examples = []
    flips = shortest(lambda b, t: not b["correct"] and t["correct"] and b["answer"] is not None)
    examples += [ex(i, "base boxed a wrong number, trained model right (shortest such pair)") for i in flips[:2]]
    nobox = [i for i in ids if base[i]["answer"] is None and tr[i]["correct"] and not eb[i] and et[i]]
    nobox = sorted(nobox, key=lambda i: tr[i]["tokens"])
    if nobox:
        examples.append(ex(nobox[0], "base gave no boxed answer (format failure or ran out of tokens), trained model right"))
    regress = shortest(lambda b, t: b["correct"] and not t["correct"])
    if regress:
        examples.append(ex(regress[0], "honesty: base right, trained model wrong (shortest such pair)"))

    bcf = L / "lm_grpo_greedy_batch_check.json"
    bc = None
    if bcf.exists():
        b = json.loads(bcf.read_text())
        bc = {"batch_sizes": b["batch_sizes"], "base": {k: r4(v) for k, v in b["base"].items()},
              "trained": {k: r4(v) for k, v in b["trained"].items()},
              "base_mean": r4(b["base_mean"]), "trained_mean": r4(b["trained_mean"]), "gain_mean": r4(b["gain_mean"]),
              "stable_fraction_base": r4(b["base_stable_fraction"]), "stable_fraction_trained": r4(b["trained_stable_fraction"]),
              "note": "same weights, greedy, only the generation batch size changes (bf16 batched decoding is not batch-invariant); the eval log used 256"}
    examples_agreement = {"base": gt.get("base_agreement_with_eval_log"), "trained": gt.get("trained_agreement_with_eval_log"),
                          "note": "example texts were re-decoded at batch 64; fraction of questions whose greedy correctness matches the eval log (batch 256)"}

    # ---- one update's ratio / clip statistics (mid-run, second minibatch)
    mid = steps[n // 2]
    update = {"step": mid["step"], "minibatches": mid["minibatches"],
              "note": "minibatch 1 is scored by the same weights that sampled it, so ratio = 1 exactly; "
                      "minibatch 2 comes after one AdamW step, so ratios spread and a few tokens are clipped"}

    # ---- extras
    dpo = json.loads((L / "lm_dpo.json").read_text()) if (L / "lm_dpo.json").exists() else None
    hacks = {}
    for run in ("lm_hack_anywhere", "lm_hack_length"):
        st = jl(L / f"{run}.jsonl")
        if not st:
            continue
        gr = jl(L / f"{run}_groups.jsonl")
        he = evals_of(run)
        nums = numbers_per_answer(gr)
        late = [g for r in gr[-5:] for g in r["groups"]]
        exploit = sorted([(g["rewards"][j] - g["correct"][j], g["tokens"][j], g["completions"][j], g["question"], g["ref"])
                          for g in late for j in range(len(g["rewards"]))
                          if g["correct"][j] == 0 and RECHECK.search(g["completions"][j])],
                         key=lambda x: (-x[0], len(x[3])))
        hacks[run] = {
            "config": json.loads((L / f"{run}_config.json").read_text())["args"],
            "steps": len(st),
            "curves": {"step": [s["step"] for s in st], "proxy_reward": [r4(s["reward"]) for s in st],
                       "true_accuracy": [r4(s["accuracy"]) for s in st],
                       "proxy_reward_ma10": [r4(x) for x in ma([s["reward"] for s in st])],
                       "true_accuracy_ma10": [r4(x) for x in ma([s["accuracy"] for s in st])],
                       "mean_len": [r4(s["mean_len"]) for s in st], "boxed_rate": [r4(s["boxed_rate"]) for s in st],
                       "truncated_rate": [r4(s["truncated_rate"]) for s in st],
                       "numbers_per_answer": [r4(x) for x in nums], "kl": [r4(s["kl"]) for s in st],
                       "main_run_true_accuracy": [r4(steps[i]["accuracy"]) for i in range(min(len(st), n))],
                       "main_run_mean_len": [r4(steps[i]["mean_len"]) for i in range(min(len(st), n))],
                       "main_run_truncated_rate": [r4(steps[i]["truncated_rate"]) for i in range(min(len(st), n))]},
            "paired_note": "same questions in the same order and the same sampling seeds as the main run; only the reward differs",
            "main_run_same_steps": {k: r4(mean([steps[i][k] for i in range(min(len(st), n))]))
                                    for k in ("accuracy", "mean_len", "truncated_rate", "boxed_rate")},
            "main_run_same_last10": {k: r4(mean([steps[i][k] for i in range(max(0, len(st) - 10), min(len(st), n))]))
                                     for k in ("accuracy", "mean_len", "truncated_rate", "boxed_rate")},
            "first10": {k: r4(mean([s[k] for s in st[:10]])) for k in ("reward", "accuracy", "mean_len", "boxed_rate", "truncated_rate")},
            "last10": {k: r4(mean([s[k] for s in st[-10:]])) for k in ("reward", "accuracy", "mean_len", "boxed_rate", "truncated_rate")},
            "numbers_per_answer_first10": r4(mean(nums[:10])), "numbers_per_answer_last10": r4(mean(nums[-10:])),
            "evals": [eval_row(e) for e in he],
            "exploit_example": ({"question": exploit[0][3], "ref": exploit[0][4], "proxy_minus_true": r4(exploit[0][0]),
                                 "tokens": exploit[0][1], "completion": exploit[0][2],
                                 "completion_short": short(exploit[0][2])} if exploit else None),
            "recheck_phrase_regex": RECHECK.pattern,
            "recheck_rate_first10": r4(recheck_rate(gr[:10])), "recheck_rate_last10": r4(recheck_rate(gr[-10:])),
            "main_run_recheck_rate_same_first10": r4(recheck_rate(groups[:10])),
            "main_run_recheck_rate_same_last10": r4(recheck_rate(groups[max(0, len(gr) - 10):len(gr)])),
            "late_truncated_count": sum(n >= a["max_tokens"] for g in late for n in g["tokens"]),
            "late_completions": sum(len(g["tokens"]) for g in late),
            "wall_s": round(sum(s["sample_s"] + s["logprob_s"] + s["update_s"] for s in st), 1)}

    head = {
        "model": "Qwen2.5-0.5B-Instruct", "mlx_weights": cfg["mlx_model"], "params_base_M": round((cfg["total_params"] - cfg["trainable_params"]) / 1e6, 1),
        "trainable_params_M": round(cfg["trainable_params"] / 1e6, 2), "finetune": f"LoRA rank {a['lora_rank']} (scale {a['lora_scale']}) on all linear layers of all 24 blocks",
        "task": "GSM8K grade-school maths (Cobbe et al. 2021)", "test_questions": a["eval_n"],
        "G": a["group"], "questions_per_step": a["prompts"], "completions_per_step": a["group"] * a["prompts"],
        "beta_kl": a["beta"], "clip_eps": a["eps"], "lr": a["lr"], "temperature_train": a["temp"],
        "max_new_tokens": a["max_tokens"], "steps": n, "optimizer_steps": n * a["minibatches"],
        "train_questions_seen": n * a["prompts"], "completions_sampled": n * a["prompts"] * a["group"],
        "wall_time_train_min": round(train_s / 60, 1), "wall_time_eval_min": round(eval_s / 60, 1),
        "wall_time_total_min": round((train_s + eval_s) / 60, 1),
        "sec_per_step": round(train_s / n, 1),
        "greedy_acc_before": r4(first["greedy_acc"]), "greedy_acc_after": r4(last["greedy_acc"]),
        "pass_at_k_before": {k: r4(v) for k, v in first["pass_at_k"].items()},
        "pass_at_k_after": {k: r4(v) for k, v in last["pass_at_k"].items()},
        "eval_temperature": first["temperature"],
        "boxed_rate_before": r4(first["greedy_boxed_rate"]), "boxed_rate_after": r4(last["greedy_boxed_rate"]),
        "train_reward_first10": r4(mean([s["reward"] for s in steps[:w]])),
        "train_reward_last10": r4(mean([s["reward"] for s in steps[-w:]])),
        "train_len_first10": round(mean([s["mean_len"] for s in steps[:w]]), 1),
        "train_len_last10": round(mean([s["mean_len"] for s in steps[-w:]]), 1),
        "greedy_len_before": round(first["greedy_mean_len"], 1), "greedy_len_after": round(last["greedy_mean_len"], 1),
        "train_reward_first25": r4(mean([s["reward"] for s in steps[:25]])),
        "train_reward_last25": r4(mean([s["reward"] for s in steps[-25:]])),
        "train_samples_format_vs_maths": {"steps_0_24": fmt(groups[:25]), "steps_175_199": fmt(groups[-25:]),
                                          "note": "strict = the reward (boxed answer); last_number = format-free check; different questions early vs late"},
        "grad_clip_active_frac": r4(mean([m["grad_norm"] > a["grad_clip"] for s in steps for m in s["minibatches"]])),
        "grad_norm_max": round(max(m["grad_norm"] for s in steps for m in s["minibatches"]), 1),
        "wall_clock_note": "00:44 -> 02:45 on 2026-10-25 (121 min) including one GPU out-of-memory crash during the step-100 eval and a resume from the step-99 checkpoint",
        "kl_last10": r4(mean([s["kl"] for s in steps[-w:]])),
        "clip_frac_mean": r4(mean([s["clip_frac"] for s in steps])),
        "zero_adv_groups_first10": r4(mean([s["zero_adv_groups"] for s in steps[:w]])),
        "zero_adv_groups_last10": r4(mean([s["zero_adv_groups"] for s in steps[-w:]])),
        "gen_tokens_per_s": round(gen_tok / gen_s), "train_tokens_per_s": round(mean([s["train_tokens_per_s"] for s in steps])),
        "peak_mem_gb": round(max(s["peak_mem_gb"] for s in steps), 1),
        "heldout_flips": table,
        "greedy_by_batch_size": bc,
        "paired_se": paired_se(first, last),
    }
    out = {
        "schema": "lm_rl/1", "generated": str(date.today()),
        "machine": "Mac Studio, Apple M4 Max, 64 GB unified memory, Apple-silicon GPU (MLX)",
        "software": {"python": platform.python_version(), "mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
        "headline": head,
        "setup": {
            "prompt_system": C.SYSTEM, "prompt_format": "Qwen2.5 chat template: system = the instruction above, user = the GSM8K question",
            "reward": "1 if the last number inside the last \\boxed{...} equals the GSM8K reference (|diff| < 1e-4), else 0; no format term",
            "boxed_regex": C.BOXED.pattern, "number_regex": C.NUMBER.pattern,
            "advantage": "A_i = (r_i - mean(r_group)) / (std_population(r_group) + 1e-4)",
            "loss": "per-token min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) - beta*k3_KL, averaged over each completion's tokens, then over completions (DeepSeekMath sequence-level averaging)",
            "kl": "k3 = exp(ref_lp - lp) - (ref_lp - lp) - 1 per token, reference = the frozen base model",
            "data": "GSM8K main: train = 7,473 problems shuffled with seed 0, steps use them in order; test = fixed 500 of 1,319 (seed 0)",
            "args": a,
        },
        "model_selection": {k: {kk: (r4(vv) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "example"} for k, v in sel.items()},
        "evals": [eval_row(e) for e in evs],
        "curves": curves(steps),
        "examples": examples,
        "examples_decode_agreement": examples_agreement,
        "groups": {"early": pick_group(groups, 0, 4), "late": pick_group(groups, n - 10, n - 1)},
        "update": update,
        "dpo": dpo,
        "reward_hacking": hacks,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(head, indent=1))


main()
