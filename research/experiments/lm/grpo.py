"""GRPO (Shao et al. 2024, DeepSeekMath) on GSM8K with mlx-lm, LoRA on a small instruct model.

One training step:
  1. take B questions, sample G completions each from the current policy (pi_old),
  2. score each with the rule-based checker (reward 0/1),
  3. advantage of completion i in its group: A_i = (r_i - mean(r)) / (std(r) + 1e-4),
  4. score the sampled tokens under pi_old (no grad) and the frozen reference pi_ref,
  5. split the B*G completions into `minibatches` and take one AdamW step per minibatch on the
     clipped-ratio objective with a k3 KL penalty to pi_ref (the loss below).

Usage (the main run):  python -I grpo.py --run lm_grpo
Resumes automatically from lm/ckpt/<run>/ if a checkpoint exists.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

import common as C  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="lm_grpo")
p.add_argument("--model", default="qwen2.5-0.5b")
p.add_argument("--reward", default="correct", choices=["correct", "anywhere", "length"],
               help="correct: the true checker; anywhere / length: deliberately flawed (episode 25)")
p.add_argument("--steps", type=int, default=200)
p.add_argument("--prompts", type=int, default=8, help="questions per step (B)")
p.add_argument("--group", type=int, default=8, help="completions per question (G)")
p.add_argument("--minibatches", type=int, default=2, help="optimizer steps per batch of rollouts")
p.add_argument("--micro", type=int, default=8, help="sequences per forward/backward pass")
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
p.add_argument("--eps", type=float, default=0.2, help="PPO clip range")
p.add_argument("--temp", type=float, default=1.0, help="sampling temperature during training")
p.add_argument("--max-tokens", type=int, default=512)
p.add_argument("--lora-rank", type=int, default=16)
p.add_argument("--lora-scale", type=float, default=2.0, help="LoRA multiplier (alpha / rank)")
p.add_argument("--grad-clip", type=float, default=1.0)
p.add_argument("--eval-every", type=int, default=50)
p.add_argument("--eval-n", type=int, default=500, help="held-out questions per evaluation")
p.add_argument("--eval-samples", type=int, default=8)
p.add_argument("--eval-temp", type=float, default=0.8)
p.add_argument("--ckpt-every", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-eval", action="store_true")
args = p.parse_args()

LOG = C.LOGS / f"{args.run}.jsonl"            # one line per step
EVAL_LOG = C.LOGS / f"{args.run}_eval.jsonl"   # one line per evaluation
GROUPS = C.LOGS / f"{args.run}_groups.jsonl"   # every sampled group, verbatim
CKPT = Path(__file__).resolve().parent / "ckpt" / args.run
CKPT.mkdir(parents=True, exist_ok=True)
SHOW_IDS = ("test-1", "test-4", "test-22", "test-36", "test-55", "test-67")  # candidates for before/after scenes

# ------------------------------------------------------------------ models
mx.set_cache_limit(8 * 1024**3)            # keep MLX's buffer cache small: the GPU is shared with other jobs
mx.random.seed(args.seed)
policy, tok = C.load_model(args.model)
ref, _ = C.load_model(args.model)          # frozen reference pi_ref = the base model
ref.freeze()
policy.freeze()
linear_to_lora_layers(policy, len(policy.layers),
                      {"rank": args.lora_rank, "scale": args.lora_scale, "dropout": 0.0})
n_train = sum(v.size for _, v in tree_flatten(policy.trainable_parameters()))
n_total = sum(v.size for _, v in tree_flatten(policy.parameters()))
opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
reward_fn = {"correct": C.reward_correct, "anywhere": C.reward_anywhere,
             "length": C.reward_correct}[args.reward]       # "length" adds 0.001 per token below
train, test = C.load_gsm8k()
test = test[:args.eval_n]
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_ids[0]

start = 0
if (CKPT / "state.json").exists():                       # resume
    st = json.loads((CKPT / "state.json").read_text())
    policy.load_weights(str(CKPT / "adapters.safetensors"), strict=False)
    opt.init(policy.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(str(CKPT / "optimizer.safetensors")).items()))
    start = st["step"] + 1
    for f in (LOG, GROUPS):                              # drop log lines written after the checkpoint
        if f.exists():
            keep = [l for l in f.read_text().splitlines() if l and json.loads(l)["step"] <= st["step"]]
            f.write_text("".join(l + "\n" for l in keep))
    print(f"resumed from step {st['step']}", flush=True)
else:
    for f in (LOG, GROUPS, EVAL_LOG):
        f.unlink(missing_ok=True)
    C.jdump({"args": vars(args), "trainable_params": n_train, "total_params": n_total,
             "mlx_model": C.MODELS[args.model]}, C.LOGS / f"{args.run}_config.json")
print(f"trainable {n_train/1e6:.2f}M of {n_total/1e6:.1f}M params", flush=True)


# ------------------------------------------------------------------ the GRPO loss
def grpo_loss(model, tokens, mask, old_lp, ref_lp, adv, n_seq):
    lp = C.token_logprobs(model, tokens)                       # log pi_theta(o_t | q, o_<t)
    ratio = mx.exp(lp - old_lp)                                # pi_theta / pi_old, per token
    A = adv[:, None]
    surr = mx.minimum(ratio * A, mx.clip(ratio, 1 - args.eps, 1 + args.eps) * A)
    kl = mx.exp(ref_lp - lp) - (ref_lp - lp) - 1               # k3 estimate of KL(pi_theta || pi_ref)
    per_token = surr - args.beta * kl
    per_seq = (per_token * mask).sum(1) / mask.sum(1)          # 1/|o_i| * sum over tokens
    loss = -per_seq.sum() / n_seq                              # 1/G * sum over completions
    clipped = ((ratio > 1 + args.eps) & (A > 0)) | ((ratio < 1 - args.eps) & (A < 0))
    return loss, (kl * mask).sum(), (clipped * mask).sum(), ratio, mask.sum()


loss_and_grad = nn.value_and_grad(policy, grpo_loss)


def evaluate(step):
    if args.no_eval:
        return
    mx.clear_cache()
    t0 = time.perf_counter()
    summ, per_q, keep = C.evaluate(policy, tok, test, args.model, n_samples=args.eval_samples,
                                   temp=args.eval_temp, max_tokens=args.max_tokens, keep_text_ids=SHOW_IDS)
    summ.update(step=step, seconds=time.perf_counter() - t0, examples=keep, per_question=per_q)
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(summ) + "\n")
    print(f"[eval step {step}] greedy {summ['greedy_acc']:.3f} pass@k {summ['pass_at_k']} "
          f"len {summ['greedy_mean_len']:.0f} ({summ['seconds']:.0f}s)", flush=True)


def micro_batches(idx, seqs):
    """Sort a minibatch by length and cut it into micro-batches (less padding)."""
    idx = sorted(idx, key=lambda i: len(seqs[i]))
    return [idx[j:j + args.micro] for j in range(0, len(idx), args.micro)]


def ratio_hist(r):
    edges = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.2, 1.5, 1e9]
    return {"edges": edges, "counts": [sum(a <= x < b for x in r) for a, b in zip(edges[:-1], edges[1:])]}


# ------------------------------------------------------------------ training loop
done_evals = {json.loads(l)["step"] for l in EVAL_LOG.read_text().splitlines() if l} if EVAL_LOG.exists() else set()
if start == 0 or (start % args.eval_every == 0 and start not in done_evals):
    evaluate(start)                                      # (re-)run an evaluation lost to a crash
t_run = time.perf_counter()
for step in range(start, args.steps):
    mx.reset_peak_memory()
    batch = [train[(step * args.prompts + j) % len(train)] for j in range(args.prompts)]
    prompts = [C.encode(tok, C.build_prompt(tok, it["question"], args.model)) for it in batch]

    # 1. sample G completions per question from pi_old (= the current policy)
    t0 = time.perf_counter()
    rep = [pr for pr in prompts for _ in range(args.group)]
    comps, fins, gst = C.generate(policy, tok, rep, args.max_tokens, temp=args.temp,
                                  seed=args.seed * 100003 + step, batch_size=128)
    t_sample = time.perf_counter() - t0

    # 2-3. rewards and group-relative advantages
    texts = [C.decode(tok, c) for c in comps]
    rewards = [reward_fn(t, batch[i // args.group]["ref"]) for i, t in enumerate(texts)]
    if args.reward == "length":                          # flawed: "longer reasoning is better"
        rewards = [r + 0.001 * len(c) for r, c in zip(rewards, comps)]
    correct = [C.reward_correct(t, batch[i // args.group]["ref"]) for i, t in enumerate(texts)]
    adv, zero_groups = [], 0
    for g in range(args.prompts):
        r = mx.array(rewards[g * args.group:(g + 1) * args.group])
        a = (r - r.mean()) / (mx.sqrt(((r - r.mean()) ** 2).mean()) + 1e-4)   # population std
        zero_groups += int(mx.all(r == r[0]).item())
        adv += a.tolist()

    # 4. sequences, completion masks, log-probs under pi_old and pi_ref (no gradient)
    t0 = time.perf_counter()
    seqs = [rep[i] + comps[i] for i in range(len(rep))]
    plen = [len(x) for x in rep]
    N = len(seqs)
    mbs = [list(range(m * N // args.minibatches, (m + 1) * N // args.minibatches))
           for m in range(args.minibatches)]
    prepared = []
    for mb in mbs:
        parts = []
        for ids in micro_batches(mb, seqs):
            toks_ = C.pad_batch([seqs[i] for i in ids], pad_id)
            L = toks_.shape[1] - 1
            mask = mx.array([[1.0 if plen[i] - 1 <= t < len(seqs[i]) - 1 else 0.0 for t in range(L)]
                             for i in ids])
            old_lp = mx.stop_gradient(C.token_logprobs(policy, toks_))
            ref_lp = mx.stop_gradient(C.token_logprobs(ref, toks_))
            mx.eval(old_lp, ref_lp)
            parts.append((ids, toks_, mask, old_lp, ref_lp, mx.array([adv[i] for i in ids])))
        prepared.append(parts)
    t_lp = time.perf_counter() - t0

    # 5. one optimizer step per minibatch
    t0 = time.perf_counter()
    upd = []
    for m, parts in enumerate(prepared):
        n_seq = sum(len(x[0]) for x in parts)
        grads, tot = None, {"loss": 0.0, "kl": 0.0, "clip": 0.0, "ntok": 0.0}
        ratios = []
        for ids, toks_, mask, old_lp, ref_lp, a in parts:
            (loss, kl_sum, clip_sum, ratio, ntok), g = loss_and_grad(policy, toks_, mask, old_lp, ref_lp, a, n_seq)
            grads = g if grads is None else tree_map(mx.add, grads, g)
            mx.eval(grads, loss, kl_sum, clip_sum, ratio)
            tot["loss"] += loss.item(); tot["kl"] += kl_sum.item()
            tot["clip"] += clip_sum.item(); tot["ntok"] += ntok.item()
            mk = mask.astype(mx.bool_)
            ratios += [x for row, mrow in zip(ratio.tolist(), mk.tolist()) for x, keep in zip(row, mrow) if keep]
        grads, gnorm = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(policy, grads)
        mx.eval(policy.trainable_parameters(), opt.state)
        upd.append({"loss": tot["loss"], "kl": tot["kl"] / tot["ntok"], "clip_frac": tot["clip"] / tot["ntok"],
                    "grad_norm": gnorm.item(), "ratio_mean": sum(ratios) / len(ratios),
                    "ratio_min": min(ratios), "ratio_max": max(ratios),
                    "outside_band": sum(abs(x - 1) > args.eps for x in ratios) / len(ratios),
                    "ratio_hist": ratio_hist(ratios)})
    t_update = time.perf_counter() - t0

    n_tok_train = sum(len(s) for s in seqs)
    lens = [len(c) for c in comps]
    rec = {"step": step, "reward": sum(rewards) / N, "accuracy": sum(correct) / N,
           "zero_adv_groups": zero_groups / args.prompts,
           "boxed_rate": sum(C.extract_answer(t) is not None for t in texts) / N,
           "truncated_rate": sum(f == "length" for f in fins) / N,
           "mean_len": sum(lens) / N, "max_len": max(lens),
           "kl": sum(u["kl"] for u in upd) / len(upd),
           "clip_frac": sum(u["clip_frac"] for u in upd) / len(upd),
           "loss": sum(u["loss"] for u in upd) / len(upd),
           "sample_s": t_sample, "logprob_s": t_lp, "update_s": t_update,
           "gen_tokens": gst["tokens"], "gen_tokens_per_s": gst["tokens_per_s"],
           "train_tokens": n_tok_train, "train_tokens_per_s": n_tok_train / (t_lp + t_update),
           "peak_mem_gb": mx.get_peak_memory() / 1e9, "minibatches": upd,
           "elapsed_s": time.perf_counter() - t_run}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    with open(GROUPS, "a") as f:
        f.write(json.dumps({"step": step, "groups": [
            {"id": batch[g]["id"], "question": batch[g]["question"], "ref": batch[g]["ref"],
             "completions": texts[g * args.group:(g + 1) * args.group],
             "tokens": lens[g * args.group:(g + 1) * args.group],
             "rewards": rewards[g * args.group:(g + 1) * args.group],
             "correct": correct[g * args.group:(g + 1) * args.group],
             "advantages": [round(x, 4) for x in adv[g * args.group:(g + 1) * args.group]]}
            for g in range(args.prompts)]}) + "\n")
    print(f"step {step:4d} reward {rec['reward']:.3f} zero {rec['zero_adv_groups']:.2f} len {rec['mean_len']:.0f} "
          f"kl {rec['kl']:.4f} clip {rec['clip_frac']:.4f} | sample {t_sample:.1f}s lp {t_lp:.1f}s "
          f"upd {t_update:.1f}s {gst['tokens_per_s']:.0f} tok/s mem {rec['peak_mem_gb']:.1f}GB", flush=True)

    if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
        mx.save_safetensors(str(CKPT / "adapters.safetensors"), dict(tree_flatten(policy.trainable_parameters())))
        mx.save_safetensors(str(CKPT / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
        (CKPT / "state.json").write_text(json.dumps({"step": step}))
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        evaluate(step + 1)
print("done", flush=True)
