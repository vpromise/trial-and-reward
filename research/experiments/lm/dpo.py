"""Episode 22 extra: DPO (Rafailov et al. 2023) on preference pairs made from the base model's own
answers (chosen = a correct answer, rejected = a wrong answer to the same question).

  1. build pairs: 4 samples (T = 1.0) per question for 400 GSM8K train questions the GRPO run
     never saw (train[2000:2400]); keep questions with at least one right and one wrong answer;
  2. one showcase pair: sequence log-probs under pi_ref (base), the GRPO-trained policy and the
     DPO-trained policy, and the DPO loss for several beta;
  3. a short DPO run (LoRA, same config as GRPO), then greedy accuracy on the 500 held-out problems.

DPO loss:  -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )
Usage: python -I dpo.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

import common as C  # noqa: E402

KEY, BETA, LR, EPOCHS, PAIRS_PER_STEP, SEED = "qwen2.5-0.5b", 0.1, 1e-5, 2, 8, 0
OUT = C.LOGS / "lm_dpo.json"
mx.random.seed(SEED)
ref, tok = C.load_model(KEY)
ref.freeze()
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_ids[0]
train, test = C.load_gsm8k()

# ---------------------------------------------------------------- 1. pairs from the base model
t0 = time.perf_counter()
qs = train[2000:2400]
prompts = [C.encode(tok, C.build_prompt(tok, q["question"], KEY)) for q in qs]
rep = [p for p in prompts for _ in range(4)]
outs, fins, _ = C.generate(ref, tok, rep, 512, temp=1.0, seed=777, batch_size=64, chunk=64)
rng = random.Random(SEED)
pairs = []
for i, q in enumerate(qs):
    cs = outs[4 * i:4 * i + 4]
    good = [c for c in cs if C.reward_correct(C.decode(tok, c), q["ref"]) == 1.0]
    bad = [c for c in cs if C.reward_correct(C.decode(tok, c), q["ref"]) == 0.0]
    if good and bad:
        pairs.append({"q": q, "prompt": prompts[i], "chosen": rng.choice(good), "rejected": rng.choice(bad)})
print(f"{len(pairs)} pairs from {len(qs)} questions ({time.perf_counter() - t0:.0f}s)", flush=True)
# showcase: the pair with the shortest chosen+rejected whose rejected answer still boxes a (wrong) number
show = min((p for p in pairs if C.extract_answer(C.decode(tok, p["rejected"])) is not None),
           key=lambda p: len(p["chosen"]) + len(p["rejected"]))


def batch_of(ps):
    """Stack [chosen_1..chosen_n, rejected_1..rejected_n] with completion masks."""
    seqs = [p["prompt"] + p["chosen"] for p in ps] + [p["prompt"] + p["rejected"] for p in ps]
    plen = [len(p["prompt"]) for p in ps] * 2
    toks = C.pad_batch(seqs, pad_id)
    L = toks.shape[1] - 1
    mask = mx.array([[1.0 if plen[i] - 1 <= t < len(seqs[i]) - 1 else 0.0 for t in range(L)]
                     for i in range(len(seqs))])
    return toks, mask


def seq_logp(model, toks, mask):
    return (C.token_logprobs(model, toks) * mask).sum(1)        # log pi(y | x) = sum of token log-probs


def dpo_loss(model, toks, mask, ref_lp, n_total):
    d = seq_logp(model, toks, mask) - ref_lp                     # log pi/ref for each answer
    n = d.shape[0] // 2
    margin = d[:n] - d[n:]                                       # chosen minus rejected
    return -nn.log_sigmoid(BETA * margin).sum() / n_total, (BETA * margin, d)


def score_pair(model, p):
    toks, mask = batch_of([p])
    s = seq_logp(model, toks, mask)
    mx.eval(s)
    return s.tolist()                                            # [log p(chosen), log p(rejected)]


def pair_report(name, model):
    lw, ll = score_pair(model, show)
    rw, rl = ref_show
    row = {"policy": name, "logp_chosen": lw, "logp_rejected": ll, "ref_logp_chosen": rw, "ref_logp_rejected": rl,
           "delta_chosen": lw - rw, "delta_rejected": ll - rl}
    for b in (0.05, 0.1, 0.5):
        m = b * ((lw - rw) - (ll - rl))
        row[f"dpo_loss_beta_{b}"] = -math.log(1 / (1 + math.exp(-m)))
        row[f"implicit_reward_margin_beta_{b}"] = m
    return row


ref_show = score_pair(ref, show)
report = {"n_pairs": len(pairs), "n_questions": len(qs), "samples_per_question": 4, "beta_train": BETA,
          "lr": LR, "epochs": EPOCHS, "pairs_per_step": PAIRS_PER_STEP,
          "showcase": {"question": show["q"]["question"], "ref_answer": show["q"]["ref"],
                       "chosen": C.decode(tok, show["chosen"]), "rejected": C.decode(tok, show["rejected"]),
                       "chosen_tokens": len(show["chosen"]), "rejected_tokens": len(show["rejected"])},
          "pair_scores": []}

# ---------------------------------------------------------------- 2. the pair under pi_ref and the GRPO policy
policy, _ = C.load_model(KEY)
policy.freeze()
linear_to_lora_layers(policy, len(policy.layers), {"rank": 16, "scale": 2.0, "dropout": 0.0})
report["pair_scores"].append(pair_report("base (policy = reference, before any training)", policy))
grpo_ckpt = Path(__file__).resolve().parent / "ckpt" / "lm_grpo" / "adapters.safetensors"
init = dict(tree_flatten(policy.trainable_parameters()))
if grpo_ckpt.exists():
    policy.load_weights(str(grpo_ckpt), strict=False)
    report["pair_scores"].append(pair_report("GRPO-trained (lm_grpo, final)", policy))
    policy.load_weights(list(init.items()), strict=False)       # back to the fresh adapters

# ---------------------------------------------------------------- 3. a short DPO run
opt = optim.AdamW(learning_rate=LR, weight_decay=0.0)
lg = nn.value_and_grad(policy, dpo_loss)
curve, step = [], 0
t0 = time.perf_counter()
for ep in range(EPOCHS):
    order = list(range(len(pairs)))
    random.Random(SEED + ep).shuffle(order)
    for s in range(0, len(order) - PAIRS_PER_STEP + 1, PAIRS_PER_STEP):
        idx = order[s:s + PAIRS_PER_STEP]
        grads, loss_sum, acc, margins, dch, drj = None, 0.0, 0.0, [], [], []
        for j in range(0, PAIRS_PER_STEP, 2):                     # 2 pairs = 4 sequences per pass
            sub = idx[j:j + 2]
            toks, mask = batch_of([pairs[k] for k in sub])
            r = mx.stop_gradient(seq_logp(ref, toks, mask))   # same padded batch -> exact
            (loss, (m, d)), g = lg(policy, toks, mask, r, PAIRS_PER_STEP)
            grads = g if grads is None else tree_map(mx.add, grads, g)
            mx.eval(grads, loss, m, d)
            loss_sum += loss.item()
            margins += m.tolist(); n = len(sub)
            dch += d.tolist()[:n]; drj += d.tolist()[n:]
        grads, gn = optim.clip_grad_norm(grads, 1.0)
        opt.update(policy, grads)
        mx.eval(policy.trainable_parameters(), opt.state)
        sw, sl = score_pair(policy, show)
        curve.append({"step": step, "loss": loss_sum, "reward_acc": sum(x > 0 for x in margins) / len(margins),
                      "margin": sum(margins) / len(margins), "delta_logp_chosen": sum(dch) / len(dch),
                      "delta_logp_rejected": sum(drj) / len(drj), "show_logp_chosen": sw, "show_logp_rejected": sl})
        print(f"dpo step {step} loss {loss_sum:.4f} acc {curve[-1]['reward_acc']:.2f} "
              f"dchosen {curve[-1]['delta_logp_chosen']:.2f} drejected {curve[-1]['delta_logp_rejected']:.2f}", flush=True)
        step += 1
report["train_seconds"] = time.perf_counter() - t0
report["curve"] = curve
report["pair_scores"].append(pair_report(f"DPO-trained ({step} steps, beta {BETA})", policy))

# ---------------------------------------------------------------- held-out greedy accuracy after DPO
items = test
ps = [C.encode(tok, C.build_prompt(tok, it["question"], KEY)) for it in items]
g, gf, _ = C.generate(policy, tok, ps, 512, temp=0.0, seed=1234, batch_size=64)
texts = [C.decode(tok, x) for x in g]
report["heldout_greedy_acc_after_dpo"] = sum(C.reward_correct(t, it["ref"]) for t, it in zip(texts, items)) / len(items)
report["heldout_greedy_mean_len_after_dpo"] = sum(map(len, g)) / len(g)
report["heldout_n"] = len(items)
C.jdump(report, OUT)
print("held-out greedy after DPO", report["heldout_greedy_acc_after_dpo"], flush=True)
