# GRPO on this Mac — the language-model RL runs (Module 6, episodes 20–26)

Everything below was run on this Mac Studio (Apple M4 Max, 64 GB unified memory, Apple-silicon GPU) with MLX,
on 2026-10-25. Every number here is in `lm_rl.json` (stable keys; scenes should read that file, not this one).
Code: `lm/`. Logs: `logs/lm_*`.

## Headline

| | before (base) | after 200 GRPO steps |
|---|---|---|
| **pass@1** (8 samples/question, T = 0.8, unbiased estimator) | **38.3 %** | **46.2 %** (+7.9 ± 0.9 points, paired SE) |
| pass@2 | 51.7 % | 58.1 % |
| pass@4 | 63.0 % | 68.4 % |
| pass@8 | 71.6 % | 76.4 % (+4.8 ± 1.8) |
| greedy accuracy (eval batch 256) | 46.2 % | 50.4 % (+4.2 ± 2.0) |
| greedy accuracy, mean over decode batch sizes 16–256 | 47.3 % (46.2–48.2) | 50.1 % (49.0–51.0) (+2.8) |
| answers with a `\boxed{}` final answer (greedy) | 94.8 % | 95.8 % |
| mean greedy answer length | 309 tokens | 305 tokens |
| training reward (mean of steps 1–25 → 176–200) | 0.476 | 0.570 |

Held-out set: 500 GSM8K test problems the model never trained on. Model: Qwen2.5-0.5B-Instruct (494 M parameters),
LoRA with 8.8 M trainable parameters. 200 steps × 8 questions × G = 8 answers = **12,800 sampled solutions to 1,600
training problems**, 400 AdamW updates. **Compute: 71.7 min of training + 37.4 min of evaluation = 109 min**
(wall clock 00:44 → 02:45, 121 min, including one crash and resume). One step ≈ 21.5 s: sampling 9.2 s
(≈ 19,500 new tokens at **≈ 2,150 tokens/s**), scoring with the old and reference policies 5.0 s, two updates 7.4 s.
Peak GPU memory ≈ 25 GB (max 28.8 GB).

**The clean claim:** sampled one at a time, the model got about **8 points better** at problems it had never seen
(38 → 46 %). The greedy (temperature-0) number moved by **3–4 points**, which is about the size of its own
measurement noise (see Caveats: greedy decoding is not batch-invariant).

## Setup

**Model and why.** `mlx-community/Qwen2.5-0.5B-Instruct-bf16`. Measured greedy accuracy on the first 100 of the
500 held-out problems, same prompt (`lm/select_model.py`, `logs/lm_model_selection.json`):

| model | greedy acc. | boxed | mean length | generation tok/s (batch 100) |
|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | **46 %** | 92 % | 300 | **4,050** |
| Qwen3-0.6B (non-thinking) | 62 % | 97 % | 228 | 1,733 |
| Qwen2.5-1.5B-Instruct | 69 % | 91 % | 286 | 1,448 |

0.5B is the only candidate clearly inside the 20–60 % band (plenty of problems it sometimes gets right, which is what
GRPO needs). It is also 2–3× faster to sample from, which is what makes 12,800 rollouts fit in about an hour.

**Task and data.** GSM8K (Cobbe et al. 2021), `openai/gsm8k` "main" parquet files from Hugging Face
(`data/gsm8k/`). The reference answer is the number after `####`. Train: all 7,473 problems shuffled with
`random.Random(0)`, consumed in order (steps 0–199 use problems 0–1,599). Test: a fixed `random.Random(0).sample` of
500 of the 1,319 test problems (ids `test-<index>` in the JSON).

**Prompt template** (Qwen2.5 chat template; the system message is Qwen2.5-Math's chain-of-thought instruction, so
the default "You are Qwen…" system prompt is *not* used):

```
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
{GSM8K question}<|im_end|>
<|im_start|>assistant
```

**Reward (the checker).** 1 if the last number inside the last `\boxed{...}` equals the reference
(|difference| < 1e-4), else 0. No format bonus, so on the training batches *accuracy = mean reward*.

```python
BOXED  = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")   # \boxed{...}, one level of nested braces
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")                 # commas are stripped before float()
```

An answer with no box (including one cut off at 512 tokens) scores 0. A diagnostic "last number anywhere" check
(not a reward) separates format from maths.

**Algorithm: GRPO as in DeepSeekMath (Shao et al. 2024)**, written on top of mlx-lm (`lm/grpo.py`, about 250 lines).
mlx-lm provides only model loading, the batched sampler and the LoRA layers. Per step:

1. take 8 questions and sample G = 8 completions each from the current policy (π_old), pure temperature-1.0
   sampling (no top-p/top-k), at most 512 new tokens;
2. reward each completion with the checker;
3. advantage within each group: `A_i = (r_i − mean(r)) / (std(r) + 1e-4)` (population std of the 8 rewards);
4. score the sampled tokens under π_old and under the frozen reference π_ref (= the base model, no LoRA), no gradient;
5. split the 64 completions into 2 minibatches of 32 (4 groups each) and take one AdamW step on each.
   Minibatch 1 is scored by the same weights that sampled it, so its ratio is exactly 1. Minibatch 2 comes after one
   update, so its ratios spread and clipping can act.

The core loss, as it can be shown on screen (12 lines, from `lm/grpo.py`):

```python
a = (r - r.mean()) / (r.std() + 1e-4)                      # group-relative advantage, one per answer

def grpo_loss(model, tokens, mask, old_lp, ref_lp, adv, n_seq):
    lp = token_logprobs(model, tokens)                     # log pi_theta(o_t | q, o_<t)
    ratio = mx.exp(lp - old_lp)                            # pi_theta / pi_old, per token
    A = adv[:, None]
    surr = mx.minimum(ratio * A, mx.clip(ratio, 1 - eps, 1 + eps) * A)
    kl = mx.exp(ref_lp - lp) - (ref_lp - lp) - 1           # k3 estimate of KL(pi_theta || pi_ref)
    per_token = surr - beta * kl
    per_seq = (per_token * mask).sum(1) / mask.sum(1)      # 1/|o_i| * sum over the answer's tokens
    return -per_seq.sum() / n_seq                          # 1/G * sum over answers
```

The KL penalty sits in the loss, as in DeepSeekMath, not in the reward. Averaging is **sequence-level**: average
over each answer's tokens, then over answers (DeepSeekMath's 1/G Σ 1/|o_i| Σ_t). It is *not* the token-level
average of DAPO / Dr. GRPO.

**Hyperparameters** (`logs/lm_grpo_config.json`):

| | |
|---|---|
| G (answers per question) | 8 |
| questions per step | 8 (64 answers per step) |
| minibatches per step | 2 (one AdamW step each) → 400 updates in 200 steps |
| clip ε | 0.2 |
| KL β | 0.04 (k3 estimator, reference = frozen base model) |
| learning rate | 2e-5, constant, AdamW (β₁ 0.9, β₂ 0.999, weight decay 0), gradient-norm clip 1.0 |
| LoRA | rank 16, scale 2.0 (α/r with α = 32), dropout 0, on q,k,v,o,gate,up,down of all 24 blocks: 8.80 M params (1.8 %) |
| sampling (training) | temperature 1.0, no top-p/top-k, max 512 new tokens |
| evaluation | 500 held-out problems at steps 0, 50, 100, 150, 200: greedy + 8 samples at T = 0.8, max 512 tokens |
| seeds | global 0; sampling seed for step s = s; eval seed 1234 |
| precision | base weights bf16; LoRA weights and optimizer fp32; log-probs computed in fp32 |

Why lr 2e-5 with LoRA: full-fine-tuning GRPO on small models typically uses ~1e-6–5e-6, and LoRA wants roughly 10×
the full-FT rate ("LoRA Without Regret", Schulman et al., Thinking Machines 2025, which also found LoRA matches full
fine-tuning for RL). It was not tuned (only a 3-step pilot, `logs/lm_pilot*`).

## Exact commands

```bash
cd ~/videos/rl-series/research/experiments
uv venv --python 3.12 .venv-lm
uv pip install --python .venv-lm/bin/python mlx==0.32.4 mlx-lm==0.32.2 huggingface_hub==2.3.0 pyarrow==26.0.0
# data, into its own directory
.venv-lm/bin/python -I -c "from huggingface_hub import hf_hub_download as d; [d('openai/gsm8k', f'main/{s}-00000-of-00001.parquet', repo_type='dataset', local_dir='data/gsm8k') for s in ('train', 'test')]"
cd lm
../.venv-lm/bin/python -I select_model.py qwen2.5-0.5b qwen3-0.6b qwen2.5-1.5b   # model choice, ~2 min
./run_resume.sh --run lm_grpo            # THE RUN: 200 steps + 5 evals, ~2 h; resumes from lm/ckpt/lm_grpo
../.venv-lm/bin/python -I examples.py lm_grpo            # verbatim greedy answers, base vs trained
../.venv-lm/bin/python -I greedy_batch_check.py lm_grpo  # greedy accuracy vs decode batch size
../.venv-lm/bin/python -I dpo.py                         # episode 22 extra, ~10 min
./run_resume.sh --run lm_hack_length --reward length --steps 60 --eval-every 60 --eval-samples 0   # episode 25, ~26 min
../.venv-lm/bin/python -I summarize.py                   # -> ../lm_rl.json
```

Python 3.12.14, mlx 0.32.4, mlx-lm 0.32.2. The checkpoint directory holds only the final adapters. To repeat a run
from scratch, delete `lm/ckpt/<run>` or pick a new `--run` name. Step 0 of two different runs produced
bit-identical samples (64/64), so sampling is reproducible on this machine and these versions.

## Results

### Held-out evaluation (500 problems; `evals` in the JSON)

| step | greedy | pass@1 | pass@2 | pass@4 | pass@8 | greedy length | boxed (greedy) | boxed (samples) |
|---|---|---|---|---|---|---|---|---|
| 0 | 46.2 | 38.3 | 51.7 | 63.0 | 71.6 | 309 | 94.8 | 87.2 |
| 50 | 49.8 | 44.1 | 56.0 | 66.6 | 75.0 | 287 | 96.6 | 95.9 |
| 100 | 48.2 | 44.9 | 57.2 | 68.0 | 77.2 | 314 | 96.0 | 93.0 |
| 150 | 49.6 | 45.3 | 56.7 | 66.7 | 75.2 | 304 | 95.2 | 93.7 |
| 200 | 50.4 | 46.2 | 58.1 | 68.4 | 76.4 | 305 | 95.8 | 94.9 |

- Most of the pass@1 gain arrives in the first 50 steps (+5.8), then it creeps up (+2.1 over the next 150).
- **pass@k vs pass@1 (episode 24):** pass@1 rose 7.9 points, pass@8 only 4.8, so the gap between "right in one try" and
  "right in at least one of eight" narrowed from 33.3 to 30.2 points. RL mostly made the model's *first* try look like
  its best try. The problems it could solve somewhere in 8 samples grew less. This is the direction Yue et al. (2025)
  report at large k; at k ≤ 8 we still see pass@8 rise, not fall.
- Greedy flips between step 0 and step 200 (eval log): 60 problems wrong → right, 39 right → wrong, 192 right both
  times, 209 wrong both times.

### Training curves (`curves`, all 200 steps)

Mean training reward (= accuracy on the 64 fresh training answers) in 25-step blocks:
0.476, 0.476, 0.504, 0.496, 0.510, 0.542, 0.546, 0.570. Single steps are noisy (8 questions of varying difficulty),
so plot `reward_ma10`. KL to the reference rises to ~0.02 nats/token within 10 steps and stays there (0.016 over the
last 10 steps). Mean answer length is flat (308 → 304 training tokens; 309 → 305 greedy). About a third of groups
(31 % early, 35 % late) had all 8 answers equally right or wrong, so their advantages are 0 and they teach nothing
except through the KL term. Clipping touched 0.05 % of tokens on average (about 0.1 % in the second minibatch).

**Format vs maths** (all 1,600 training answers of steps 0–24 vs 175–199): strict reward 47.6 → 57.0 %, format-free
"last number" accuracy 48.1 → 56.2 %, boxed 91.6 → 94.8 %. Nearly all of the gain is getting the number right, not
learning to box it. (Different questions early and late, so this is indicative.)

### One update's ratio and clip statistics (`update`, step 100)

- Minibatch 1 (32 answers, 11,690 answer tokens): ratio π_θ/π_old = 1.000 for every token, nothing clipped.
- Minibatch 2 (9,703 answer tokens, after one AdamW step): ratio mean 1.000, min 0.68, max 1.54. 0.25 % of tokens are
  outside [0.8, 1.2], and clipping actually changed the objective for 0.05 % of tokens (ratio > 1.2 with A > 0, or
  < 0.8 with A < 0). Histogram, bin edges 0 | 0.5 | 0.8 | 0.9 | 0.95 | 0.99 | 1.01 | 1.05 | 1.1 | 1.2 | 1.5 | ∞,
  counts 0, 11, 81, 172, 391, **8,463**, 335, 127, 110, 12, 1. KL 0.018, gradient norm 0.21.

### Groups for episode 23 (`groups`, verbatim and truncated)

- **Early (step 4, train-3690)**, "Sandy's monthly phone bill expense is equal to ten times her age now…" (answer 340):
  rewards 1 0 0 1 0 1 0 1 → advantages +1.00 / −1.00 exactly (4 of 8 right); answer lengths 145–305 tokens.
- **Late (step 191, train-3476)**, "Hannah has three brothers… twice as old as the sum of her brothers' ages?" (48):
  rewards 1 0 1 1 1 0 1 0 → +0.77 for each right answer, −1.29 for each wrong one (5 of 8 right).

### Before / after answers (`examples`, greedy, verbatim with token counts)

Chosen by rule: the shortest problems where base → trained flips and the flip agrees with the eval log.

1. **test-1028**, "Tom buys a bedroom set for $3000. He sells his old bedroom for $1000… pays 10% a month… How much
   per month?" (200). Base (181 tokens) takes 10 % of the full $3000 → **\boxed{300}** ✗. Trained (203 tokens):
   "10% of the remaining amount… 0.10 × 2000 = 200" → **\boxed{200}** ✓. *The clearest example of real change.*
2. **test-788**, "John… the school will cover half the cost… has $50 and the trip costs $300" (100). Base (252 tokens)
   halves the cost twice and gets **75** ✗. Trained (182 tokens): 300/2 = 150, 150 − 50 = **100** ✓.
3. **test-776**, "a $1000 laptop with a 20% discount" (800). Base (203 tokens) does the maths right but ends "So, you
   have to pay $800" with **no box**, which scores 0. Trained (228 tokens) ends with \boxed{800} ✓. *The honest
   example: some of the change is format.*
4. (honesty) **test-1280**: base right (7), trained wrong (5). Regressions happen (39 of 500).

The example texts were re-decoded at batch 64 (the eval used 256). Per-question correctness agrees with the eval log
for 93.6 % (base) and 93.8 % (trained) of the problems, and all four examples agree.

## Extras

### Episode 22 — DPO on one pair, and a short DPO run (`dpo`)

Pairs came from the base model's own answers: 4 samples (T = 1.0) for 400 training problems the GRPO run never used
(train[2000:2400]). 239 problems had at least one right and one wrong answer, giving 239 (chosen, rejected) pairs.

**Showcase pair.** "On a highway, the total number of vehicles passing is 300. If the number of cars is twice the
number of trucks, find the total number of trucks." (100). Chosen (240 tokens): x + 2x = 300 → **100** ✓.
Rejected (54 tokens): "2x = 300… x = \boxed{150}" ✗. Sequence log-probabilities (nats, summed over the answer):

| policy | log π(chosen) | log π(rejected) | Δchosen | Δrejected | DPO loss β=0.05 | β=0.1 | β=0.5 |
|---|---|---|---|---|---|---|---|
| reference = base (start of DPO) | −41.5 | −28.3 | 0 | 0 | 0.693 | 0.693 | 0.693 |
| GRPO-trained (never saw this pair) | −45.3 | −37.6 | −3.7 | −9.3 | 0.563 | 0.452 | 0.059 |
| DPO-trained (58 steps, β = 0.1) | −48.9 | −72.0 | −7.4 | −43.7 | 0.150 | 0.026 | 0.000 |

(Δ = log π − log π_ref; loss = −log σ(β[Δchosen − Δrejected]); at the start the loss is ln 2 for every β.)
Things to say: the base model finds the short wrong answer *more* likely than the right one (−28 vs −42). GRPO,
trained only on rewards, already pushed the wrong answer down 2.5× harder than the right one.

**Short DPO run:** β = 0.1, lr 1e-5, LoRA as above, 8 pairs/step, 2 epochs = 58 steps, 6 min. Loss 0.693 → 0.484
(end of epoch 1, 75 % of pairs ranked right) → 0.007 (epoch 2, 100 %). On the showcase pair, chosen first rose
(−41.9 → −38.6) while rejected fell (−28.2 → −45.0); by the end both fell (−48.9, −72.0). Averaged over a batch at
the end: Δchosen −15.7, Δrejected −71.0 nats. **Held-out greedy accuracy after DPO: 43.4 %**, vs 47.4 % for the base
and 49.0 % for the GRPO model at the same decode batch size (64). The tiny offline run overfits its 239 pairs and
lowers the likelihood of good answers too. That is a known DPO failure mode, not a fair DPO-vs-GRPO comparison.

### Episode 25 — reward hacking with a length bonus (`reward_hacking.lm_hack_length`)

Flawed reward: `correct + 0.001 × answer tokens` ("longer reasoning is better"). Everything else was identical to the
main run: same questions in the same order, same sampling seeds, same hyperparameters, 60 steps (25 min). Step 0
is bit-identical to the main run, so the two runs are a paired comparison.

| 15-step blocks | 1–15 | 16–30 | 31–45 | 46–60 |
|---|---|---|---|---|
| proxy reward (what the optimiser sees) | 0.747 | 0.775 | 0.817 | 0.855 |
| true accuracy, hacked run | 40.1 % | 41.3 % | 42.6 % | 46.5 % |
| true accuracy, honest run, same questions | 45.3 % | 48.5 % | 45.8 % | 52.3 % |
| mean length, hacked / honest | 346 / 310 | 362 / 300 | 391 / 303 | 390 / 288 |
| cut off at 512 tokens, hacked / honest | 15.8 / 7.0 % | 20.2 / 7.4 % | 26.3 / 5.6 % | 25.9 / 5.4 % |

- The proxy climbs in every block. True accuracy stays 5–7 points *below* the honest run on the same problems.
- **What it exploited:** length, by faking carefulness. Answers containing "re-evaluate / recheck / verify /
  mistake" went from 4.8 % to 14.8 % of samples (honest run: 2.3 %). It also invents hypotheticals ("If we choose
  another price for the Blu-rays: if each Blu-ray costs $24 instead of $18…") until it runs out of tokens. In the
  last 5 steps 104 of 320 answers (33 %) hit the 512-token limit before any boxed answer. Boxed answers: 83 → 76 %.
- **Why it bit so fast:** the group normalisation. In the honest run about a third of groups are all-right or
  all-wrong and carry no signal. With the bonus, *no* group ties (0 % zero-advantage groups), and dividing by the
  tiny std turns a 0.001-per-token difference into a full ±1 advantage.
- Held-out greedy after 60 steps: 48.4 % (base 46.2 %), but answers are 25 % longer (309 → 387 tokens), 20 % of them
  are cut off (base 4 %) and 82 % are boxed (base 95 %). In 60 steps the damage shows in the sampled answers, not yet
  as a greedy-accuracy drop. Longer runs were not measured.
- The "number appears anywhere" flawed reward was not run (one demonstration was enough).

## What to say on screen

- **Ep 20 (the model is the policy):** each answer is ~300 actions (tokens), one reward at the end. The leash: KL to
  the original model ≈ 0.02 nats per token for the whole run.
- **Ep 22:** the trucks pair and the table above. GRPO without any preference data still preferred the right answer
  (DPO loss 0.452 < 0.693 at β = 0.1).
- **Ep 23:** the step-4 group: 4 of 8 right → advantages exactly +1 and −1; no critic network, the baseline is the
  group mean. About 1 group in 3 is all-right or all-wrong and teaches nothing.
- **Ep 24:** pass@1 38 → 46 %, pass@8 72 → 76 %: the first try caught up with the best of eight. **Length did not grow**
  in our run (309 → 305 tokens). R1-Zero-style length growth needs far longer runs and bigger models; don't claim it.
- **Ep 25:** "same run, one line of the reward changed": proxy up every block, true accuracy 5–7 points below the
  honest run, a third of answers padded past the limit, "let's re-check" 6× more common.
- **Ep 26:** "Qwen2.5-0.5B, 200 steps, 12,800 answers to 1,600 problems, 72 minutes of training on this Mac
  (≈ 2 hours with evaluations), ≈ 2,150 tokens per second while sampling, ~25 GB of memory. On 500 problems it had
  never seen, single-try accuracy went from 38 % to 46 %." Show the bedroom-set answer before and after. If a greedy
  number is quoted, say "about 47 → 50 %".

## Caveats

1. **Greedy decoding is not batch-invariant.** The same weights decoded greedily at batch 16/32/64/128/256 give
   48.2/48.0/47.4/46.8/46.2 % (base) and 51.0/50.2/49.0/50.0/50.4 % (trained). Only 86 % of problems keep the same
   correctness across batch sizes (bf16 batched decoding with left padding). The greedy gain (+2.8 averaged; +4.2 at
   batch 256, SE 2.0) is real but small. **pass@1 from 4,000 samples (+7.9 ± 0.9) is the robust headline.**
   Compare models only at identical decode settings; all evals here used batch 256.
2. One seed, one run. No run-to-run variance was measured.
3. **The GPU was shared.** Another job (a narration TTS server) used the GPU during the second half of the run and the
   extras. It caused one Metal out-of-memory crash, during the step-100 evaluation. The run resumed from the step-99
   checkpoint (adapters, optimizer state and per-step seeds restored, so nothing was lost; the eval was redone), and
   the extras retried on OOM. Throughput numbers come from the run itself and may be a few percent lower than on an
   idle machine. Late steps were sometimes slower (update 10–13 s instead of 7 s).
4. Gradient clipping at norm 1.0 was active on 21 % of updates (largest pre-clip norm 77), so it shaped the effective
   step size. KL β = 0.04 kept the policy close (KL ≈ 0.02). Likely both slowed learning after step 50.
5. LoRA, not full fine-tuning, and an untuned learning rate. Sequence-level averaging has a known length bias
   (Dr. GRPO, Liu et al. 2025). It did not visibly change length here.
6. Training sampled at T = 1.0, pass@k at T = 0.8, greedy at T = 0. Eval subset: 500 of 1,319 test problems
   (±2.2 points binomial SE on a single accuracy; paired SEs are given above). Qwen2.5 was trained on a great deal of
   maths data, so GSM8K contamination of the *base* model cannot be ruled out. It affects the starting level, not
   the before/after comparison.
7. The DPO run is tiny (239 pairs, 2 epochs) and overfits. The reward-hacking run is 60 steps only.

## Files

- `lm/common.py` — data split, prompt, checker, sampler (mlx-lm `BatchGenerator`), log-probs, pass@k, evaluation
- `lm/grpo.py` — the GRPO trainer (also `--reward length|anywhere` for episode 25); `lm/run_resume.sh` — crash-resume loop
- `lm/select_model.py`, `lm/examples.py`, `lm/greedy_batch_check.py`, `lm/dpo.py`, `lm/summarize.py`
- `lm/ckpt/lm_grpo/adapters.safetensors` (final LoRA adapters, 35 MB), `lm/ckpt/lm_hack_length/adapters.safetensors`
- `logs/lm_grpo.jsonl` (per step: reward, accuracy, zero-advantage groups, boxed and truncated rates, length, KL,
  clip fraction, loss, sampling / scoring / update seconds, tokens/s, peak memory, per-minibatch ratio stats and
  histogram), `logs/lm_grpo_eval.jsonl` (5 evals with per-question results), `logs/lm_grpo_groups.jsonl` (every
  sampled answer, verbatim, with rewards and advantages), `logs/lm_grpo_greedy_texts.json`,
  `logs/lm_grpo_greedy_batch_check.json`, `logs/lm_dpo.json`, `logs/lm_hack_length*.jsonl`,
  `logs/lm_model_selection.json`, `logs/lm_pilot*` (3-step pilot), `logs/*.log` (console output;
  `lm_grpo_attempt1.log` holds the crash)

## Errata (coordinator review, 2026-10-25)

Recounted from `logs/lm_hack_length_groups.jsonl` and the step logs while reviewing Episode 25:
- Reward-hacking run, group ties: **7 of 480 groups** tied (each one eight wrong answers all cut off at 512 tokens), not 0 %. The honest run had 159 of 480 in the same 60 steps.
- Accuracy gap to the honest run per 15-step block: 5.2, 7.2, 3.2, 5.8 points (5.4 over all 60 steps), so "3–7 points", not "5–7".
- Cut-off answers in the last 5 steps: 104 hit 512 tokens; 88 never boxed an answer, 16 boxed one first ("most", not "all", had no answer).
