# Trial and Reward — reinforcement learning, from slot machines to reasoning models

A video course in 26 short episodes (3.5–4.5 minutes each), narrated and animated. It follows **Rover**, a small robot that learns only from reward: first at a row of slot machines, then in a small grid world with a charger and a puddle, then balancing a pole, playing games against itself — and finally as the policy inside a language model that learns to solve maths problems with GRPO, trained for real on this Mac.

Part of a family of narrated, animated video courses, rendered with [ManimGX](https://pypi.org/project/manimgx/) and voiced by a local narrator ([tutorial-video](https://github.com/vpromise/tutorial-video)):

- **AI, Explained** — 46 episodes on modern AI, from first principles to frontier research · [ai-explained](https://github.com/vpromise/ai-explained)
- **Agents, Animated** — how AI agents work, from one loop to a team · [agents-animated](https://github.com/vpromise/agents-animated)
- **Tokens per Second** — how LLM inference really works · [tokens-per-second](https://github.com/vpromise/tokens-per-second)
- **Trial and Reward** — reinforcement learning, from slot machines to reasoning models *(this repository)*
- **Out of the Noise** — how diffusion models make pictures and video · [out-of-the-noise](https://github.com/vpromise/out-of-the-noise)

This repository holds the **source code** of the course: one scene per episode, the series kit, reference code, research notes and the production tools. The rendered videos are not included.

## Episodes


### Module 1 · Trial and Reward

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 1 | The Reward Loop | Reinforcement learning is learning what to do from the consequences of doing it | 3:35 | [`ep01-reward-loop/scene.py`](../ep01-reward-loop/scene.py) |
| 2 | Explore or Exploit | Always picking the best-looking option can lock you out of the best one | 4:08 | [`ep02-explore-or-exploit/scene.py`](../ep02-explore-or-exploit/scene.py) |
| 3 | Optimism and Luck | Smarter exploration: be optimistic about the uncertain, or sample your beliefs | 4:14 | [`ep03-optimism-and-luck/scene.py`](../ep03-optimism-and-luck/scene.py) |
| 4 | Bandits with Context | The best action can depend on who is asking | 4:23 | [`ep04-bandits-with-context/scene.py`](../ep04-bandits-with-context/scene.py) |

### Module 2 · Planning in a Known World

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 5 | States, Actions, Returns | An MDP describes a world where actions change what happens next | 3:52 | [`ep05-states-actions-returns/scene.py`](../ep05-states-actions-returns/scene.py) |
| 6 | The Bellman Equation | A state's value is the reward now plus the value of where you land | 3:45 | [`ep06-bellman-equation/scene.py`](../ep06-bellman-equation/scene.py) |
| 7 | Value Iteration | Repeating the Bellman backup until nothing changes solves the world | 3:37 | [`ep07-value-iteration/scene.py`](../ep07-value-iteration/scene.py) |

### Module 3 · Learning from Experience

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 8 | Learning by Playing | Without a model of the world, average what actually happened | 3:41 | [`ep08-learning-by-playing/scene.py`](../ep08-learning-by-playing/scene.py) |
| 9 | Learning from a Guess | Update a guess from the next guess, without waiting for the end | 3:57 | [`ep09-learning-from-a-guess/scene.py`](../ep09-learning-from-a-guess/scene.py) |
| 10 | Q-Learning and SARSA | Learn the values of actions, on-policy or off-policy | 4:09 | [`ep10-q-learning-and-sarsa/scene.py`](../ep10-q-learning-and-sarsa/scene.py) |
| 11 | When Rewards Are Rare | Sparse rewards make credit assignment and exploration hard | 4:25 | [`ep11-when-rewards-are-rare/scene.py`](../ep11-when-rewards-are-rare/scene.py) |

### Module 4 · Deep Value Learning

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 12 | From Tables to Networks | When states are too many to list, a function must generalise | 3:48 | [`ep12-tables-to-networks/scene.py`](../ep12-tables-to-networks/scene.py) |
| 13 | Deep Q-Networks | A neural network can learn Q-values if you stabilise the targets | 4:10 | [`ep13-deep-q-networks/scene.py`](../ep13-deep-q-networks/scene.py) |
| 14 | Fixing Overconfidence | Max over noisy estimates overestimates; several fixes stack | 4:27 | [`ep14-fixing-overconfidence/scene.py`](../ep14-fixing-overconfidence/scene.py) |
| 15 | Search and Self-Play | Planning with a learned evaluation beats either alone | 4:13 | [`ep15-search-and-self-play/scene.py`](../ep15-search-and-self-play/scene.py) |

### Module 5 · Learning the Policy

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 16 | Policy Gradients | Change the policy directly in the direction that makes good actions likelier | 3:59 | [`ep16-policy-gradients/scene.py`](../ep16-policy-gradients/scene.py) |
| 17 | Baselines and Critics | Subtracting a baseline keeps the signal and cuts the noise | 4:11 | [`ep17-baselines-and-critics/scene.py`](../ep17-baselines-and-critics/scene.py) |
| 18 | Proximal Policy Optimization | Reuse each batch several times without stepping too far | 4:26 | [`ep18-ppo/scene.py`](../ep18-ppo/scene.py) |
| 19 | Continuous Control | Robots need actions that are numbers, not choices | 4:17 | [`ep19-continuous-control/scene.py`](../ep19-continuous-control/scene.py) |

### Module 6 · Teaching Language Models with Reward

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 20 | The Model Is the Policy | A language model is a policy; a completion is an episode | 4:05 | [`ep20-model-is-the-policy/scene.py`](../ep20-model-is-the-policy/scene.py) |
| 21 | Learning from Human Preferences | People compare answers; a reward model learns their taste | 4:19 | [`ep21-human-preferences/scene.py`](../ep21-human-preferences/scene.py) |
| 22 | Direct Preference Optimization | The preference objective can be optimised directly, without a reward model | 4:31 | [`ep22-dpo/scene.py`](../ep22-dpo/scene.py) |
| 23 | GRPO | Compare a group of answers to each other instead of training a critic | 4:03 | [`ep23-grpo/scene.py`](../ep23-grpo/scene.py) |
| 24 | Rewards You Can Check | Correct answers and passing tests are rewards no one can argue with | 4:20 | [`ep24-rewards-you-can-check/scene.py`](../ep24-rewards-you-can-check/scene.py) |
| 25 | Reward Hacking | An optimiser finds every gap between the reward and what you meant | 4:20 | [`ep25-reward-hacking/scene.py`](../ep25-reward-hacking/scene.py) |
| 26 | GRPO on This Mac | The whole loop, trained for real, and what every piece did | 4:19 | [`ep26-grpo-on-this-mac/scene.py`](../ep26-grpo-on-this-mac/scene.py) |

## Repository layout

| Path | What |
|---|---|
| `COURSE.md` | the full syllabus: key concepts, what each episode animates, the running example |
| `kit.py` | the series kit: palette, fonts, components, intro and outro cards |
| `rl_reference.py` | canonical algorithms and environments, seeded and self-checked |
| `epNN-*/scene.py` | one narrated scene per episode (`self.say("…[cue]…", animation)`: the narration drives the timing) |
| `assets/` | icons and the backdrop |
| `lexicon.json` | pronunciation overrides for the narrator |
| `research/` | research notes (sources, dated facts) and the measurement or experiment scripts |
| `briefs/` | the production briefs each episode was made from, and the QA tools (`check.sh`, `make_index.py`) |
| `LISTEN.md` | narration lines flagged for a human listen |

## Rendering an episode

Requirements: macOS on Apple silicon (the narrator runs Chatterbox on MPS and Whisper on MLX), ManimGX (`uv tool install --python 3.13 manimgx`) and the [tutorial-video](https://github.com/vpromise/tutorial-video) toolkit (set it up with its `scripts/setup_tts.sh`).

```sh
cd epNN-*/
VOICE=draft manimgx inspect scene.py      # silent draft: layout check + storyboards
# in the course folder, start the narrator (it reads lexicon.json):
#   <tutorial-video>/scripts/narrator.sh
manimgx render scene.py                    # MP4 + SRT captions (speaks new lines once, then caches them)
manimgx present scene.py --no-open         # HTML slides
```

## Not in this repository

Rendered MP4s, captions, HTML slides, the narration cache (`voice/`), storyboards and logs.

Experiment summaries are included (`research/experiments/*.md`, `*.json`). Trained weights, the language-model LoRA checkpoints, training logs and the GSM8K data are not — regenerate them with `research/experiments/scripts/` and `research/experiments/lm/` (MLX on Apple silicon). Scenes that show trained agents or the language-model runs read those files.
