# Trial and Reward — reinforcement learning, from slot machines to reasoning models

A video course in 26 short episodes (3.5–4.5 minutes each), narrated and animated. It follows **Rover**, a small
robot that learns only from reward: first at a row of slot machines, then in a small grid world with a charger and
a puddle, then balancing a pole, playing games against itself — and finally as the policy inside a language model
that learns to solve maths problems with GRPO, trained for real on this Mac.

- **One idea, six settings:** an agent acts, the world answers with a reward, and the agent changes so that more
  reward follows. Every episode is a new answer to the same two questions: *which action was good?* (credit) and
  *what haven't I tried?* (exploration).
- **Rover, the running example:** Rover's five slot machines (Module 1); Rover's world, the classic 4×3 grid with a
  charger (+1), a puddle (−1), a rock and slippery wheels (Modules 2–3); the cliff walk; CartPole and tic-tac-toe
  (Modules 4–5); a small open language model learning grade-school maths (Module 6).
- **Computed and measured, not just quoted:** every algorithm the course shows is implemented in `rl_reference.py`
  (numpy, seeded, self-checked against textbook values), and the training curves — bandit testbeds, the cliff walk,
  DQN / REINFORCE / PPO on CartPole, and a GRPO run on a small language model — were run on this Mac Studio (Apple M4
  Max, 64 GB) (`research/experiments/`). Every other number is cited (`research/notes.md`).
- **Who it's for:** developers who know what a neural network and a gradient are. "AI, Explained" introduced RL in
  one episode (8), RLHF and DPO (30), reasoning models (43) and reward hacking (46); "Agents, Animated" episode 30
  trained agents with RL; "Tokens per Second" explains why rollouts are expensive. This course is the full story.
- **Visual language:** a deep plum stage; colours fixed across the course —
  reward GOLD · values (V, Q, returns) BLUE · policy and actions CORAL · states and the environment TEAL ·
  exploration and uncertainty VIOLET · the learning signal (TD error, advantage, gradient) PINK · good GREEN · bad RED.
- **Format:** 1080p60 MP4 with captions (.srt) and HTML slides; series kit in `kit.py`.

Status: ✅ produced · 🎬 in production · ○ planned

## Module 1 · Trial and Reward — gold

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 1 | **The Reward Loop** | Reinforcement learning is learning what to do from the consequences of doing it | agent, environment, state, action, reward, policy; RL vs supervised learning; delayed reward; a short history (TD-Gammon → DQN → AlphaGo → RLHF → reasoning models) | the agent–environment loop turning; a reward arriving late and travelling back to the actions that earned it | Rover meets its first slot machine; the course's map of six worlds | ✅ `ep01-reward-loop` |
| 2 | **Explore or Exploit** | Always picking the best-looking option can lock you out of the best one | multi-armed bandit, action values, sample averages, incremental update, greedy vs ε-greedy, regret | five machines, estimate bars growing with each pull; a greedy Rover stuck on a mediocre machine; regret piling up | Rover's five slot machines; the 10-armed testbed run 2,000 times on this Mac | ✅ `ep02-explore-or-exploit` |
| 3 | **Optimism and Luck** | Smarter exploration: be optimistic about the uncertain, or sample your beliefs | UCB1 confidence bounds, Thompson sampling, Beta posteriors, logarithmic regret | confidence bars shrinking around each estimate; Beta curves sharpening; regret curves of ε-greedy vs UCB vs Thompson | the same five machines; regret measured over 1,000 pulls | ✅ `ep03-optimism-and-luck` |
| 4 | **Bandits with Context** | The best action can depend on who is asking | contextual bandits, linear payoff models, LinUCB, A/B tests vs bandits, offline evaluation | a context vector entering, per-arm lines, the chosen arm changing with the context | Rover recommending a study tip to different students; Yahoo! front-page news (Li et al., 2010) | ✅ `ep04-bandits-with-context` |

## Module 2 · Planning in a Known World — teal

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 5 | **States, Actions, Returns** | An MDP describes a world where actions change what happens next | Markov decision process, Markov property, transition probabilities, episodes, return, discount γ | Rover's grid: an intended move splitting 80/10/10; a return summed with shrinking weights | Rover's world (4×3, charger +1, puddle −1, −0.04 per step) | ✅ `ep05-states-actions-returns` |
| 6 | **The Bellman Equation** | A state's value is the reward now plus the value of where you land | V and Q, policy evaluation, Bellman expectation and optimality equations, backup diagrams | one cell's value assembled from its neighbours' values, weighted by the slip probabilities | Rover's world: the value of one cell, by hand | ✅ `ep06-bellman-equation` |
| 7 | **Value Iteration** | Repeating the Bellman backup until nothing changes solves the world | value iteration, convergence, policy extraction, policy iteration | value colour spreading out from the charger sweep by sweep; arrows snapping into the optimal policy | Rover's world solved: the textbook's values to three decimals | ✅ `ep07-value-iteration` |

## Module 3 · Learning from Experience — blue

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 8 | **Learning by Playing** | Without a model of the world, average what actually happened | model-free learning, Monte Carlo prediction, first-visit averaging, exploring starts, ε-soft policies | episodes played out, returns written back along each path, averages settling | Rover's world without the map: values learned from played episodes | ✅ `ep08-learning-by-playing` |
| 9 | **Learning from a Guess** | Update a guess from the next guess, without waiting for the end | temporal-difference learning, TD(0), bootstrapping, TD error, bias vs variance, n-step returns | the TD error as a pink spark between two neighbouring estimates; MC vs TD learning curves | the five-state random walk (Sutton & Barto 6.2), measured here | ✅ `ep09-learning-from-a-guess` |
| 10 | **Q-Learning and SARSA** | Learn the values of actions, on-policy or off-policy | Q-learning, SARSA, on- vs off-policy, ε-greedy behaviour, convergence | two Rovers on the cliff: one hugs the edge and sometimes falls, one takes the safe road | the cliff walk: rewards per episode measured here (≈ −25 vs ≈ −50) | ✅ `ep10-q-learning-and-sarsa` |
| 11 | **When Rewards Are Rare** | Sparse rewards make credit assignment and exploration hard | sparse vs dense reward, reward shaping (potential-based), eligibility traces, count-based and curiosity bonuses | a long corridor with one reward at the end; shaping arrows; a novelty glow fading where Rover has been | Rover in a bigger maze; Montezuma's Revenge (cited) | ✅ `ep11-when-rewards-are-rare` |

## Module 4 · Deep Value Learning — violet

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 12 | **From Tables to Networks** | When states are too many to list, a function must generalise | function approximation, features, semi-gradient TD, generalisation, the deadly triad | a table dissolving into a smooth value surface; one update moving many states at once | CartPole: four numbers per state, no table can hold them | ✅ `ep12-tables-to-networks` |
| 13 | **Deep Q-Networks** | A neural network can learn Q-values if you stabilise the targets | DQN, experience replay, target networks, ε decay, Atari results | a replay buffer filling and being sampled; a frozen target net; CartPole's score climbing | DQN learning CartPole on this Mac; Mnih et al. 2015 (49 Atari games) | ✅ `ep13-deep-q-networks` |
| 14 | **Fixing Overconfidence** | Max over noisy estimates overestimates; several fixes stack | overestimation bias, Double DQN, dueling networks, prioritised replay, Rainbow | noisy bars and the max picking the luckiest; two networks splitting choose and evaluate | the overestimation measured on a toy problem; Rainbow's ablations (cited) | ✅ `ep14-fixing-overconfidence` |
| 15 | **Search and Self-Play** | Planning with a learned evaluation beats either alone | Monte Carlo tree search, UCT, self-play, AlphaGo, AlphaZero, MuZero | a search tree growing, visit counts and values; two copies of the same player | tic-tac-toe searched live; AlphaGo → AlphaZero → MuZero (cited) | ✅ `ep15-search-and-self-play` |

## Module 5 · Learning the Policy — coral

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 16 | **Policy Gradients** | Change the policy directly in the direction that makes good actions likelier | stochastic policies, the objective J(θ), the log-derivative trick, REINFORCE | action probabilities nudged up after a good episode and down after a bad one | REINFORCE learning CartPole on this Mac | ✅ `ep16-policy-gradients` |
| 17 | **Baselines and Critics** | Subtracting a baseline keeps the signal and cuts the noise | variance, baselines, advantage, actor-critic, A2C, generalised advantage estimation | noisy returns vs advantages around zero; an actor and a critic trading signals | REINFORCE vs actor-critic on CartPole, measured | ✅ `ep17-baselines-and-critics` |
| 18 | **Proximal Policy Optimization** | Reuse each batch several times without stepping too far | importance ratios, trust regions (TRPO), the PPO clipped objective, minibatch epochs | the ratio moving inside a clip band; the objective going flat outside it | PPO on CartPole measured; why PPO became the default (OpenAI Five, RLHF) | ✅ `ep18-ppo` |
| 19 | **Continuous Control** | Robots need actions that are numbers, not choices | Gaussian policies, deterministic policy gradients, DDPG/TD3, soft actor-critic and entropy, sim-to-real | a Gaussian over torque narrowing; an arm reaching; noise in simulation for robustness | the pendulum swing-up, measured here | ✅ `ep19-continuous-control` |

## Module 6 · Teaching Language Models with Reward — pink

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 20 | **The Model Is the Policy** | A language model is a policy; a completion is an episode | tokens as actions, the prompt and prefix as state, reward at the end, pretraining → SFT → RL, the KL leash to a reference | a completion generated token by token as actions; one reward at the end flowing back to every token | a small model answering a maths question; the training pipeline | ✅ `ep20-model-is-the-policy` |
| 21 | **Learning from Human Preferences** | People compare answers; a reward model learns their taste | preference pairs, Bradley–Terry, reward models, PPO with a KL penalty, InstructGPT | two answers, a human's choice, a reward model's score; four models in memory during PPO | InstructGPT: a 1.3B model preferred over the 175B one (Ouyang et al., 2022) | ✅ `ep21-human-preferences` |
| 22 | **Direct Preference Optimization** | The preference objective can be optimised directly, without a reward model | the KL-regularised objective, the implicit reward, the DPO loss, β, variants | the derivation in four steps; chosen and rejected log-probabilities pulled apart | one preference pair through the DPO loss, computed | ✅ `ep22-dpo` |
| 23 | **GRPO** | Compare a group of answers to each other instead of training a critic | group sampling, group-relative advantages, the clipped ratio, KL to the reference, DeepSeekMath | eight answers to one question, scored, normalised into advantages; the critic network disappearing | one maths question, eight sampled answers, their advantages | ✅ `ep23-grpo` |
| 24 | **Rewards You Can Check** | Correct answers and passing tests are rewards no one can argue with | RL with verifiable rewards, answer checkers, unit tests, DeepSeek-R1-Zero, longer reasoning, pass@k vs pass@1 | a checker stamping answers; response length growing over training; pass@k curves crossing | DeepSeek-R1 (cited); our run's accuracy on held-out problems | ✅ `ep24-rewards-you-can-check` |
| 25 | **Reward Hacking** | An optimiser finds every gap between the reward and what you meant | Goodhart's law, specification gaming, reward-model overoptimisation, length bias, the KL leash | true quality rising then falling as the proxy reward keeps climbing | the boat that circles for points (cited); overoptimisation curves (Gao et al., 2022) | ✅ `ep25-reward-hacking` |
| 26 | **GRPO on This Mac** | The whole loop, trained for real, and what every piece did | a full GRPO run: sampling, checking, advantages, update; the course revisited | the reward curve of our run; answers before and after; the loop from episode 1 again | a small model learning grade-school maths on this Mac | ✅ `ep26-grpo-on-this-mac` |

## Paths through the course

- **Just the core:** 1 → 2 → 5 → 6 → 9 → 10 → 13 → 16 → 18 → 23.
- **Classic RL:** 1 → 11 and 12 → 15.
- **RL for language models:** 1 → 16 → 17 → 18 → 20 → 26.
- **Bandits for product people:** 1 → 2 → 3 → 4.

## Production notes

- Series kit: `kit.py` (palette, RL components, Rover, cards); component tour: `kit_demo.py`; canonical algorithms
  and numbers: `rl_reference.py`; research: `research/notes.md`; runs on this Mac: `research/experiments/` (load with
  `measured(...)`).
- Episodes are made by subagents from `briefs/epNN.md` (generated by `briefs/make_briefs.py` from `briefs/spec.py`),
  one shared narrator; each episode is reviewed by the coordinator (numbers, storyboards, `briefs/check.sh N`).
