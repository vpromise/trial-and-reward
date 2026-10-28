# Trial and Reward: research notes (factual backbone)

**Fact date: October 2026** (compiled 2026-10-25). All web sources were read on 2026-10-25.
Fast-moving facts (model releases, benchmark and leaderboard numbers, library defaults) are tagged **[DATED]**. Anything not confirmed against a primary source is tagged **[UNVERIFIED]**. My own derivations are tagged **[DERIVED]**, with the arithmetic shown; every [DERIVED] number was re-computed with a short Python script.

**Conventions**
- Book references are to Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd edition (MIT Press, 2018; free PDF at http://incompleteideas.net/book/RLbook2020.pdf, the 2020 corrected printing). "S&B §6.5" means section 6.5 of that book. Figure and example numbers are the book's.
- Timing follows S&B: in state S_t the agent takes A_t, then receives R_{t+1} and lands in S_{t+1}. (Some papers, e.g. DQN, write r_t for the same reward. The meaning is the same.)
- Percentages of "human-normalised score" are 100 × (agent − random) / (human − random), unless a section says otherwise.
- arXiv IDs are given as arXiv:YYMM.NNNNN; the URL is https://arxiv.org/abs/<id>.
- Notation (one symbol per idea across the course; papers' own symbols are mapped in each section):

| Symbol | Meaning | Notes |
|---|---|---|
| s, S_t | state (S_t: the random state at time t) | in a language model: the prompt plus the tokens so far |
| a, A_t | action | in a language model: the next token |
| r, R_{t+1} | reward received after A_t | sequence-level reward for LMs arrives only at the end |
| p(s′, r \| s, a) | environment dynamics (transition + reward probabilities) | Russell & Norvig write P(s′ \| s, a) and R(s) |
| γ | discount factor, 0 ≤ γ ≤ 1 | γ = 1 only for episodic tasks |
| G_t | return: R_{t+1} + γR_{t+2} + γ²R_{t+3} + … | **overloaded:** GRPO papers use G (no subscript) for the **group size**; we keep their G there |
| π(a \| s), π_θ | policy (probability of a in s), parameterised by θ | π_ref = the frozen reference policy (RLHF, DPO, GRPO); π_θold = the policy that generated the batch |
| V(s), v_π(s), v*(s) | state value: expected return from s (under π; under the optimal policy) | Russell & Norvig call v* the "utility" U(s) |
| Q(s, a), q_π, q* | action value: expected return after taking a in s | in bandits, q*(a) is the true mean reward of arm a; Q_t(a) its estimate |
| A(s, a) | advantage: Q(s, a) − V(s) | Â_t = an estimate of it (GAE, GRPO's group-normalised reward) |
| δ_t | TD error: R_{t+1} + γV(S_{t+1}) − V(S_t) | the PINK learning signal |
| α | step size (learning rate) | **overloaded:** LinUCB's exploration width α; PER's priority exponent α; SAC's entropy temperature α; Gao et al.'s fit coefficient α |
| ε | exploration rate in ε-greedy | **overloaded:** PPO/GRPO clip range ε (0.2) |
| θ | policy parameters | w = value-function weights (S&B); φ(s) = feature vector |
| β | KL-penalty coefficient (RLHF, DPO, GRPO) | **overloaded:** PER's importance-sampling exponent; A3C's entropy weight; Gao et al.'s fit coefficient |
| λ | trace-decay / GAE parameter | 0 = one-step TD, 1 = Monte Carlo |
| ρ | importance-sampling ratio π(a\|s) / b(a\|s), or π_θ / π_θold | PPO writes r_t(θ) for this ratio; we write ρ_t to avoid a clash with reward |
| N_t(a) | number of times arm a was tried before time t | UCB, sample averages |
| c | UCB exploration constant | UCB1 uses c = √2 in the S&B form |
| k | number of arms | the 10-armed testbed has k = 10; Rover's machines k = 5 |
| x, y | prompt, completion (LM setting) | y_w / y_l = preferred ("winner") / dispreferred ("loser") completion |
| r_φ(x, y) | learned reward model | σ = logistic sigmoid; Z(x) = partition function in DPO |
| D_KL | Kullback–Leibler divergence | GRPO estimates it per token with "k3" (§23) |

**Running examples (from COURSE.md)**: Rover's five slot machines (§2–§3) · Rover's world = the Russell & Norvig 4×3 grid: charger +1, puddle −1, rock = the wall, slippery wheels = 0.8/0.1/0.1, −0.04 per step (§5–§8) · the five-state random walk (§9) · the cliff walk (§10) · CartPole (§12–§18) · tic-tac-toe (§15) · the pendulum swing-up (§19) · a small open LM on grade-school maths (GSM8K) with GRPO (§20–§26).

Five numbers carry the course: **0.705** (the value of Rover's start cell), **−25 vs −50** (SARSA vs Q-learning on the cliff), **29 of 49** (Atari games where DQN reached 75% of the human tester), **1.3B > 175B** (InstructGPT), **15.6% → 71.0%** (R1-Zero on AIME 2024, Jan 2025 preprint).

[DERIVED] numbers were re-computed with short scripts kept in the research scratch folder (/Users/vpromisex/.claude/jobs/61935fea/tmp/rl-research/scripts/; not part of the course repo). The course's canonical implementations are in `rl_reference.py`, and its measured runs are in `research/experiments/`. Where a worked example here and a measured run disagree, the measured run wins on screen and this file explains the textbook value.

---

## §0 Cheat sheet (the numbers and formulas worth memorising)

| # | Fact | Value | Source | § |
|---|---|---|---|---|
| 1 | Return and Bellman optimality | G_t = R_{t+1} + γG_{t+1}; V*(s) = max_a Σ p(s′,r\|s,a)[r + γV*(s′)] | S&B eqs. 3.9, 3.19 | 5–6 |
| 2 | The update rule behind everything | new = old + step × (target − old); bandit: Q ← Q + (1/n)(R − Q) | S&B eqs. 2.3–2.4 | 2 |
| 3 | 10-armed testbed | 2,000 tasks, q*(a) ~ N(0,1), rewards N(q*,1), 1,000 steps; best possible ≈ 1.54; greedy ≈ 1 and finds the best arm in ~⅓ of tasks; ε = 0.1 is capped at 91% optimal | S&B §2.3, Fig. 2.2 | 2 |
| 4 | UCB1 and its guarantee | pick argmax Q(a) + √(2 ln t / N(a)); expected regret ≤ 8 Σ ln n/Δ_a + (1 + π²/3) Σ Δ_a (rewards in [0,1]); no method beats log T (Lai & Robbins 1985) | Auer, Cesa-Bianchi & Fischer 2002, Fig. 1, Thm 1 | 2–3 |
| 5 | Thompson sampling, Bernoulli arms | sample θ_a ~ Beta(1 + wins, 1 + losses), pull the largest; idea from 1933, log-regret proof 2012 | Thompson 1933; Agrawal & Goyal 2012 | 3 |
| 6 | LinUCB on Yahoo! Today Module | **12.5% click lift** over context-free ε-greedy, offline replay on "over 33 million events" (disjoint model, deployment bucket, 1.795 vs 1.596 relative CTR) | Li, Chu, Langford & Schapire 2010, abstract, Table 1 | 4 |
| 7 | Rover's world (R&N 4×3, 0.8/0.1/0.1, R = −0.04, γ = 1) | top 0.812 0.868 0.918 **+1** · middle 0.762 rock 0.660 **−1** · bottom **0.705** 0.655 0.611 0.388; at (3,1) take the long way (Left 0.651 vs Up 0.632) | R&N 3rd ed. Figs. 17.2–17.3; recomputed exactly [DERIVED] | 5–7 |
| 8 | TD(0) | δ = r + γV(s′) − V(s); V(s) ← V(s) + αδ; "a guess from a guess" | Sutton 1988; S&B eqs. 6.2, 6.5 | 9 |
| 9 | Random walk (S&B Ex. 6.2) | true values 1/6 … 5/6; every curve starts at RMS √(1/18) = 0.236; TD α = .05/.1/.15 beats MC α = .01–.04 | S&B p. 125 | 9 |
| 10 | Q-learning vs SARSA, cliff walk (4×12, −1/step, −100 cliff, ε = 0.1) | online reward per episode: SARSA ≈ **−25**, Q-learning ≈ **−50** (our re-run at α = 0.5: −27.3 vs −50.3); optimal path 13 steps = −13 | S&B Ex. 6.6; [DERIVED] | 10 |
| 11 | Deadly triad | function approximation + bootstrapping + off-policy ⇒ possible divergence; "If any two … are present, but not all three, then instability can be avoided"; Baird's example diverges for any α > 0 | S&B §11.2–11.3 | 12 |
| 12 | DQN (Nature 2015) | replay 1,000,000; minibatch 32; target copy every C = 10,000 updates; ε 1 → 0.1 over 1M frames; 50M frames; γ = 0.99; ≥ 75% of the human tester on **29 of 49** games; median 93.5% | Mnih et al. 2015, Ext. Data Tables 1–2; van Hasselt et al. 2016 | 13 |
| 13 | Maximization bias | E[max of 10 unbiased N(0,1) estimates] = **1.539** [DERIVED]; Double DQN target r + γQ(s′, argmax_a Q(s′,a; θ); θ⁻); Rainbow median **223%** (no-ops) | S&B §6.7; van Hasselt et al. 2016; Hessel et al. 2018 | 14 |
| 14 | Montezuma's Revenge ladder | DQN **0** → pseudo-counts 3,439 → RND **8,152** (first above average human) → Go-Explore **43,791** (no domain knowledge). Human: 4,367 (DQN paper) or 4,753 (later papers) | Mnih 2015; Bellemare 2016; Burda 2018; Ecoffet 2021 | 11 |
| 15 | Go and chess | AlphaGo beat Fan Hui 5–0 (Oct 2015) and Lee Sedol 4–1 (Mar 2016); AlphaGo Zero 100–0 vs AlphaGo Lee after 3 days, 4.9M self-play games; AlphaZero vs Stockfish: 155 wins, 6 losses, 839 draws in 1,000 games | Silver et al. 2016, 2017, 2018 | 15 |
| 16 | Tic-tac-toe | 5,478 legal positions (765 up to symmetry); 255,168 complete games; perfect play is a draw [DERIVED] | own enumeration | 15 |
| 17 | Policy gradient (REINFORCE) | ∇J = E[Σ_t G_t ∇ln π_θ(a_t\|s_t)]; softmax score ∇ln π = one-hot(a) − π; a state-only baseline adds no bias | Williams 1992; Sutton et al. 2000; S&B ch. 13 | 16–17 |
| 18 | GAE | Â_t = Σ_l (γλ)^l δ_{t+l}; λ = 0 → TD error, λ = 1 → MC return − V; usual γ = 0.99, λ = 0.95 | Schulman et al. 2016; PPO Table 3 | 17 |
| 19 | PPO clip | L = E[min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t)], ρ_t = π_θ/π_θold; ε = 0.2 scored 0.82 vs 0.76 (ε = 0.1), 0.70 (ε = 0.3), −0.39 (no clip) | Schulman et al. 2017, Eq. 7, Table 1 | 18 |
| 20 | The KL-regularised objective and its optimum | max E[r] − β KL(π‖π_ref) ⇒ π*(y\|x) = π_ref(y\|x) e^{r(x,y)/β} / Z(x) | Ziegler et al. 2019; Rafailov et al. 2023 eqs. 3–4 | 20, 22 |
| 21 | Bradley–Terry and InstructGPT | P(y_w ≻ y_l) = σ(r_w − r_l); InstructGPT 1.3B preferred to 175B GPT-3; RM 6B; ~40 labelers; prompts SFT 12,725 / RM 33,207 / PPO 31,144; β = 0.02; RL used 60 of GPT-3's 3,640 petaflop/s-days (1.65%) | Bradley & Terry 1952; Ouyang et al. 2022 | 21 |
| 22 | DPO loss | −log σ(β[log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)]); default β = 0.1 (0.5 on TL;DR); implicit reward β log(π_θ/π_ref) | Rafailov et al. 2023, eq. 7, App. B | 22 |
| 23 | GRPO | Â_i = (r_i − mean(r))/std(r) over a group of G answers, the same for every token; no critic; KL via k3 = r − 1 − ln r with r = π_ref/π_θ. DeepSeekMath: G = 64, β = 0.04, lr 1e-6; GSM8K 82.9 → 88.2% | Shao et al. 2024 eqs. 3–4, §4.2, Table 5; Schulman 2020 | 23 |
| 24 | DeepSeek-R1-Zero | AIME 2024 pass@1 **15.6% → 71.0%** (arXiv v1, Jan 2025; Nature/v2: 77.9%); cons@64 86.7%; rule rewards = accuracy + format; G = 16, β = 0.001 | DeepSeek-AI 2025 | 24 |
| 25 | GSM8K | 8.5K grade-school problems, 2–8 steps; files: **7,473 train / 1,319 test**; final answer after `####` | Cobbe et al. 2021; GitHub repo | 24 |
| 26 | Overoptimisation and Goodhart | with d = √KL: best-of-n gold R = d(a − b·d), RL gold R = d(a − b·ln d); "When a measure becomes a target, it ceases to be a good measure" is **Strathern 1997**, not Goodhart | Gao, Schulman & Hilton 2022; Strathern 1997 | 25 |
| 27 | Where RL time goes (LLMs) | rollout generation "often accounts for over 90%" of RLHF/RLVR runtime at scale; on this Mac, with a 0.5B model and short answers, batching rollouts matters most [DERIVED] | Hu et al. (OpenRLHF); §26 worked example | 26 |

---

## §1 What reinforcement learning is: the reward loop, and a dated timeline

### Mechanism (plain words)
- **An agent acts, and the world answers with a number.** At each step t the agent sees a state s_t and picks an action a_t. One step later the environment returns a reward r_{t+1} and a new state s_{t+1}. Then the loop repeats.
- **The goal is not the next reward but the sum of all rewards to come.** That sum is the return G. An action can be good because of what it sets up, even if its own reward is zero. This is **delayed reward**, and it creates the **credit-assignment problem**: which of many earlier decisions earned the reward that just arrived?
- **Nobody tells the agent the right action.** It only learns how good the action it actually took turned out to be. To find better actions it must try ones it has not tried yet. This is the **exploration–exploitation trade-off**, and it does not exist in supervised learning.
- **The two questions of the course** are credit ("which action was good?") and exploration ("what haven't I tried?"). S&B name the matching features "trial-and-error search and delayed reward" (§1.1).
- **Four ingredients** (S&B §1.3): a **policy** (how the agent picks actions), a **reward signal** (what is good right now), a **value function** (what is good in the long run), and, optionally, a **model** of the environment (used for planning).
- **The "agent" boundary is a choice.** S&B define the environment as "everything outside the agent". For a language model (§20), the state is the prompt plus the tokens so far, an action is the next token, and the reward usually arrives only at the end of the completion.

### Key formulas (S&B ch. 3 notation → ours)
| S&B | Ours | Meaning |
|---|---|---|
| S_t, A_t, R_{t+1} | s_t, a_t, r_{t+1} | state, action, reward. S&B index the reward that follows a_t as **R_{t+1}**, and the course keeps that convention. |
| S_0, A_0, R_1, S_1, A_1, R_2, … | same | the trajectory (S&B eq. 3.1) |
| G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + … | G_t = Σ_{k≥0} γ^k r_{t+k+1} | the return (S&B eq. 3.8), 0 ≤ γ ≤ 1 |
| π(a\|s) | π(a\|s) | the policy: the probability of choosing a in s |
| J(θ) = v_{π_θ}(s_0) | J(θ) = E_{π_θ}[G_0] | the objective (S&B eq. 13.4, episodic case) |

- The **reward hypothesis** (S&B §3.2, attributed to Michael Littman): "That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)."

### RL vs supervised vs unsupervised (S&B §1.1)
| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Feedback | the correct label for each input | none | a scalar reward for the action actually taken |
| Says what *would* have been right? | yes | n/a | no: it is evaluative, not instructive |
| Data | a fixed dataset, i.i.d. | a fixed dataset | generated by the agent's own behaviour, so the policy changes the data |
| Timing | immediate | n/a | often delayed by many steps |
| Exploration needed? | no | no | yes |

- S&B: supervised learning is "learning from a training set of labeled examples provided by a knowledgable external supervisor". S&B also say RL is not unsupervised learning: "reinforcement learning is trying to maximize a reward signal instead of trying to find hidden structure."

### Timeline (dates checked against the primary source; tags as noted)
| Date | Event | Headline fact | Source |
|---|---|---|---|
| 1911 | Thorndike's **Law of Effect** | Responses "accompanied or closely followed by satisfaction … will be more likely to recur" | Thorndike 1911, p. 244, quoted in S&B §1.7 |
| 1957 | Bellman, *Dynamic Programming*; "A Markov decision process" | The Bellman equation, and MDPs | Bellman 1957a,b (S&B refs) |
| 1959 (Jul) | **Samuel's checkers player** | The first learning method with temporal-difference ideas (S&B §1.7). His first program was for the IBM 701 in 1952, the first *learning* program was completed in 1955 and shown on TV in 1956 (S&B §16.2). | Samuel 1959, IBM J. R&D 3(3):210–229 |
| 1960 | Howard, *Dynamic Programming and Markov Processes* | Policy iteration | Howard 1960 |
| 1961 (Jan) | Minsky, "Steps Toward Artificial Intelligence" | Names the "credit-assignment problem" | Proc. IRE 49; S&B §1.7 |
| 1983 (Sep) | Barto, Sutton & Anderson, "Neuronlike adaptive elements…" | Actor–critic learns **pole balancing**, the ancestor of CartPole | IEEE Trans. SMC-13:834–846 |
| 1988 (Aug) | Sutton, "Learning to predict by the methods of temporal differences" | TD(λ) | Machine Learning 3:9–44 |
| 1989 | Watkins, PhD thesis *Learning from Delayed Rewards* (Cambridge) | **Q-learning**. S&B: TD and optimal control "were fully brought together in 1989 with Chris Watkins's development of Q-learning" | S&B §1.7 |
| 1992 (May) | Watkins & Dayan, "Q-learning" (convergence); Williams, REINFORCE; Tesauro, "Practical issues in TD learning" | **TD-Gammon** first published | Machine Learning 8 (1992): 279–292, 229–256, 257–277 |
| 1995 (Mar) | Tesauro, "Temporal difference learning and TD-Gammon" | Plays "near the level of the world's strongest grandmasters" (S&B §16.1) | CACM 38(3):58–68 |
| 1998 / 2018 | Sutton & Barto, *Reinforcement Learning: An Introduction*, 1st and 2nd editions | The standard textbook | MIT Press |
| 2006 | Coulom coins "Monte-Carlo tree search"; Kocsis & Szepesvári introduce **UCT** | MCTS begins | Coulom, CG 2006; K&S, ECML 2006 |
| 2013 (19 Dec) | Mnih et al., "Playing Atari with Deep RL" (arXiv 1312.5602) | Beats all previous approaches on 6 of 7 games, and a human expert on 3 | arXiv abstract |
| 2015 (26 Feb) | **DQN in Nature** | "a level comparable to that of a professional human games tester across a set of 49 games" | Nature 518:529–533 |
| 2015 (Oct) | AlphaGo vs Fan Hui | **5–0**, played 5–9 Oct 2015 | Silver et al. 2016 |
| 2016 (28 Jan) | **AlphaGo** in Nature | 99.8% win rate vs other programs | Nature 529:484–489 |
| 2016 (9–15 Mar) | AlphaGo vs Lee Sedol, Seoul | **4–1**; "watched by over 200 million people"; Move 37 in game 2; Lee won game 4 (13 Mar) with his move 78 | DeepMind AlphaGo page; the dates and the game-4 result are from contemporaneous news (China Daily, 15 Mar 2016) and Wikipedia, not a DeepMind primary |
| 2017 (Jan) | AlphaGo Master online | 60–0 against top professionals | AGZ paper |
| 2017 (12 Jun) | Christiano et al., "Deep RL from human preferences" | RLHF begins | arXiv 1706.03741 |
| 2017 (20 Jul) | **PPO** | Clipped surrogate objective | arXiv 1707.06347 |
| 2017 (19 Oct) | **AlphaGo Zero** in Nature | No human data; **100–0** vs AlphaGo Lee after 3 days | Nature 550:354–359 |
| 2017 (5 Dec) | AlphaZero preprint | Chess, shogi and Go "within 24 hours" | arXiv 1712.01815 |
| 2018 (7 Dec) | **AlphaZero** in Science | 155 wins, 6 losses in 1,000 games vs Stockfish | Science 362:1140–1144 |
| 2019 (13 Apr) | **OpenAI Five** beats OG, the Dota 2 world champions | "the first AI system to defeat the world champions at an esports game" | arXiv 1912.06680 abstract |
| 2019 (16 Oct) | Dactyl solves a Rubik's Cube one-handed | Sim-to-real with domain randomization (§19) | arXiv 1910.07113 |
| 2019 (30 Oct) | **AlphaStar** in Nature | "Grandmaster level for all three StarCraft races and above 99.8% of officially ranked human players" | Nature 575:350–354 |
| 2019 (19 Nov) / 2020 (23 Dec) | **MuZero** on arXiv, then in Nature | Learns the model; matches AlphaZero without being given the rules | arXiv 1911.08265; Nature 588:604–609 |
| 2022 (Feb) | GT Sophy (Gran Turismo, 9 Feb) and tokamak plasma control (16 Feb) | Deep RL beyond games | Nature 602 |
| 2022 (4 Mar) | **InstructGPT** | 1.3B InstructGPT preferred to 175B GPT-3 | arXiv 2203.02155 |
| 2022 (5 Oct) | AlphaTensor | RL finds faster matrix-multiplication algorithms | Nature 610:47–53 |
| 2022 (30 Nov) | **ChatGPT** | "We trained this model using Reinforcement Learning from Human Feedback (RLHF)" | OpenAI blog, 30 Nov 2022 |
| 2023 (29 May) | DPO | RLHF objective without a reward model or RL loop | arXiv 2305.18290 |
| 2023 (30 Aug) | Swift drone racing | Champion-level drone racing with deep RL | Nature 620 |
| 2024 (5 Feb) | **GRPO** (DeepSeekMath) | Removes the critic | arXiv 2402.03300 |
| 2024 (Jul) | AlphaProof + AlphaGeometry 2 | IMO 2024 silver-medal standard, 28/42 | DeepMind IMO-2025 blog recap; Nature paper published online 12 Nov 2025 |
| 2024 (12 Sep) | **OpenAI o1** | "trained with reinforcement learning to perform complex reasoning"; AIME 2024 74% pass@1 | OpenAI, "Learning to Reason with LLMs" |
| 2024 (Nov) | Tülu 3 coins "RL with verifiable rewards" (RLVR) | see §24 | arXiv 2411.15124 |
| 2024 (20 Dec) | o3 previewed | ARC-AGI-1 semi-private: 75.7% (high-efficiency), 87.5% (172× compute) | ARC Prize blog, 20 Dec 2024 [DATED] |
| 2025 (20–22 Jan) | **DeepSeek-R1 / R1-Zero** (weights on HF 20 Jan; arXiv 22 Jan) and **Kimi k1.5** (arXiv 22 Jan) | RL alone elicits long reasoning (§24). Kimi: "77.5 on AIME … matching OpenAI's o1" | arXiv 2501.12948, 2501.12599; HF createdAt |
| 2025 (5 Mar) | **2024 ACM Turing Award to Barto & Sutton** | "for developing the conceptual and algorithmic foundations of reinforcement learning"; $1 million | ACM press release |
| 2025 (Apr) | Silver & Sutton, "Welcome to the Era of Experience" | Preprint of a chapter for *Designing an Intelligence* (MIT Press) | DeepMind-hosted PDF (dated by its PDF metadata, 10 Apr 2025) |
| 2025 (16 Apr) | o3 and o4-mini released | "an additional order of magnitude in both training compute and inference-time reasoning" in RL | OpenAI blog [DATED] |
| 2025 (29 Apr) | **Qwen3** | Four-stage post-training, two stages of it RL. The reasoning-RL stage used **3,995** query–verifier pairs with GRPO; Qwen3-235B-A22B's AIME'24 rose **70.1 → 85.1 in 170 RL steps** | Qwen blog; Qwen3 Tech. Report §4.2 (arXiv 2505.09388) |
| 2025 (19–21 Jul) | **IMO 2025 gold-medal level** for two LLMs, both at 35/42 | Google DeepMind's Gemini Deep Think, officially graded, "end-to-end in natural language", "within the 4.5-hour competition time limit". OpenAI's experimental model was graded by former medallists, per Alexander Wei's X thread **[UNVERIFIED primary: X post]** | DeepMind blog, 21 Jul 2025 [DATED] |
| 2025 (17 Sep) | DeepSeek-R1 published in **Nature** | "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning" | Nature 645:633–638 |
| 2025 (12 Nov) | AlphaProof in Nature | "Olympiad-level formal mathematical reasoning with reinforcement learning" | Nature 651:607–613 (print issue 19 Mar 2026) |
| 2026 (15–16 Jul) | IMO 2026, Shanghai | Several AI systems reported perfect scores. Xiaohongshu's dots team says "dots-note-3.0 received the full 7 points on all six problems" (42/42). NVIDIA's open Nemotron 3 Ultra pipeline (SFT + RL) "scored 30 out of 42 points at IMO 2026, reaching the gold-medal threshold". Claims about other labs come from social media **[UNVERIFIED]** | dots team page; arXiv 2609.10712 [DATED] |
| 2026 (8 Oct) | Xiaomi MiMo-V2.6 report | Scaled agentic RL: each step "consumes 1,568 samples and 2.7-3.7B tokens per step at context lengths of up to 1M"; a "multi-layer defense against reward hacking" | arXiv 2610.11959 [DATED] |

### Numbers with sources
- **Samuel 1959, abstract**: the checkers program "can learn to do this in a remarkably short period of time (8 or 10 hours of machine-playing time)".
- **o1, AIME 2024** (OpenAI, 12 Sep 2024): "GPT-4o only solved on average 12% (1.8/15) of problems. o1 averaged 74% (11.1/15) with a single sample per problem, 83% (12.5/15) with consensus among 64 samples, and 93% (13.9/15) when re-ranking 1000 samples with a learned scoring function." [DATED]
- **ChatGPT methods** (OpenAI, 30 Nov 2022): "We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup." The reward models came from AI trainers ranking sampled completions, and "Using these reward models, we can fine-tune the model using Proximal Policy Optimization. We performed several iterations of this process."
- **OpenAI Five** (arXiv 1912.06680): it learned "from batches of approximately 2 million frames every 2 seconds", and the team trained it "for 10 months". Scale numbers are in §18.
- **Turing Award** (ACM, March 2025): Barto and Sutton are recognised "for developing the conceptual and algorithmic foundations of reinforcement learning". The award "carries a $1 million prize with financial support provided by Google, Inc." ACM's description cites RLHF in ChatGPT and AlphaGo as the prominent examples.

### Worked example [DERIVED] (Rover's first slot machine, and a reward that arrives late)
1. **Bandit (one step).** Rover pulls lever a, the machine pays r = 1 or 0, and the episode ends. Then G_0 = r_1. No credit-assignment problem exists, only exploration (§2).
2. **Delayed reward (four steps).** In the grid, Rover makes 3 moves that earn 0 and then reaches the charger for +1, so the rewards are (0, 0, 0, +1).
   - γ = 1: G_0 = 0 + 0 + 0 + 1 = **1**.
   - γ = 0.9: G_0 = 0.9³ × 1 = **0.729**.
   - Every one of the four actions gets credit for that single +1, which is the arrow travelling back in the ep. 1 animation.
   - (scripts/f1_arith.py)
3. **Course map: one loop, six settings.** In each one, the state, action and reward are:
   - **Slot machines:** no state; which lever; payout.
   - **Grid world:** cell; N/E/S/W; +1/−1/−0.04.
   - **Cliff:** cell; move; −1 per step, −100 for the cliff.
   - **CartPole:** 4 numbers; push left or right; +1 per step upright.
   - **Tic-tac-toe:** board; move; win/draw/loss at the end.
   - **Language model:** prompt + tokens so far; next token; checker score at the end.

### Common misconceptions
- **"RL is unsupervised learning."** S&B call it a third paradigm. There are no labels, but there is a reward signal to maximise.
- **"The reward tells the agent what to do."** It says how good the outcome was, never which action would have been best. S&B: the reward signal is "your way of communicating to the agent what you want achieved, not how you want it achieved" (S&B §3.2).
- **"RL started with deep learning."** TD learning, Q-learning and policy gradients all come from 1959–1992. DQN (2013/15) added deep networks and the tricks that keep them stable.
- **"ChatGPT was trained by RL."** It was pretrained on next-token prediction and then fine-tuned with SFT and RLHF. RL is the last stage, not the whole thing (§20).
- **"o1/R1 invented RL for reasoning."** They scaled policy-gradient RL with checkable rewards. The ingredients are old: REINFORCE 1992, PPO 2017, GRPO 2024.

### Quotable
- "Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal." (S&B 2018, §1.1)
- "These two characteristics—trial-and-error search and delayed reward—are the two most important distinguishing features of reinforcement learning." (S&B 2018, §1.1)
- "The agent has to exploit what it has already experienced in order to obtain reward, but it also has to explore in order to make better action selections in the future." (S&B 2018, §1.1)
- Minsky's credit-assignment problem, in S&B's words: "How do you distribute credit for success among the many decisions that may have been involved in producing it?" (S&B §1.7)
- "a computer can be programmed so that it will learn to play a better game of checkers than can be played by the person who wrote the program." (Samuel 1959, abstract)
- "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin." (Sutton, "The Bitter Lesson", 13 Mar 2019)
- "while imitating humans is enough to reproduce many human capabilities to a competent level, this approach in isolation has not and likely cannot achieve superhuman intelligence across many important topics and tasks." (Silver & Sutton, "Welcome to the Era of Experience", 2025)
- "We have found that the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)." (OpenAI, 12 Sep 2024)

---

## §2 Multi-armed bandits: explore or exploit

### Mechanism (plain words)
- **The problem.** Rover faces k slot machines ("arms"). Each pull of arm a pays a random reward drawn from a fixed distribution with unknown mean q*(a). Rover wants the most total reward over, say, 1,000 pulls. There is only one situation (no state), so this is RL with the "which state am I in?" part removed. What is left is the evaluative-feedback core: Rover only learns how good the arm it pulled was, never which arm would have been best.
- **Action values.** Rover keeps an estimate Q(a) of each arm's mean. The simplest estimate is the **sample average** of the rewards that arm has paid so far.
- **Greedy vs ε-greedy.** Greedy always pulls the arm with the highest Q. It can lock onto a mediocre arm whose first payouts happened to be good, while the best arm's first pulls were unlucky and it never gets tried again. ε-greedy pulls a uniformly random arm with probability ε (the random pick can land on the greedy arm too) and otherwise acts greedily. Every arm is then sampled forever, so every estimate converges.
- **Incremental update.** You never need to store the rewards. You nudge the estimate toward each new reward by a step of size 1/n. This "new = old + step × (target − old)" shape recurs in the rest of the course: TD, Q-learning, SGD.
- **Constant step size α** gives an exponential recency-weighted average. It never fully converges, which is useful when the arms drift (nonstationary).
- **Optimistic initial values.** Start every Q at a value above any plausible reward. Every pulled arm then "disappoints", so even a greedy agent tries everything. This is a trick for stationary problems only: the drive to explore happens once, at the beginning.
- **Regret** is the reward lost by not always pulling the best arm. Constant-ε exploration has **linear** regret: it keeps paying ε × (average gap) per pull forever. The best possible is **logarithmic** (Lai & Robbins 1985).

### Key formulas (S&B 2nd ed. notation → ours)
S&B write A_t and R_t for the action and reward at step t, q*(a) = E[R_t | A_t = a] for the true value, and Q_t(a) for the estimate. We use the same symbols. N_t(a) is the pull count. Auer et al.'s μ_i is q*(a), x̄_j is Q(a), T_j(n) is N_n(a), and Δ_i is our Δ_a.
- **Sample average (S&B eq. 2.1).** Q_t(a) = (sum of rewards when a was taken before t) / (number of times a was taken before t).
- **Greedy (2.2).** A_t = argmax_a Q_t(a), ties broken arbitrarily (in practice at random).
- **ε-greedy.** With probability 1 − ε pull argmax_a Q(a); with probability ε pull a uniformly random arm. The greedy arm's total probability is therefore 1 − ε + ε/k.
- **Incremental sample average (2.3).** Q_{n+1} = Q_n + (1/n)[R_n − Q_n]. Its general form (2.4) is NewEstimate ← OldEstimate + StepSize [Target − OldEstimate].
- **Constant step (2.5–2.6).** Q_{n+1} = Q_n + α[R_n − Q_n] = (1−α)^n Q_1 + Σ_{i=1}^{n} α(1−α)^{n−i} R_i. The weights sum to 1, and the weight on R_i decays geometrically with its age n − i.
- **Robbins–Monro conditions (2.7)** for convergence with probability 1: Σ_n α_n(a) = ∞ and Σ_n α_n²(a) < ∞. α_n = 1/n meets both; a constant α fails the second.
- **Regret (Auer, Cesa-Bianchi & Fischer 2002, §1).** regret after n plays = μ* n − Σ_{j=1}^{K} μ_j E[T_j(n)], with μ* = max_i μ_i. In our notation, Regret_n = Σ_a Δ_a E[N_n(a)], where Δ_a = q*(a*) − q*(a).
- **Lai & Robbins (1985) lower bound.** Read here via Auer et al. 2002 and Ren & Zhang 2024; the original PDF was not accessible (ScienceDirect 403).
  - The setting is a one-parameter family of reward densities, and the bound applies to every "consistent" policy (regret o(T^p) for every p > 0).
  - lim inf_{T→∞} Regret_T / ln T ≥ Σ_{a: Δ_a>0} Δ_a / KL(p_a ‖ p*).
  - Equivalently, every suboptimal arm must be pulled at least (1 + o(1)) ln T / KL(p_a ‖ p*) times.
  - For Bernoulli arms, KL(p ‖ q) = p ln(p/q) + (1−p) ln((1−p)/(1−q)).
- **UCB action selection (S&B eq. 2.10).** A_t = argmax_a [Q_t(a) + c √(ln t / N_t(a))]; an arm with N_t(a) = 0 counts as maximizing. UCB1 is the special case c = √2 (see §3).
- **Gradient bandit (S&B 2.11–2.12), for reference** (it previews policy gradients, ep 16).
  - π_t(a) = e^{H_t(a)} / Σ_b e^{H_t(b)}.
  - H_{t+1}(A_t) = H_t(A_t) + α(R_t − R̄_t)(1 − π_t(A_t)).
  - H_{t+1}(a) = H_t(a) − α(R_t − R̄_t)π_t(a) for every a ≠ A_t.
  - The average reward R̄_t is a **baseline**.

### Numbers with sources
**The 10-armed testbed.** Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018, §2.3, pp. 28–30; http://incompleteideas.net/book/RLbook2020.pdf
- Setup, exact:
  - "a set of 2000 randomly generated k-armed bandit problems with k = 10."
  - The true values q*(a), a = 1…10, "were selected according to a normal (Gaussian) distribution with mean 0 and variance 1."
  - Each reward R_t is drawn from "a normal distribution with mean q*(A_t) and variance 1."
  - Performance is measured "over 1000 time steps … This makes up one run. Repeating this for 2000 independent runs, each with a different bandit problem."
- **Figure 2.2 (p. 29): greedy vs ε = 0.01 vs ε = 0.1.**
  - All three use sample averages with initial estimate 0. The panels show average reward and % optimal action against steps 1–1000.
  - Headline sentence: "It achieved a reward-per-step of only about 1, compared with the best possible of about 1.54 on this testbed." (the greedy method)
  - "the greedy method found the optimal action in only approximately one-third of the tasks."
  - "The ε = 0.1 method explored more, and usually found the optimal action earlier, but it never selected that action more than 91% of the time."
  - "The ε = 0.01 method improved more slowly, but eventually would perform better than the ε = 0.1 method on both performance measures shown in the figure."
- [DERIVED] 1.54 is the expected maximum of 10 independent N(0,1) draws, which is 1.539. Our re-run gives 1.535 averaged over its 2,000 sampled problems.
- [DERIVED] The 91% ceiling is 1 − ε + ε/k = 0.9 + 0.01 = 0.91. For ε = 0.01 the ceiling is 0.991.
- **Figure 2.3 (p. 34): optimistic initial values.**
  - Optimistic: greedy, Q_1(a) = +5. Realistic: ε-greedy, ε = 0.1, Q_1(a) = 0. Caption: "Both methods used a constant step-size parameter, α = 0.1."
  - Y-axis is % optimal action over 1000 steps. "Initially, the optimistic method performs worse because it explores more, but eventually it performs better because its exploration decreases with time."
  - Exercise 2.6 asks about "oscillations and spikes in the early part of the curve for the optimistic method".
- **Figure 2.4 (p. 36): UCB c = 2 vs ε-greedy ε = 0.1**, average reward.
  - Caption: "UCB generally performs better than ε-greedy action selection, except in the first k steps, when it selects randomly among the as-yet-untried actions."
  - Exercise 2.8: "the UCB algorithm shows a distinct spike in performance on the 11th step … Hint: If c = 1, then the spike is less prominent."
- **Figure 2.6 (p. 42): parameter study.**
  - Each point is the average reward over the first 1000 steps, with each method's parameter (ε, α, c, Q_0) varied by factors of 2 from 1/128 to 4.
  - Every method shows an inverted U. "Overall, on this problem, UCB seems to perform best."
  - Read off the rendered figure on PDF page 64 [approximate, ±0.01]:
    - UCB peaks at **≈ 1.47** for c = 1/2–1, and falls to ≈ 1.15 at c = 4.
    - Optimistic greedy (α = 0.1) peaks at ≈ 1.43 for Q_0 = 1.
    - The gradient bandit peaks at ≈ 1.38 for α = 1/4.
    - ε-greedy peaks at ≈ 1.31 for ε = 1/16.
- **Our re-run, for sanity-checking the figures** [DERIVED]. This is script `scripts/F2_testbed.py`: numpy, 2000 runs × 1000 steps, seed 0. The course's official measurements live in `research/experiments/`; these numbers only show what the textbook curves look like.

| Method (S&B figure) | Avg reward, steps 980–1000 | % optimal, steps 980–1000 | Mean reward over steps 1–1000 |
|---|---|---|---|
| greedy, sample avg (2.2) | 1.01 | 35% | 1.009 |
| ε = 0.01, sample avg (2.2) | 1.30 | 59% | 1.184 |
| ε = 0.1, sample avg (2.2) | 1.37 | 80% | 1.300 |
| optimistic greedy Q_1=5, α=0.1 (2.3) | 1.49 | 86% | 1.292 |
| realistic ε=0.1, Q_1=0, α=0.1 (2.3) | 1.35 | 76% | 1.245 |
| UCB c = 2 (2.4) | 1.48 | 85% | 1.386 |
| UCB c = √2 (UCB1 form) | 1.51 | 91% | 1.444 |

- The re-run reproduces each textbook sentence: greedy plateaus near 1.0 and finds the best arm in about a third of problems; ε = 0.1 is far below its 91% ceiling at step 1000.
- UCB c = 2 reward is 0.03 at step 10, **1.08 at step 11**, then 0.92 and 0.83 at steps 12–13. That is the "spike on the 11th step" of Exercise 2.8: after the 10 forced pulls, step 11 goes to the arm whose single sample was highest.
- The optimistic method's % optimal action spikes to 44% at step 11, then drops to 25%.

**Lai & Robbins.**
- Citation: T. L. Lai and H. Robbins, "Asymptotically efficient adaptive allocation rules", *Advances in Applied Mathematics* 6(1):4–22, 1985, doi:10.1016/0196-8858(85)90002-8.
- Auer et al. 2002 summarise it: "Lai and Robbins were the first ones to show that the regret for this problem has to grow at least logarithmically in the number of plays."
- Ren & Zhang 2024 (arXiv:2410.02279) restate the bound in the form given above. They add that it "must sample each inferior arm a at least log(T)/KL(θa, θ∗) times asymptotically".

**History.** Thompson (1933) and Robbins (1952) studied the problem first, under the heading of "sequential design of experiments". S&B §2.1 bibliographic remarks: "introduced by Thompson (1933, 1934) and Robbins (1952), and studied by Bellman (1956)".

**Attribution.** S&B's bibliographic remarks credit the term "action value" to Watkins (1989). On ε-greedy they say: "The first to use ε-greedy methods may also have been Watkins (1989, p. 187), but the idea is so simple that some earlier use seems likely."

### Worked example: Rover's five machines [DERIVED; script `scripts/F2_rover.py`]
Arms come from `rl_reference.py`, ROVER_ARMS = (0.25, 0.45, 0.30, 0.60, 0.50). Each is a Bernoulli payout of 1 coin. Machine 4 (code arm 3) is best at 0.60.
1. **Incremental average.** Machine 4 pays 1, 0, 1, 1, 0.
   - Q = 0 → 1.000 → 0.500 → 0.667 → 0.750 → **0.600**.
   - For example, step 4: 0.667 + (1/4)(1 − 0.667) = 0.750.
   - With a constant α = 0.1 from Q_1 = 0, the same rewards give 0.100, 0.090, 0.181, 0.263, 0.237. Constant-α estimates move slowly and carry the initial bias.
2. **ε-greedy with k = 5, ε = 0.1.** The greedy arm is pulled with probability 0.9 + 0.1/5 = **0.92**. Each other arm gets 0.02.
3. **Regret bookkeeping.**
   - Gaps Δ = (0.35, 0.15, 0.30, 0, 0.10).
   - Uniformly random pulling loses mean(Δ) = 0.18 per pull, which is **180 coins per 1,000 pulls**.
   - A greedy Rover stuck on machine 2 (0.45) loses 0.15 per pull, **150 per 1,000**.
   - An ε = 0.1 Rover that already knows the best arm still loses ε × mean(Δ) = 0.018 per pull, so 18 per 1,000 and 18,000 per million. That is **linear regret**.
4. **The Lai–Robbins floor for these machines** (Bernoulli KL to p* = 0.6).
   - KL(0.25‖0.6) = 0.2526, KL(0.45‖0.6) = 0.0457, KL(0.30‖0.6) = 0.1838, KL(0.50‖0.6) = 0.0204.
   - So asymptotically machine 5 (0.50) must be tried ≥ 49.0 ln T times, machine 2 (0.45) 21.9 ln T, machine 3 (0.30) 5.4 ln T, and machine 1 (0.25) 4.0 ln T. **The near-miss machine needs the most testing.**
   - The regret constant is C = Σ Δ/KL = **11.2**, so regret ≳ 11.2 ln T. That is ≈ 77 coins at T = 1,000 and ≈ 155 at T = 10^6, against 18,000 for constant ε = 0.1.
   - Caveat: this is an asymptotic statement, not a guarantee at T = 1,000.

### Common misconceptions
- "Greedy is fine if you start with good estimates." Greedy on the 10-armed testbed finds the best arm in only about a third of problems (S&B Fig. 2.2).
- "ε-greedy explores the non-greedy arms with probability ε." It explores **all** arms with probability ε, including the greedy one. So the greedy arm's probability is 1 − ε + ε/k. [DERIVED] answer to Exercise 2.1: two arms with ε = 0.5 gives 0.5 + 0.5 × 0.5 = 0.75.
- "Optimistic initial values are a general exploration method." S&B call it "a simple trick that can be quite effective on stationary problems". With sample averages (α = 1/n), the optimism is wiped out after the **first** pull of each arm. S&B's Fig. 2.3 uses α = 0.1, which makes the optimism decay gradually.
- "Constant α is just a worse average." It is deliberately non-convergent: it tracks drifting arms. It violates Σα² < ∞ on purpose.
- "Regret is how often you pick the wrong arm." Regret weights each wrong pull by its gap Δ_a. Pulling a near-miss arm is cheap per pull, but the arm is hard to rule out.
- "The 1.54 line is an achievable score." It is the average of the best arm's *true* value; even an oracle's realised reward is noisy around it.

### Quotable
- "The most important feature distinguishing reinforcement learning from other types of learning is that it uses training information that evaluates the actions taken rather than instructs by giving correct actions." (S&B 2018, ch. 2 opening, p. 25)
- "This is the original form of the k-armed bandit problem, so named by analogy to a slot machine, or “one-armed bandit,” except that it has k levers instead of one." (S&B §2.1)
- "Because it is not possible both to explore and to exploit with any single action selection, one often refers to the “conflict” between exploration and exploitation." (S&B §2.1)
- "In this book we do not worry about balancing exploration and exploitation in a sophisticated way; we worry only about balancing them at all." (S&B §2.1)
- "Reinforcement learning requires a balance between exploration and exploitation." (S&B §2.3)
- "The beginning of time occurs only once, and thus we should not focus on it too much." (S&B §2.6, on optimistic initial values)
- Whittle's quip (Discussion of Gittins, JRSS B 41, 1979), as quoted in Scott 2015: "... [the bandit problem] was formulated during the [second world] war, and efforts to solve it so sapped the energies and minds of Allied analysts that the suggestion was made that the problem be dropped over Germany, as the ultimate instrument of intellectual sabotage." [Bracketed insertions are Scott's. The original JRSS B page was not read; some secondary sources say "Allied scientists".]

---

## §3 Optimism and luck: UCB1 and Thompson sampling

### Mechanism (plain words)
- **UCB (optimism in the face of uncertainty).** Score every arm by its estimate plus a bonus that is large when the arm has been tried rarely, then pull the top score. That is a deterministic rule.
  - The bonus √(2 ln t / N(a)) is the width of a Chernoff–Hoeffding confidence interval.
  - It shrinks every time the arm is pulled (N grows). It grows slowly, logarithmically, while the arm is ignored (t grows).
  - So every arm is eventually re-checked, and bad arms ever more rarely.
- **Thompson sampling (probability matching).** Keep a posterior belief over each arm's mean. Each round, **draw one plausible mean from each posterior** and pull the arm whose draw is highest.
  - Each arm is then pulled with exactly the posterior probability that it is the best arm.
  - Uncertain arms sometimes produce high draws, so they get explored. Confident bad arms almost never do.
  - For 0/1 rewards with a Beta prior, the update is counting: Beta(1 + wins, 1 + losses).
- **Both reach logarithmic regret.** UCB1 does so with a finite-time guarantee for any rewards in [0,1]. Thompson sampling was shown to be asymptotically optimal for Bernoulli arms in 2012. Thompson is randomised, which makes it robust to delayed or batched feedback (Chapelle & Li 2011).

### Key formulas (paper notation → ours)
- **UCB1 (Auer, Cesa-Bianchi & Fischer 2002, Figure 1).**
  - The policy, exact: "Initialization: Play each machine once. Loop: Play machine j that maximizes x̄_j + √(2 ln n / n_j), where x̄_j is the average reward obtained from machine j, n_j is the number of times machine j has been played so far, and n is the overall number of plays done so far."
  - In our notation: A_t = argmax_a [Q(a) + √(2 ln t / N(a))]. This is S&B eq. 2.10 with c = √2.
  - S&B's Figure 2.4 uses c = 2 on Gaussian (unbounded) rewards. That is a different setting from UCB1's [0,1] assumption.
- **UCB1 regret bound (Theorem 1)**, exact form: "For all K > 1, if policy UCB1 is run on K machines having arbitrary reward distributions P_1, …, P_K with support in [0, 1], then its expected regret after any number n of plays is at most
  [8 Σ_{i:μ_i<μ*} (ln n / Δ_i)] + (1 + π²/3)(Σ_{j=1}^{K} Δ_j)."
  - The proof gives E[T_j(n)] ≤ 8 ln n / Δ_j² + 1 + π²/3 for each suboptimal j.
  - The bound holds "uniformly over time" (for every n), not only asymptotically.
  - Ren & Zhang 2024: "the leading constant factor 8 of log T in (4) is bigger than the optimal constant factor 1/2 for this index".
- **UCB1-TUNED (Auer et al. §4).** Replace √(2 ln n / n_j) with √((ln n / n_j) · min{1/4, V_j(n_j)}), where V_j(s) is the sample variance + √(2 ln t / s). Quote: "This variant, which we call UCB1-TUNED, performs substantially better than UCB1 in essentially all of our experiments. However, we are not able to prove a regret bound."
- **εn-greedy (Auer et al. Theorem 3).** ε_n = min{1, cK/(d²n)}, where 0 < d ≤ min Δ_i. A decaying ε also achieves logarithmic regret, but it must know a lower bound d on the gap.
- **Thompson sampling, Beta–Bernoulli (Chapelle & Li 2011, Algorithm 2).** S_i, F_i are success and failure counters. Each round, draw θ_i ~ Beta(S_i + α, F_i + β) for every arm, pull argmax_i θ_i, and increment S or F.
  - In that paper α, β are the *Beta prior parameters*. They are not our step size α or KL coefficient β; on screen, write Beta(1 + wins, 1 + losses).
  - The prior Beta(1,1) is uniform.
- **Beta(a, b) facts.**
  - mean = a/(a+b); mode = (a−1)/(a+b−2).
  - sd = √(ab / ((a+b)²(a+b+1))).
  - The posterior after w wins and l losses from Beta(1,1) is Beta(1+w, 1+l).
- **Probability matching.** P(pull a) = P(θ_a = max_b θ_b | data).
- **Thompson sampling regret.**
  - Agrawal & Goyal 2012, COLT, arXiv:1111.1797, abstract: "for the stochastic two-armed bandit problem, the expected regret in time T is O(ln T/Δ + 1/Δ³). And, for the stochastic N-armed bandit problem, the expected regret in time T is O([Σ_{i=2}^{N} 1/Δ_i²]² ln T)."
  - Kaufmann, Korda & Munos 2012 (ALT; arXiv:1205.4217) give the first finite-time analysis that "matches the asymptotic rate given in the Lai and Robbins lower bound" for Bernoulli rewards.
  - Agrawal & Goyal 2013 (arXiv:1209.3353, AISTATS 2013) prove a problem-independent bound O(√(NT ln T)), which "solves a COLT 2012 open problem of Chapelle and Li".

### Numbers with sources
**Auer, Cesa-Bianchi & Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem", Machine Learning 47:235–256, 2002.** PDF: https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
- Abstract: "In this work we show that the optimal logarithmic regret is also achievable uniformly over time, with simple and efficient policies, and for all reward distributions with bounded support."
- Experiments use Bernoulli arms over 100,000 plays, averaged over 100 runs. Two-armed problems: (0.9, 0.6), (0.9, 0.8), (0.55, 0.45). Ten-armed problems include (0.9, 0.6×9) and the "hard" (0.55, 0.45×9).
- Result: "An optimally tuned εn-GREEDY performs almost always best", but "if εn-GREEDY is not well tuned its performance degrades rapidly". Also: "In most cases, UCB1-TUNED performs comparably to a well-tuned εn-GREEDY."

**Thompson 1933.**
- W. R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples", *Biometrika* 25(3/4):285–294, 1933, doi:10.1093/biomet/25.3-4.285.
- It was framed around comparing two treatments. Chapelle & Li 2011: "The idea of this heuristic is old and dates back to [16]. This is the reason why this scheme is also referred to as Thompson sampling." I did not read the original text, so there is no direct quote from Thompson.

**Chapelle & Li, "An Empirical Evaluation of Thompson Sampling", NeurIPS 2011.** https://papers.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
- **Simulations (Fig. 1)**: K ∈ {10, 100} arms. The best arm pays 0.5; the others pay 0.5 − ε with ε ∈ {0.02, 0.1}. Results are averaged over 100 repetitions and compared with a tight Chernoff-based UCB. Conclusion: "Thompson sampling is asymptotically optimal and achieves a smaller regret than the popular UCB algorithm."
- **Delayed feedback (Table 1).** 10 items, each retiring with probability 10⁻³ per step, true rates ~ Beta(4,4). Average regret at T = 10⁶ over 100 runs:

| feedback every δ steps | 1 | 3 | 10 | 32 | 100 | 316 | 1000 |
|---|---|---|---|---|---|---|---|
| UCB | 24,145 | 24,695 | 25,662 | 28,148 | 37,141 | 77,687 | 226,220 |
| Thompson | 9,105 | 9,199 | 9,049 | 9,451 | 11,550 | 21,594 | 59,256 |
| ratio | 2.65 | 2.68 | 2.84 | 2.98 | 3.22 | 3.60 | 3.82 |

- **Display ads (Table 2).** The context and ads are real but the clicks are simulated from a perturbed learned weight vector. CTR regret in %:
  - Thompson: 4.45 (α=0.25), **3.72 (α=0.5)**, 3.81 (α=1)
  - LinUCB: 4.99 (α=0.5), 4.22 (α=1), 4.14 (α=2)
  - ε-greedy: 5.05 (ε=0.005), 4.98 (ε=0.01), 5.22 (ε=0.02)
  - Exploit-only 5.00; Random 31.95
  - For Thompson, α < 1 shrinks the posterior, which means less exploration.
- **Yahoo! news (Fig. 5).** "From 7 days in June 2009, over 34M randomized serving events were obtained". These were evaluated with the replay method (§4). "While the deterministic UCB works well with short delay, its performance drops significantly as the delay increases."

**S&B §2.10** on Thompson: "This method, sometimes called posterior sampling or Thompson sampling, often performs similarly to the best of the distribution-free methods we have presented in this chapter." The book does not include a Thompson curve in Figs. 2.2–2.6.

### Worked example: Rover's five machines [DERIVED; `scripts/F2_rover.py`]
**A. UCB score table.** The illustrative history is t = 20 pulls so far, N = (2, 4, 2, 8, 4), wins = (0, 2, 1, 4, 3), ln 20 = 2.996.

| machine | true p | N | Q | UCB1 bonus √(2 ln t/N) | UCB1 score | S&B c=2 score |
|---|---|---|---|---|---|---|
| 1 | 0.25 | 2 | 0.000 | 1.731 | 1.731 | 2.448 |
| 2 | 0.45 | 4 | 0.500 | 1.224 | 1.724 | 2.231 |
| 3 | 0.30 | 2 | 0.500 | 1.731 | **2.231** | **2.948** |
| 4 | 0.60 | 8 | 0.500 | 0.865 | 1.365 | 1.724 |
| 5 | 0.50 | 4 | 0.750 | 1.224 | 1.974 | 2.481 |

- UCB pulls machine 3 under either c: it is under-explored, and its 1-in-2 record could still hide a good rate.
- The truly best machine 4 has the *lowest* score right now. It has had 8 pulls and an unlucky 4/8 record. Its bonus will grow again as t rises while N(4) stays put, so UCB will come back to it.

**B. The UCB1 guarantee is loose for small T.**
- Theorem 1's bound for Rover's machines at n = 1,000 is 8 × Σ(ln 1000/Δ) + (1 + π²/3) × ΣΔ = 8 × 6.908 × 22.86 + 4.29 × 0.90 ≈ **1,267 coins**.
- That is larger than the worst possible regret, 0.35 × 1000 = 350.
- At n = 10^6 the bound is ≈ 2,530, while the Lai–Robbins asymptotic floor is ≈ 155 (§2).
- The bound's value is its *shape* (log n, uniformly in n), not its constant.

**C. Beta posteriors.** Machine 4 starts at Beta(1,1): mean 0.500, sd 0.289.
- After 1, 0, 1, 1, 0 it is Beta(4, 3): mean 0.571, mode 0.600, sd 0.175.
- After 60 wins in 100 pulls it is Beta(61, 41): mean 0.598, sd 0.048. The curve narrows roughly like 1/√n.

**D. Probability matching, one round.** The posteriors are (a, b) = machine 1 (1,2) [0 of 1], machine 2 (2,2) [1 of 2], machine 3 (1,1) [untried], machine 4 (4,3) [3 of 5], machine 5 (3,2) [2 of 3].
- P(machine is best) = (0.066, 0.169, 0.257, 0.211, **0.296**). This comes from numerical integration and was checked by 400k Monte Carlo draws: (0.067, 0.170, 0.256, 0.211, 0.297).
- Thompson pulls each machine with exactly these probabilities. The untried machine 3 gets a quarter of the pulls because its flat posterior can produce high draws. Machine 1, at 0 of 1, still gets 7%.

### Common misconceptions
- "UCB adds noise to explore." UCB is deterministic. Its exploration comes from the bonus, and given the same history it always makes the same choice. Thompson sampling is the randomised one.
- "UCB1 is S&B's UCB with c = 2." UCB1 is c = √2 with rewards in [0,1] (Auer et al., Fig. 1). S&B's c = 2 is a tuned value on unit-variance Gaussian rewards.
- "Logarithmic regret means few mistakes early." The log bound is about growth over time. Early on every method must try every arm. UCB1 even pulls every arm once before anything else, which causes the step-11 spike.
- "Thompson sampling needs a correct prior." It works well with the flat Beta(1,1) prior. Chapelle & Li also report it robust to "prior mismatch" (best arm 0.1 vs nine arms at 0.08, still better than UCB). Shrinking the posterior (α < 1) can lower finite-time regret at the cost of variance.
- "Thompson sampling is a recent idea." It dates from 1933. Chapelle & Li (2011) called it "surprisingly unpopular in the literature". Its regret was first proved logarithmic only in 2012 (Agrawal & Goyal).
- "Bayesian means slow." The Beta–Bernoulli update is just two counters per arm.

### Quotable
- "Lai and Robbins were the first ones to show that the regret for this problem has to grow at least logarithmically in the number of plays." (Auer, Cesa-Bianchi & Fischer 2002, abstract)
- "In this work we show that the optimal logarithmic regret is also achievable uniformly over time, with simple and efficient policies, and for all reward distributions with bounded support." (Auer et al. 2002, abstract)
- "Thompson sampling is one of oldest heuristic to address the exploration / exploitation trade-off, but it is surprisingly unpopular in the literature." (Chapelle & Li 2011, abstract; the grammar is the original's)
- "The idea of Thompson sampling is to randomly draw each arm according to its probability of being optimal." (Chapelle & Li 2011, §1)
- "Also, since it is a randomized algorithm, it is robust in the case of delayed feedback." (Chapelle & Li 2011, §6)
- "The question of the optimality of Thompson Sampling for solving the stochastic multi-armed bandit problem had been open since 1933." (Kaufmann, Korda & Munos 2012, abstract)
- "Thompson sampling is an algorithm for online decision problems where actions are taken sequentially in a manner that must balance between exploiting what is known to maximize immediate performance and investing to accumulate new information that may improve future performance." (Russo, Van Roy, Kazerouni, Osband & Wen, "A Tutorial on Thompson Sampling", arXiv:1707.02038, abstract)

---

## §4 Contextual bandits: LinUCB, offline replay, and A/B tests

### Mechanism (plain words)
- **Contextual bandits.** Before each choice Rover sees a **context** x: who is asking, such as a student's level or a user's features. The best arm can depend on x.
  - Each action still affects only the immediate reward, never the next context. This sits between bandits and full RL.
  - S&B §2.9 call it "associative search": "Associative search tasks are often now called contextual bandits in the literature."
- **LinUCB** assumes each arm's expected reward is linear in the context: E[r | x, a] = xᵀθ*_a. Per arm, it runs:
  - **ridge regression**, with A_a = I + Σ x xᵀ and b_a = Σ r x, so θ̂_a = A_a⁻¹ b_a;
  - **a UCB bonus** α √(xᵀ A_a⁻¹ x), which is the standard deviation of the prediction in direction x. The bonus is large for contexts unlike anything this arm has seen.
  - Rover picks argmax_a [xᵀθ̂_a + bonus]. The chosen arm changes as x changes.
  - The **hybrid** variant adds features shared by all arms, so what is learned about one article transfers to the others.
- **Offline replay evaluation (Li et al. 2010 §4; 2011).** Log traffic where the arm was chosen **uniformly at random**. Stream the log through the new policy and keep an event only when the policy picks the same arm the log shows.
  - Because logging was uniform, each event survives with probability exactly 1/K. The kept events are distributed exactly as if the policy had run live, so the average reward is **unbiased**.
  - The cost is that it needs ≈ K × T logged events to simulate T decisions.
- **A/B test vs bandit.**
  - An A/B test keeps a fixed traffic split until it reaches statistical power. It optimises *learning which arm is best* and pays the full opportunity cost meanwhile.
  - A bandit shifts traffic toward winners as evidence arrives. It optimises *reward during the experiment*, with lower regret and often an earlier stop.
  - The price is weaker classical inference, plus extra sensitivity to time-varying effects: weekday vs weekend users, delayed conversions.

### Key formulas (Li, Chu, Langford & Schapire 2010 → ours)
- **Disjoint model (eq. 2):** E[r_{t,a} | x_{t,a}] = x_{t,a}ᵀ θ*_a.
- **Ridge estimate (eq. 3):** θ̂_a = (D_aᵀ D_a + I_d)⁻¹ D_aᵀ c_a.
- **Confidence bound (eq. 4).** With probability ≥ 1 − δ, |x_{t,a}ᵀθ̂_a − E[r_{t,a}|x_{t,a}]| ≤ α √(x_{t,a}ᵀ (D_aᵀD_a + I_d)⁻¹ x_{t,a}), "where α = 1 + √(ln(2/δ)/2) is a constant."
  - [DERIVED] δ = 0.05 gives α = 2.358.
  - The paper notes this α "may be conservatively large in some applications", so in practice α is tuned.
- **Arm choice (eq. 5):** a_t = argmax_{a∈A_t} (x_{t,a}ᵀθ̂_a + α √(x_{t,a}ᵀ A_a⁻¹ x_{t,a})), where A_a = D_aᵀD_a + I_d.
- **Algorithm 1, per round.**
  - For a new arm, initialise A_a ← I_d and b_a ← 0.
  - θ̂_a ← A_a⁻¹ b_a; p_{t,a} ← θ̂_aᵀx_{t,a} + α√(x_{t,a}ᵀA_a⁻¹x_{t,a}).
  - Choose the argmax, observe r_t, then update A_{a_t} += x xᵀ and b_{a_t} += r_t x.
  - The only input is α.
- **Notation clashes.** LinUCB's α is an exploration width, not a step size. Its δ is a confidence level, not a TD error. Its θ*_a are per-arm regression weights.
- **Hybrid model (eq. 6):** E[r_{t,a} | x_{t,a}] = z_{t,a}ᵀβ* + x_{t,a}ᵀθ*_a. Here z is a shared user×article feature and β* is common to all arms; it is not the KL β. The UCB is computed with block-matrix inversion (Algorithm 2).
- **Properties claimed.** Computation is "linear in the number of arms and at most cubic in the number of features". With a fixed set of K arms the analysis gives regret Õ(√(KdT)).
- **Replay evaluator (Li et al. 2010, Algorithm 3 / Theorem 1).**
  - For every distribution D of contexts, every policy π, every T and every history h_T: Pr_{π,D}(h_T) = Pr_{Policy_Evaluator(π,S)}(h_T), where S is a stream logged by a uniform random policy.
  - "Furthermore, the expected number of events obtained from the stream to gather a history h_T of length T is KT."
  - Li et al. 2011 add a finite-stream version (their Algorithm 2: about L/K valid events from a log of length L) and a high-probability bound.
  - **Unbiasedness does not imply concentration for every learning algorithm**: "such a conjecture is false for general bandit algorithms" (Li et al. 2011, §3.1).
- **A/B sample size** (two-proportion, one-sided): n per arm ≈ (z_{1−α} + z_{power})² [p₁(1−p₁) + p₂(1−p₂)] / (p₂ − p₁)². Here α is the significance level, not our step size.

### Numbers with sources
**Li, Chu, Langford & Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010.** arXiv:1003.0146, https://arxiv.org/abs/1003.0146
- **Headline (abstract)**: "using this offline evaluation method, we successfully applied our new algorithm to a Yahoo! Front Page Today Module dataset containing over 33 million events. Results showed a 12.5% click lift compared to a standard context-free bandit algorithm, and the advantage becomes even greater when data gets more scarce."
- **The 12.5% is real, but it is specific.** It is Table 1's **linucb (disjoint)** in the **deployment bucket** with **100% of the data**: CTR 1.795 vs **ε-greedy** (context-free) 1.596. [DERIVED] 1.795/1.596 − 1 = 12.47%.
  - CTRs are relative to the random policy's CTR, which is 1 by definition: "To protect business-sensitive information, we report an algorithm's relative CTR".
- **Dataset.**
  - Events come from a "random bucket" in May 2009, using only the F1 article at the story position.
  - Tuning data: "about 4.7 million events in the random bucket on May 01".
  - Evaluation data: "a one-week event set … from May 03–09, which contained about 36 million events".
  - **Discrepancy:** the abstract says "over 33 million events" and §5.2.1 says "about 36 million". Both are in the same paper; use "over 33 million" when quoting the abstract.
- **Features.**
  - Users start as 1,193-dimensional vectors (over 1,000 binary categories: demographics, ~200 locations, behavioural categories) and articles as 83-dimensional.
  - These were reduced to **6-dim** vectors (5 user clusters + a constant) via a bilinear logistic model and k-means.
  - The hybrid model's shared feature z is the 6 × 6 = **36-dim** outer product.
- **Two "buckets".** The *learning bucket* is a small share of traffic where the bandit explores. The *deployment bucket* is served greedily from the learning bucket's estimates.
- **Table 1 (CTR, lift vs ε-greedy), selected cells:**

| algorithm | 100% data: deploy | 100%: learn | 1% data: deploy | 1%: learn |
|---|---|---|---|---|
| ε-greedy | 1.596 (0%) | 1.326 (0%) | 1.234 (0%) | 1.139 (0%) |
| ucb | 1.594 (0%) | 1.569 (18.3%) | 1.354 (9.7%) | 1.22 (7.1%) |
| ucb (seg) | 1.781 (11.6%) | 1.677 (26.5%) | 1.398 (13.3%) | 1.25 (9.7%) |
| linucb (disjoint) | **1.795 (12.5%)** | 1.647 (24.2%) | 1.382 (12%) | 1.197 (5.1%) |
| linucb (hybrid) | 1.73 (8.4%) | 1.663 (25.4%) | 1.482 (20.1%) | 1.446 (27%) |

  - "The CTR of omniscient was 1.615". Omniscient is the best single article in hindsight, with no context.
  - "by considering user features, both ε-greedy (seg/disjoint/hybrid) and UCB methods (ucb (seg) and linucb (disjoint/hybrid)) were able to achieve a CTR lift of around 10%, compared to the baseline ε-greedy."
  - One article's CTR was "lifted from 1.31 to 3.03—a 132% improvement" (Figure 3(c), learning bucket).
  - **Internal inconsistency:** the text says at 1% data "a 10.3% improvement of linucb (hybrid)'s CTR in the deployment bucket (1.493) over ucb's (1.354)", but Table 1 lists linucb (hybrid) at 1% deploy as **1.482**. 1.493/1.354 = 1.103 matches the text, and 1.482/1.354 = 1.095. Quote the table, or say "about 10%".
- **Conclusion quote**: "we found that upper confidence bound methods generally outperform the simpler yet unguided ε-greedy methods. Furthermore, our new algorithm LinUCB shows advantages when data are sparse".

**Li, Chu, Langford & Wang, "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms", WSDM 2011.** arXiv:1003.5956, https://arxiv.org/abs/1003.5956
- Abstract: "More importantly, our method can provide provably unbiased evaluations."
- Data: a random bucket from Nov 1–10, 2009, "about 40 million events in the offline evaluation data set, and about 20 articles available in the pool at every moment."
- **Offline vs online validation.** For each day they compare ρ_A = (offline CTR estimate) / (online bucket CTR).
  - EMP vs SEMP over 16 days (May 3–18, 2009, ~2,000,000 views per bucket per day): slope 1.019, residual sd 0.0563.
  - CEMP (logistic-regression contextual policy) over 18 days (May 22–June 8, 2010): slope 1.113, residual sd 0.075.
  - Their reading: "the relative performance of bandit models in offline evaluation is reserved in online buckets". "Reserved" is the paper's wording, presumably for "preserved".
- **Limit (Chapelle & Li 2011, §4).** The replay method "cannot be used in our case here because it reduces the effective data size substantially when the number of arms K is large". Their display-ads problem had a mean of 1,364 eligible ads per context.

**Industry contextual bandits [DATED].**
- Agarwal et al. 2016, "Making Contextual Decisions with Low Technical Debt" (Microsoft Decision Service; arXiv:1606.03966) reports "two live production deployments for content recommendation that achieved click-through improvements of 25-30%, another with 18% revenue lift in the landing page".
- The service became Azure AI Personalizer. Microsoft's archived Personalizer page (updated 2026-09-01) says "Starting 20 September 2023, you can't create new Personalizer resources. The Personalizer service is retiring 25 August 2026."
- **Sources disagree on the date:** several third-party retirement trackers give 1 October 2026. Either way it is retired as of this writing.
- Microsoft points users to the open-source microsoft/learning-loop. https://learn.microsoft.com/en-us/azure/ai-services/personalizer/what-is-personalizer

**A/B tests vs bandits: Steven L. Scott, "Multi-armed bandit experiments in the online service economy", Applied Stochastic Models in Business and Industry 31:37–45, 2015.** PDF: https://research.google.com/pubs/archive/42550.pdf
- Abstract: "the cost of experimenting is dominated by opportunity cost rather than the cost of obtaining experimental units."
- **Worked case (§4.1).** Conversion rates are p = 0.001 vs 0.0011.
  - A classical test "with 95% power under a one-sided alternative we would need roughly 4.5 million observations (2,270,268 in each arm). If we decreased the power to .5 we would need slightly more than 1.1 million (567,568 in each arm)."
  - [DERIVED] the two-proportion formula with z = 1.645 reproduces 2,270,265 and 567,566 per arm.
- **Bandit simulation** (100 visits/day, daily Thompson updates, 100 runs). "The simulation found the correct version in 84 out of 100 runs."
  - "Roughly 2/3 of the simulation runs resulted in single digits of lost conversions compared to hundreds of lost conversions for the traditional experiment with comparable power."
  - "Under the 95% power calculation the experiment would take roughly 125 “years,” while 29 of 100 bandit simulations finished within one “year”."
- **Caveat (§4.2).** If arm A is slightly better on weekdays and B much better on weekends, the binomial model can declare A the winner before any weekend traffic arrives. "the bandit experiment is more susceptible because it will be run for a shorter period of time." The fix he gives is contextual: a logistic regression on known covariates.

### Worked example: Rover recommends a study tip [DERIVED; `scripts/F2_rover.py`]
- **Setup.** Two arms: "flashcards" and "practice problems". Context x = [1, beginner?], i.e. a bias term plus one feature. α = 1, ridge prior A = I.
- **Data so far.**
  - Flashcards: (x = [1,1], clicked 1) and (x = [1,0], 0).
  - Practice: (x = [1,0], 1) and (x = [1,1], 0).
- **Each arm's A.** For both arms, A = I + Σxxᵀ = [[3,1],[1,2]], so A⁻¹ = (1/5)[[2,−1],[−1,3]].
- **Each arm's weights.**
  - Flashcards: b = [1,1], θ̂ = (1/5)[2−1, −1+3] = **[0.2, 0.4]**.
  - Practice: b = [1,0], θ̂ = **[0.4, −0.2]**.
- **Beginner (x = [1,1]).** The bonus is √(xᵀA⁻¹x) = √(3/5) = 0.775 for both arms.
  - Flashcards: 0.6 + 0.775 = **1.375**. Practice: 0.2 + 0.775 = 0.975.
  - **Choose flashcards.**
- **Experienced student (x = [1,0]).** The bonus is √(2/5) = 0.632.
  - Flashcards: 0.2 + 0.632 = 0.832. Practice: 0.4 + 0.632 = **1.032**.
  - **Choose practice.**
- **The animation beat.** Same arms and same data, but a different context flips the choice.
- **Replay cost.** With 5 arms logged uniformly at random, about 5,000 logged events are needed to replay 1,000 decisions of a new policy (the KT rule).

### Common misconceptions
- **"LinUCB gave Yahoo! a 12.5% lift in production."** The 12.5% is an *offline replay* estimate, normalised to a random policy. It is for the disjoint model in the deployment bucket, against context-free ε-greedy, not against context-free UCB. The hybrid model, which the authors hoped would be best, scored only 8.4% at full data. It led at 1% data (20.1%).
- **"A contextual bandit is full RL."** Actions do not change future contexts; there is no credit assignment through time. S&B: if actions affect the next situation as well, "then we have the full reinforcement learning problem."
- **"You can evaluate a new policy on any logs."** Replay needs logs with known, randomised action probabilities: uniform in the simple version, or any randomised logging policy via rejection sampling at lower data efficiency. It keeps only 1/K of the data, so it breaks down when K is large.
- **"Bandits replace A/B tests."** They answer a different question: maximise reward while learning, rather than estimate an effect with guaranteed power. They are more exposed to time-varying effects and to delayed feedback. The deterministic UCB is hurt most by delay (Chapelle & Li Table 1).
- **"Exploration in a contextual bandit is ε of the traffic."** That was Yahoo!'s ε-greedy baseline. LinUCB explores in a *directed* way: the bonus is large only for contexts the arm has not seen.

### Quotable
- "Results showed a 12.5% click lift compared to a standard context-free bandit algorithm, and the advantage becomes even greater when data gets more scarce." (Li, Chu, Langford & Schapire 2010, abstract)
- "Associative search tasks are often now called contextual bandits in the literature." (S&B §2.9)
- "More importantly, our method can provide provably unbiased evaluations." (Li, Chu, Langford & Wang 2011, abstract)
- "In particular, the cost of experimenting is dominated by opportunity cost rather than the cost of obtaining experimental units." (Scott 2015, abstract)
- "When there is a difference to be found, the multi-armed bandit approach is dramatically more efficient at finding the best arm than traditional statistical experiments, and its advantage increases as the number of arms grows (Berry, 2011)." (Scott 2015, §4.1)

---

## §5 Markov decision processes: states, actions, returns (Rover's world)

### Mechanism (plain words)
- An **MDP** is a world described by five things: states S, actions A(s), transition probabilities p(s′|s,a), rewards, and a discount γ. At each step Rover sees the state s and picks an action a. The world answers with a reward r and a next state s′, drawn at random from p.
- **Markov property.** The next state and reward depend only on the current state and action, not on the path that led there. This is a property of *how you describe the state*, not of the world. If the description leaves out something that matters for the future (Rover's battery level, a moving obstacle), it is not Markov.
- **Episodes vs continuing tasks.**
  - An *episodic* task ends in a terminal state, then resets. Examples: a game ends; Rover reaches the charger or the puddle.
  - A *continuing* task never ends (a thermostat, a server). The plain sum of its rewards can be infinite, so we discount.
- **Return.** The return G_t is the sum of the rewards from time t onward, each multiplied by γ once per step of delay.
  - γ = 0 makes a "myopic" agent; γ near 1 makes a far-sighted one.
  - Rule of thumb: γ sets an effective horizon of about 1/(1−γ) steps, i.e. 10 steps for γ = 0.9 and 100 for γ = 0.99 [DERIVED from Σγ^k = 1/(1−γ)].
- **Reward ≠ instructions.** The reward says *what* we want, not *how* to do it. Rewarding sub-goals invites the agent to farm the sub-goal (see §11 shaping, §25 reward hacking).
- **Policy.** A policy π(a|s) is a rule for choosing actions. In a finite MDP an optimal policy can always be chosen deterministic and dependent only on the current state (standard result; S&B §3.6).

### Key formulas (S&B 2nd ed. notation → ours)
- Dynamics (S&B eq. 3.2): p(s′, r | s, a) = Pr{S_t = s′, R_t = r | S_{t−1} = s, A_{t−1} = a}. Transition probabilities: p(s′|s,a) = Σ_r p(s′,r|s,a).
- Episodic return (3.7): G_t = R_{t+1} + R_{t+2} + … + R_T.
- Discounted return (3.8): G_t = Σ_{k=0}^{∞} γ^k R_{t+k+1}, with 0 ≤ γ ≤ 1.
- Recursion (3.9): **G_t = R_{t+1} + γ G_{t+1}** (with G_T = 0). This one line is the seed of every Bellman equation.
- Constant reward +1 forever (3.10): G_t = 1/(1−γ).
- Unified episodic/continuing notation (3.11): G_t = Σ_{k=t+1}^{T} γ^{k−t−1} R_k. T = ∞ or γ = 1 is allowed, but not both.
- Value functions (3.12, 3.13): V^π(s) = E_π[G_t | S_t = s]; Q^π(s,a) = E_π[G_t | S_t = s, A_t = a].
- **Notation map, Russell & Norvig (R&N) → ours.**
  - R&N's U(s) ("utility") = our V(s); U^π = V^π.
  - R&N 3rd ed. uses a state reward R(s), collected in every non-terminal state. S&B use R_{t+1} on each transition.
  - With γ = 1 the two conventions give identical numbers for Rover's world if we read them as "−0.04 per move, plus ±1 on the move that enters a terminal" (checked by our script, below).

### Rover's world = Russell & Norvig's 4×3 world (exact setup)
Source: Russell & Norvig, *Artificial Intelligence: A Modern Approach*, 3rd ed. (2010), Ch. 17 "Making Complex Decisions":
- Fig. 17.1: the world and its transition model.
- Fig. 17.2: optimal policies.
- Fig. 17.3: utilities.

Other editions:
- **2nd ed. (2003):** same chapter and figure numbers. The official errata list "Chapter 17 Page 616 … Figure 17.2(b)".
- **US 4th ed. (2020):** still Ch. 17 "Making Complex Decisions", with the policy figure as Fig. 17.2. I saw this only in a search-engine excerpt of a 4th-ed. preview [UNVERIFIED].
- The US 4th ed. does **not** put it in Ch. 16 (that chapter is "Making Simple Decisions") [UNVERIFIED for international printings].

The setup:
- **Grid:** 4 columns × 3 rows, coordinates (column, row), with (1,1) at bottom-left. Rock/wall at **(2,2)**. Rover can occupy 11 cells, of which **9 are non-terminal**.
- **Terminals:** **+1 at (4,3)** (charger) and **−1 at (4,2)** (puddle).
- **Start:** **(1,1)**.
- **Actions:** Up, Down, Left, Right.
- **Slippery wheels:** the intended move happens with probability **0.8**. With probability **0.1 each**, Rover moves at right angles to the intended direction. Bumping into the outer wall or the rock leaves Rover where it is.
- **Reward:** **R(s) = −0.04** in every non-terminal state.
- **Discount:** **γ = 1** (no discounting).
- **Reference code:** aima-python defines the same grid, `[[-0.04, -0.04, -0.04, +1], [-0.04, None, -0.04, -1], [-0.04, -0.04, -0.04, -0.04]]`, and cites "[Figure 17.1]" (aimacode/aima-python, `aima/mdp.py`). But its `GridMDP` default is **γ = 0.9**, which gives different numbers (see misconceptions).

**Optimal utilities, γ = 1, R = −0.04.** These are R&N 3rd ed. Fig. 17.3, reproduced exactly by our value iteration ([DERIVED] 4-dp values in brackets).

| row \ col | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 3 | 0.812 (0.8116) | 0.868 (0.8678) | 0.918 (0.9178) | **+1** |
| 2 | 0.762 (0.7616) | rock | 0.660 (0.6603) | **−1** |
| 1 | **0.705** (0.7053) start | 0.655 (0.6553) | 0.611 (0.6114) | 0.388 (0.3879) |

The same table (to 3 dp) appears in Alice Gao's CS 486/686 Lecture 18 notes (2021, hosted at cs.toronto.edu). Those notes also give the same Q-values at (3,1): 0.6323 (Up, past the puddle), 0.6511 (Left), 0.4375 (Right) and 0.5931 (Down). Their row labels are flipped, so they name these actions differently.

**Optimal policy, R = −0.04.** This is R&N 3rd ed. Fig. 17.2(a), reproduced [DERIVED].

| row \ col | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 3 | → | → | → | +1 |
| 2 | ↑ | rock | ↑ | −1 |
| 1 | ↑ | ← | ← | ← |

The famous detail:
- At **(3,1)** Rover takes the long way round (Left, Q = 0.651) instead of the shortcut Up past the puddle (Q = 0.632).
- At **(4,1)** it also goes Left.
- R&N's text is often paraphrased as calling the (3,1) choice "conservative", taking "the long way round" [UNVERIFIED wording: not checked against the book; say it in your own words].

**The policy depends on the step reward R(s)** (R&N Fig. 17.2(b), 4 panels). The printed thresholds come from the 2nd-ed. official errata, which corrected missing minus signs: "Should be: -1.6284, -0.4278, -0.0850, -0.0221". The four panels:
- R < −1.6284: "life is painful"; head for the nearest exit, even the −1.
- −0.4278 < R < −0.0850: take the shortest route to +1, accepting the risk of the −1.
- −0.0221 < R < 0: avoid the −1 at almost any cost, even by bumping into walls.
- R > 0: life is good; avoid both exits forever (with γ = 1, values are infinite).

**Our exact recomputation [DERIVED].** We ran value iteration to 1e-13 and confirmed it with exact policy iteration. There are 8 switch points between −2 and 0, not 4:

| R range | what changes (state: action) | R&N panel |
|---|---|---|
| R < −1.6497 | 11:→ 12:↑ 13:→ 21:→ 23:→ 31:→ 32:→ 33:→ 41:↑ | "painful" (matches the panel) |
| −1.6497 … −1.5643 | (3,2) → ↑ | |
| −1.5643 … −0.7311 | (3,1) → ↑ | |
| −0.7311 … −0.4526 | (1,1) → ↑ | |
| −0.4526 … −0.0850 | (4,1) → ← ; policy = 11:↑ 12:↑ 13:→ 21:→ 23:→ 31:↑ 32:↑ 33:→ 41:← | "quite unpleasant" (matches the panel) |
| −0.0850 … −0.0448 | (2,1) → ← | |
| −0.0448 … −0.0274 | (3,1) → ← (contains R = −0.04: the book's policy) | Fig. 17.2(a) |
| −0.0274 … −0.0221 | (3,2) → ← (bump into the rock rather than risk the puddle) | |
| −0.0221 … 0 | (4,1) → ↓ (bump into the bottom wall rather than risk the puddle) | "slightly dreary" (matches) |

**Two disagreements with the book:**
- **Thresholds.** Our switch points reproduce the book's **−0.0850 and −0.0221** to 4 dp. But we put the other two at **≈ −1.6497** (not −1.6284) and **≈ −0.4526** (not −0.4278).
  - The model variants we tried (terminal reward shifted by R; γ = 0.999) do not move them to the printed values.
  - For the video: quote the book's thresholds with a citation, or ours labelled "computed". Do not mix them.
- **The 4th-ed. tie.** The 4th-ed. caption reportedly says "There are two policies because in state (3,1) both Left and Up are optimal" (search excerpt) [UNVERIFIED].
  - Under the 3rd-ed. model we find **no tie** at (3,1): Left 0.6514 vs Up 0.6325.
  - The 4th ed.'s per-transition reward does not create one either: it shifts every path's return by the same constant.
  - Use the 3rd-ed. figures for the course.

### Numbers with sources
- **MDP formalism.** Bellman (1957b) "also introduced the discrete stochastic version of the optimal control problem known as Markov decision processes (MDPs)" (S&B 2018, §1.7).
  - Citation: Bellman, R. (1957). "A Markovian Decision Process." *Journal of Mathematics and Mechanics* 6(5):679–684 (JSTOR 24900506).
  - S&B's bibliography gives the title as "A Markov decision process"; the journal title is "A Markovian Decision Process".
- **S&B Example 3.5 gridworld** (5×5, γ = 0.9; leaving A moves you to A′ for +10, leaving B moves you to B′ for +5, stepping off the grid costs −1). Both values below are reproduced by our script [DERIVED].
  - Under the equiprobable random policy, V(A) = **8.8**, *less* than its immediate reward of 10 (Fig. 3.2).
  - Under the optimal policy, V*(A) = **24.4** (Fig. 3.5).
- **Blackjack as an MDP:** 200 states (S&B Example 5.1; see §8).

### Worked example [DERIVED] (scripts grid43.py, f3_worked.py, mc43.py)
1. **One lucky run.** Rover goes Up, Up, Right, Right, Right from (1,1) and every move works.
   - γ = 1: 5 moves × (−0.04) + 1 = **G = 0.80**.
   - γ = 0.9: G = −0.04(1 + 0.9 + 0.81 + 0.729 + 0.6561) + 0.9⁴ × 1 = −0.1638 + 0.6561 = **0.4923**.
2. **How lucky is that?**
   - All five moves succeed with probability 0.8⁵ = **0.32768**.
   - If Rover executes that fixed five-move plan blindly, it reaches the charger with probability **0.32776**. That is 0.8⁵ + 0.1⁴ × 0.8, the extra term being a run of slips the long way round.
   - It ends in the puddle with probability 0.014.
   - R&N 3rd ed. §17.1 states the 0.32776 figure [UNVERIFIED wording; the arithmetic is ours].
3. **Why V(1,1) = 0.705 and not 0.80.** The optimal *policy* reacts to slips.
   - Following it, Rover reaches the charger 98.63% of the time and the puddle 1.37%, in **6.68 moves on average**.
   - Check: 0.9863 − 0.0137 − 0.04 × 6.682 = 0.7053 ✓.
   - In simulation, 32.7% of episodes take the perfect 5-move path, matching 0.8⁵.
4. **A random Rover** (each action with probability ¼) starting from (1,1):
   - reaches the charger only 35.4% of the time and the puddle 64.6%;
   - takes 32.4 moves on average;
   - so V^random(1,1) = **−1.587**, worse than falling straight into the puddle.

Animation idea: the intended arrow splits into a fat 0.8 arrow and two thin 0.1 arrows. A return bar shrinks by 0.04 per step, then jumps by +1 at the charger.

### Common misconceptions
- "Markov means the world has no memory." No: it means the *state* carries all the memory that matters. Add the relevant history to the state and almost anything becomes Markov.
- "γ is only a mathematical trick." It is also a modelling choice: how much the future matters, or equivalently a constant probability 1−γ that the episode ends at each step. It sets the effective horizon ≈ 1/(1−γ).
- "γ = 1 is not allowed." It is fine in episodic tasks where good policies end with probability 1; Rover's world uses γ = 1. Policies that never end (e.g. always bump Left in column 1) then have value −∞ when R < 0.
- "−0.04 is an arbitrary detail." The optimal behaviour changes 8 times as R goes from −2 to 0 (table above).
- "R&N and S&B disagree on the numbers." They use different reward-timing conventions. With γ = 1 and "−0.04 per move, ±1 on entering a terminal", the numbers are identical.
- **Code warning.** aima-python's `GridMDP` defaults to **γ = 0.9** [DERIVED]:
  - V(1,1) becomes 0.296 (not 0.705);
  - the bottom row becomes ↑ → ↑ ←, which is exactly the policy in aima-python's doctest.

  This is a common source of "the textbook is wrong" confusion.

### Quotable
- "The state must include information about all aspects of the past agent–environment interaction that make a difference for the future. If it does, then the state is said to have the Markov property." (Sutton & Barto 2018, §3.1, p. 49)
- Reward hypothesis: "That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)." (S&B 2018, §3.2, p. 53. S&B credit the explicit statement to Michael Littman.)
- "The reward signal is your way of communicating to the agent what you want achieved, not how you want it achieved." (S&B 2018, §3.2, p. 54)
- "For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as taking its opponent's pieces or gaining control of the center of the board." (S&B 2018, §3.2, p. 54)

---

## §6 The Bellman equations

### Mechanism (plain words)
- **A state's value = the reward you get now + the (discounted) value of where you land.** This is averaged over what the policy does and over where the slippery wheels take you. That is the Bellman expectation equation; it follows from G_t = R_{t+1} + γG_{t+1}.
- **V vs Q.**
  - V^π(s) scores a state.
  - Q^π(s,a) scores "take a now, then follow π".
  - They are linked by V^π(s) = Σ_a π(a|s) Q^π(s,a).
- **Expectation vs optimality.**
  - For a fixed policy we *average* over its actions. The expectation equation is a linear system, one equation per state.
  - For the best possible behaviour we take the *max* over actions. The optimality equation is non-linear because of the max.
- **Greedy on V\* is optimal.**
  - Once you know V*, a one-step look-ahead picks optimal actions. That look-ahead needs the model p.
  - Once you know Q*, you just take argmax_a Q*(s,a); no model is needed.
  - This is why model-free methods (Q-learning, DQN) learn Q.
- **Backup diagrams** (S&B §3.5).
  - Open circle = state; solid dot = state–action pair.
  - For V^π, the root state branches into actions (weighted by π). Each action branches into (r, s′) outcomes (weighted by p). Values flow *back up* to the root.
  - The optimality diagrams (S&B Fig. 3.4) add an arc at the action level, meaning "max" instead of "average".
- **Principle of optimality** (Bellman). The tail of an optimal plan is itself optimal from wherever it starts. This is what lets one-step equations describe whole futures.
- **Why solving works: contraction.** The Bellman backup operator T pulls any two value functions closer together by at least a factor γ (in the max norm). Repeating it therefore converges to a unique fixed point, from any start.

### Key formulas (S&B equation numbers; ours: V, Q, π, γ, s, a, r, s′)
- Bellman expectation, V (S&B 3.14): **V^π(s) = Σ_a π(a|s) Σ_{s′,r} p(s′,r|s,a) [ r + γ V^π(s′) ]**.
- Bellman expectation, Q: Q^π(s,a) = Σ_{s′,r} p(s′,r|s,a) [ r + γ Σ_{a′} π(a′|s′) Q^π(s′,a′) ].
- Q from V (4.6): Q^π(s,a) = Σ_{s′,r} p(s′,r|s,a)[r + γV^π(s′)].
- Optimal values (3.15, 3.16): V*(s) = max_π V^π(s); Q*(s,a) = max_π Q^π(s,a).
- Link between them (3.17): Q*(s,a) = E[R_{t+1} + γV*(S_{t+1}) | S_t = s, A_t = a].
- Bellman optimality, V (3.19): **V\*(s) = max_a Σ_{s′,r} p(s′,r|s,a) [ r + γ V\*(s′) ]**.
- Bellman optimality, Q (3.20): **Q\*(s,a) = Σ_{s′,r} p(s′,r|s,a) [ r + γ max_{a′} Q\*(s′,a′) ]**.
- R&N 3rd-ed. form (state reward): U(s) = R(s) + γ max_a Σ_{s′} P(s′|s,a) U(s′), "the Bellman equation" (R&N eq. 17.5) [UNVERIFIED equation number].
- **Matrix form for a fixed policy:** V^π = r^π + γP^π V^π, so V^π = (I − γP^π)^{−1} r^π.
  - This is well defined if γ < 1 or every state eventually terminates under π.
  - S&B §4.1: "The existence and uniqueness of v_π are guaranteed as long as either γ < 1 or eventual termination is guaranteed from all states under the policy π."
- **Contraction** (Denardo 1967, "Contraction mappings in the theory underlying dynamic programming", SIAM Review 9(2):165–177). For T*V(s) = max_a Σ p [r + γV(s′)]: ‖T*V − T*V′‖_∞ ≤ γ‖V − V′‖_∞. The same holds for T^π. Consequences [DERIVED from the contraction]:
  - ‖V_k − V*‖_∞ ≤ γ^k ‖V_0 − V*‖_∞, i.e. value iteration converges geometrically.
  - **Stopping rule:** if ‖V_{k+1} − V_k‖_∞ < ε(1−γ)/γ, then ‖V_{k+1} − V*‖_∞ < ε.
    - Proof: ‖V_{k+1}−V*‖ ≤ γ‖V_k−V*‖ ≤ γ‖V_k−V_{k+1}‖ + γ‖V_{k+1}−V*‖.
    - R&N 3rd ed. §17.2 states this rule [UNVERIFIED equation number].
  - **Sweeps to shrink the error 1,000×:** ln(1000)/ln(1/γ) = **66 sweeps at γ = 0.9, 688 at γ = 0.99**.
- **Greedy-policy loss** (Singh & Yee 1994, *Machine Learning* 16:227–233). If ‖V − V*‖_∞ ≤ ε, the greedy policy π w.r.t. V satisfies V*(s) − V^π(s) ≤ 2γε/(1−γ) for all s. Szepesvári's RL-theory lecture notes show the bound is tight.

### Numbers with sources
- **Bellman's equation and DP.** S&B §1.7: the value function, or "optimal return function", defines "a functional equation, now often called the Bellman equation. The class of methods for solving optimal control problems by solving this equation came to be known as dynamic programming (Bellman, 1957a)."
- "For finite MDPs, the Bellman optimality equation for v* (3.19) has a unique solution." (S&B §3.6, p. 64)
- **S&B Example 3.5/3.8 gridworld:** V^random(A) = 8.8, less than the immediate reward of 10; V*(A) = 24.4 (Figs. 3.2, 3.5). Reproduced [DERIVED].
- **Q-functions predate Watkins.** S&B's ch. 3 bibliographic remarks note that Shannon (1950) suggested a function h(P, M) for chess, and that Denardo (1967) used action values in his contraction-mapping treatment of DP.

### Worked example: the value of one cell, by hand [DERIVED] (Rover's world, converged values)
**The start cell (1,1).** Bellman optimality: V(1,1) = −0.04 + the maximum, over the four actions, of the slip-weighted neighbour values. We use the 3-dp values from the table; a bump leaves Rover at (1,1).
- **Up**: 0.8·V(1,2) + 0.1·V(1,1) [Left slip bumps] + 0.1·V(2,1) [Right slip] = 0.8·0.762 + 0.1·0.705 + 0.1·0.655 = 0.6096 + 0.0705 + 0.0655 = **0.7456**
- Left: 0.8·0.705 + 0.1·0.762 + 0.1·0.705 = 0.7107
- Down: 0.8·0.705 + 0.1·0.705 + 0.1·0.655 = 0.7000
- Right: 0.8·0.655 + 0.1·0.762 + 0.1·0.705 = 0.6707
- Result: V(1,1) = −0.04 + 0.7456 = **0.7056 ≈ 0.705** ✓. The exact value is 0.7053; the 0.0003 gap is 3-dp rounding. Best action: **Up**.

**The decision that makes the policy interesting, (3,1):**
- Q(Left) = 0.8·0.655 + 0.1·0.660 + 0.1·0.611 = **0.6511**
- Q(Up) = 0.8·0.660 + 0.1·0.655 + 0.1·0.388 = **0.6323**. Up passes next to the puddle, and its Right slip lands on (4,1), the cell under the puddle.
- Q(Down) = 0.5931; Q(Right) = 0.4375.
- V(3,1) = −0.04 + 0.6511 = 0.611 ✓.

**Next to the charger, (3,3):** Q(Right) = 0.8·1 + 0.1·0.918 (Up bumps) + 0.1·0.660 (Down) = 0.9578, so V = 0.918 ✓.

Animation: the cell's value is assembled from three neighbours, pulled with coloured weights 0.8, 0.1 and 0.1, minus a −0.04 toll.

### Common misconceptions
- "The Bellman equation is an algorithm." It is a consistency condition. Turning it into an update gives policy evaluation and value iteration (§7), TD (§9) and Q-learning (§10).
- "Knowing V\* is enough to act." Only with a model, because the one-step look-ahead needs p. With Q* you can act without a model.
- "The optimal policy is unique." V* is unique, but several policies can share it when actions tie. R&N 4th ed. reportedly shows such a tie at (3,1); our computation does not reproduce it (see §5).
- "Contraction needs γ < 1." The γ-contraction argument does. But undiscounted episodic problems (stochastic shortest path; Bertsekas & Tsitsiklis 1996) still converge when every policy considered terminates.
  - Rover's world at γ = 1 converges, with the error shrinking by about ×0.44–0.5 per sweep [DERIVED].
  - The rate is set by the spectral radius of the optimal policy's non-terminal transition matrix, 0.437.
- "Bellman equations are for discrete worlds only." Bellman's own work was largely continuous control (Hamilton–Jacobi–Bellman); see S&B §1.7.

### Quotable
- Principle of optimality: "An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision." (Bellman 1957, *Dynamic Programming*, Ch. III §3, as cited by Wikipedia "Bellman equation"; page 83 per a secondary citation.) [UNVERIFIED against the book. The 1954 *Bull. AMS* paper "The theory of dynamic programming" also states it; not checked.]
- "Equation (3.14) is the Bellman equation for v_π. It expresses a relationship between the value of a state and the values of its successor states." (S&B 2018, §3.5)
- "any policy that is greedy with respect to the optimal evaluation function v* is an optimal policy." (S&B 2018, §3.6, p. 64)
- "Hence, a one-step-ahead search yields the long-term optimal actions." (S&B 2018, §3.6, p. 65)
- On the name: "Thus, I thought dynamic programming was a good name. It was something not even a Congressman could object to. So I used it as an umbrella for my activities." (Bellman, autobiography *Eye of the Hurricane*, 1984, p. 159, as quoted in Dreyfus 2002, *Operations Research* 50(1):48–51.) [UNVERIFIED: the primary was not accessible, and a blog post disputes the chronology of the Wilson story.]

---

## §7 Dynamic programming: policy evaluation, value iteration, policy iteration

### Mechanism (plain words)
DP means solving the Bellman equations when you **know the model** (p and the rewards). It is planning, not learning.
- **Iterative policy evaluation.** Start with any V (e.g. 0). Sweep through all states, replacing each value by the Bellman *expectation* backup. Repeat. This converges to V^π.
- **Policy improvement.** Given V^π, act greedily: π′(s) = argmax_a Q^π(s,a). The policy improvement theorem guarantees π′ is at least as good everywhere.
- **Policy iteration** (Howard 1960): evaluate → improve → evaluate → …
  - Each new policy is strictly better until it is optimal.
  - There are finitely many deterministic policies, so it stops after finitely many rounds, usually very few.
- **Value iteration.** Do just one sweep of evaluation between improvements. This is the same as applying the Bellman *optimality* backup repeatedly. Values spread outward from the rewards, sweep by sweep.
- **In-place (Gauss–Seidel) vs two-array (Jacobi) sweeps.** In-place uses new values immediately and usually converges faster.
- **Asynchronous DP.** Update states in any order, as long as every state keeps getting updated.
- **Generalised policy iteration (GPI).** Evaluation and improvement interact at any granularity. This is the template for almost all RL.
- **Limits.** Each sweep touches every state (the curse of dimensionality).
  - Model-free methods (MC, TD) remove the need for p.
  - Function approximation (§12) removes the need to list the states.

### Key formulas (S&B equation numbers)
- Policy evaluation update (4.5): V_{k+1}(s) = Σ_a π(a|s) Σ_{s′,r} p(s′,r|s,a)[r + γV_k(s′)].
- Policy improvement theorem (4.7–4.8): if Q^π(s, π′(s)) ≥ V^π(s) for all s, then V^{π′}(s) ≥ V^π(s) for all s.
- Greedy improvement (4.9): π′(s) = argmax_a Σ_{s′,r} p(s′,r|s,a)[r + γV^π(s′)].
- Value iteration (4.10): **V_{k+1}(s) = max_a Σ_{s′,r} p(s′,r|s,a)[r + γV_k(s′)]**. In S&B's words, "value iteration is obtained simply by turning the Bellman optimality equation into an update rule." (§4.4, p. 83)
- Stop when max_s |V_{k+1}(s) − V_k(s)| falls below a small threshold. S&B call that threshold θ; in this course θ means policy parameters, so call it "tol".
- Policy iteration sequence (S&B §4.3): π_0 →E V^{π_0} →I π_1 →E V^{π_1} →I π_2 → … →I π* →E V*.
- **Complexity** (S&B §4.7).
  - DP finds an optimal policy in time polynomial in the numbers of states n and actions k, although there are k^n deterministic policies.
  - Policy iteration is *strongly* polynomial for a fixed discount rate (Ye 2011, *Math. of OR* 36(4):593–603).
  - Rover's world has 4^9 = **262,144** deterministic policies [DERIVED]; policy iteration needs 3–4 improvements (below).

### Numbers with sources
**S&B Example 4.1 (4×4 gridworld).**
- Setup: 14 non-terminal states; the terminal corners are one terminal state shown twice; reward −1 on every transition; undiscounted; equiprobable random policy.
- Values from iterative policy evaluation (Fig. 4.1, printed to 2 significant digits):
  - k = 1: all −1.0.
  - k = 2: −1.7 next to the terminals, −2.0 elsewhere. The exact value of the −1.7 cells is −1.75.
  - k = 3: −2.4, −2.9, −3.0.
  - k = 10: −6.1, −7.7, −8.4, −9.0.
- **k = ∞ (= V^π).** The values are exact integers, reproduced [DERIVED]:

  | 0 | −14 | −20 | −22 |
  |---|---|---|---|
  | −14 | −18 | −20 | −20 |
  | −20 | −20 | −18 | −14 |
  | −22 | −20 | −14 | 0 |

- "The final estimate is in fact v_π, which in this case gives for each state the negation of the expected number of steps from that state until termination." (S&B Example 4.1)
- The greedy policy w.r.t. these random-policy values is already optimal. Fig. 4.1 caption: "all policies after the third iteration, are optimal".
- [DERIVED] Sweeps until the largest change < 1e-4: **173 two-array sweeps vs 114 in-place sweeps** (row-major order).

**S&B Example 4.2 (Jack's Car Rental).**
- Setup:
  - two locations, at most 20 cars each;
  - at most 5 cars moved overnight, at $2 per car; $10 per rental;
  - Poisson rental requests with λ = 3 and 4, returns with λ = 3 and 2;
  - γ = 0.9.
- Policy iteration from "never move any cars" (π_0) produces π_1 … π_4, and **π_4 is optimal**: **4 improvement steps**. Fig. 4.2: "Each successive policy is a strict improvement over the previous policy, and the last policy is optimal."
- The final value function in the figure ranges from about 420 to 612 [read from the figure's axis labels; treat as approximate].

**Other S&B statements:**
- "Policy iteration often converges in surprisingly few iterations" (§4.3, p. 82).
- Finite termination: "Because a finite MDP has only a finite number of deterministic policies, this process must converge to an optimal policy and the optimal value function in a finite number of iterations." (§4.3)
- Value iteration "formally requires an infinite number of iterations to converge exactly to v*. In practice, we stop once the value function changes by only a small amount in a sweep." (§4.4, p. 83)
- Scale: "the game of backgammon has over 10^20 states. Even if we could perform the value iteration update on a million states per second, it would take over a thousand years to complete a single sweep." (§4.5, p. 85)
  - [DERIVED] 10^20 states ÷ 10^6 per second = 10^14 s ≈ **3.2 million years**, so "over a thousand years" is a big understatement.
- "In practice, DP methods can be used with today's computers to solve MDPs with millions of states." (§4.7, p. 87)

**History:**
- The policy improvement theorem and policy iteration "are due to Bellman (1957a) and Howard (1960)" (S&B ch. 4 remarks). Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press.
- Treating value iteration as truncated policy iteration follows Puterman & Shin (1978), "modified policy iteration".
- The phrase "curse of dimensionality" is due to Bellman (1957a) (S&B §4.7 remarks).

### Worked example 1: value iteration on Rover's world [DERIVED]
Setup: two-array sweeps; V_0 = 0 for non-terminals; terminals fixed at ±1; γ = 1; R = −0.04.

| sweep | row 3 (cols 1–3) | row 2 (cols 1, 3) | row 1 (cols 1–4) | max change |
|---|---|---|---|---|
| 1 | −0.040, −0.040, **0.760** | −0.040, −0.040 | −0.040, −0.040, −0.040, −0.040 | 0.760 |
| 2 | −0.080, **0.560**, 0.832 | −0.080, **0.464** | −0.080 ×4 | 0.600 |
| 3 | **0.392**, 0.738, 0.890 | −0.120, 0.572 | −0.120, −0.120, **0.315**, −0.120 | 0.472 |
| 4 | 0.577, 0.819, 0.906 | 0.250, 0.629 | −0.160, 0.188, 0.394, 0.100 | 0.370 |
| 5 | 0.698, 0.849, 0.914 | 0.472, 0.648 | 0.162, 0.313, 0.492, 0.185 | 0.323 |
| ∞ | 0.812, 0.868, 0.918 | 0.762, 0.660 | 0.705, 0.655, 0.611, 0.388 | 0 |

- **Checks.**
  - Sweep 1 at (3,3): −0.04 + 0.8·(+1) + 0.1·0 + 0.1·0 = 0.760.
  - Sweep 2 at (3,2): −0.04 + [0.8·0.76 + 0.1·(−0.04) + 0.1·(−1)] = −0.04 + 0.504 = 0.464.
- The value "wave" leaves the charger and reaches the start cell (1,1) with a positive value at sweep 5.
- **Convergence.**
  - The largest change drops below 1e-2 at sweep 13, below 1e-3 at sweep **19**, and below 1e-4 at sweep **22**.
  - The max error to V* crosses the same thresholds at sweeps 15, 19 and 22.
  - Fully converged (1e-12) after 46 sweeps.
- **The policy is right long before the values are.** The greedy policy is already optimal from **sweep 12**, when the values are still off by up to 0.026.
- **In-place sweeps** (row by row from the top-left): the max error falls below 1e-4 after **14** sweeps instead of 22.
- The error shrinks by about ×0.44–0.5 per sweep even though γ = 1, because Rover's episodes end (spectral radius 0.437, see §6).

### Worked example 2: policy iteration on Rover's world [DERIVED] (exact linear-solve evaluation)
- Start from **π_0 = "always Right"**. This is a proper policy: every state eventually terminates. V^{π_0}(1,1) = −1.396.
- π_1: (1,1), (1,2), (3,1), (3,2) and (4,1) switch to ↑. V(1,1) = 0.676.
- π_2: (2,1) switches to ←, (4,1) to ←. V(1,1) = 0.705.
- π_3: (3,1) switches to ←. This is π*.
- Evaluating π_3 and finding no change ends the run: **3 improvements, 4 evaluations**. Starting from "always Up" takes 4 improvements and 5 evaluations.
- **Warning for code.** Starting from "always Left" or "always Down" is *improper*: Rover never terminates from some cells, so with γ = 1 the evaluation matrix is singular. Start from a proper policy or use γ < 1. (Our first run hung for exactly this reason.)

### Common misconceptions
- "DP is reinforcement learning." DP plans with a known model; RL learns from experience. DP is the theory that RL approximates.
- "Value iteration must converge before the policy is right." On Rover's world the greedy policy is optimal from sweep 12 on, while the values are still about 0.03 off.
- "Policy iteration is slow because each step solves a linear system." Each step is expensive, but there are very few of them: 4 on Jack's car rental, 3–4 on Rover's world.
- "DP is hopeless for big problems." Millions of states are routine (S&B §4.7). What is hopeless is enumerating the k^n policies.
- "Initial values don't matter." They do not change the fixed point, but they change how many sweeps you need.

### Quotable
- "Policy iteration often converges in surprisingly few iterations" (S&B 2018, §4.3, p. 82)
- "value iteration is obtained simply by turning the Bellman optimality equation into an update rule." (S&B 2018, §4.4, p. 83)
- "Almost all reinforcement learning methods are well described as GPI." (S&B 2018, §4.6, p. 86)
- "DP is sometimes thought to be of limited applicability because of the curse of dimensionality … Large state sets do create difficulties, but these are inherent difficulties of the problem, not of DP as a solution method." (S&B 2018, §4.7, p. 87)
- "This in-place algorithm also converges to v_π; in fact, it usually converges faster than the two-array version" (S&B 2018, §4.1, p. 75)

---

## §8 Monte Carlo methods (S&B ch. 5)

### Mechanism (plain words)
- **No map, just play.** Without p, Rover plays whole episodes and averages the returns that actually followed each state (or state–action pair). As the averages pile up, they converge to the true values (law of large numbers).
- **First-visit vs every-visit.**
  - If a state is visited twice in one episode, first-visit MC uses only the return after the first visit; every-visit MC averages all of them.
  - Both converge to V^π.
  - First-visit returns are i.i.d., so the error's standard deviation falls as 1/√n.
- **No bootstrapping.** Each state's estimate uses only real returns, never other estimates.
  - The estimates are unbiased but have high variance, and you must wait for the episode to end.
  - So MC only works for episodic tasks.
- **Control needs Q, and Q needs exploration.** Without a model, V is not enough to pick actions, so MC control learns Q(s,a). But a greedy policy never tries some actions, so their Q never improves. Two fixes:
  - **Exploring starts (ES):** every episode starts from a random state–action pair. Fine in simulation (blackjack), unrealistic for a real robot.
  - **ε-soft policies:** every action keeps probability ≥ ε/|A(s)|.
    - ε-greedy gives the greedy action 1 − ε + ε/|A(s)| and each other action ε/|A(s)|.
    - This is on-policy control. It finds the best ε-soft policy, not the best policy.
- **Off-policy MC.** Learn about a target policy π from episodes generated by a behaviour policy b. Weight each return by the importance-sampling ratio: how much more likely the trajectory is under π than under b.
  - Ordinary IS is unbiased but can have infinite variance.
  - Weighted IS is biased (the bias → 0) and has much lower variance.
  - This idea returns in PPO/GRPO as the probability ratio (§18, §23).

### Key formulas (S&B ch. 5)
- **First-visit MC prediction:** V(s) ← average of G_t over the first visits t to s.
  - Incremental form: V(s) ← V(s) + (1/N(s))(G_t − V(s)).
  - For non-stationary problems, use a constant step α (constant-α MC): V(S_t) ← V(S_t) + α[G_t − V(S_t)] (S&B eq. 6.1).
- Returns are computed backwards through an episode: G ← γG + R_{t+1}.
- **ε-greedy probabilities:** π(a|s) = 1 − ε + ε/|A(s)| for the greedy a, and ε/|A(s)| otherwise. **ε-soft** means π(a|s) ≥ ε/|A(s)| for all a.
- **Importance-sampling ratio** (5.3): ρ_{t:T−1} = Π_{k=t}^{T−1} π(A_k|S_k) / b(A_k|S_k).
  - The transition probabilities p cancel, so no model is needed.
  - E[ρ_{t:T−1}G_t | S_t = s] = V^π(s) (5.4).
- **Ordinary IS** (5.5): V(s) = Σ_{t∈𝒯(s)} ρ_{t:T(t)−1} G_t / |𝒯(s)|.
- **Weighted IS** (5.6): V(s) = Σ ρ_{t:T(t)−1} G_t / Σ ρ_{t:T(t)−1}.

### Numbers with sources (S&B 2018)
**Example 5.1, Blackjack.**
- Rules: infinite deck; the dealer sticks on 17 or more; rewards +1/−1/0 for win/lose/draw; γ = 1.
- State = player sum (12–21) × dealer's showing card (ace–10) × usable ace (yes/no) = **200 states**.
- Policy evaluated: stick only on 20 or 21.
- Fig. 5.1 shows the MC estimates **"After 10,000 episodes" vs "After 500,000 episodes"**. Usable-ace states are noisier because they are rarer.
- "In any event, after 500,000 games the value function is very well approximated." (p. 93)

**Example 5.3, Solving blackjack with Monte Carlo ES** (Fig. 5.2, p. 100).
- Exploring starts: the dealer's card, the player's sum and the usable ace are chosen uniformly. Initial policy: stick on 20/21. Q initialised to 0.
- The optimal policy, as read from Fig. 5.2 (300-dpi render):
  - **No usable ace:** stick on ≥ 12 vs dealer 4–6; on ≥ 13 vs dealer 2–3; on ≥ 17 vs dealer 7–10 or ace.
  - **Usable ace:** stick on ≥ 18 vs dealer 2–8; on ≥ 19 vs dealer 9, 10 or ace.
- Book: "This policy is the same as the 'basic' strategy of Thorp (1966) with the sole exception of the leftmost notch in the policy for a usable ace, which is not present in Thorp's strategy. We are uncertain of the reason for this discrepancy, but confident that what is shown here is indeed the optimal policy for the version of blackjack we have described."

**Example 5.4, off-policy estimation of one blackjack state** (p. 106).
- The state: the dealer shows a deuce; the player's sum is 13 with a usable ace.
- Behaviour policy: hit or stick 50/50. Target policy: stick only on 20/21.
- "The value of this state under the target policy is approximately **−0.27726** (this was determined by separately generating one-hundred million episodes using the target policy and averaging their returns)."
- Both IS methods were close after 1,000 off-policy episodes. Fig. 5.3 averages **100 runs × 10,000 episodes**; weighted IS has much lower error early on.
- pdftotext drops the minus sign; we checked the sign on the rendered page.
- [DERIVED] Our own 4,000,000-episode simulation of that state under the target policy gives **−0.27755 ± 0.00047**, consistent with −0.27726.

**Convergence.**
- "Both first-visit MC and every-visit MC converge to v_π(s) as the number of visits (or first visits) to s goes to infinity … the standard deviation of its error falls as 1/√n" (§5.1, p. 93). First-visit MC is "dating back to the 1940s".
- Monte Carlo ES convergence was an open problem: "Convergence to this optimal fixed point seems inevitable … but has not yet been formally proved. In our opinion, this is one of the most fundamental open theoretical questions in reinforcement learning (for a partial solution, see Tsitsiklis, 2002)." (§5.3, p. 99)
- Later partial answers:
  - Wang, Yuan, Shao & Ross (arXiv 2002.03585, 2020), "On the Convergence of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning".
  - Liu (arXiv 2007.10916, 2020), for the undiscounted (stochastic shortest path) case.

### Worked example [DERIVED] (Rover's world, optimal policy, scripts/mc43.py)
1. **First-visit vs every-visit on one episode.**
   - Rover at (1,1) tries Up but slips Right to (2,1). It goes Left back to (1,1), then Up, Up, Right, Right, Right to the charger: **7 moves** in all.
   - Return after the first visit to (1,1) (t = 0): 1 − 7 × 0.04 = **0.72**. After the second visit (t = 2): 1 − 5 × 0.04 = **0.80**.
   - First-visit MC records **0.72**. Every-visit MC records both, mean **0.76**. The true V(1,1) is 0.705.
2. **How noisy is one return?** Under the optimal policy, the return from (1,1) has mean 0.705 and **SD 0.251**.
   - 1.4% of episodes end in the puddle, with G < 0.
   - 32.7% take the perfect 5-move path, with G = 0.80.

   Averaging n episodes, the estimate's SD is 0.251/√n:

   | episodes n | 10 | 100 | 1,000 | 10,000 |
   |---|---|---|---|---|
   | SD of the first-visit estimate (simulated, 200 repeats) | 0.075 | 0.025 | 0.0075 | 0.0024 |

   Ten times the precision costs a hundred times the episodes. This is the variance that TD learning (§9) attacks.
3. **ε-greedy numbers for Rover** (4 actions, ε = 0.1): the greedy action gets 1 − 0.1 + 0.025 = **0.925**; each other action gets **0.025**.

### Common misconceptions
- "Monte Carlo means random search." Here it means *averaging sampled returns*.
- "MC needs a model." No: it needs only episodes (real or simulated). It is DP that needs p.
- "MC works on any task." It needs episodes that end; it cannot learn mid-episode.
- "First-visit and every-visit give different answers." Both converge to V^π. They differ in bias and variance on small samples: every-visit is biased for finite n, but it extends more naturally to function approximation and eligibility traces (S&B §5.1).
- "Unbiased ordinary importance sampling is always the better estimator." Its variance can be infinite (S&B Example 5.5); weighted IS is preferred in practice.
- "Exploring starts are how you explore." They are a convenience of simulators. ε-soft policies or off-policy learning are the general fixes.

### Quotable
- "Monte Carlo methods sample and average returns for each state–action pair much like the bandit methods we explored in Chapter 2 sample and average rewards for each action." (S&B 2018, ch. 5 intro)
- "An important fact about Monte Carlo methods is that the estimates for each state are independent. The estimate for one state does not build upon the estimate of any other state, as is the case in DP. In other words, Monte Carlo methods do not bootstrap as we defined it in the previous chapter." (S&B 2018, §5.1, p. 95)
- "the computational expense of estimating the value of a single state is independent of the number of states." (S&B 2018, §5.1, p. 95)
- "Ordinary importance sampling is unbiased whereas weighted importance sampling is biased (though the bias converges asymptotically to zero). On the other hand, the variance of ordinary importance sampling is in general unbounded" (S&B 2018, §5.5)
- "In any event, after 500,000 games the value function is very well approximated." (S&B 2018, Example 5.1, p. 93)

---

## §9 Temporal-difference learning

### Mechanism (plain words)
- Monte Carlo (§8) waits until the episode ends, then moves V(s) toward the **actual return** G.
- TD learning moves V(s) after **one step**, toward the reward just received plus the current estimate of the next state: r + γV(s′). It updates "a guess from a guess". Sutton & Barto call this **bootstrapping**.
- The gap between that one-step target and the old estimate is the **TD error δ**. This is the course's PINK learning signal: a spark between two neighbouring estimates.
- **Model-free like MC, bootstrapping like DP.** It needs no transition probabilities, it learns online during the episode, and it works on continuing tasks with no episodes at all.
- **Bias vs variance.**
  - The MC target G is unbiased, but it sums many random rewards and transitions, so its variance is high.
  - The TD target r + γV(s′) depends on one random step, so its variance is low. It is biased, though, because V(s′) is still a guess.
  - In practice TD usually learns faster on stochastic tasks. Sutton & Barto, however, say this has never been proved in general (see Numbers).
- **Batch view.** Replay a fixed set of episodes until convergence:
  - batch MC converges to the sample averages of the observed returns (minimum error on the training data);
  - batch TD(0) converges to the **certainty-equivalence** estimate, the values that would be exactly right for the maximum-likelihood Markov model of the data.
  - When the process really is Markov, TD's answer generalises better.
- **n-step returns and TD(λ)** sit between the two extremes.
  - n-step: use n real rewards, then bootstrap. n = 1 is TD(0); n = ∞ is MC.
  - TD(λ) averages all n-step returns with weights (1−λ)λ^(n−1). Implemented online, it uses an **eligibility trace**: a fading memory of recently visited states, or of recently active weights. Each δ updates all of them in proportion.
  - λ = 0 gives TD(0); λ = 1 behaves like MC.

### Key formulas (S&B 2nd ed. notation → ours: S_t→s, A_t→a, R_{t+1}→r, v̂/V→V, w→θ, z_t→z)
- Constant-α MC (S&B eq. 6.1): V(s_t) ← V(s_t) + α [G_t − V(s_t)]
- TD(0) (S&B eq. 6.2): V(s_t) ← V(s_t) + α [r_{t+1} + γV(s_{t+1}) − V(s_t)], with V(terminal) = 0
- TD error (S&B eq. 6.5): δ_t = r_{t+1} + γV(s_{t+1}) − V(s_t)
- MC error as a sum of TD errors, if V is held fixed during the episode (S&B eq. 6.6): G_t − V(s_t) = Σ_{k=t}^{T−1} γ^{k−t} δ_k
- n-step return (S&B eq. 7.1): G_{t:t+n} = r_{t+1} + γr_{t+2} + … + γ^{n−1}r_{t+n} + γ^n V(s_{t+n})
- λ-return (S&B eq. 12.2): G^λ_t = (1−λ) Σ_{n=1}^∞ λ^{n−1} G_{t:t+n}
- Semi-gradient TD(λ) with an accumulating trace (S&B eqs. 12.5–12.7):
  - z_{−1} = 0
  - z_t = γλ z_{t−1} + ∇V(s_t, θ)
  - δ_t = r_{t+1} + γV(s_{t+1}, θ) − V(s_t, θ)
  - θ ← θ + α δ_t z_t
- Sutton (1988) eq. 4, his notation (P_t = prediction, w = weights): Δw_t = α (P_{t+1} − P_t) Σ_{k=1}^{t} λ^{t−k} ∇_w P_k.
  - Tesauro (1995) prints the same rule: "w[t+1] − w[t] = alpha (Y[t+1] − Y[t]) Sum(k=1 to t) lambda^(t−k) grad[w] Y[k]".

### Numbers with sources

**S&B Example 6.2, Random Walk (5 states)** (Sutton & Barto 2018, *Reinforcement Learning: An Introduction*, 2nd ed., p. 125; http://incompleteideas.net/book/RLbook2020.pdf)
- **Setup.**
  - States A B C D E, plus a terminal at each end.
  - Every episode starts in C. Each step goes left or right with probability ½.
  - Reward +1 on terminating at the right end; every other reward is 0. Undiscounted (γ = 1).
  - Example trajectory from the book: "C, 0, B, 0, C, 0, D, 0, E, 1".
- **True values.** "The true values of all the states, A through E, are 1/6, 2/6, 3/6, 4/6, and 5/6." Also: "the true value of the center state is vπ(C) = 0.5." Each value is the probability of exiting on the right.
- **Left graph.** Values after 0, 1, 10 and 100 episodes of a single TD(0) run, with α = 0.1. "The estimates after 100 episodes are about as close as they ever come to the true values—with a constant step-size parameter (α = 0.1 in this example), the values fluctuate indefinitely".
- **Right graph.** Empirical RMS error over the five states, averaged over **100 runs**, for 0–100 episodes. All values are initialised to V(s) = 0.5.
  - Curve labels, read from the rendered figure: **TD α = .05, .1, .15** and **MC α = .01, .02, .03, .04**.
  - Book: "The TD method was consistently better than the MC method on this task."
  - Values read from the figure (approximate): all curves start at ≈ 0.236.
    - TD α = .05 falls steadily to ≈ 0.035 at 100 episodes.
    - TD α = .1 levels off at ≈ 0.05–0.06.
    - TD α = .15 drops fastest to ≈ 0.055 by about episode 20–25, then **rises again** to ≈ 0.07. Exercise 6.5 asks why.
    - The MC curves end at ≈ 0.08–0.10.
- **Batch training, Example 6.3 / Figure 6.2.** "After each new episode, all episodes seen so far were treated as a batch", replayed until convergence; RMS averaged over 100 repetitions. "the batch TD method was consistently better than the batch Monte Carlo method."
- **Example 6.4, "You are the Predictor"** (8 episodes: A,0,B,0 once; B,1 six times; B,0 once).
  - Everyone agrees V(B) = 3/4.
  - Batch TD gives V(A) = 3/4.
  - Batch MC gives V(A) = 0, which has zero error on the training data. The book still expects the TD answer to "produce lower error on future data".
- **Convergence (S&B §6.2).** "For any fixed policy π, TD(0) has been proved to converge to vπ, in the mean for a constant step-size parameter if it is sufficiently small, and with probability 1 if the step-size parameter decreases according to the usual stochastic approximation conditions".
- **Speed is not proved.** "no one has been able to prove mathematically that one method converges faster than the other." "In practice, however, TD methods have usually been found to converge faster than constant-α MC methods on stochastic tasks".

**Origin: Sutton 1988** ("Learning to predict by the methods of temporal differences", *Machine Learning* 3:9–44; http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
- The original walk had **7 states, A–G**. A and G are terminal; B–F are non-terminal; "Every walk begins in the center state D."
- Outcome z = 0 at A and z = 1 at G. Ideal predictions: 1/6, 1/3, 1/2, 2/3, 5/6 for B–F.
- Experimental design: "100 training sets, each consisting of 10 sequences".
- λ values tested: 1 (Widrow–Hoff supervised learning), 0, 0.1, 0.3, 0.5, 0.7 and 0.9.
- Under repeated presentations, error "was best at λ = 0".
- S&B's 5-state version is the same chain with the terminals unlabeled.

**n-step TD, S&B Example 7.1 / Figure 7.2**
- 19-state random walk "with a −1 outcome on the left, all values initialized to 0". The minus sign was checked on the rendered page; pdftotext drops it.
- Measure: RMS error "averaged over the first 10 episodes and 100 repetitions". "methods with an intermediate value of n worked best."
- Read from the figure (approximate):
  - best is **n = 4**, ≈ 0.27 at α ≈ 0.4;
  - n = 8 ≈ 0.28 at α ≈ 0.2; n = 2 ≈ 0.28 at α ≈ 0.6;
  - n = 1 (TD(0)) ≈ 0.35 at α ≈ 0.8;
  - n ≥ 128 (near-MC) stays ≥ ≈ 0.45.
- True values of the 19-state walk: −0.9, −0.8, …, +0.9 [DERIVED: v(i) = 2·(i/20) − 1].

**TD-Gammon** (Tesauro 1995, "Temporal difference learning and TD-Gammon", *Communications of the ACM* 38(3):58–68; HTML transcription with ACM permission: https://bkgm.com/articles/tesauro/tdl.html. Also S&B §16.1)
- **Method.**
  - A neural network value function trained by **TD(λ) self-play**.
  - Moves are chosen greedily over the ~20 afterstates of each dice roll.
  - Inputs: 198 units in the first version (S&B).
  - Output: 4 units (White/Black × normal win/gammon). Tesauro: "Y[t] is a four component vector corresponding to the four possible outcomes of either White or Black winning either a normal win or a gammon."
- **State space.** Tesauro: "the enormous number of possible states (estimated at over 10 to the power of 20)". S&B: "effective branching factor of about 400".
- **Knowledge-free version.** "The best performance obtained in the raw-encoding experiments was for a network with 40 hidden units that was trained for a total of 200,000 games. This network achieved a strong intermediate level of play approximately equal to Neurogammon." S&B's Table 16.1 lists TD-Gammon 0.0 as 40 hidden units, **300,000** games, "tied for best". The sources disagree (200k vs 300k); see Disagreements.
- **Tesauro 1995, Table 1** (results against world-class humans; read from the table image):

| Program | Training games | Opponents | Result |
|---|---|---|---|
| TDG 1.0 | 300,000 | Robertie, Davis, Magriel | −13 pts / 51 games (−0.25 ppg) |
| TDG 2.0 | 800,000 | Goulding, Woolsey, Snellings, Russell, Sylvester | −7 pts / 38 games (−0.18 ppg) |
| TDG 2.1 | 1,500,000 | Robertie | −1 pt / 40 games (−0.02 ppg) |

  - Table caption: "Version 1.0 used 1-ply search for move selection; versions 2.0 and 2.1 used 2-ply search. Version 2.0 had 40 hidden units; versions 1.0 and 2.1 had 80 hidden units."
- **S&B Table 16.1** adds TD-Gammon 3.0: 80 hidden units, 1,500,000 games, vs Kazaros, **+6 pts / 20 games**.
  - The S&B text says "TD-Gammon 3.0 and 3.1, used 160 hidden units and a selective three-ply search". That contradicts its own table (80). See Disagreements.
- **Headline sentence.** "the latest version of the program (2.1) achieved near-parity to Bill Robertie in a recent 40-game test session. Robertie actually trailed the entire session, and only in the very last game was he able to pull ahead for an extremely narrow 1-point victory." (Tesauro 1995)
- **Comparisons.** "The best previous commercial program scored -0.66 points per game on this scale." Berliner's BKG (1979): "Robertie estimates that it would have scored in the range of -0.3 to -0.4 points per game."
- **Expert assessment.** Robertie judged that TD-Gammon 2.1 "now plays at a strong master level that is extremely close (within a few hundredths of a point) to equaling the world's best human players."
- **Tesauro 2002, cited via S&B.** TD-Gammon 3.1 had a "lopsided advantage" in piece movement and a "slight edge" in doubling decisions over top humans.
- **Effect on human play.** Human experts changed some opening plays to match TD-Gammon (S&B; Tesauro 1995).
- A figure of "−0.08 ppg" (sometimes quoted for TD-Gammon; it was in this course's research request) **does not appear** in Tesauro 1995 or S&B. Use the table above.

### Worked example [DERIVED] (script: rl-research/scripts/F4_sims.py; numpy, seeded)
1. **Starting error.** With V = 0.5 everywhere, the errors are (−1/3, −1/6, 0, 1/6, 1/3). RMS = √((1/9 + 1/36 + 0 + 1/36 + 1/9)/5) = √(1/18) = **0.2357**. This is where every curve in Example 6.2 starts.
2. **The first TD(0) episode only moves one state** (Exercise 6.3).
   - Every step between non-terminal states has δ = 0 + 0.5 − 0.5 = 0.
   - Only the final step has a non-zero δ.
   - If the walk exits left from A: δ = 0 + 0 − 0.5 = −0.5, so V(A) ← 0.5 + 0.1 × (−0.5) = **0.45**.
   - If it exits right from E: V(E) ← 0.55.
   - MC on the same left-exiting episode moves *every* visited state toward G = 0.
3. **Re-run of Example 6.2, 100 runs** (agrees with the book's curves):

| Method, α | RMS @10 | @25 | @50 | @100 | minimum (episode) |
|---|---|---|---|---|---|
| TD 0.05 | 0.174 | 0.109 | 0.053 | **0.033** | 0.032 (91) |
| TD 0.10 | 0.127 | 0.055 | 0.050 | 0.055 | 0.043 (34) |
| TD 0.15 | 0.092 | 0.057 | 0.068 | 0.071 | 0.055 (24), then rises |
| MC 0.01 | 0.212 | 0.182 | 0.146 | 0.094 | 0.094 (100) |
| MC 0.02 | 0.195 | 0.151 | 0.113 | 0.077 | 0.076 (99) |
| MC 0.03 | 0.183 | 0.134 | 0.106 | 0.088 | 0.088 (99) |
| MC 0.04 | 0.174 | 0.127 | 0.111 | 0.101 | 0.100 (91) |

4. **Batch version** (TD computed as the certainty-equivalence solution; MC as every-visit return averages):
   - batch TD RMS 0.069 / 0.052 / **0.036** at 25 / 50 / 100 episodes;
   - batch MC 0.109 / 0.075 / **0.052**.
5. **TD-Gammon points per game.** 13/51 = 0.255, 7/38 = 0.184, 1/40 = 0.025. Tesauro's table rounds these to 0.25, 0.18, 0.02. The +6/20 vs Kazaros is +0.30 ppg.

### Common misconceptions
- "TD is always better than MC." Bootstrapping helps when the state is Markov and recognisable. With poor features, TD's bias can hurt. The S&B bound (9.14) lets TD's asymptotic error be up to 1/(1−γ) × the best possible (§12).
- "The TD error is the error versus the true value." No: it compares two successive *estimates* and is available only one step later (S&B: "δt is the error in V(St), available at time t + 1").
- "TD(0) needs a model." No: it is model-free, like MC. DP is the method that needs p(s′, r | s, a).
- "Smaller α is always more accurate." With constant α the estimates never settle: large α learns fast and then plateaus higher (TD α = .15 above). Decaying α gives convergence.
- "TD-Gammon was the first TD program." Sutton 1988's abstract notes that TD methods were already "used in Samuel's checker player, Holland's bucket brigade, and the author's Adaptive Heuristic Critic". TD-Gammon made the idea famous.

### Quotable
- "If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-difference (TD) learning." (Sutton & Barto 2018, ch. 6 opening)
- "They learn a guess from a guess—they bootstrap." (Sutton & Barto 2018, §6.2)
- "Whereas conventional prediction-learning methods assign credit by means of the difference between predicted and actual outcomes, the new methods assign credit by means of the difference between temporally successive predictions." (Sutton 1988, abstract; OCR "tile" corrected to "the")
- "Robertie actually trailed the entire session, and only in the very last game was he able to pull ahead for an extremely narrow 1-point victory." (Tesauro 1995)
- "That it was able to do as well as Neurogammon and all other approaches is striking testimony to the potential of self-play learning methods." (Sutton & Barto 2018, §16.1)

---

## §10 Q-learning and SARSA

### Mechanism (plain words)
- **Learn action values Q(s, a)** instead of state values. A greedy policy can then be read directly off Q, with no model of the world.
- **SARSA (on-policy).** Its target uses the action a′ that the agent actually takes next, chosen by its own ε-greedy policy. It learns the value of the policy it is following, *including its own exploration mistakes*.
  - The name comes from the quintuple (s, a, r, s′, a′).
- **Q-learning (off-policy).** Its target uses max over a′ of Q(s′, a′), the value of the greedy action, whatever the agent actually does next. It learns q* (the optimal values) while behaving exploratorily.
- **Expected SARSA.** Its target is the *expected* next value under the policy, Σ_a′ π(a′|s′) Q(s′, a′).
  - This removes the noise from sampling a′.
  - With a greedy target policy it **is** Q-learning.
- **The cliff walk** is the classic contrast. Q-learning learns the shortest route along the cliff edge, then keeps falling off because it still explores 10% of the time. SARSA learns a detour that is safe *given* that it explores. SARSA collects more reward while learning.
- **Convergence.**
  - Tabular Q-learning converges to q* with probability 1 if every (s, a) is tried infinitely often and the step sizes satisfy Σα = ∞, Σα² < ∞ (Watkins & Dayan 1992).
  - SARSA converges to an optimal policy if, in addition, the policy becomes greedy in the limit (e.g. ε = 1/t). Singh, Jaakkola, Littman & Szepesvári (2000), via S&B.

### Key formulas (S&B notation → ours)
- SARSA (S&B eq. 6.7): Q(s, a) ← Q(s, a) + α [r + γQ(s′, a′) − Q(s, a)], with Q(terminal, ·) = 0
- Q-learning (S&B eq. 6.8): Q(s, a) ← Q(s, a) + α [r + γ max_a′ Q(s′, a′) − Q(s, a)]
- Expected SARSA (S&B eq. 6.9): Q(s, a) ← Q(s, a) + α [r + γ Σ_a′ π(a′|s′) Q(s′, a′) − Q(s, a)]
- ε-greedy expected value used in Expected SARSA, with k actions: Σ_a′ π(a′|s′)Q(s′, a′) = (1−ε) max_a′ Q(s′, a′) + (ε/k) Σ_a′ Q(s′, a′)
- **Watkins & Dayan (1992) theorem**, their notation: x state, y next state, a action, r_n reward, α_n step size, γ discount; n^i(x, a) is the index of the i-th time action a is tried in state x.
  - "Given bounded rewards |r_n| ≤ ℛ, learning rates 0 ≤ α_n < 1, and Σ_{i=1}^∞ α_{n^i(x,a)} = ∞, Σ_{i=1}^∞ [α_{n^i(x,a)}]² < ∞, ∀x, a, then Q_n(x, a) → Q*(x, a) as n → ∞, ∀x, a, with probability 1."
  - Equation (3), read from the rendered page 282.

### Numbers with sources

**Q-learning origin**
- Watkins, C. J. C. H. (May 1989), *Learning from Delayed Rewards*, PhD thesis, King's College, Cambridge (https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf; title page checked: "Thesis Submitted for Ph.D. May, 1989").
- Watkins & Dayan (1992), "Q-learning", *Machine Learning* 8:279–292 (http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
  - Abstract: "We show that Q-learning converges to the optimum action-values with probability 1 so long as all actions are repeatedly sampled in all states and the action-values are represented discretely."
  - Caveat: "Watkins (1989) shows that Q-learning may not converge correctly for other representations." This is an early warning about function approximation (§12).

**SARSA origin**
- Rummery & Niranjan (September 1994), "On-line Q-learning using connectionist systems", Technical Report CUED/F-INFENG/TR 166, Cambridge University Engineering Department (https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf; title page checked).
- S&B bibliographic note 6.4: "They explored it in conjunction with artificial neural networks and called it 'Modified Connectionist Q-learning'. The name 'Sarsa' was introduced by Sutton (1996)."
  - Sutton (1996): "Generalization in reinforcement learning: Successful examples using sparse coarse coding", NIPS 8, pp. 1038–1044.

**Cliff walking, S&B Example 6.6** (p. 132)
- **Grid.** 4 rows × 12 columns (counted on the rendered figure). Start S at bottom-left, goal G at bottom-right. The 10 cells between them on the bottom row are "The Cliff".
- **Rewards.** "Reward is −1 on all transitions except those into the region marked 'The Cliff.' Stepping into this region incurs a reward of −100 and sends the agent instantly back to the start."
  - Undiscounted and episodic. Actions: up, down, right, left.
- **Exploration.** "ε-greedy action selection, ε = 0.1."
- **α is not stated** in the text or caption of Example 6.6.
  - The widely used reproduction of the book's figures (S. Zhang, github.com/ShangtongZhang/reinforcement-learning-an-introduction, chapter06/cliff_walking.py) uses **α = 0.5, 50 runs, 500 episodes**.
  - Treat α = 0.5 as conventional, not as the book's stated value [UNVERIFIED as the book's own setting].
- **Plot** ("Sum of rewards during episode", episodes 0–500). Read from the rendered figure:
  - SARSA settles at **≈ −25**, fluctuating roughly −22 to −29;
  - Q-learning settles at **≈ −50**, roughly −41 to −53.
  - The book does not say how much the curves are smoothed or averaged.
- **Book text.** "After an initial transient, Q-learning learns values for the optimal policy, that which travels right along the edge of the cliff. Unfortunately, this results in its occasionally falling off the cliff because of the ε-greedy action selection. Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path through the upper part of the grid."
- "Of course, if ε were gradually reduced, then both methods would asymptotically converge to the optimal policy."

**Expected SARSA, S&B §6.6, Figure 6.3** (adapted from van Seijen, van Hasselt, Whiteson & Wiering 2009, "A theoretical and empirical analysis of Expected Sarsa", IEEE ADPRL, pp. 177–184)
- Caption: "Asymptotic performance is an average over 100,000 episodes whereas interim performance is an average over the first 100 episodes. These data are averages of over 50,000 and 10 runs for the interim and asymptotic cases respectively." ε = 0.1.
- Read from the figure (approximate):
  - **Asymptotic:** Expected SARSA ≈ −21 at every α; Q-learning ≈ −51 at every α; SARSA ≈ −22 at small α, degrading above α ≈ 0.6 and collapsing at α ≥ 0.9.
  - **Interim** (best α marked by solid circles): Expected SARSA ≈ −45 at α = 1.0; SARSA ≈ −68 at α ≈ 0.9; Q-learning ≈ −73 at α ≈ 0.9.
- Book: "Expected Sarsa can safely set α = 1 without suffering any degradation of asymptotic performance". "Expected Sarsa subsumes and generalizes Q-learning while reliably improving over Sarsa."

### Worked example: the cliff walk [DERIVED] (rl-research/scripts/F4_sims.py; 100 seeded runs × 500 episodes, ε = 0.1, γ = 1, Q₀ = 0)
- **Route lengths on the 4×12 grid.** The start is in the bottom row; the goal is 11 columns to the right.
  - Optimal edge path: up 1, right 11, down 1 = **13 steps → return −13**.
  - Path through the second row from the top-left: up 2, right 11, down 2 = 15 steps → −15.
  - Top-row path: up 3, right 11, down 3 = **17 steps → −17**.
- **Learning curves, mean "sum of rewards per episode":**

| α | Method | episodes 1–100 | episodes 100–500 | greedy route at episode 500 |
|---|---|---|---|---|
| 0.5 | SARSA | −71.4 | **−27.3** | 17-step top row in 77/100 runs (19-step in 7; 16 runs' greedy rollout never reached G within 100 steps) |
| 0.5 | Q-learning | −80.4 | **−50.3** | 13-step edge in 100/100 |
| 0.5 | Expected SARSA | −56.1 | **−21.2** | 15-step in 100/100 |
| 0.1 | SARSA | −140.9 | −29.1 | 15-step in 91/100 |
| 0.1 | Q-learning | −150.7 | −50.0 | 13-step in 100/100 |
| 0.1 | Expected SARSA | −134.9 | −27.9 | 15-step in 95/100 |

  - This reproduces the book's ≈ −25 vs ≈ −50.
  - The episodes 1–100 values match Figure 6.3's interim curves at α = 0.5 (≈ −73 / −81 / −57).
  - **The safe route SARSA finds depends on α:** top row at α = 0.5, second row at α = 0.1.
- **Why −50.** Follow each fixed route with ε = 0.1 random actions (20,000 episodes each):
  - edge route **−46.1**;
  - second-row route **−20.8**;
  - top-row route −22.5.
  - Q-learning's ≈ −50 is the price of exploring next to the edge. SARSA's values *include* that price, so it steps back from the edge.
- **One Q-learning update** (α = 0.5, using optimal values):
  - Rover is in the cell directly above the first cliff cell (row 3 of 4, column 2 of 12), 11 steps from G, so V* = −11.
  - Q*(s, right) = −1 + V*(next) = −1 + (−10) = −11. Stepping right gives target −11 and δ = 0: no change.
  - If instead it explores *down*, it lands in the cliff: r = −100, back to S, where V*(S) = −13. Target = −100 + (−13) = −113.
  - Starting from Q(s, down) = 0: δ = −113, so Q(s, down) ← 0 + 0.5 × (−113) = **−56.5**.

### Common misconceptions
- "Q-learning is better because it finds the optimal path." It finds the optimal *greedy* policy, but its *online* return while exploring is worse (−50 vs −25). Which one you want depends on whether mistakes during learning cost anything.
- "SARSA learns a worse Q-function." It learns the correct Q for the ε-greedy policy it actually runs. If ε decays to 0, it too converges to the optimal policy.
- "Off-policy means it learns from a replay buffer." Off-policy only means the target policy (greedy) differs from the behaviour policy (ε-greedy). Replay (§13) is one use of that property.
- "With greedy action selection Q-learning and SARSA are identical." Nearly, but not exactly; S&B pose this as Exercise 6.12. SARSA chooses a′ *before* updating Q(s, a), so when s′ = s the chosen action can differ from Q-learning's post-update argmax [DERIVED reasoning, not quoted from the book].
- "Q-learning's convergence proof covers neural networks." No: the 1992 theorem is for a lookup table (§12).

### Quotable
- "One of the early breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as Q-learning (Watkins, 1989)" (Sutton & Barto 2018, §6.5)
- "Although Q-learning actually learns the values of the optimal policy, its online performance is worse than that of Sarsa, which learns the roundabout policy." (Sutton & Barto 2018, Example 6.6)
- "Q-learning (Watkins, 1989) is a simple way for agents to learn how to act optimally in controlled Markovian domains." (Watkins & Dayan 1992, abstract)
- "It works by successively improving its evaluations of the quality of particular actions at particular states." (Watkins & Dayan 1992, abstract)
- "This quintuple gives rise to the name Sarsa for the algorithm." (Sutton & Barto 2018, §6.4)

---

## §11 Sparse rewards & credit assignment

### Mechanism (plain words)
- **Sparse reward.** A reward arrives only at the end of a long chain of actions. Learning then needs two things. First, the agent must *find* the reward at least once (exploration). Second, it must work out which of the many earlier actions earned it (credit assignment).
- **Why random exploration fails.** Suppose Rover random-walks along a corridor. The expected time to reach the end grows with the *square* of the corridor's length (worked example below). In Atari's Montezuma's Revenge, the first point needs a long, precise sequence of jumps and climbs. DQN, which explores with ε-greedy, scores **0** there (Mnih et al. 2015).
- **Reward shaping.** Add extra reward for progress. Done naively, this changes *which policy is optimal*: an agent paid for "moving toward the goal" learned to ride a bicycle in circles. **Potential-based shaping** pays the *change in a potential*, F = γΦ(s′) − Φ(s). Around any loop these payments cancel, and the optimal policy is provably unchanged (Ng, Harada & Russell 1999).
- **Eligibility traces.** These spread credit backwards in time without storing the whole episode. Each recently visited state (or weight) keeps a fading trace, and every TD error is applied to all of them in proportion to their trace. λ = 0 gives one-step TD; λ = 1 behaves like Monte Carlo (S&B ch. 12). See §9 for TD(λ).
- **Exploration bonuses (intrinsic reward).** Pay the agent for novelty, so that it explores systematically instead of at random:
  - **Count-based:** bonus ∝ 1/√N(s). This needs *pseudo-counts* when states are pixels (Bellemare et al. 2016).
  - **Curiosity / prediction error:** bonus = how badly the agent predicts the next state, measured in a learned feature space (Pathak et al. 2017, ICM).
  - **Random Network Distillation (RND):** bonus = how badly a trained network imitates a *fixed random* network on this observation. Novel observations are imitated poorly (Burda et al. 2018).
- **Go-Explore** (Ecoffet et al. 2021) tackles the problem differently. It *remembers* promising states in an archive, *returns* to one (by restoring the simulator, or with a goal-conditioned policy), and only *then* explores. Afterwards it "robustifies" the found trajectories into a policy with imitation learning.

### Key formulas
- **Potential-based shaping** (Ng, Harada & Russell 1999, Theorem 1; their notation already matches ours):
  - F(s, a, s′) = γΦ(s′) − Φ(s) for some real-valued Φ: S → ℝ.
  - Learning happens in M′ = (S, A, T, γ, R + F) instead of M = (S, A, T, γ, R).
  - **Sufficiency:** if F is potential-based, "every optimal policy in M′ will also be an optimal policy in M (and vice versa)".
  - **Necessity:** if F is not potential-based, "then there exist (proper) transition functions T and a reward function R … such that no optimal policy in M′ is optimal in M."
  - Key identity (their proof): Q*_M′(s, a) = Q*_M(s, a) − Φ(s). The shift does not depend on a, so the argmax is unchanged.
  - Their suggested potential is Φ(s) = V*_M(s), or a crude estimate of it.
  - Telescoping form [DERIVED]: for an episode s₀ … s_T, Σ_t γ^t F_t = γ^T Φ(s_T) − Φ(s₀). With Φ(terminal) = 0, *every* policy's return shifts by the same −Φ(s₀).
- **Eligibility trace, semi-gradient TD(λ)** (S&B §12.2, eqs. 12.5–12.7):
  - z₋₁ = 0, and z_t = γλ z_{t−1} + ∇v̂(S_t, w_t)
  - δ_t = R_{t+1} + γ v̂(S_{t+1}, w_t) − v̂(S_t, w_t)
  - w_{t+1} = w_t + α δ_t z_t
- **Pseudo-count** (Bellemare et al. 2016, eqs. 1–2):
  - Let ρ_n(x) be a density model's probability of x after n observations, and ρ′_n(x) its "recoding" probability after one more observation of x.
  - Then N̂_n(x) = ρ_n(x)(1 − ρ′_n(x)) / (ρ′_n(x) − ρ_n(x)) = n̂ ρ_n(x).
  - The bonus used with DQN is R⁺_n(x, a) = β_bonus (N̂_n(x) + 0.01)^(−1/2), with **β_bonus = 0.05** "selected from a coarse parameter sweep". Their A3C+ variant used β = 0.01.
- **ICM curiosity** (Pathak et al. 2017, eq. 6):
  - Intrinsic reward r^i_t = (η/2) ‖φ̂(s_{t+1}) − φ(s_{t+1})‖².
  - φ is a feature encoder trained by an *inverse* model (predict a_t from s_t, s_{t+1}).
  - The total loss is −λ E_π[Σ r_t] + (1 − β_ICM) L_I + β_ICM L_F, with **β_ICM = 0.2 and λ = 0.1**. (Their β and λ are not our β and λ.)
- **RND** (Burda et al. 2018): bonus = ‖f̂(x; θ) − f(x)‖², where f is a fixed, randomly initialised target network and f̂ is a predictor trained by MSE on the agent's own observations.
  - RND uses two value heads: extrinsic with γ_E = 0.999, intrinsic with γ_I = 0.99. The intrinsic return is treated as non-episodic.

### Numbers with sources
**Montezuma's Revenge, the benchmark for sparse reward** (scores are game points; human baselines differ between papers, see the note at the end).

| Agent | Score | Source |
|---|---|---|
| Random play | 0 | Mnih et al. 2015, Extended Data Table 2 |
| DQN (Nature) | **0** (± 0), i.e. 0.0% of human | Mnih et al. 2015, Extended Data Table 2 |
| Human tester (DQN paper) | **4,367** | Mnih et al. 2015, Extended Data Table 2 |
| DQN, DDQN, Dueling, Prioritized (no-op starts) | 0.0 each | Wang et al. 2016, appendix Table 2 (human 4,753.3) |
| Rainbow | 384.0 (no-ops); 154.0 (human starts) | Hessel et al. 2018, appendix Tables 6 and 5 |
| DQN + pseudo-count bonus | avg **2,461** at 50M frames, **3,439** at 100M frames; one run "consistently achieving 6600 points"; **15 rooms** explored vs **2** without the bonus | Bellemare et al. 2016, §6 and appendix |
| Forward-dynamics curiosity (RND paper's baseline) | 400 | Burda et al. 2018, Table 1 |
| PPO without bonus | 2,497 | Burda et al. 2018, Table 1 |
| RND (CNN policy, ~1.97 B frames) | **8,152** (avg human listed as 4,753) | Burda et al. 2018, Table 1 |
| RND (RNN policy, 1,024 parallel envs) | mean **10,070**; one run with mean 14,415; best single episode **17,500** (32 envs, "visited all 24 rooms, and passed the first level once") | Burda et al. 2018, §3.4 |
| Agent57 (as listed by Go-Explore) | 9,352 | Ecoffet et al. 2021, SI Table 1 |
| Go-Explore, no domain knowledge | exploration phase 24,758; **robustified policy 43,791** (their "SOTA" column: 11,618) | Ecoffet et al. 2021, Extended Data Table 3 |
| Go-Explore with domain-knowledge cells | **1,731,645** mean ("exceeding the previous state of the art score by a factor of 150"); human world record cited as 1.2 million | Ecoffet et al. 2021, main text |
| Policy-based Go-Explore (no simulator restore) | 97,728 | Ecoffet et al. 2021, main text |

- **Human baselines disagree.** The DQN Nature paper's tester scored **4,367**. The DDQN, Dueling, PER, RND and Go-Explore papers use **4,753(.3)**, and Dueling's human-starts table uses 4,182.0. Always say which baseline a percentage uses.
- **Bellemare et al.'s 3,439 vs RND's "3,700".** RND's Table 1 lists the prior SOTA as 3,700, citing Bellemare et al. 2016. The arXiv text of Bellemare et al. says 3,439 at 100M frames. I could not trace the 3,700 figure [UNVERIFIED]. It may come from a different table or version.
- **Go-Explore's factor.** "quadruples the state-of-the-art score": 43,791 / 11,618 = 3.77× [DERIVED].
- **Go-Explore's time limit.** With the default 400,000-frame time limit removed, "one agent frequently reaching a score of over 40 million after 12.5 million frames (the equivalent of about 58 hours of continuous game play)."
- **ICM on Mario:** "Without any extrinsic reward from environment, our Mario agent can learn to cross over 30% of Level-1." ICM was evaluated on VizDoom and Super Mario Bros, not on Montezuma.
- **Ng et al. 1999 experiments:**
  - A 10×10 grid, start and goal in opposite corners, −1 per step, no discounting.
  - Moves succeed 80% of the time; 20% of the time the agent moves in a random direction.
  - Potential Φ₀(s) = −MANHATTAN(s, GOAL)/0.8, i.e. minus a crude estimate of the steps to the goal. They also used 0.5Φ₀ and ran a 50×50 grid and a 5×5 grid with 5 ordered subgoals.
  - The figures plot steps-to-goal against trial number. Shaping learns much faster; no single headline number is given.
- **The bicycle story (secondary, as told by Ng et al. 1999, citing Randløv & Alstrøm 1998):** the agent "learned to ride in tiny circles near the start state because no penalty was incurred for riding away from the goal." I did not read Randløv & Alstrøm directly. A second story is the soccer robot rewarded for touching the ball, which "remained next to the ball and 'vibrated'" (personal communication, D. Andre & A. Teller).

### Worked example [DERIVED] (scripts/f5_checks.py)
**Rover's corridor.** Cells 0 … 9, Rover starts at 0, and the charger at cell 9 pays +1 on arrival and ends the episode. Rover moves left or right with probability ½ each; bumping the wall at cell 0 leaves it in place.
1. **Pure random exploration.** The expected steps to reach the charger is N(N+1) for N = 9 cells of distance: **90 steps**. N = 20 gives 420 steps, and N = 99 gives **9,900**. Doubling the distance roughly quadruples the wait. This is why sparse rewards get exponentially worse once several "doors" must be passed in sequence. (The numbers come from solving the linear hitting-time equations; they match the closed form N(N+1).)
2. **Naive progress bonus (the bicycle bug).** Pay +0.1 for every step toward the charger and 0 for steps away; γ = 0.99.
   - Going straight (9 steps): Σ_{t=0}^{8} 0.99^t·0.1 + 0.99^8·1 = 0.865 + 0.923 = **1.788**.
   - Wiggling forever (toward, back, toward …): 0.1/(1 − 0.99²) = **5.025**.
   - The "helpful" bonus makes **wiggling optimal**. At γ = 0.9 the straight path still wins (1.043 vs 0.526), so the bug can appear just from raising γ.
3. **Potential-based fix.** Let Φ(s) = −0.1 × (distance to charger), with Φ(charger) = 0, and F = γΦ(s′) − Φ(s).
   - Every policy's return shifts by −Φ(s₀) = +0.9.
   - Straight: 0.923 + 0.9 = **1.823** (checked by summing the shaped rewards step by step).
   - Wiggling forever: 0 + 0.9 = **0.900**.
   - Going straight is optimal again. With γ = 1, a toward-and-back loop earns F = +0.1 − 0.1 = **0**.
4. **Novelty glow (pseudo-count bonus, β = 0.05):**

| Visits N | Bonus 0.05·(N + 0.01)^−½ |
|---|---|
| 0 | 0.500 |
| 1 | 0.0498 |
| 4 | 0.025 |
| 10 | 0.0158 |
| 100 | 0.005 |
| 1,000 | 0.0016 |

   Animation: a cell's violet glow dims by 10× after the first visit, then slowly as 1/√N.

### Common misconceptions
- **"More reward is always more helpful."** Non-potential bonuses can change the optimal policy: the bicycle circling, the vibrating soccer robot, and wiggling Rover above. Sutton & Barto: well-intentioned supplemental rewards "may lead the agent to behave differently from what is intended".
- **"Potential-based shaping makes learning faster."** It guarantees the optimal policy is *unchanged*. Speed depends on how good Φ is.
  - S&B (ch. 17 bibliographical remarks on §17.4) report that Wiewiora (2003) showed it equivalent "to the simpler idea of providing an initial approximation to the value function".
  - It is also different from Skinner's behavioural "shaping", which gradually changes the task.
- **"Curiosity = seek randomness."** Pure prediction-error curiosity *is* drawn to unpredictable things: the "noisy-TV problem". RND's forward-dynamics baseline oscillated between two rooms in Montezuma because sticky actions make the room crossing unpredictable.
  - ICM's inverse-model features ignore what the agent cannot affect.
  - RND's target is deterministic, so its error can be driven to zero on familiar states.
- **"RND solved Montezuma's Revenge."** It was the first method above *average human* (8,152 vs 4,753) without demonstrations or RAM state, and it "occasionally completes the first level". The game has 3 levels of 24 rooms each.
- **"Go-Explore's 1.7 million is a pure-RL result."** That run used domain-knowledge cells (room, x, y, level, keys) and simulator restoration during exploration. Without domain knowledge it scored 43,791; the policy-based variant, without restore, scored 97,728.
- **"DQN failed Montezuma because the network was too small."** The problem is exploration. ε-greedy almost never stumbles on the first key, so there is no reward to learn from.

### Quotable
- "Even when there is a simple and easily identifiable goal, the problem of sparse reward often arises." (Sutton & Barto 2018, §17.4)
- "The agent may wander aimlessly for long periods of time (what Minsky, 1961, called the “plateau problem”)." (Sutton & Barto 2018, §17.4)
- "But augmenting the reward signal with well-intentioned supplemental rewards may lead the agent to behave differently from what is intended; the agent may end up not achieving the overall goal." (Sutton & Barto 2018, §17.4)
- "The agent learned to ride in tiny circles near the start state because no penalty was incurred for riding away from the goal." (Ng, Harada & Russell 1999, §1; text read via OCR of the scanned PDF)
- "Whereas the weight vector is a long-term memory, accumulating over the lifetime of the system, the eligibility trace is a short-term memory, typically lasting less time than the length of an episode." (Sutton & Barto 2018, §12.2)
- "Due to its sparse reward function, most published agents achieve an average score close to zero and completely fail to explore most of the 24 rooms that constitute the first level" (Bellemare et al. 2016, §6)
- "We formulate curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model." (Pathak et al. 2017, abstract)
- "The bonus is the error of a neural network predicting features of the observations given by a fixed randomly initialized neural network." (Burda et al. 2018, abstract)
- "To the best of our knowledge, this is the first method that achieves better than average human performance on this game without using demonstrations or having access to the underlying state of the game, and occasionally completes the first level." (Burda et al. 2018, abstract)
- "We hypothesise that the main impediment to effective exploration originates from algorithms forgetting how to reach previously visited states (“detachment”) and from failing to first return to a state before exploring from it (“derailment”)." (Ecoffet et al. 2021, arXiv v6 abstract)

---

## §12 Function approximation and the deadly triad

### Mechanism (plain words)
- A table needs one entry per state. CartPole's state is four real numbers, and backgammon has over 10^20 positions (Tesauro 1995). Instead, learn a **parametrised function** V(s; θ) or Q(s, a; θ):
  - **linear:** V = θᵀx(s), on hand-made features such as tile coding;
  - **neural:** a network with weights θ.
- **Generalisation.** One update changes θ, which moves the values of *many* states at once. That is the point, and it is also the danger.
- **Semi-gradient TD.**
  - Do SGD on the squared TD error, but treat the target r + γV(s′; θ) as a constant. Only V(s; θ) is differentiated.
  - It is not true gradient descent, because the target moves with θ. S&B call it "semi-gradient".
  - It is fast. In the linear on-policy case it converges, to the **TD fixed point**.
- **Tile coding.**
  - Overlay several offset grids ("tilings"). A state activates exactly one tile per tiling, giving a sparse binary x(s) with exactly n ones.
  - Nearby states share tiles, so they generalise to each other.
  - Rule of thumb: α = 1/(10n) moves the trained state one-tenth of the way to its target.
- **The deadly triad.** Function approximation + bootstrapping + off-policy training together can make values **diverge to infinity**, even with zero rewards, a perfect representation and expected (DP) updates. Any two of the three are safe.
- **The w-to-2w toy** (the simplest animation of divergence):
  - Two states with values w and 2w (features 1 and 2) and a reward-0 transition from the first to the second.
  - The update raises w to reduce the TD error, which also raises the second state's value, so the TD error grows.
  - If only this transition is ever trained (possible off-policy), w → ∞ whenever γ > 0.5.

### Key formulas (S&B notation: w → θ, x(s) features)
- Gradient MC (S&B eq. 9.7 with U_t = G_t): θ ← θ + α [G_t − V(s_t; θ)] ∇V(s_t; θ)
- Semi-gradient TD(0): θ ← θ + α [r + γV(s′; θ) − V(s; θ)] ∇V(s; θ)
- Linear case: ∇V = x(s). S&B eq. 9.9: θ ← θ + α (r + γθᵀx′ − θᵀx) x
- TD fixed point (S&B eqs. 9.11–9.12): A = E[x_t (x_t − γx_{t+1})ᵀ], b = E[r_{t+1} x_t], θ_TD = A⁻¹b
- Bound at the TD fixed point (S&B eq. 9.14, continuing case, on-policy distribution μ): VE(θ_TD) ≤ (1/(1−γ)) min_θ VE(θ), where VE(θ) = Σ_s μ(s)[v_π(s) − V(s; θ)]² (S&B eq. 9.1)
- **Tsitsiklis & Van Roy (1997), Theorem 1.** Their notation: α is the **discount**, γ_t the **step size** (swapped relative to ours); J* is the cost-to-go, Φr* the approximation, Π the projection under the stationary distribution D.
  - "For any λ ∈ [0, 1], the TD(λ) algorithm with linear function approximators … converges with probability one."
  - The limit satisfies ΠT^(λ)(Φr*) = Φr*.
  - Error bound: ‖Φr* − J*‖_D ≤ ((1 − λα)/(1 − α)) ‖ΠJ* − J*‖_D
  - Footnote 1 (improvement credited to V. Papavassiliou): ‖Φr* − J*‖_D ≤ ((1 − λα)/√((1 − α)(1 + α − 2λα))) ‖ΠJ* − J*‖_D
  - Step sizes: Σγ_t = ∞, Σγ_t² < ∞.
- w-to-2w (S&B §11.2):
  - δ = 0 + γ·2w − w = (2γ − 1)w
  - off-policy semi-gradient update: w ← [1 + α(2γ − 1)] w
  - "this constant is greater than 1 whenever γ > 0.5"

### Numbers with sources

**The deadly triad** (S&B §11.3, p. 264; exact wording)
- "the danger of instability and divergence arises whenever we combine all of the following three elements, making up what we call the deadly triad:"
  - "**Function approximation** A powerful, scalable way of generalizing from a state space much larger than the memory and computational resources (e.g., linear function approximation or ANNs)."
  - "**Bootstrapping** Update targets that include existing estimates (as in dynamic programming or TD methods) rather than relying exclusively on actual rewards and complete returns (as in MC methods)."
  - "**Off-policy training** Training on a distribution of transitions other than that produced by the target policy. Sweeping through the state space and updating all states uniformly, as in dynamic programming, does not respect the target policy and is an example of off-policy training."
- "If any two elements of the deadly triad are present, but not all three, then instability can be avoided."
- "the danger is not due to control or to generalized policy iteration … The danger is also not due to learning or to uncertainties about the environment, because it occurs just as strongly in planning methods, such as dynamic programming".
- **History** (S&B note 11.3): "The deadly triad was first identified by Sutton (1995b) and thoroughly analyzed by Tsitsiklis and Van Roy (1997). The name 'deadly triad' is due to Sutton (2015a)." Sutton 2015a is a NIPS 2015 tutorial.

**Baird's counterexample** (S&B §11.2, Figure 11.1 and Figure 11.2, pp. 261–262)
- Note: in S&B, **Example 11.1 is *Tsitsiklis and Van Roy's* counterexample**. Baird's counterexample is the unnumbered example in §11.2 with Figures 11.1–11.2.
- **MDP.** "the episodic seven-state, two-action MDP".
  - "The dashed action takes the system to one of the six upper states with equal probability, whereas the solid action takes the system to the seventh state."
  - Behaviour policy: dashed 6/7, solid 1/7, "so that the next-state distribution under it is uniform".
  - Target policy: "always takes the solid action".
  - "The reward is zero on all transitions." "The discount rate is γ = 0.99."
- **Features.** w ∈ ℝ⁸.
  - Each upper state i = 1…6 has value **2w_i + w_8**, e.g. x(1) = (2, 0, 0, 0, 0, 0, 0, 1)ᵀ.
  - The seventh state has value **w_7 + 2w_8**.
  - The true values are all 0, exactly representable with w = 0. The features are linearly independent: 8 weights for 7 states.
- **Result.** "If we apply semi-gradient TD(0) to this problem (11.2), then the weights diverge to infinity … The instability occurs for any positive step size, no matter how small. In fact, it even occurs if an expected update is done as in dynamic programming (DP)".
- **Figure 11.2 settings.** "The step size was α = 0.01, and the initial weights were w = (1, 1, 1, 1, 1, 1, 10, 1)ᵀ."
  - x-axes: 1,000 steps (semi-gradient off-policy TD) and 1,000 sweeps (semi-gradient DP).
  - w_8 shoots past 200–300. w_1–w_6 climb to ≈ 100+. w_7 drifts slowly down from 10.
- **Fix.** "If we alter just the distribution of DP updates in Baird's counterexample, from the uniform distribution to the on-policy distribution … then convergence is guaranteed".
- **Version note** (S&B note 11.2): "Baird's counterexample is due to Baird (1995), though the version we present here is slightly modified." Baird 1995: "Residual algorithms: Reinforcement learning with function approximation", ICML, pp. 30–37.

**Tsitsiklis & Van Roy (1997)** ("An analysis of temporal-difference learning with function approximation", *IEEE Transactions on Automatic Control* 42(5):674–690; http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf)
- Abstract: "We present a proof of convergence (with probability one), a characterization of the limit of convergence, and a bound on the resulting approximation error."
- "First, we prove that divergence may occur when updates are not based on trajectories of the Markov chain."
- "Second, we present an example illustrating the possibility of divergence when temporal-difference learning is used in the presence of a nonlinear function approximator."

**Deep RL check of the triad** (van Hasselt, Doron, Strub, Hessel, Sonnerat & Modayil 2018, "Deep Reinforcement Learning and the Deadly Triad", arXiv:1812.02648)
- "When these three properties are combined, learning can diverge with the value estimates becoming unbounded. However, several algorithms successfully combine these three properties, which indicates that there is at least a partial gap in our understanding."
- DQN combines all three; its stabilisers are covered in §13.

**Tile coding** (S&B §9.5.4)
- "Tile coding … may be the most practical feature representation for modern sequential digital computers."
- Four 4×4 tilings give "4 × 4 × 4 = 64 components, all of which will be 0 except for the four corresponding to the tiles that s falls within".
- "choosing α = 1/n, where n is the number of tilings, results in exact one-trial learning". One might choose α = 1/(10n) instead.

**CartPole-v1** (Gymnasium docs, https://gymnasium.farama.org/environments/classic_control/cart_pole/, and source gymnasium/envs/classic_control/cartpole.py + envs/__init__.py on GitHub main; read Oct 2026; PyPI gymnasium 1.4.0) **[DATED]**
- **Origin.** "This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson". Barto, Sutton & Anderson 1983, IEEE Trans. SMC 13(5) (pp. 834–846 per the DOI record; S&B print 835–846).
- **Actions.** Discrete {0: push left, 1: push right}, fixed force magnitude 10.0 (source: force_mag = 10.0).
- **Observation**, shape (4,):

| # | Variable | Observation bounds |
|---|---|---|
| 0 | cart position | ±4.8 |
| 1 | cart velocity | ±∞ |
| 2 | pole angle | ±0.418 rad (±24°) |
| 3 | pole angular velocity | ±∞ |

- **Termination.** Pole angle outside ±0.2095 rad (**±12°**), or cart position outside **±2.4**.
- **Truncation.** "Episode length is greater than 500 (200 for v0)".
- **Reward.** "+1 is given for every step taken, including the termination step". Optional sutton_barto_reward=True gives 0 per step and −1 on failure (added in Gymnasium 1.0.0a2).
- **Start state.** "All observations are assigned a uniformly random value in (-0.05, 0.05)".
- **Physics** (source): gravity 9.8, cart mass 1.0, pole mass 0.1, length 0.5 ("actually half the pole's length"), τ = 0.02 s, Euler integration.
- **Reward threshold.**
  - Registry: CartPole-v1 has max_episode_steps = 500 and **reward_threshold = 475.0** (v0: 200 and 195.0).
  - The docs page text instead says "The default reward threshold is 500 for v1 and 200 for v0". See Disagreements.

### Worked example [DERIVED] (rl-research/scripts/F4_sims.py)
1. **w-to-2w** (α = 0.1, γ ≈ 1, w₀ = 10): w = 10 → 11 → 12.1 → 13.31. Each update multiplies w by 1 + α(2γ − 1) = 1.1. This matches S&B's narrative "nearly 11 … ≈12.1".
2. **Baird, re-run** (α = 0.01, w₀ = (1, 1, 1, 1, 1, 1, 10, 1), γ = 0.99; values start at 3 for the upper states and 12 for the seventh):
   - Semi-gradient DP (S&B eq. 11.9, α/|S| per sweep), after 1,000 sweeps: w_1…w_6 = **114.4**, w_7 = **6.9**, w_8 = **335**.
   - Semi-gradient off-policy TD(0) (ρ = 7 for solid and 0 for dashed), one seed, after 1,000 steps: w_1…6 ≈ 103–142, w_7 = 5.5, w_8 = 376.
   - Both match Figure 11.2's shapes. The rewards are all zero, yet the values head for infinity.
3. **How loose is the TD bound?** At γ = 0.9:
   - TVR's original factor on the error *norm* is 1/(1−γ) = 10; the Papavassiliou factor (λ = 0) is 1/√(1−γ²) = 2.29.
   - On the *squared* error VE, the improved bound gives 1/(1−γ²) = 5.26, which is ≤ S&B's 1/(1−γ) = 10. So S&B eq. 9.14 is a valid (looser) statement in VE terms.
   - At γ = 0.99 the three factors are 100, 7.09 and 50.25.
4. **CartPole numbers.**
   - 500 steps × τ = 0.02 s = **10 s** of balancing for a perfect episode.
   - A table over 4 variables with 10 bins each needs 10⁴ = 10,000 cells; with 100 bins each, 10⁸. A small network (e.g. 4→64→64→2) has 4,610 parameters [(4·64+64) + (64·64+64) + (64·2+2) = 320 + 4,160 + 130]. It generalises across the continuum instead of memorising cells.

### Common misconceptions
- "Divergence comes from noise or a too-large step size." Baird's example diverges "for any positive step size, no matter how small", and even with noise-free DP updates.
- "Divergence needs a bad representation." Baird's features are linearly independent and the true solution w = 0 is exactly representable.
- "The deadly triad is about control." S&B: the instability "arises in the simpler prediction case".
- "Linear TD converges to the best linear fit." It converges to the TD fixed point, whose error can be up to 1/(1−γ) times larger in S&B's VE statement (eq. 9.14). Gradient MC converges to the best fit.
- "Semi-gradient TD is gradient descent on some loss." It ignores the target's dependence on θ, so it is not the gradient of any fixed objective in general. That is why S&B call it "semi-gradient".
- "Neural networks make RL unstable, full stop." The triad says *which* combination is risky. DQN (§13) keeps all three ingredients but tames them with replay and a target network. Van Hasselt et al. (2018) studied when deep Q-learning actually diverges.

### Quotable
- "If any two elements of the deadly triad are present, but not all three, then instability can be avoided." (Sutton & Barto 2018, §11.3)
- "The instability occurs for any positive step size, no matter how small." (Sutton & Barto 2018, §11.2, on Baird's counterexample)
- "Bootstrapping methods are not in fact instances of true gradient descent (Barnard, 1993). They take into account the effect of changing the weight vector wt on the estimate, but ignore its effect on the target." (Sutton & Barto 2018, §9.3)
- "We know from reinforcement learning theory that temporal difference learning can fail in certain cases." (van Hasselt et al. 2018, abstract)
- "First, we prove that divergence may occur when updates are not based on trajectories of the Markov chain." (Tsitsiklis & Van Roy 1997, abstract)

### Disagreements and corrections (§9, §10, §12)
1. **TD-Gammon 0.0 training games.** Tesauro 1995 text says the best raw-encoding net (40 hidden units) "was trained for a total of 200,000 games". S&B Table 16.1 says 300,000 games for TD-Gammon 0.0. (S&B's text also says "After playing about 300,000 games against itself, TD-Gammon 0.0 …".)
2. **TD-Gammon 3.0 hidden units.** S&B Table 16.1 says 80. S&B's text says "TD-Gammon 3.0 and 3.1, used 160 hidden units".
3. **The "−0.08 ppg"** figure (from the research request) appears in neither source. Tesauro 1995's Table 1 gives −0.25, −0.18 and −0.02 ppg. S&B adds +6/20 (+0.30 ppg) for TD-Gammon 3.0.
4. **Baird ≠ Example 11.1** (the research request called it that). In S&B 2nd ed., Example 11.1 is Tsitsiklis & Van Roy's counterexample. Baird's counterexample is in §11.2 with Figures 11.1–11.2.
5. **CartPole-v1 reward threshold.** The docs text says 500; the registry code says reward_threshold = 475.0.
6. **Cliff-walk α** is not given in S&B. 0.5 is from the community reproduction.
7. **TD bound norm vs squared.** TVR's bound is on the D-norm. S&B state it on VE, which is squared. S&B's version holds via the Papavassiliou-improved bound (worked example 3).

---

## §13 Deep Q-Networks (DQN)

### Mechanism (plain words)
- **One network instead of a table.** A convolutional network reads the last 4 screen frames and outputs one Q-value per joystick action, all in one forward pass. It is trained with the Q-learning target r + γ max_a′ Q(s′, a′).
- **Why naive deep Q-learning breaks.** Mnih et al. 2015 list three causes:
  1. consecutive frames are highly correlated;
  2. a small change to Q can change the policy and hence the data distribution;
  3. the target uses the same network that is being updated.
- **Two stabilisers**, both shown by the paper's ablation (Extended Data Table 3, below) to matter a great deal:
  1. **Experience replay.** Store the last 1,000,000 transitions (s, a, r, s′) and train on uniformly sampled minibatches of 32. This breaks correlations, reuses data, and averages over many past behaviours. Because the data come from older policies, the method must be off-policy, "which motivates the choice of Q-learning".
  2. **Target network.** Compute targets with a frozen copy Q̂ (parameters θ⁻), refreshed from the online network every C = 10,000 updates. The target then stops chasing itself.
- **Small but important tricks:**
  - Rewards clipped to {−1, 0, +1}, so one learning rate works for every game.
  - TD-error clipping to [−1, 1], equivalent to a Huber-style loss.
  - Action repeat (frame skip) of 4.
  - Life loss treated as the end of an episode during training.
  - Up to 30 random "no-op" actions at the start of evaluation episodes.

### Key formulas (Mnih et al. 2015, notation mapped)
- Q*(s, a) = max_π E[r_t + γ r_{t+1} + γ² r_{t+2} + … | s_t = s, a_t = a, π]
- Loss at iteration i, with transitions drawn uniformly from replay memory D:
  - L_i(θ_i) = E_{(s,a,r,s′)~U(D)} [ (r + γ max_a′ Q(s′, a′; θ_i⁻) − Q(s, a; θ_i))² ]
  - θ_i⁻ are the target-network parameters, "only updated with the Q-network parameters (θi) every C steps and are held fixed between individual updates".
- Target for a terminal s′: y = r.
- **Human-normalised score:** 100 × (DQN score − random play score) / (human score − random play score). 0% = random, 100% = the human tester.
- **ε schedule:** ε_t = max(0.1, 1 − 0.9·t/10⁶), linear over the first million frames. Evaluation uses ε = 0.05.

### Numbers with sources
**Mnih et al. 2013**, "Playing Atari with Deep Reinforcement Learning", arXiv:1312.5602 (19 Dec 2013, NIPS Deep Learning Workshop):
- 7 games: Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders.
- Smaller network: conv 16 filters 8×8 stride 4 → conv 32 filters 4×4 stride 2 → fully connected 256 → outputs.
- 10 million frames of training, replay memory of 1 million, ε annealed 1 → 0.1 over the first million frames, RMSProp, minibatch 32. No separate target network in 2013.
- Abstract: "it outperforms all previous approaches on six of the games and surpasses a human expert on three of them." The three are Breakout, Enduro and Pong (§5, Table 1).
- Table 1, average DQN scores vs human:

| Game | DQN | Human |
|---|---|---|
| Breakout | 168 | 31 |
| Enduro | 470 | 368 |
| Pong | 20 | −3 |
| Beam Rider | 4,092 | 7,456 |
| Q*bert | 1,952 | 18,900 |
| Seaquest | 1,705 | 28,010 |
| Space Invaders | 581 | 3,690 |

**Mnih et al. 2015**, "Human-level control through deep reinforcement learning", *Nature* 518, 529–533, doi:10.1038/nature14236. I read DeepMind's hosted PDF.

**Architecture (Methods):**
- Input 84×84×4: luminance, rescaled from the 210×160 frames, with a per-pixel max over 2 consecutive frames to remove flicker, and the last 4 frames stacked.
- conv 32 filters 8×8 stride 4 + ReLU → conv 64 filters 4×4 stride 2 + ReLU → conv 64 filters 3×3 stride 1 + ReLU → fully connected 512 ReLU → linear output, one per action.
- "The number of valid actions varied between 4 and 18 on the games we considered."
- A *different network was trained per game*, with the same architecture and hyperparameters.

**Extended Data Table 1 (hyperparameters, exact):**

| Hyperparameter | Value |
|---|---|
| minibatch size | 32 |
| replay memory size | 1,000,000 |
| agent history length | 4 |
| target network update frequency C | 10,000 (measured in parameter updates) |
| discount γ | 0.99 |
| action repeat | 4 |
| update frequency | 4 actions between SGD updates |
| learning rate (RMSProp) | 0.00025 |
| gradient momentum | 0.95 |
| squared gradient momentum | 0.95 |
| min squared gradient | 0.01 |
| initial exploration ε | 1 |
| final exploration ε | 0.1 |
| final exploration frame | 1,000,000 |
| replay start size | 50,000 (uniform random policy) |
| no-op max | 30 |

- Hyperparameters were "selected by performing an informal search on the games Pong, Breakout, Seaquest, Space Invaders and Beam Rider."
- Training ran "for a total of 50 million frames (that is, around 38 days of game experience in total)". See the frame arithmetic below.

**Headline result (main text):**
- "Our DQN method outperforms the best existing reinforcement learning methods on 43 of the games without incorporating any of the additional prior knowledge about Atari 2600 games used by other approaches"
- "our DQN agent performed at a level that was comparable to that of a professional human games tester across the set of 49 games, achieving more than 75% of the human score on more than half of the games (29 games; see Fig. 3, Supplementary Discussion and Extended Data Table 2)."
- The Fig. 3 caption makes the threshold explicit: "broadly comparable with or superior to a professional human games tester (that is, operationalized as a level of 75% or above) in the majority of games."

**How "human" was measured (Methods):**
- "The professional human tester used the same emulator engine as the agents, and played under controlled conditions. The human tester was not allowed to pause, save or reload games."
- "The human performance is the average reward achieved from around 20 episodes of each game lasting a maximum of 5 min each, following around 2 h of practice playing each game."
- Audio was off for both.
- Agents were evaluated on 30 episodes of up to 5 min, with ε = 0.05 and random no-op starts.

**Extended Data Table 2 recount [DERIVED].** I transcribed the 49 rows by OCR from the table image. My counts: **29** games ≥ 75% (matching the paper), **23** games ≥ 100%, median **93.5%**, mean **241.1%**. Van Hasselt et al. 2016 (Table 1) independently report DQN's median 93.5% and mean 241.1% on these 49 games. Individual games:

| Game | Human | DQN | % of human |
|---|---|---|---|
| Video Pinball (best) | 17,298 | 42,684 | 2,539.4% |
| Boxing | 4.3 | 71.8 | 1,707.9% |
| Breakout | 31.8 | 401.2 | 1,327.2% |
| Pong | 9.3 | 18.9 | 132.0% |
| Space Invaders | 1,652 | 1,976 | 121.5% |
| Seaquest | 20,182 | 5,286 | 25.9% |
| Private Eye | 69,571 | 1,788 | 2.5% |
| Montezuma's Revenge (worst) | 4,367 | 0 | 0.0% |

**Extended Data Table 3: replay and target-network ablation.** Settings: 10M frames, best of 3 learning rates, scores not truncated at 5 min.

| Game | Replay + target | Replay, no target | No replay, target | Neither |
|---|---|---|---|---|
| Breakout | 316.8 | 240.7 | 10.2 | 3.2 |
| Enduro | 1006.3 | 831.4 | 141.9 | 29.1 |
| River Raid | 7446.6 | 4102.8 | 2867.7 | 1453.0 |
| Seaquest | 2894.4 | 822.6 | 1003.0 | 275.8 |
| Space Invaders | 1088.9 | 826.3 | 373.2 | 302.0 |

- Breakout: full DQN / neither = 316.8 / 3.2 ≈ **99×** [DERIVED].
- Removing replay hurts far more than removing the target network on Breakout, Enduro and Space Invaders. On Seaquest the two are comparable.
- Extended Data Table 4: a linear function approximator, with replay and a target net, does much worse on these 5 games (values not transcribed here).

### Worked example [DERIVED] (scripts/f5_checks.py)
1. **Network size.** Feature maps are 84 → 20×20×32 → 9×9×64 → 7×7×64 (3,136 values). Parameters:
   - conv1: 32·(8·8·4) + 32 = 8,224
   - conv2: 64·(4·4·32) + 64 = 32,832
   - conv3: 64·(3·3·64) + 64 = 36,928
   - fully connected: 3,136·512 + 512 = 1,606,144
   - output: 512·18 + 18 = 9,234
   - **Total ≈ 1.69 M parameters** (1,693,362 for 18 actions). About 95% of them sit in the single fully connected layer. That is tiny by today's standards: roughly 1/5,000 of an 8B LLM.
2. **What "50 million frames" means.** 50M × 4 (action repeat) / 60 Hz / 86,400 s = **38.6 days**, which matches the paper's "around 38 days". 50M / 60 Hz is only 9.6 days. So the 50M counts *agent steps*, i.e. 200M emulator frames. Later papers (e.g. Rainbow) count DQN's budget as 200M frames. *This is my interpretation of the arithmetic, not a sentence from the paper.*
3. **Target-network period in frames.** 10,000 updates × 4 actions per update × 4 frames per action = **160,000 emulator frames**, about 44 minutes of play at 60 Hz. Rainbow's table lists its own target period as "32K frames".
4. **Replay memory size.** One 84×84 uint8 frame is 7,056 bytes. A million frames stored once is **7.1 GB**; storing (s, s′) as two 4-frame stacks naively would be 56 GB. Implementations store each frame once and rebuild the stacks.
5. **ε at frame t.** t = 0 → 1.0; 250k → 0.775; 500k → 0.55; ≥ 1M → 0.1.
6. **Normalised score.** Breakout: 100 × (401.2 − 1.7)/(31.8 − 1.7) = **1,327.2%**. Montezuma: 100 × (0 − 0)/(4,367 − 0) = 0%.
- For the course's CartPole run, take its own hyperparameters from `research/experiments/`; the Atari values above are the reference point.

### Common misconceptions
- **"DQN beat humans at Atari."** DQN was ≥ 75% of one tester's score on 29 of 49 games and above 100% on 23 [DERIVED]. It scored 0% on Montezuma's Revenge, 2.5% on Private Eye and 5.3% on Gravitar. "Human-level" meant one professional tester after about 2 hours of practice per game, with 5-minute episodes.
- **"One network played all 49 games."** No. One *architecture and hyperparameter set*; a separate network was trained per game.
- **"DQN used only pixels."** It also used the game score (rewards, clipped to ±1 during training), the number of legal actions, and the life counter as an episode-end signal during training.
- **"The target network is updated every 10,000 frames."** It is every 10,000 *parameter updates* = 40,000 agent steps = 160,000 emulator frames [DERIVED].
- **"Experience replay was invented for DQN."** It is due to Lin (1992), and DQN cites it. DQN's contribution was making replay plus a target network work with a deep convolutional network.
- **"The 2013 and 2015 DQNs are the same."** The 2013 workshop paper had 2 conv layers (16 and 32 filters) and FC-256, no target network, and 7 games. The 2015 Nature paper had 3 conv layers (32, 64, 64) and FC-512, a target network, and 49 games.
- **"Scores across papers are comparable."** Evaluation regimes differ: up to 30 no-ops vs human starts, 5-minute vs 30-minute caps, 49 vs 57 games. So DQN's "median" is 93.5% in the Nature setting, 79% in the 57-game no-op setting used by Rainbow and Dueling, and 47.5% (68% in Rainbow's re-run) with human starts.

### Quotable
- "We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning." (Mnih et al. 2013, abstract)
- "Reinforcement learning is known to be unstable or even to diverge when a nonlinear function approximator such as a neural network is used to represent the action-value (also known as Q) function" (Mnih et al. 2015)
- "This work bridges the divide between high-dimensional sensory inputs and actions, resulting in the first artificial agent that is capable of learning to excel at a diverse array of challenging tasks." (Mnih et al. 2015, abstract)
- "…achieving more than 75% of the human score on more than half of the games (29 games…)" (Mnih et al. 2015)
- "Note that when learning by experience replay, it is necessary to learn off-policy (because our current parameters are different to those used to generate the sample), which motivates the choice of Q-learning." (Mnih et al. 2015, Methods)
- "Generating the targets using an older set of parameters adds a delay between the time an update to Q is made and the time the update affects the targets yj, making divergence or oscillations much more unlikely." (Mnih et al. 2015, Methods)

---

## §14 Overestimation and its fixes (Double Q, Double DQN, Dueling, Prioritized replay, Rainbow)

### Mechanism (plain words)
- **Maximization bias.** Q-learning's target uses max_a Q(s′, a). When the estimates are noisy, the max picks whichever action got *luckiest*, so the max of the estimates exceeds the max of the true values on average. This holds even when every individual estimate is unbiased.
  - S&B: "The maximum of the true values is zero, but the maximum of the estimates is positive, a positive bias. We call this maximization bias."
  - The bias then propagates backwards through bootstrapping.
- **Double learning.** Use one estimate to *choose* the best action and an independent estimate to *evaluate* it. The evaluation's error is then not correlated with the choice, so the bias disappears (it can even turn slightly negative).
  - Double Q-learning keeps two tables and updates one at random each step.
  - **Double DQN** reuses the DQN target network as the second estimator: the online net chooses and the target net evaluates. This needs no new network.
- **Dueling networks** change the architecture. Two heads compute V(s) and A(s, a), and are recombined as Q = V + (A − mean A). V can be learned from every transition, even when the actions don't matter much.
- **Prioritized replay** samples transitions with large |TD error| more often, and corrects the resulting bias with importance weights.
- **Rainbow** stacks six DQN extensions: Double Q, prioritized replay, dueling, multi-step returns, distributional RL (C51) and Noisy Nets. The ablations show which ones matter most.

### Key formulas
- **Q-learning target (DQN):** Y^DQN = r + γ max_a Q(s′, a; θ⁻)
- **Double Q-learning** (van Hasselt 2010, Alg. 1; S&B eq. 6.10). With probability ½ update Q₁, else Q₂ (symmetric):
  - Q₁(s, a) ← Q₁(s, a) + α [ r + γ Q₂(s′, argmax_a′ Q₁(s′, a′)) − Q₁(s, a) ]
- **Double DQN** (van Hasselt, Guez & Silver 2016):
  - Y^DoubleDQN = r + γ Q(s′, argmax_a Q(s′, a; θ_t); θ_t⁻)
  - "the weights of the second network θ′_t are replaced with the weights of the target network θ_t⁻ for the evaluation of the current greedy policy."
- **Size of the bias:**
  - Thrun & Schwartz (1993), as reported by van Hasselt et al. 2016: errors uniform in [−ε, ε] give each target an overestimation of "up to γε (m−1)/(m+1)", where m = number of actions.
  - Van Hasselt et al. 2016, Theorem 1: if the true values are all equal to V*(s), the estimates are unbiased overall (Σ_a (Q_t − V*) = 0), and the mean squared error is C > 0, then max_a Q_t(s, a) ≥ V*(s) + √(C/(m − 1)). The bound is tight. "Under the same conditions, the lower bound on the absolute error of the Double Q-learning estimate is zero."
- **Dueling aggregation** (Wang et al. 2016, eq. 9; their stream parameters α, β renamed θ_A, θ_V):
  - Q(s, a) = V(s; θ_V) + ( A(s, a; θ_A) − (1/|A|) Σ_a′ A(s, a′; θ_A) )
  - Eq. 7 (plain V + A) is "unidentifiable". Eq. 8 subtracts max_a′ A instead. The paper uses the mean version, eq. 9, for stability.
- **Prioritized replay** (Schaul et al. 2016; their α and β renamed α_PER and β_IS to avoid clashing with our step size and KL coefficient):
  - Sampling probability P(i) = p_i^α_PER / Σ_k p_k^α_PER.
  - Proportional variant: p_i = |δ_i| + ε. Rank-based variant: p_i = 1/rank(i).
  - Importance weights: w_i = (1/N · 1/P(i))^β_IS, normalised by 1/max_i w_i, and the update uses w_i δ_i.
  - β_IS is annealed linearly from β₀ to 1 by the end of training.
  - Tuned values: rank-based α = 0.7, β₀ = 0.5; proportional α = 0.6, β₀ = 0.4. The step size was reduced 4× vs (Double) DQN.
- **Rainbow hyperparameters** (Hessel et al. 2018, Table 1):

| Hyperparameter | Value |
|---|---|
| multi-step n | 3 |
| distributional atoms | 51, support [−10, 10] |
| prioritization | proportional, exponent ω = 0.5, IS β from 0.4 → 1.0 |
| optimiser | Adam, lr 0.0000625, Adam ε 1.5 × 10⁻⁴ |
| Noisy Nets σ₀ | 0.5 |
| ε (exploration) | 0.0 |
| target period | 32K frames |
| learning starts after | 80K frames |
| replay buffer | 1M transitions (Table 4) |

### Numbers with sources
**S&B Example 6.7 / Figure 6.5** (Sutton & Barto 2018, §6.7, p. 135):
- Two non-terminal states, A and B. From A: *right* → terminal, reward 0. *left* → B, reward 0.
- From B, "many possible actions all of which cause immediate termination with a reward drawn from a normal distribution with mean −0.1 and variance 1.0". So *left* has expected return −0.1 and is always a mistake.
- Settings: ε = 0.1, α = 0.1, γ = 1, initial Q = 0, random tie-breaking, averaged over 10,000 runs, 300 episodes.
- Result: Q-learning "initially learns to strongly favor the left action". "Even at asymptote, Q-learning takes the left action about 5% more often than is optimal". The optimal rate is **5%**, the ε/2 minimum forced by ε-greedy with 2 actions.
- "Double Q-learning is essentially unaffected by maximization bias."
- Reading the figure: the Q-learning curve peaks near **~95% left at around episode 20** and ends near 10% at 300. Double Q drops from 50% toward ~5–7%.
- Note: pdftotext loses the minus sign; the rendered page shows N(−0.1, 1).

**Van Hasselt 2010** (NeurIPS), "Double Q-learning", roulette experiment:
- 170 betting actions plus one walk-away action; almost all bets have expected payout $0.947 per $1, i.e. −$0.053 per play; γ = 0.95.
- "After 100,000 trials Q-learning valued all non-terminating actions between $22.63 and $22.67 for linear learning rates and between $9.58 to $9.64 for polynomial rates."
- True value of betting forever = −0.053/(1 − 0.95) = **−$1.06** [DERIVED], vs Q-learning's +$22.6.

**Double DQN (van Hasselt, Guez & Silver 2016, AAAI; arXiv:1509.06461):**
- "overestimations were observed for DQN in all 49 tested Atari games, albeit in varying amounts."
- Median human-normalised scores:

| Setting | DQN | Double DQN | Double DQN (tuned) |
|---|---|---|---|
| 49 games, 5-min, no-op (Table 1), median | 93.5% | **114.7%** | — |
| 49 games, 5-min, no-op (Table 1), mean | 241.1% | 330.3% | — |
| 49 games, 30-min, human starts (Table 2), median | 47.5% | 88.4% | **116.7%** |
| 49 games, 30-min, human starts (Table 2), mean | 122.0% | 273.1% | 475.2% |

**Dueling (Wang et al. 2016, ICML; arXiv:1511.06581).** Table 1, 57 games, mean / median % of human:

| Agent | 30 no-ops | Human starts |
|---|---|---|
| Prior. Duel Clip | 591.9% / **172.1%** | 567.0% / 115.3% |
| Prior. Single | 434.6% / 123.7% | 386.7% / 112.9% |
| Duel Clip | 373.1% / 151.5% | 343.8% / 117.1% |
| Single Clip | 341.2% / 132.6% | 302.8% / 114.1% |
| Single (Double DQN) | 307.3% / 117.8% | 332.9% / 110.9% |
| Nature DQN | 227.9% / 79.1% | 219.6% / 68.5% |

- Value and advantage streams each have a 512-unit fully connected layer.
- With human starts: "Duel Clip does better than Single Clip on 75.4% of the games (43 out of 57). It also achieves higher scores compared to the Single baseline on 80.7% (46 out of 57) of the games."

**Prioritized replay (Schaul et al. 2016, ICLR; arXiv:1511.05952):**
- Abstract: "DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games."
- Table 1, human starts:

| Agent | Median | Mean | Games |
|---|---|---|---|
| DQN | 48% | 122% | 49 |
| DQN rank-based | 106% | 355% | 49 |
| Double DQN (tuned) | 111% | 418% | 57 |
| Double DQN rank-based | 113% | 454% | 57 |
| Double DQN proportional | **128%** | **551%** | 57 |

- "…on aggregate, learning is twice as fast".

**Rainbow (Hessel et al. 2018, AAAI; arXiv:1710.02298).** Table 2, median human-normalised score:

| Agent | No-ops | Human starts |
|---|---|---|
| DQN | 79% | 68% |
| DDQN | 117% | 110% |
| Prioritized DDQN | 140% | 128% |
| Dueling DDQN | 151% | 117% |
| A3C | — | 116% |
| Noisy DQN | 118% | 102% |
| Distributional DQN | 164% | 125% |
| **Rainbow** | **223%** | **153%** |

- Exact sentence: "Rainbow achieves a median score of 223% in the no-ops regime; in the human starts regime we measured a median score of 153%."
- Data efficiency: "we match final performance of DQN after 7M frames, surpass the best final performance of these baselines in 44M frames, and reach substantially improved final performance."
- Wall clock: "The 7M frames required to match DQN’s final performance correspond to less than 10 hours of wall-clock time. A full run of 200M frames corresponds to approximately 10 days" on a single GPU.
- Ablations, Fig. 3 and §"Ablation studies":
  1. "Prioritized replay and multi-step learning were the two most crucial components of Rainbow, in that removing either component caused a large drop in median performance." The full Rainbow beat either ablation "in 53 games out of 57".
  2. Distributional RL "ranked immediately below". There is no difference for the first 40M frames, then it lags.
  3. Noisy Nets: removing them gave worse aggregate performance.
  4. Dueling: "we did not observe a significant difference when removing the dueling network".
  5. Double Q: the difference "is limited". The authors hypothesise that clipping the value distribution to [−10, 10] already "counteracts the overestimation bias".

### Worked example [DERIVED] (scripts/f5_checks.py, f5_per.py)
1. **"Max over noisy bars."** All true action values are 0; each estimate = truth + N(0, 1) noise. The expected max of k estimates:

| k | 1 | 2 | 3 | 5 | 10 | 18 (Atari's full action set) | 100 |
|---|---|---|---|---|---|---|---|
| E[max] | 0 | 0.564 (= 1/√π) | 0.846 | 1.163 | **1.539** | 1.820 | 2.508 |

   - With 10 actions the max overshoots the truth by **1.54 standard deviations**, even though every bar is unbiased.
   - With uniform [−1, 1] errors the overshoot is (m−1)/(m+1): 0.333 (m = 2), 0.600 (m = 4), **0.818 (m = 10)**. Monte Carlo agrees to 3 decimals.
2. **Double estimate.** Choose with set 1, evaluate with an independent set 2. E[Q₂(argmax Q₁)] = 0 exactly, because Q₂'s noise is independent of the choice. Animation: two rows of bars; the coral arrow picks the tallest in row 1, and the blue reading is taken from row 2.
3. **S&B Example 6.7 simulated** (10 actions in B is my assumption; the book says "many"; 10,000 runs):

| Episode | 1 | 5 | 10 | 20 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| Q-learning % left | 50 | 69 | 87 | 94 | 82 | 47 | 16 | 12 |
| Double Q % left | 50 | 49 | 46 | 32 | 15 | 9 | 7 | 7 |

   - Q-learning peaks at **94.1% at episode 21**, matching the book's figure. Optimal is 5%.
4. **Dueling aggregation.** V = 2 and raw advantages A = [1, 0, −1, 2], so mean(A) = 0.5 and Q = [2.5, 1.5, 0.5, 3.5]. The greedy action is unchanged; only the offset is pinned.
5. **Prioritized sampling.** |δ| = [2, 1, 0.5, 0.1] with α_PER = 0.6 gives P = [0.442, 0.292, 0.193, 0.073], vs 0.25 each under uniform.
   - IS weights (normalised by the max): β = 0.4 → [0.49, 0.58, 0.68, 1.0]; β = 1 → [0.17, 0.25, 0.38, 1.0].
   - Rank-based with α = 0.7: P = [0.407, 0.25, 0.189, 0.154].

### Common misconceptions
- **"Overestimation means the estimates are biased."** Each estimate can be unbiased; the *max* of them is biased. Theorem 1 holds even when errors are not independent: "estimation errors of any source can drive the estimates up and away from the true optimal values".
- **"Double DQN adds a second network."** It reuses the existing target network as the evaluator; only the target formula changes.
- **"Double learning doubles the compute."** In tabular Double Q only one table is updated per step: "double learning doubles the memory requirements, but does not increase the amount of computation per step" (S&B §6.7).
- **"Dueling is a new learning algorithm."** It is only an architecture (two streams plus mean-subtraction), usable with any Q-learning variant. In Rainbow's ablation it made no significant difference to the median.
- **"Prioritized replay is just sampling by TD error."** Non-uniform sampling biases the expectation. The IS weights, with β annealed to 1, are what make the converged solution correct.
- **"Rainbow's gain comes from Double Q fixing overestimation."** In Rainbow's own ablations, prioritized replay and multi-step returns mattered most; Double Q and Dueling mattered least on the median.
- **"Q-learning's overestimation always makes it fail."** Uniform overestimation is harmless to the policy. Van Hasselt et al.: "Overoptimistic value estimates are not necessarily a problem in and of themselves." It hurts when it is uneven across actions and states.

### Quotable
- "The maximum of the true values is zero, but the maximum of the estimates is positive, a positive bias. We call this maximization bias." (Sutton & Barto 2018, §6.7)
- "These overestimations result from a positive bias that is introduced because Q-learning uses the maximum action value as an approximation for the maximum expected action value." (van Hasselt 2010, abstract)
- "The popular Q-learning algorithm is known to overestimate action values under certain conditions." (van Hasselt, Guez & Silver 2016, abstract)
- "…it is known to sometimes learn unrealistically high action values because it includes a maximization step over estimated action values, which tends to prefer overestimated to underestimated values." (van Hasselt, Guez & Silver 2016, §1)
- "Our dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function." (Wang et al. 2016, abstract)
- "In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently." (Schaul et al. 2016, abstract)
- "Prioritized replay and multi-step learning were the two most crucial components of Rainbow, in that removing either component caused a large drop in median performance." (Hessel et al. 2018)

---

## §15 Search and self-play: MCTS, UCT, PUCT, AlphaGo → AlphaGo Zero → AlphaZero → MuZero

### Mechanism (plain words)
- **Search uses a model to look ahead before acting.** Exhaustive minimax over b^d move sequences is hopeless for Go (b ≈ 250, d ≈ 150). AlphaGo's Nature paper names the "two general principles" for shrinking the search: cut **depth** with a value function v(s), and cut **breadth** with a policy p(a|s).
- **Monte Carlo tree search (MCTS)** (S&B §8.11) grows a tree from the current state. Each iteration has four steps:
  1. **Selection:** walk down the tree with a *tree policy* that balances exploitation and exploration.
  2. **Expansion:** add one or more child nodes.
  3. **Simulation:** finish the game with a cheap *rollout policy*, or, in AlphaGo Zero and later, ask a value network instead.
  4. **Backup:** add the outcome to the visit count N and the value sum W of every edge on the path.

  After the budget runs out, play the most-visited root move.
- **UCT** (Kocsis & Szepesvári 2006) treats every tree node as a bandit and applies UCB1 (§3) at each node. "Optimism in the face of uncertainty" therefore drives the search.
- **PUCT** (Rosin 2011; AlphaGo 2016) adds a learned prior P(s,a) from a policy network. Moves the network likes are explored first, and the bonus decays with visits.
- **Self-play:** the agent plays both sides, so its opponent improves exactly as fast as it does.
- **AlphaGo Zero's loop is policy iteration** (§7).
  - Search is the *policy improvement operator*: MCTS visit counts π are stronger than the raw network policy p.
  - Playing games to the end is the *policy evaluation operator*: the winner z is a sample of the value.
  - The network is trained to imitate π and to predict z. The better network makes the next search better, and so on.
- **The lineage:**
  - **AlphaGo** (2016): human games for SL, then RL by self-play, then MCTS with rollouts + value net.
  - **AlphaGo Zero** (2017): no human data and one network with two heads.
  - **AlphaZero** (2017/18): the same algorithm for chess, shogi and Go, with no game-specific tuning.
  - **MuZero** (2019/20): also *learns the model*. It plans in a learned latent space and is never told the rules.
- **Language models are the counterpoint:** DeepSeek-R1 tried MCTS and listed it under "Unsuccessful Attempts" (see Quotable). The token search space is enormous, and a fine-grained value model is hard to train.

### Key formulas (paper notation → ours)
**UCB1 / UCT tree policy.** At node s, pick the child a that maximises

  Q̄(s,a) + c · √( ln N(s) / N(s,a) ),  where Q̄ ∈ [0,1].

- UCB1 (Auer et al. 2002) uses c = √2, i.e. √(2 ln n / n_j).
- Browne et al. (2012, IEEE TCIAIG survey, §3.3) write UCT as "X̄_j + 2C_p √(2 ln n / n_j)". They add that "The value C_p = 1/√2 was shown by Kocsis and Szepesvári to satisfy the Hoeffding ineqality with rewards in the range [0, 1]".
- **Note:** the survey's form with C_p = 1/√2 equals √(4 ln n/n_j), not UCB1's √(2 ln n/n_j). Search-engine extracts of K&S 2006 itself give the UCT bias term as **c_{t,s} = 2C_p √(ln t / s)**, which with C_p = 1/√2 is exactly UCB1's √(2 ln t / s); so the survey's extra 2 under the root looks like a transcription difference. The K&S PDF itself could not be downloaded from here (all three hosts refused), so the exact wording stays **[UNVERIFIED]**. For the course, use UCB1's √(2 ln N / n).
- Unvisited children are tried first (an infinite bonus).

**PUCT, AlphaGo (Silver et al. 2016, Methods).**
- a_t = argmax_a [Q(s_t,a) + u(s_t,a)], with u(s,a) = c_puct · P(s,a) · √(Σ_b N_r(s,b)) / (1 + N_r(s,a)).
- The prior P(s,a) = p_σ(a|s) comes from the **SL** policy network.
- Leaf evaluation mixes the value net and a fast rollout: V(s_L) = (1 − λ) v_θ(s_L) + λ z_L.
- Extended Data Table 5 values: **c_puct = 5**, **λ = 0.5**, softmax temperature β = 0.67, virtual loss n_vl = 3, expansion threshold n_thr = 40. (This β is a temperature, not the course's KL β.)

**PUCT, AlphaGo Zero (Silver et al. 2017).**
- U(s,a) = c_puct P(s,a) √(Σ_b N(s,b)) / (1 + N(s,a)).
- Q(s,a) = W(s,a)/N(s,a) is the mean value of the simulations through that edge.
- The move is chosen from π(a|s_0) = N(s_0,a)^{1/τ} / Σ_b N(s_0,b)^{1/τ}. τ = 1 for the first 30 moves; after that τ → 0, i.e. play the most-visited move.
- **Root exploration noise:** P(s,a) = (1 − ε) p_a + ε η_a, with η ~ Dir(0.03) and **ε = 0.25**.
- AlphaZero scales the Dirichlet α "in inverse proportion to the approximate number of legal moves": α = 0.3 (chess), 0.15 (shogi), 0.03 (Go).

**PUCT, AlphaZero (Science 2018) and MuZero.**
- AlphaZero: U(s,a) = C(s) P(s,a) √N(s) / (1 + N(s,a)), with C(s) = log((1 + N(s) + c_base)/c_base) + c_init, which "is essentially constant during the fast training games".
- MuZero writes the same rule with c1 = **1.25** and c2 = **19652**.

**AlphaGo Zero / AlphaZero training loss** (one network, (p, v) = f_θ(s)):

  l = (z − v)² − πᵀ log p + c‖θ‖²

- The value head regresses on the game outcome z ∈ {−1, 0, +1}.
- The policy head is pulled toward the search probabilities π (cross-entropy).
- c is the L2 weight on θ.
- **Notation clash:** in these papers **π means the MCTS visit distribution**, not the network policy. The network policy is **p**. In course notation, p = π_θ(·|s), their π = π_MCTS, and z = G (the final, undiscounted outcome).

**AlphaGo's RL policy step is REINFORCE** (§16): Δρ ∝ ∂log p_ρ(a_t|s_t)/∂ρ · z_t, where z_t = ±1 is the game result. Methods: "using the REINFORCE algorithm with baseline v(s_t) for variance reduction" (on the second pass).

**MuZero's learned model.** It has three functions, unrolled for K = 5 steps:
- representation h: observations → s^0;
- dynamics g: (s^{k−1}, a^k) → (r^k, s^k);
- prediction f: s^k → (p^k, v^k).

The hidden state "s^k" is not a real board. It is trained only so that predicted rewards, values and policies match the real ones.

### Numbers with sources
**AlphaGo** (Silver et al. 2016, "Mastering the game of Go with deep neural networks and tree search", Nature 529:484–489, 28 Jan 2016)
- SL policy network: 13 layers, trained on **30 million positions** from the KGS server. Methods: 29.4M positions from 160,000 games by 6–9 dan players. Move-prediction accuracy "57.0% using all input features, and 55.7% using only raw board position and move history". Training took "around 3 weeks for 340 million training steps" on 50 GPUs.
- Fast rollout policy: 24.2% accuracy, "using just 2 μs to select an action, rather than 3 ms for the policy network".
- RL policy network: trained by self-play policy gradient for "10,000 minibatches of 128 games, using 50 GPUs, for one day". Results: "the RL policy network won more than 80% of games against the SL policy network", and "Using no search at all, the RL policy network won 85% of games against Pachi". The previous SL-only CNN state of the art won 11%.
- Value network: trained on 30 million positions, "each sampled from a separate game", using "50 million mini-batches of 32 positions, using 50 GPUs, for one week".
- Compute and strength (Extended Data Table 6): single-machine AlphaGo had 48 CPUs and 8 GPUs, Elo 2890. **Distributed AlphaGo had 1,202 CPUs and 176 GPUs** (40 search threads), Elo 3140. Fan Hui is rated 2,908. All programs got 5 s per move.
- Results: "winning 494 out of 495 games (99.8%) against other Go programs". With 4 handicap stones it won 77%, 86% and 99% against Crazy Stone, Zen and Pachi. The distributed version beat single-machine AlphaGo 77% of the time.
- Fan Hui (European champion 2013–15, professional 2 dan): "Over 5–9 October 2015 … AlphaGo won the match 5 games to 0."
- Leaf-evaluation ablation: the mixed λ = 0.5 "performed best, winning ≥95% of games against other variants".
- Lee Sedol (DeepMind AlphaGo page): "AlphaGo's 4-1 victory in Seoul, South Korea, in March 2016 was watched by over 200 million people worldwide". Move 37 in game 2 "had a 1 in 10,000 chance of being used". AlphaGo Lee ran on **48 TPUs**, distributed (AGZ paper).

**AlphaGo Zero** (Silver et al. 2017, "Mastering the game of Go without human knowledge", Nature 550:354–359, 19 Oct 2017; quotes from the UCL-hosted author manuscript)
- Abstract: "Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100-0 against the previously published, champion-defeating AlphaGo."
- 3-day run: "Over the course of training, 4.9 million games of self-play were generated, using 1,600 simulations for each MCTS, which corresponds to approximately 0.4s thinking time per move. Parameters were updated from 700,000 mini-batches of 2,048 positions." The network had 20 residual blocks.
- "AlphaGo Zero outperformed AlphaGo Lee after just 36 hours". At 72 h it beat AlphaGo Lee **100–0** under match conditions (2 h time controls). It used "a single machine with 4 Tensor Processing Units (TPUs)", where AlphaGo Lee "used 48 TPUs".
- 40-day run: 40 residual blocks, **29 million** self-play games, 3.1 million mini-batches of 2,048.
- Elo: AlphaGo Zero 5,185, AlphaGo Master 4,858, AlphaGo Lee 3,739, AlphaGo Fan 3,144. The **raw network with no search scored 3,055**.
- Head-to-head vs AlphaGo Master: "AlphaGo Zero won by 89 games to 11" over 100 games at 2 h time controls.
- Pipeline details:
  - Each iteration plays 25,000 self-play games.
  - A new network replaces the best one only if it wins by more than 55%.
  - Training samples come from the most recent 500,000 games.
  - Optimisation ran on "64 GPU workers and 19 CPU parameter servers".

**AlphaZero.** There are two versions, and their numbers differ (see the disagreements below).
- **Preprint** (Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", arXiv 1712.01815, 5 Dec 2017):
  - "Starting from random play, and given no domain knowledge except the game rules, AlphaZero achieved within 24 hours a superhuman level of play in the games of chess and shogi (Japanese chess) as well as Go".
  - Training: 700,000 steps with batches of 4,096, on "5,000 first-generation TPUs" for self-play and **64** second-generation TPUs for training.
  - It outperformed Stockfish after 4 h (300k steps), Elmo after < 2 h (110k), and AlphaGo Lee after 8 h (165k).
  - Table 1 (100-game matches, 1 min per move):
    - chess vs Stockfish: 25W/25D/0L as White and 3W/47D/0L as Black, so **28 W, 72 D, 0 L** [DERIVED sum];
    - shogi vs Elmo: 90 W, 2 D, 8 L;
    - Go vs AlphaGo Zero (3 days): 60 W, 40 L.
  - Search speed: "80 thousand positions per second in chess and 40 thousand in shogi, compared to 70 million for Stockfish and 35 million for Elmo".
- **Science version** (Silver et al. 2018, "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play", Science 362:1140–1144, 7 Dec 2018; quotes from the DeepMind-hosted preprint):
  - Training used "5,000 first-generation tensor processing units (TPUs) … to generate self-play games, and 16 second-generation TPUs" to train. It "lasted for approximately 9 hours in chess, 12 hours in shogi and 13 days in Go".
  - Table S3:

    | Game | Self-play games | Simulations per move | Time per move |
    |---|---|---|---|
    | Chess | 44 million | 800 | ~40 ms |
    | Shogi | 24 million | 800 | ~80 ms |
    | Go | 140 million | 800 | ~200 ms |

  - Time to first beat the champion program: Stockfish after 4 h (300k steps), Elmo after 2 h (110k), AlphaGo Lee after **30 h (74k steps)**.
  - Chess: "AlphaZero defeated Stockfish, winning 155 games and losing 6 games out of 1,000", so **839 draws** [DERIVED]. Time control was 3 h + 15 s per move against Stockfish 8, the 2016 TCEC champion.
  - Shogi vs Elmo: "winning 98.2% of games". Go vs AlphaGo Zero: "winning 61% of games".
  - Search speed: "AlphaZero searches just 60,000 positions per second in chess and shogi, compared with 60 million for Stockfish and 25 million for Elmo", which is **1,000× fewer** in chess [DERIVED].
  - How it differs from AlphaGo Zero:
    - it optimises the expected outcome, so draws count;
    - it does not augment the data with the 8 board symmetries;
    - it has no evaluator gate ("simply maintains a single neural network that is updated continually");
    - it uses the same hyperparameters for every game.

**MuZero** (Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", arXiv 1911.08265, 19 Nov 2019; Nature 588:604–609, 23 Dec 2020)
- Abstract: "When evaluated on Go, chess and shogi, without any knowledge of the game rules, MuZero matched the superhuman performance of the AlphaZero algorithm that was supplied with the game rules."
- Setup: K = 5 unroll steps and 1 million mini-batches (batch 2,048 for board games, 1,024 for Atari). Search used 800 simulations per move for board games and **50** for Atari. Hardware: "16 TPUs for training and 1000 TPUs for selfplay" per board game, and 8 + 32 TPUs per Atari game.
- Atari (Table 1, human-normalised):
  - **MuZero: median 2041.1%, mean 4999.2%** (20.0B frames, 12 h, 1M steps).
  - R2D2: median 1920.6%, mean 4024.9%.
  - Ape-X: median 434.1%, mean 1695.6%.
  - Small-data setting (200M frames): MuZero Reanalyze median **731.1%**, vs Rainbow 231.1%, IMPALA 191.8% and LASER 431%.
  - MuZero beats R2D2 "in 42 out of 57 games".

**Context** (S&B §8.11): "MCTS is largely responsible for the improvement in computer Go from a weak amateur level in 2005 to a grandmaster level (6 dan or more) in 2015."

### Tic-tac-toe facts [DERIVED, exhaustive enumeration, scripts/ttt.py; all match the standard published counts]
| Quantity | Value |
|---|---|
| Naive board fillings 3⁹ | 19,683 |
| **Legal positions reachable in play** (incl. empty board) | **5,478** |
| … of which terminal / non-terminal | 958 / 4,520 |
| Distinct positions up to the 8 board symmetries | **765** (138 of them terminal) |
| Upper bound on move orders 9! | 362,880 |
| **Complete games** (sequences to a win or full board) | **255,168** |
| … X wins / O wins / draws | 131,184 / 77,904 / 46,080 |
| Games by length (5, 6, 7, 8, 9 moves) | 1,440 · 5,328 · 47,952 · 72,576 · 127,872 |
| **Minimax value with perfect play** | **draw (0)** |
| X's openings (corner / edge / centre) | all draws with perfect play |
| After X takes the centre, O replies… | corner → draw; edge → **X wins** |
| After X takes a corner, O replies… | centre → draw; **any other reply loses** |

- S&B §1.5 uses tic-tac-toe as the first RL example. Every state gets a value table entry, initialised to 0.5, with 1 for X wins and 0 for losses and draws. Updates use V(S_t) ← V(S_t) + α[V(S_{t+1}) − V(S_t)] after greedy moves. S&B note: "Because a skilled player can play so as never to lose, let us assume that we are playing against an imperfect player".
- Search-space comparison [DERIVED]: Go 250^150 ≈ 10^359.7, chess 35^80 ≈ 10^123.5 (b and d from AlphaGo 2016), and tic-tac-toe ≤ 9! = 362,880 orders, 255,168 actual games. A tabular agent can hold all 5,478 tic-tac-toe states (765 with symmetry), but not Go's.

### Worked examples [DERIVED, scripts/f1_arith.py and scripts/ttt_mcts.py]
1. **UCT picks the under-explored move.** The parent has been visited n = 10 times, and rewards are in [0,1].

   | Child | Wins/visits | Q̄ | Bonus √(2 ln 10 / n_j) | Score |
   |---|---|---|---|---|
   | a | 4/6 | 0.667 | 0.876 | 1.543 |
   | b | 2/3 | 0.667 | 1.239 | 1.906 |
   | c | 0/1 | 0 | 2.146 | **2.146** |

   UCT tries c next, even though it lost its only playout.
2. **PUCT trusts the prior.** Use c_puct = 1 and ΣN = 10, so √10 = 3.162.

   | Child | P | N | Q | U = P·√10/(1+N) | Score |
   |---|---|---|---|---|---|
   | a | 0.5 | 6 | 0.20 | 0.226 | 0.426 |
   | b | 0.3 | 3 | 0.40 | 0.237 | **0.637** |
   | c | 0.2 | 1 | −0.10 | 0.316 | 0.216 |

   PUCT picks b: a high value plus a decent prior. The low-prior move c is no longer forced.
3. **Plain UCT on the empty tic-tac-toe board** (UCB1 with c = √2, random rollouts, seed 1).
   - Root visit counts for cells 0–8:
     - 1,000 sims: [143, 68, 96, 86, **219**, 56, 141, 86, 105]
     - 10,000 sims: [1195, 493, 760, 524, **5173**, 315, 656, 305, 579]
     - 100,000 sims: [6773, 5928, 7212, 6391, **48398**, 5680, 6945, 5726, 6947]
   - The centre (cell 4) dominates. Its mean reward for X falls from 0.77 (1k) to 0.755 (10k) to **0.598** (100k), moving toward the minimax value 0.5 (a draw).
   - Lesson for the video: early estimates reflect how *random* play goes (random opponents blunder). As the tree deepens, the estimates move toward *perfect-play* values. These are illustrative numbers, not the experiments folder's.
4. **Throughput** [DERIVED].
   - AlphaGo Zero 3-day run: 4.9M games ÷ 72 h = **18.9 games/s**. The 40-day run: 29M ÷ 40 d = 8.4 games/s.
   - One search: 1,600 simulations ÷ 0.4 s = 4,000 network evaluations/s.
   - AlphaZero Go: 140M games ÷ 13 d ≈ 125 games/s.
5. **Elo sanity check** [DERIVED]: P(win) = 1/(1 + 10^(−Δ/400)).
   - Δ = 3140 − 2908 = 232 gives 0.79. The paper says "a 230 point gap corresponds to a 79% probability of winning".
   - AlphaGo Zero vs Master: Δ = 327 gives 0.87 expected; the observed score was 0.89.

### Common misconceptions
- **"AlphaGo learned only from self-play."** That was AlphaGo *Zero*. AlphaGo (2016) started from 30M human positions (SL) and then improved by RL self-play. AlphaGo also notes that "the SL policy network pσ performed better in AlphaGo than the stronger RL policy network pρ, presumably because humans select a diverse beam of promising moves, whereas RL optimizes for the single best move." The RL policy was used to train the value network.
- **"MCTS means random rollouts."** AlphaGo mixed rollouts with a value net (λ = 0.5). AlphaGo Zero, AlphaZero and MuZero use **no rollouts at all**: the leaf is evaluated by the value head.
- **"The network alone is the player."** Search adds a lot: AlphaGo Zero's raw network is Elo 3,055, while network + MCTS is 5,185.
- **"AlphaZero searches more than Stockfish."** It searches about 1,000× fewer positions per second (60k vs 60M) and relies on a better evaluation.
- **"MuZero is given the rules."** It is not. It learns a latent model that predicts reward, value and policy, never the next board.
- **"AlphaZero trained in 24 hours for Go."** The preprint said "within 24 hours a superhuman level", which meant beating AlphaGo Lee after 8 h. The Science version trained Go for **13 days** and first beat AlphaGo Lee after 30 h.
- **"Tic-tac-toe is a win for the first player."** Perfect play is a draw, whichever opening X picks.

### Disagreements between sources
- **AlphaZero preprint (Dec 2017) vs Science (Dec 2018).**
  - Training TPUs: 64 vs 16 second-generation TPUs.
  - Time for Go to beat AlphaGo Lee: 8 h vs 30 h.
  - Search speed: 80k/40k vs 60k positions/s, with Stockfish at 70M vs 60M and Elmo at 35M vs 25M.
  - Matches: 100 games at 1 min/move vs 1,000 games at 3 h + 15 s.
  - Go result vs AlphaGo Zero: 60/100 vs 61%.
  - Cite the Science numbers as final, and the preprint's only when you say so.
- **UCT constant:** see the note under Key formulas. The Browne et al. survey's expression does not reduce to UCB1 with C_p = 1/√2, while K&S's own 2C_p√(ln t/s) (seen only in search extracts) does.

### Quotable
- "Without any lookahead search, the neural networks play Go at the level of state-of-the-art Monte Carlo tree search programs that simulate thousands of random games of self-play." (Silver et al. 2016, abstract)
- "This is the first time that a computer program has defeated a human professional player in the full-sized game of Go, a feat previously thought to be at least a decade away." (Silver et al. 2016, abstract)
- "During the match against Fan Hui, AlphaGo evaluated thousands of times fewer positions than Deep Blue did in its chess match against Kasparov" (Silver et al. 2016)
- "AlphaGo becomes its own teacher: a neural network is trained to predict AlphaGo's own move selections and also the winner of AlphaGo's games." (Silver et al. 2017, abstract)
- "MCTS may therefore be viewed as a powerful policy improvement operator." (Silver et al. 2017)
- "AlphaZero searches just 60,000 positions per second in chess and shogi, compared with 60 million for Stockfish and 25 million for Elmo" (Silver et al. 2018)
- "The two methods that seem to scale arbitrarily in this way are search and learning." (Sutton, "The Bitter Lesson", 2019)
- "while MCTS can improve performance during inference when paired with a pre-trained value model, iteratively boosting model performance through self-search remains a significant challenge." (DeepSeek-AI 2025, R1 paper, §4.2 "Unsuccessful Attempts")

---

## §16 Policy gradients: REINFORCE and the log-derivative trick

**Notation mapping.** Sutton et al. 2000 write ρ for the objective (ours: J(θ)), d^π(s) for the on-policy state distribution (S&B: μ(s)), and π(s,a) for π(a|s). Williams 1992 writes r for the reinforcement, b for the baseline and e_ij = ∂ln g_i/∂w_ij for the "characteristic eligibility" (ours: ∇_θ ln π(a|s)). S&B ch. 13 writes h(s,a,θ) for action preferences (logits) and x(s,a) for feature vectors.

### Mechanism (plain words)
- Value methods (§9–§14) learn Q and then act greedily. Policy-gradient methods skip that step: they give the policy its own parameters θ, write the policy as a probability distribution π_θ(a|s), and climb the gradient of expected return J(θ) = E[G_0] directly.
- The obstacle is that the return depends on θ only through *which actions get sampled*. You cannot backpropagate through a dice roll, and you do not have the environment's equations.
- **The log-derivative trick** turns the gradient of an expectation into an expectation of something you can sample: ∇π = π·∇ln π. The environment's transition probabilities do not depend on θ, so they drop out of ∇ln p_θ(trajectory). The method needs **no model and no differentiable reward**. This is exactly why it works later on with answer checkers and unit tests (§24).
- **REINFORCE** (Williams 1992): play a whole episode. For every step t, nudge θ along ∇ln π(a_t|s_t), scaled by the return G_t that followed. Actions followed by high returns become more probable, and actions followed by low returns less probable.
- It is "supervised learning on your own samples, weighted by how well they turned out": for a softmax policy, ∇ln π is exactly the cross-entropy gradient "one-hot(action) − probabilities".
- Why parameterise the policy at all? S&B §13.1 gives three reasons:
  - stochastic optimal policies (Example 13.1 below);
  - continuous actions (§19);
  - smooth changes: a small change in θ changes action probabilities smoothly, whereas an argmax over Q can flip.

### Key formulas
1. **Policy gradient theorem** (Sutton, McAllester, Singh & Mansour 2000, Theorem 1, Eq. 2; checked against the page image):
   ∂ρ/∂θ = Σ_s d^π(s) Σ_a ∂π(s,a)/∂θ · Q^π(s,a)
   In our notation: **∇J(θ) = Σ_s d^π(s) Σ_a ∇π(a|s) Q^π(s,a)**. This holds "for any MDP, in either the average-reward or start-state formulations".
   S&B 2nd ed. Eq. (13.5) writes it as ∇J(θ) ∝ Σ_s μ(s) Σ_a q_π(s,a) ∇π(a|s,θ). "In the episodic case, the constant of proportionality is the average length of an episode, and in the continuing case it is 1."
2. **Log-derivative trick** [standard derivation]:
   ∇_θ E_{x∼p_θ}[f(x)] = Σ_x f(x) ∇p_θ(x) = Σ_x f(x) p_θ(x) ∇ln p_θ(x) = E[f(x) ∇ln p_θ(x)].
   For a trajectory τ, ln p_θ(τ) = ln p(s_0) + Σ_t ln π_θ(a_t|s_t) + Σ_t ln P(s_{t+1}|s_t,a_t). Only the middle sum depends on θ, so
   **∇J = E_τ[ Σ_t G_t ∇ln π_θ(a_t|s_t) ]**. Using the reward-to-go G_t instead of the full return G_0 is allowed because rewards before t cannot depend on a_t.
3. **REINFORCE update** (S&B Eq. 13.8): θ ← θ + α G_t ∇ln π(A_t|S_t,θ).
   The boxed pseudocode (p. 328) carries an extra γ^t because "in the text we are treating the non-discounted case (γ = 1)". Most deep-RL code drops the γ^t factor [practice; see S&B Exercise 13.2].
4. **Williams's original form** (Williams 1992, §4): Δw_ij = α_ij (r − b_ij) e_ij, with e_ij = ∂ln g_i/∂w_ij. The baseline b_ij is part of REINFORCE from the very first paper.
5. **Softmax policy** (S&B 13.2, 13.3, 13.9):
   - π(a|s) = e^{h(s,a)} / Σ_b e^{h(s,b)}, with linear preferences h = θᵀx(s,a);
   - **∇ln π(a|s) = x(s,a) − Σ_b π(b|s) x(s,b)**;
   - with one logit per action (tabular, or the last layer of a network): ∂ln π(a)/∂h_b = 1[a = b] − π(b).
6. **Gaussian policy** (S&B 13.19–13.20, Exercise 13.4):
   - π(a|s) = (1/(σ√(2π))) exp(−(a−μ)²/(2σ²)), with μ = θ_μᵀx_μ(s) and σ = exp(θ_σᵀx_σ(s));
   - **∇_{θμ} ln π = (a−μ)/σ² · x_μ(s)**;
   - **∇_{θσ} ln π = ((a−μ)²/σ² − 1) · x_σ(s)**.

### Numbers with sources
- **Short corridor with switched actions** (S&B Example 13.1, p. 323): reward −1 per step, 3 non-terminal states that all look identical (x(s,right) = [1,0]ᵀ, x(s,left) = [0,1]ᵀ), and actions reversed in the 2nd state.
  - With ε = 0.1, ε-greedy can only choose between "right with prob. 1 − ε/2" and "left with prob. 1 − ε/2". These achieve start-state values of less than −44 and −82.
  - "The best probability is about 0.59, which achieves a value of about −11.6." (minus signs restored; pdftotext drops them)
  - A stochastic policy beats every near-deterministic one, and only a policy-gradient method can represent it here.
- **S&B Figure 13.1** (REINFORCE on the short corridor, 100 runs, 1,000 episodes): step sizes α = 2⁻¹², 2⁻¹³, 2⁻¹⁴. Figure 13.2 says 2⁻¹³ is the best power of two.
- **S&B Figure 13.2**: REINFORCE with a learned baseline (v̂(s,w) = w, one scalar), with α^θ = 2⁻⁹ and α^w = 2⁻⁶, learns much faster than plain REINFORCE at α = 2⁻¹³.
- **Williams 1992**: Ronald J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", *Machine Learning* 8, 229–256 (1992). Theorem 1: "if α_ij = α is independent of i and j, then E{ΔW|W} = α∇_W E{r|W}". REINFORCE follows the true gradient in expectation. (Read from the author's PDF via OCR.)
- **Sutton et al. 2000** (NIPS 12; conference Dec 1999, proceedings 2000): "we prove for the first time that a version of policy iteration with arbitrary differentiable function approximation is convergent to a locally optimal policy."
- **REINFORCE is back in LLM training** [DATED]: Ahmadian et al. 2024, "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (arXiv 2402.14740; this is RLOO). Quote: "far simpler REINFORCE-style optimization variants outperform both PPO and newly proposed "RL-free" methods such as DPO and RAFT." GRPO (§23) is likewise REINFORCE with a group-mean baseline plus PPO's clip.

### Worked example [DERIVED] (checked in scripts/f6_checks.py)
**A. The log-derivative trick, on two of Rover's slot machines.**
- Rover pulls machine 1 with probability π₁ = σ(θ) and machine 2 with 1 − σ(θ). Machine 1 pays 1, machine 2 pays 0. So J(θ) = σ(θ) and the true gradient is σ(1−σ) = **0.25** at θ = 0.
- Score function: pulling machine 1 gives ∇ln π₁ = 1 − σ = 0.5; pulling machine 2 gives ∇ln(1−σ) = −σ = −0.5.
- E[score × reward] = 0.5·(0.5·1) + 0.5·(−0.5·0) = **0.25**. Same number, and no derivative of the reward was needed.

**B. One REINFORCE step on CartPole's two actions (left, right).**
- Logits θ = [0, 0], so π = [0.5, 0.5]. Rover pushed *right*, and the episode's return from that step was G = 20. Step size α = 0.01.
- ∇ln π(right) = one-hot − π = [−0.5, +0.5].
- Δθ = α·G·∇ln π = 0.01 × 20 × [−0.5, 0.5] = **[−0.1, +0.1]**.
- New π(right) = 1/(1 + e^{−0.2}) = **0.5498**: up from 0.5000.

**C. Why every CartPole action gets pushed up, and the fix (preview of §17).**
- CartPole pays +1 per step, so every G_t > 0 and every sampled action is reinforced. Only the *differences* between returns carry the signal.
- Take a baseline b = 15 (the average return):
  - with G = 20: Δθ = 0.01 × (20 − 15) × [−0.5, 0.5] = [−0.025, +0.025], so π(right) = 0.5125;
  - with G = 10: the update reverses, Δθ = [+0.025, −0.025], so π(right) = 0.4875.

**D. Variance with and without a baseline** (two-arm example A, θ = 0):
- Rewards (1, 0): the single-sample gradient is 0.5 or 0. Mean 0.25, variance 0.0625.
- Rewards (101, 100): same gradient, because adding a constant changes nothing. The samples are now **+50.5 or −50.0**: mean still 0.25, but **std 50.25, 200× the signal**.
- Subtract the baseline b = 100.5 (the mean reward): both samples become exactly 0.25, so the variance is **0**.

### Common misconceptions
- *"You need to differentiate through the environment or the reward."* No. The dynamics terms vanish from ∇ln p_θ(τ), and the reward is only a multiplier. This is why a 0/1 answer checker can train a language model.
- *"The policy gradient involves how the state distribution changes."* The theorem's point is the opposite. Sutton et al.: "their are no terms of the form ∂d^π(s)/∂θ" (sic).
- *"REINFORCE pushes up every action in a good episode equally."* The push is G_t·∇ln π. For a softmax, the factor (1 − π(a)) means already-likely actions get small pushes.
- *"Policy gradients find the best policy."* They find a local optimum (Sutton et al. 2000; S&B §13.3). Step size matters: S&B Fig. 13.1 shows α = 2⁻¹² doing worse than 2⁻¹³.
- *"REINFORCE is a historical curiosity."* RLOO (2024) and GRPO (2024) are REINFORCE-family estimators with baselines, used to train LLMs [DATED].

### Quotable
- "The name is an acronym for "REward Increment = Nonnegative Factor times Offset Reinforcement times Characteristic Eligibility," which describes the form of the algorithm." (Williams 1992, §4)
- "In any event, the key aspect of both expressions for the gradient is that their are no terms of the form ∂d^π(s)/∂θ: the effect of policy changes on the distribution of states does not appear." (Sutton et al. 2000; "their" sic)
- "However, as a Monte Carlo method REINFORCE may be of high variance and thus produce slow learning." (S&B 2nd ed., §13.3)
- "The best probability is about 0.59, which achieves a value of about −11.6." (S&B Example 13.1)

---

## §17 Baselines, advantages, actor–critic, A2C/A3C, GAE

**Notation mapping.** A3C uses β for the entropy weight; we write **c_ent** (PPO calls it c₂) to keep β for the KL coefficient. GAE's δ^V_t is our TD error δ_t. GAE's "γ-just" means "unbiased for the γ-discounted policy gradient".

### Mechanism (plain words)
- **Baseline.** Subtract from the return any quantity b(s) that does not depend on the action. The gradient stays the same in expectation, but its noise drops, sometimes enormously (§16, example D).
- **Advantage.** The natural baseline is V(s), which turns G_t − V(s_t) into an estimate of the advantage A(s,a) = Q(s,a) − V(s): "how much better than usual was this action?" Positive pushes the action up, negative pushes it down, and zero means no signal.
- **Actor–critic.** Learn V with a second network or head, the **critic**, by TD (§9). Then replace the Monte Carlo return with a bootstrapped one: the one-step TD error δ_t = r + γV(s') − V(s) is itself an advantage estimate.
  - The actor no longer waits for the episode to end, and the variance falls.
  - The price is bias whenever the critic is wrong.
- **A3C** (Mnih et al. 2016): many CPU threads run their own copy of the environment, compute n-step advantage gradients (t_max = 5), and apply them asynchronously to shared parameters. There is no replay buffer, and the parallel actors decorrelate the data instead.
  - An **entropy bonus** keeps the policy from collapsing too early.
  - **A2C** is the synchronous version: wait for all actors, then average. OpenAI found it equal or better, and more GPU-friendly.
- **GAE** (Schulman et al. 2016) is the dial between the two extremes:
  - λ = 0: the one-step TD error (low variance, biased if V is wrong);
  - λ = 1: the full Monte Carlo return minus V (unbiased, high variance).
  Everything in between is an exponentially weighted average of k-step advantage estimates.

### Key formulas
1. **Baseline is unbiased** (S&B Eq. 13.10 and the line below it): Σ_a b(s) ∇π(a|s,θ) = b(s) ∇Σ_a π(a|s,θ) = b(s) ∇1 = 0.
2. **REINFORCE with baseline** (S&B 13.11): θ ← θ + α (G_t − b(S_t)) ∇ln π(A_t|S_t,θ).
3. **One-step actor–critic** (S&B 13.12–13.14):
   θ ← θ + α (R_{t+1} + γ v̂(S_{t+1},w) − v̂(S_t,w)) ∇ln π(A_t|S_t,θ) = θ + α δ_t ∇ln π(A_t|S_t,θ)
   The critic update is w ← w + α_w δ_t ∇v̂(S_t,w).
4. **A3C gradient** (Mnih et al. 2016, §4 and Algorithm S3): ∇_θ log π(a_t|s_t) · Â_t + c_ent ∇_θ H(π(·|s_t)), with the k-step advantage Â_t = Σ_{i=0}^{k−1} γ^i r_{t+i} + γ^k V(s_{t+k}) − V(s_t), k ≤ t_max.
5. **GAE** (Schulman, Moritz, Levine, Jordan & Abbeel 2016):
   - definition (Eq. 16): **Â_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}**, with δ_t = r_t + γV(s_{t+1}) − V(s_t);
   - (17) GAE(γ, 0): Â_t = δ_t = r_t + γV(s_{t+1}) − V(s_t);
   - (18) GAE(γ, 1): Â_t = Σ_l γ^l δ_{t+l} = Σ_l γ^l r_{t+l} − V(s_t).
   - Truncated version for length-T segments (PPO paper Eq. 11): Â_t = δ_t + (γλ)δ_{t+1} + … + (γλ)^{T−t+1}δ_{T−1}.
   - Implemented as a backward recursion: Â_t = δ_t + γλ Â_{t+1}.
   - Value targets are usually Â_t + V(s_t), the "λ-return" [standard practice].
6. **Effective horizon** [DERIVED]: weights decay like (γλ)^l, so the horizon is ≈ 1/(1 − γλ).
   - γ = 0.99, λ = 0.95: γλ = 0.9405, so ≈ 16.8 steps for credit.
   - γ alone sets the value scale: ≈ 1/(1 − γ) = 100 steps.

### Numbers with sources
- **A3C**: Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver & Kavukcuoglu 2016, "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016, https://arxiv.org/abs/1602.01783
  - Abstract: "The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU."
  - Table 1 (57 Atari games, human starts, human-normalised score, mean / median):

    | Method | Training time | Mean | Median |
    |---|---|---|---|
    | DQN | 8 days on GPU | 121.9% | 47.5% |
    | Gorila | 4 days, 100 machines | 215.2% | 71.3% |
    | D-DQN | 8 days on GPU | 332.9% | 110.9% |
    | Dueling D-DQN | 8 days on GPU | 343.8% | 117.1% |
    | Prioritized DQN | 8 days on GPU | 463.6% | 127.6% |
    | A3C, FF | 1 day on CPU | 344.1% | 68.2% |
    | A3C, FF | 4 days on CPU | 496.8% | 116.6% |
    | A3C, LSTM | 4 days on CPU | 623.0% | 112.6% |

  - Setup (Supplementary §8):
    - "16 actor-learner threads running on a single machine and no GPUs";
    - updates every 5 actions (t_max = 5), shared RMSProp, γ = 0.99;
    - "Advantage actor-critic used entropy regularization with a weight β = 0.01 for all Atari and TORCS experiments";
    - initial learning rate sampled from LogUniform(10⁻⁴, 10⁻²) and annealed to 0;
    - the LSTM agent adds 256 LSTM cells.
- **A2C**: OpenAI blog, "OpenAI Baselines: ACKTR & A2C", 18 Aug 2017. Original URL blog.openai.com/baselines-acktr-a2c/, now openai.com/index/openai-baselines-acktr-a2c/; read via the Wayback Machine snapshot of 13 Feb 2019.
  - "A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which we've found gives equal performance."
  - "Our synchronous A2C implementation performs better than our asynchronous implementations — we have not seen any evidence that the noise introduced by asynchrony provides any performance benefit."
- **GAE**: Schulman, Moritz, Levine, Jordan & Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016, https://arxiv.org/abs/1506.02438
  - Best (γ, λ) found:
    - cart-pole: "γ ∈ [0.96, 0.99] and λ ∈ [0.92, 0.99]" (21 seeds);
    - 3D biped: "γ ∈ [0.99, 0.995], λ ∈ [0.96, 0.99]" (9 seeds);
    - quadruped: best with λ = 0.96 at γ = 0.995.
  - Conclusion: "choosing an appropriate intermediate value of λ in the range [0.9, 0.99] usually results in the best performance."
  - Biped experience: "0.01 seconds/timestep × 50000 timesteps/batch × 1000 batches / 3600·24 seconds/day = 5.8 days". The abstract says 3D bipeds need "1-2 weeks of real time" of simulated experience.
  - Each biped trial took "about 2 hours to run on a 16-core machine".
- **The defaults everyone copies**: PPO's Table 3 uses γ = 0.99, λ = 0.95 (§18). OpenAI Five used GAE λ = 0.95 (Berner et al. 2019, Table 2).

### Worked example [DERIVED]: GAE at the end of a CartPole episode
Three last steps before the pole falls: rewards r = (1, 1, 1). The critic's estimates are V(s₀..s₃) = (10, 9, 8, 0); s₃ is terminal, so V = 0. γ = 0.99.
- TD errors:
  - δ₀ = 1 + 0.99·9 − 10 = **−0.09**
  - δ₁ = 1 + 0.99·8 − 9 = **−0.08**
  - δ₂ = 1 + 0 − 8 = **−7.00**
- Advantage of the first action at three settings of λ:
  - λ = 0: Â₀ = δ₀ = **−0.09**. The critic says "fine": the fall is invisible from s₀.
  - λ = 0.95 (γλ = 0.9405): Â₀ = −0.09 + 0.9405·(−0.08) + 0.9405²·(−7) = **−6.357**.
  - λ = 1: Â₀ = (1 + 0.99 + 0.9801) − 10 = **−7.030**, the plain MC return minus the baseline.
- Animation idea: the pink δ spark at the fall travels back two steps, shrinking by 0.9405 per step.
- CartPole scale [DERIVED]: a perfect 500-step episode with γ = 0.99 has G₀ = (1 − 0.99⁵⁰⁰)/(1 − 0.99) = **99.34**. A good critic's V(s₀) is therefore ~99, not 500.

### Common misconceptions
- *"Subtracting a baseline biases the gradient."* Not if b depends only on the state (formula 1). A baseline that depends on the action *does* bias it.
- *"The critic makes the gradient exact."* A critic trades variance for bias. GAE: "GAE(γ, 0) is γ-just for V = V^{π,γ} and otherwise induces bias".
- *"A3C works because asynchrony adds useful noise."* OpenAI found no evidence of that, and synchronous A2C matched or beat it (blog, 2017).
- *"γ is just a technical knob."* γ < 1 changes the objective. GAE: "Taking γ < 1 introduces bias into the policy gradient estimate, regardless of the value function's accuracy." λ only adds bias when V is inaccurate.
- *"Advantage normalisation (subtract the mean, divide by the std) is part of the algorithm."* It is a code-level detail. In OpenAI baselines' PPO it happens per minibatch (Huang et al. 2022, "The 37 Implementation Details of PPO"). GRPO (§23) promotes a group-wise version of it to the core of the method.

### Quotable
- "In general, the baseline leaves the expected value of the update unchanged, but it can have a large effect on its variance." (S&B 2nd ed., §13.4)
- "GAE(γ, 1) is γ-just regardless of the accuracy of V, but it has high variance due to the sum of terms. GAE(γ, 0) is γ-just for V = V^{π,γ} and otherwise induces bias, but it typically has much lower variance." (Schulman et al. 2016)
- "Empirically, we find that the best value of λ is much lower than the best value of γ, likely because λ introduces far less bias than γ for a reasonably accurate value function." (Schulman et al. 2016)
- "A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which we've found gives equal performance." (OpenAI, 2017)

---

## §18 Trust regions: TRPO, PPO, OpenAI Five, and why PPO ran RLHF

**Notation mapping.**
- PPO writes r_t(θ) for the **probability ratio**. We write **ρ_t(θ) = π_θ(a_t|s_t)/π_θold(a_t|s_t)** so that r stays the reward.
- PPO's clip parameter ε is written **ε_clip** here; it is not ε-greedy.
- TRPO's trust-region radius δ is written **δ_KL**; it is not the TD error.
- TRPO's η(π) is our J(π), and its L_π(π̃) is the "surrogate" objective.
- PPO's β is a KL-penalty coefficient, the same role as our β.

### Mechanism (plain words)
- Vanilla policy gradient uses each batch of experience **once**, and a too-large step can wreck the policy. The next batch is then collected by the wrecked policy, so the method cannot easily recover.
- **Importance ratio.** To reuse a batch for several gradient steps, re-weight each old sample by ρ_t = π_new/π_old. The surrogate objective E[ρ_t Â_t] equals the true objective's gradient at ρ = 1, but becomes untrustworthy as the policy drifts away from the one that collected the data.
- **TRPO** (Schulman et al. 2015): maximise the surrogate subject to a hard limit on the average KL between old and new policies (δ_KL = 0.01).
  - It solves this approximately with a natural-gradient step, using conjugate gradient with Fisher-vector products, then a line search.
  - Theory (Theorem 1) guarantees monotonic improvement for a KL-*penalised* version. The practical algorithm relaxes it.
- **PPO** (Schulman et al. 2017) gets most of the benefit with first-order SGD. It **clips** the ratio to [1 − ε, 1 + ε] and takes the **minimum** of the clipped and unclipped terms.
  - If an action's probability has already moved far enough in the direction its advantage asks for, the gradient for that sample switches off.
  - If the move went the *wrong* way, the full (unclipped) term stays active, so the bound is pessimistic.
  - Then it runs K epochs of minibatch Adam on the same batch, recollects data, and repeats.
  - Alternatively, it can use an adaptive KL penalty, which performed worse in their tests.

### Key formulas
1. **TRPO Theorem 1** (Schulman et al. 2015, Eq. 8 onward): with ε_A = max_{s,a}|A_π(s,a)| and α = D_TV^max(π_old, π_new),
   J(π_new) ≥ L_{π_old}(π_new) − (4 ε_A γ / (1 − γ)²) α².
   Using D_TV² ≤ D_KL gives J(π̃) ≥ L_π(π̃) − C·D_KL^max(π, π̃), with C = 4ε_Aγ/(1−γ)².
   **Practical TRPO**: maximise L_θold(θ) subject to the average KL D̄_KL^ρθold(θold, θ) ≤ δ_KL.
2. **PPO surrogate** (Eq. 6): L^CPI(θ) = Ê_t[ρ_t(θ) Â_t].
3. **PPO clipped objective** (Eq. 7): **L^CLIP(θ) = Ê_t[ min( ρ_t(θ)Â_t , clip(ρ_t(θ), 1 − ε, 1 + ε) Â_t ) ]**.
4. **PPO adaptive KL** (Eq. 8): L^KLPEN = Ê_t[ρ_t Â_t − β KL[π_θold(·|s_t), π_θ(·|s_t)]].
   After each update, compute d = Ê[KL]. "If d < d_targ/1.5, β ← β/2; If d > d_targ × 1.5, β ← β × 2".
5. **Combined loss** with a shared network (Eq. 9): L_t^{CLIP+VF+S}(θ) = Ê_t[ L_t^CLIP(θ) − c₁ L_t^VF(θ) + c₂ S[π_θ](s_t) ], with L^VF = (V_θ(s_t) − V_t^targ)² and S the entropy bonus.
6. **Algorithm 1** (PPO, actor–critic style): N parallel actors each collect T steps. Compute GAE advantages, then "Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT", then set θ_old ← θ.

### Numbers with sources
- **TRPO**: Schulman, Levine, Moritz, Jordan & Abbeel, "Trust Region Policy Optimization", ICML 2015, https://arxiv.org/abs/1502.05477
  - Locomotion experiments: "We used δ = 0.01 for all experiments."
  - Abstract: "We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement."
- **PPO**: Schulman, Wolski, Dhariwal, Radford & Klimov, "Proximal Policy Optimization Algorithms", arXiv 1707.06347 (v2, 28 Aug 2017), https://arxiv.org/abs/1707.06347
  - **Table 1**: 7 MuJoCo tasks × 3 seeds = 21 runs, 1M timesteps each. Scores are normalised so that random = 0 and best = 1; β initialised at 1.

    | Variant | Avg normalised score |
    |---|---|
    | No clipping or penalty | −0.39 |
    | Clipping, ε = 0.1 | 0.76 |
    | **Clipping, ε = 0.2** | **0.82** |
    | Clipping, ε = 0.3 | 0.70 |
    | Adaptive KL, d_targ = 0.003 / 0.01 / 0.03 | 0.68 / 0.74 / 0.71 |
    | Fixed KL, β = 0.3 / 1 / 3 / 10 | 0.62 / 0.71 / 0.72 / 0.69 |

    The score without clipping is negative "because for one environment (half cheetah) it leads to a very negative score, which is worse than the initial random policy."
  - **Table 3 (MuJoCo, 1M steps)**:
    - horizon T = 2048; Adam step size 3 × 10⁻⁴; **10 epochs**; minibatch 64; γ = 0.99; λ = 0.95;
    - policy is an MLP with 2 hidden layers of 64 tanh units outputting a Gaussian mean with "variable standard deviations";
    - no parameter sharing and no entropy bonus.
  - **Table 4 (Roboschool humanoid)**: T = 512, 15 epochs, minibatch 4096, γ = 0.99, λ = 0.95, Adam step size adapted by KL.
  - **Table 5 (Atari)**:
    - T = 128; Adam 2.5 × 10⁻⁴ × α; **3 epochs**; minibatch 32 × 8; γ = 0.99; λ = 0.95;
    - 8 actors; **clipping 0.1 × α**; c₁ = 1; c₂ = 0.01;
    - "α is linearly annealed from 1 to 0 over the course of learning."
  - **Table 2 (Atari, 49 games, games "won")**:
    - average reward over all of training: A2C 1, ACER 18, **PPO 30**, tie 0;
    - over the last 100 episodes: A2C 1, **ACER 28**, PPO 19, tie 1.
    - PPO learns faster; ACER ends slightly better.
- **"PPO became the default"**: OpenAI blog "Proximal Policy Optimization", 20 Jul 2017 (blog.openai.com/openai-baselines-ppo/, read via the Wayback Machine): "PPO has become the default reinforcement learning algorithm at OpenAI because of its ease of use and good performance."
- **OpenAI Five**: Berner et al. (OpenAI), "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv 1912.06680 (13 Dec 2019), https://arxiv.org/abs/1912.06680
  - "On April 13th, 2019, OpenAI Five became the first AI system to defeat the world champions at an esports game." It beat OG "winning a best-of-three (2-0)".
  - "OpenAI Five is a single training run that ran from June 30th, 2018 to April 22nd, 2019. After ten months of training using 770±50 PFlops/s·days of compute…"
  - "OpenAI Five trained for 180 days (spread over 10 months of real time due to restarts and reverts)."
  - Scale:
    - batch sizes of 1 to 3 million timesteps; "a model with over 150 million parameters";
    - a single-layer **4096-unit LSTM**: "The LSTM composes 84% of the model's total parameter count.";
    - "Compared AlphaGo, we use 50 to 150 times larger batch size, 20 times larger model, and 25 times longer training time."
  - Batch: 120 samples × 16 timesteps per GPU × up to 1,536 optimizer GPUs = **2,949,120 timesteps** ([DERIVED] check: 120·16·1536 = 2,949,120 ✓).
  - Table 2 hyperparameters (final run):
    - optimizer GPUs 480↔1,536; rollout GPUs 500↔1,440; **rollout CPUs 80,000↔172,800**;
    - **PPO clipping 0.2**; **GAE λ 0.95**; GAE horizon 60 → 840 s;
    - entropy coefficient 0.01 → 0.001; learning rate 5e-5 ↔ 1e-6;
    - 20% of games against past opponents, 80% against itself.
  - Game: Dota "games run at 30 frames per second for approximately 45 minutes"; Five acts every 4th frame, so "approximately 20,000 steps per episode" ("chess usually lasts 80 moves, Go 150 moves").
  - Horizon H = T/(1 − γ), with T = 0.133 s per step. The base agent's 180 s horizon corresponds to γ = 0.9993 ([DERIVED]: 0.133/0.0007 = 190 s ≈ the stated 180 s).
  - **Arena** (April 18–21, 2019): "Five played 3,193 teams in 7,257 total games, winning 99.4%". "Twenty-nine teams managed to defeat OpenAI Five for a total of 42 games". Abandoned games were counted as wins for Five.
  - Footnote 6: "Early on in the project, we considered other algorithms including other policy gradient methods, q-learning, and evolutionary strategies. PPO was the first to show initial learning progress."
  - **OpenAI blog "OpenAI Five", 25 Jun 2018** (Wayback): "OpenAI Five plays 180 years worth of games against itself every day … running on 256 GPUs and 128,000 CPU cores". The table there gives 256 P100 GPUs on GCP and 128,000 preemptible CPU cores, batch 1,048,576 observations, ~60 batches per minute. These are the **2018** numbers, not the final run's.
  - **OpenAI blog "OpenAI Five Defeats Dota 2 World Champions", 15 Apr 2019** (Wayback; URL slug how-to-train-your-openai-five): "the current version of OpenAI Five has consumed 800 petaflop/s-days and experienced about 45,000 years of Dota self-play over 10 realtime months … for an average of 250 years of simulated experience per day."
  - **Disagreements**:
    - Compute: the blog says 800 PF/s-days and the paper says 770 ± 50, so the blog's figure is inside the paper's range.
    - Experience: the blog says ~45,000 years, while the Rubik's-cube paper (Akkaya et al. 2019) refers to "the 40 thousand years used by OpenAI Five".
- **PPO in the RLHF line**:
  - Ziegler et al. 2019 ("Fine-Tuning Language Models from Human Preferences", arXiv 1909.08593): "Train π via Proximal Policy Optimization (PPO, Schulman et al. (2017)) with reward R" plus an adaptive KL coefficient (K_β = 0.1).
  - Stiennon et al. 2020 (arXiv 2009.01325), Fig. 1: "Train policy with PPO".
  - Ouyang et al. 2022 (InstructGPT, arXiv 2203.02155): "Once again following Stiennon et al. (2020), we fine-tuned the SFT model on our environment using PPO (Schulman et al., 2017)."
  - Why PPO and not TRPO: the PPO paper notes TRPO "is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing". PPO is first-order, works with huge shared transformer networks, and was already OpenAI's default when the RLHF work began. The cost is a value network (critic) as large as the policy, which GRPO removes (§23); the four-models-in-memory picture is in §21.
- **Implementation matters**:
  - Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO", ICLR 2020, arXiv 2005.12729: code-level optimisations "are responsible for most of PPO's gain in cumulative reward over TRPO".
  - Huang, Dossa, Raffin, Kanervisto & Wang, "The 37 Implementation Details of Proximal Policy Optimization", ICLR Blog Track, 25 Mar 2022, https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

### Worked example [DERIVED]: the clip, one sample at a time (ε = 0.2)
| ρ = π_new/π_old | Â | unclipped ρÂ | clipped term | min (objective) | gradient? |
|---|---|---|---|---|---|
| 1.0 | +2 | 2.0 | 2.0 | 2.0 | flows (start of the update) |
| 1.5 | +2 | 3.0 | 1.2·2 = 2.4 | **2.4** | **0**: already moved enough in the good direction |
| 0.7 | +2 | 1.4 | 0.8·2 = 1.6 | **1.4** | flows: a good action became *less* likely, so it is pushed back |
| 0.7 | −1 | −0.7 | 0.8·(−1) = −0.8 | **−0.8** | **0**: a bad action is already down enough |
| 1.5 | −1 | −1.5 | 1.2·(−1) = −1.2 | **−1.5** | flows: a bad action became more likely, so it is pulled down |
- Ratios come from log-probabilities: if log π_new − log π_old = 0.3, then ρ = e^0.3 = 1.350. That is clipped at 1.2 when Â > 0.
- Why TRPO's theory step is useless as-is: C = 4ε_Aγ/(1−γ)² with ε_A = 1 and γ = 0.99 gives C = 3.96/0.0001 = **39,600**. The penalty is so large that steps would be microscopic. Hence a *constraint* δ_KL = 0.01 in practice (TRPO §4).
- CartPole with PPO (for the episode's measured run): the CartPole-v1 return is capped at 500 (500-step time limit; gymnasium registers reward_threshold = 475). SB3 RL-Zoo's PPO reaches 500.0 ± 0.0 at 100k steps [DATED, rl-baselines3-zoo benchmark.md].

### Common misconceptions
- *"PPO guarantees monotonic improvement."* No. That is TRPO's theorem, for a penalised objective, and even TRPO's practical algorithm approximates it. PPO's clip is a heuristic "pessimistic bound" on the surrogate.
- *"Clipping clips the ratio or the gradient."* It removes the *incentive* to move ρ further outside [1−ε, 1+ε] in the direction that would improve the objective. Samples that moved the wrong way keep their full gradient, and ρ can still drift outside the band through other samples and shared parameters.
- *"PPO is off-policy because it reuses data."* It is on-policy with limited reuse: a few epochs on the latest batch, then the data is thrown away.
- *"ε = 0.2 is universal."* It is the MuJoCo winner (Table 1). Atari used 0.1 × α, annealed to 0. RLHF recipes often use 0.2 [practice].
- *"PPO's success over TRPO is due to the clip."* Engstrom et al. 2020 attribute most of the gain to code-level optimisations: value clipping, reward/observation normalisation, learning-rate annealing, orthogonal initialisation and so on.
- *"OpenAI Five learned from human games."* It used self-play (80%) plus past versions of itself (20%). The 2018 blog: "no human data".

### Quotable
- "…we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse." (Schulman et al. 2017, §3)
- "[PPO has] some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically)." (PPO abstract; bracket mine)
- "PPO has become the default reinforcement learning algorithm at OpenAI because of its ease of use and good performance." (OpenAI blog, 20 Jul 2017)
- "PPO was the first to show initial learning progress." (Berner et al. 2019, footnote 6)
- "On April 13th, 2019, OpenAI Five became the first AI system to defeat the world champions at an esports game." (Berner et al. 2019)

---

## §19 Continuous control: Gaussian policies, DPG/DDPG/TD3, SAC, sim-to-real, Pendulum

**Notation mapping.**
- DPG/DDPG write the deterministic policy as μ_θ(s). Here **μ** means the policy's output action (not S&B's state distribution), and ρ^μ is its discounted state distribution.
- SAC's temperature α is written **α_H**, so as not to clash with step size α. SAC writes H for entropy and H̄ for the target entropy.
- DDPG/TD3 write τ for the soft target-update rate, and θ′, φ′ for target-network parameters.
- The Ornstein–Uhlenbeck θ (= 0.15) is a mean-reversion rate, not a policy parameter.

### Mechanism (plain words)
- Rover's pendulum asks for a **torque**, a real number in [−2, 2]. You cannot take a max over infinitely many actions, so DQN-style argmax fails. There are two answers.
- **(a) Stochastic Gaussian policy + policy gradient.** The network outputs a mean μ(s), plus a state-independent learned log σ. Sample a ~ N(μ, σ²) and use REINFORCE/PPO with the Gaussian ∇ln π (§16).
  - Exploration is built in, and σ shrinks as the policy grows confident.
  - This is what PPO does on MuJoCo.
- **(b) Deterministic policy + learned Q (DPG).** The actor outputs one action a = μ_θ(s). The critic Q(s,a) is differentiable in a, so the actor follows ∇_a Q: "turn the knob the way the critic says the value goes up".
  - **DDPG** = DPG + DQN's tricks: replay buffer, target networks with slow "soft" updates, batch norm, and temporally correlated Ornstein–Uhlenbeck exploration noise.
- **TD3** fixes DDPG's overestimation (the continuous cousin of §14's max bias) with three tricks:
  1. two critics, using the **minimum** for the target;
  2. **delayed** actor and target updates (once every 2 critic updates);
  3. **target policy smoothing**: clipped noise added to the target action, so the critic cannot exploit sharp peaks.
- **SAC** keeps a *stochastic* actor but learns off-policy from replay. It maximises reward **plus entropy**: "succeed at the task while acting as randomly as possible". The temperature can be tuned automatically to hold the entropy near a target of −dim(A).
- **Sim-to-real.** Train in simulation, but randomise what you cannot model (textures, friction, masses, delays) so that "the real world may appear to the model as just another variation" (Tobin et al. 2017).
  - OpenAI's Dactyl (2018) learned in-hand cube reorientation this way.
  - The 2019 Rubik's-cube work grew the randomisation ranges automatically (ADR).

### Key formulas
1. **Deterministic policy gradient theorem** (Silver, Lever, Heess, Degris, Wierstra & Riedmiller 2014, Theorem 1, Eq. 9):
   **∇_θ J(μ_θ) = E_{s∼ρ^μ}[ ∇_θ μ_θ(s) ∇_a Q^μ(s,a)|_{a=μ_θ(s)} ]**.
   §3.3: as σ → 0 the stochastic policy gradient converges to this.
2. **DDPG** (Lillicrap et al. 2015/ICLR 2016, Algorithm 1):
   - critic target y_i = r_i + γ Q′(s_{i+1}, μ′(s_{i+1}));
   - critic minimises (y_i − Q(s_i,a_i))²;
   - actor follows ∇_a Q(s,a)|_{a=μ(s)} ∇_θ μ(s);
   - soft target update θ′ ← τθ + (1 − τ)θ′.
3. **TD3** (Fujimoto, van Hoof & Meger 2018):
   - target action ã = μ_φ′(s′) + ε̃, with ε̃ ~ clip(N(0, σ̃), −c, c);
   - **y = r + γ min_{i=1,2} Q_θ′_i(s′, ã)** (Eq. 10 is the un-smoothed form);
   - update the actor and targets every d iterations.
4. **SAC objective** (Haarnoja, Zhou, Abbeel & Levine 2018, Eq. 1): **J(π) = Σ_t E_{(s_t,a_t)∼ρ_π}[ r(s_t,a_t) + α_H H(π(·|s_t)) ]**.
   - "the conventional objective can be recovered in the limit as α → 0."
   - Soft value (Haarnoja et al. 2018b): V(s_t) = E_{a_t∼π}[Q(s_t,a_t) − α_H log π(a_t|s_t)].
   - Two Q-functions are trained, and the minimum is used.
5. **SAC automatic temperature** (Haarnoja et al. 2018b, "Soft Actor-Critic Algorithms and Applications"): J(α) = E_{a_t∼π_t}[−α log π_t(a_t|s_t) − α H̄], with target entropy **H̄ = −dim(A)** ("e.g., −6 for HalfCheetah-v1"; for Pendulum, −1).
6. **Gaussian policy score**: see §16, formula 6.
7. **Domain randomisation**: train on a distribution over simulator parameters ξ ~ P(ξ) and maximise E_ξ[J_ξ(π)]. ADR widens P(ξ) automatically when the policy succeeds at its boundary (Akkaya et al. 2019) [paraphrase].

### Numbers with sources
- **DPG** (ICML 2014, PMLR 32, https://proceedings.mlr.press/v32/silver14.pdf):
  - "The deterministic policy gradient has a particularly appealing form: it is the expected gradient of the action-value function."
  - Deterministic gradients "can significantly outperform their stochastic counterparts in high-dimensional action spaces"; the demo is a 6-segment octopus arm.
- **DDPG**: Lillicrap, Hunt, Pritzel, Heess, Erez, Tassa, Silver & Wierstra, "Continuous Control with Deep Reinforcement Learning", ICLR 2016, https://arxiv.org/abs/1509.02971
  - Supplementary hyperparameters:
    - Adam, actor lr 10⁻⁴, critic lr 10⁻³, critic L2 weight decay 10⁻², γ = 0.99, **τ = 0.001**;
    - networks of 400 and 300 units (≈ 130,000 params); actions enter Q at the 2nd hidden layer;
    - minibatch 64 (low-dim) / 16 (pixels); **replay buffer 10⁶**;
    - **Ornstein–Uhlenbeck noise with θ = 0.15, σ = 0.2**.
  - The same algorithm and hyper-parameters solve "more than 20 simulated physics tasks".
  - "Nearly all of the problems we looked at were solved within 2.5 million steps of experience (and usually far fewer), a factor of 20 fewer steps than DQN requires for good Atari solutions."
- **TD3**: "Addressing Function Approximation Error in Actor-Critic Methods", ICML 2018, https://arxiv.org/abs/1802.09477
  - Setup: target policy smoothing "N(0, 0.2) … clipped to (−0.5, 0.5)"; "delayed policy updates … every d iterations, with d = 2"; "Both target networks are updated with τ = 0.005"; exploration N(0, 0.1); batch 100; Adam 10⁻³ for both networks.
  - **Table 1** (max average return over 10 trials, 1M steps):

    | Task | TD3 | DDPG (orig.) | Our DDPG | PPO | TRPO | ACKTR | SAC |
    |---|---|---|---|---|---|---|---|
    | HalfCheetah | 9636.95 ± 859.065 | 3305.60 | 8577.29 | 1795.43 | −15.57 | 1450.46 | 2347.19 |
    | Hopper | 3564.07 ± 114.74 | 2020.46 | 1860.02 | 2164.70 | 2471.30 | 2428.39 | 2996.66 |
    | Walker2d | 4682.82 ± 539.64 | 1843.85 | 3098.11 | 3317.69 | 2321.47 | 1216.70 | 1283.67 |
    | Ant | 4372.44 ± 1000.33 | 1005.30 | 888.77 | 1083.20 | −75.85 | 1821.94 | 655.35 |
    | Reacher | −3.60 ± 0.56 | −6.51 | −4.01 | −6.18 | −111.43 | −4.26 | −4.44 |
    | InvPendulum | 1000.00 ± 0.00 | 1000.00 | 1000.00 | 1000.00 | 985.40 | 1000.00 | 1000.00 |
    | InvDoublePendulum | 9337.47 ± 14.96 | 9355.52 | 8369.95 | 8977.94 | 205.85 | 9081.92 | 8487.15 |

    These are TD3's runs of every baseline, at 1M steps, on the old Gym "-v1" tasks. Baseline numbers from other papers differ.
  - TD3's own **"Our DDPG"** re-implementation gets 8577.29 on HalfCheetah. That shows how much implementation matters.
- **SAC**: Haarnoja, Zhou, Abbeel & Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", ICML 2018, https://arxiv.org/abs/1801.01290
  - Its abstract emphasises stability across seeds.
  - The first version needed a **reward scale** "tuned for each environment"; the reward scale plays the role of an inverse temperature.
- **SAC 2018b**: Haarnoja, Zhou, Hartikainen, Tucker, Ha, Tan, Kumar, Zhu, Gupta, Abbeel & Levine, "Soft Actor-Critic Algorithms and Applications", arXiv 1812.05905
  - Table 1: Adam 3·10⁻⁴; γ = 0.99; replay 10⁶; 2 hidden layers × 256; minibatch 256; **entropy target −dim(A)**; τ = 0.005.
  - Real Minitaur quadruped: walking learned "from 160k environment steps, or approximately 400 episodes with maximum length of 500 steps, which amount to approximately 2 hours of real-world training time."
- **Domain randomisation**: Tobin, Fong, Ray, Schneider, Zaremba & Abbeel, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017, https://arxiv.org/abs/1703.06907
  - Result: "a real-world object detector that is accurate to 1.5 cm and robust to distractors and partial occlusions using only data from a simulator with non-realistic random textures."
  - Note this is a vision detector used for grasping, not an RL policy.
- **Dactyl**: OpenAI (Andrychowicz et al.), "Learning Dexterous In-Hand Manipulation", arXiv 1808.00177 (2018; IJRR 2020)
  - Hardware: a Shadow Dexterous Hand trained entirely in simulation, with "the same distributed RL system that was used to train OpenAI Five" (PPO).
  - Compute: "384 worker machines, each with 16 CPU cores" (= **6,144 CPU cores** [DERIVED]); optimisation on "a single machine with 8 GPUs" (V100); "about 2 years of simulated experience per hour".
  - Cost of randomisation: "Learning to rotate an object in simulation without randomizations requires about 3 years of simulated experience, while achieving the same performance in a fully randomized simulation requires about 100 years of experience. This corresponds to a wall-clock time of around 1.5 hours and 50 hours".
  - **Table 3** (consecutive successful rotations; 50 = cap):
    - Block (state): simulation median **50** (mean 43.4); physical median **13** (mean 18.8).
    - Block (vision): physical median 11.5.
    - Locked wrist: physical median 28.5.
- **Rubik's cube**: OpenAI (Akkaya et al.), "Solving Rubik's Cube with a Robot Hand", arXiv 1910.07113 (Oct 2019)
  - Method: ADR, "automatic domain randomization".
  - Compute: "8 × 8 = 64 NVIDIA V100 GPUs and 8 × 115 = 920 worker machines with 32 CPU cores each" (= 29,440 cores); "roughly 13 thousand years" of cumulative experience; trained "continuously for several months".
  - Result: the best policy "achieves 26.80 successes on average over 10 trials. This corresponds to successfully solving a Rubik's cube that requires 15 face rotations 60% of the time and to solve a Rubik's cube that requires 26 face rotations 20% of the time."
  - **The solution sequence comes from the Kociemba solver**, not from RL. RL and ADR learn the *manipulation*.
- **Pendulum-v1** (Gymnasium source, `gymnasium/envs/classic_control/pendulum.py`, main branch, read Oct 2026; latest PyPI release 1.4.0) [DATED for version]:

  | Item | Value |
  |---|---|
  | Observation | [cos θ, sin θ, θ̇]; bounds [−1,1], [−1,1], [−8, 8] |
  | Action | torque u ∈ **[−2.0, 2.0]** (shape (1,)) |
  | Reward | **r = −(θ² + 0.1·θ̇² + 0.001·u²)**, θ normalised to [−π, π], 0 = upright |
  | Reward range | min **−16.2736044** = −(π² + 0.1·8² + 0.001·2²); max 0 |
  | Dynamics | θ̇ ← clip(θ̇ + (3g/(2l)·sin θ + 3/(m l²)·u)·dt, −8, 8); θ ← θ + θ̇·dt |
  | Constants | g = 10.0 (default), m = 1.0, l = 1.0, dt = 0.05, max_speed = 8, max_torque = 2.0 |
  | Start | θ ~ U[−π, π], θ̇ ~ U[−1, 1] |
  | Episode end | "truncated at 200 time steps. There is no termination condition." (registered max_episode_steps = 200; no reward_threshold) |
- **What "solved" looks like** (SB3 RL-Zoo benchmark.md, Pendulum-v1, mean ± std over 750 eval episodes) [DATED]:
  - TD3 −151.9 ± 90.2 (20k steps); DDPG −152.1 ± 94.3 (20k); SAC −157.0 ± 88.7 (20k); TQC −151.3 (20k);
  - PPO −172.2 ± 104.2 (100k); A2C −163.0 ± 103.2 (1M); TRPO −174.6 ± 127.6 (100k).
  - Off-policy learners need ~5× fewer steps than PPO here.

### Worked example [DERIVED] (scripts/f6_checks.py)
**A. Why Pendulum is a swing-up, not a push-up.**
- Angular acceleration from gravity is (3g/2l)·sin θ = **15·sin θ** rad/s². The most the motor can add is (3/ml²)·2 = **6** rad/s².
- A steady push can therefore only hold the pendulum where |sin θ| ≤ 6/15 = 0.4, i.e. within **23.6°** of vertical.
- From hanging down, Rover must *pump*, swinging back and forth to build energy, then catch the pendulum at the top. This is an exploration and credit problem.

**B. Reward scale of Pendulum.**
- Episode = 200 steps × 0.05 s = **10 s**.
- Worst possible: 200 × (−16.27) = **−3,254.7**.
- Hanging motionless at the bottom with zero torque: θ = π, so −π² = −9.87 per step and **−1,973.9** per episode.
- Good learned policies average ≈ **−150** (RL-Zoo above): roughly 1–2 s of swinging, then near-zero cost.

**C. One Gaussian policy-gradient step on a torque.**
- μ = 0.5, σ = 1.0; Rover sampled u = 1.3; advantage Â = +2; step α = 0.1.
- ∂ln π/∂μ = (u − μ)/σ² = **0.8**, so μ ← 0.5 + 0.1·2·0.8 = **0.66**: the mean moves toward the good action.
- ∂ln π/∂log σ = (u−μ)²/σ² − 1 = 0.64 − 1 = **−0.36**, so σ ← exp(0.1·2·(−0.36)) = **0.931**.
- A good action *inside* one σ narrows the Gaussian; a good action *outside* one σ would widen it.

**D. Slow targets.**
- With soft update rate τ, the target net's half-life is ln 2/τ updates.
- DDPG (τ = 0.001): **≈ 693 updates**. TD3/SAC (τ = 0.005): **≈ 139 updates**.
- Compare DQN's hard copy every C steps (§13).

**E. SAC's entropy target for Pendulum.**
- H̄ = −dim(A) = −1 nat.
- For an *unsquashed* 1-D Gaussian, H = ½ln(2πeσ²), so H = −1 ⇔ σ = e^{−1}/√(2πe) ≈ **0.089** (in pre-scaling action units).
- This ignores SAC's tanh squashing, so it is only indicative.

### Common misconceptions
- *"Continuous actions need a different kind of RL."* The same policy-gradient theorem applies; only ∇ln π changes (Gaussian instead of softmax). DPG is the σ → 0 limit (Silver et al. 2014, §3.3).
- *"Deterministic policies can't explore."* DDPG and TD3 explore with *added* noise (OU or Gaussian). Learning is off-policy from replay, so the behaviour policy can differ from the learned one.
- *"TD3's min of two critics is the same as Double DQN."* Related, but it is a deliberate *underestimate*. TD3: "this is far preferable to overestimation bias".
- *"Maximum entropy means random behaviour."* The entropy term is traded against reward through α_H (or a target entropy). The policy commits where it matters and stays diffuse where actions are equally good.
- *"OpenAI's robot hand learned to solve the Rubik's cube."* It learned to *manipulate* the cube. The move sequence came from the Kociemba solver, and success was 60% for 15-rotation scrambles and 20% for the hardest 26-rotation ones.
- *"Domain randomisation is free."* Dactyl needed ~33× more experience with full randomisation: 100 years vs 3 years.

### Quotable
- "The deterministic policy gradient has a particularly appealing form: it is the expected gradient of the action-value function." (Silver et al. 2014)
- "In this framework, the actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible." (Haarnoja et al. 2018, SAC abstract)
- "While this update rule may induce an underestimation bias, this is far preferable to overestimation bias, as unlike overestimated actions, the value of underestimated actions will not be explicitly propagated through the policy update." (Fujimoto et al. 2018)
- "With enough variability in the simulator, the real world may appear to the model as just another variation." (Tobin et al. 2017)
- "Learning to rotate an object in simulation without randomizations requires about 3 years of simulated experience, while achieving the same performance in a fully randomized simulation requires about 100 years of experience." (OpenAI, Dactyl, 2018)

---

## §20 The language model as a policy

### Mechanism (plain words)
- **Same loop, new world.** The prompt x is the start state. Each step the policy (the LM) picks one **action**: the next token from the vocabulary. The world's response is trivial and deterministic: append the token to the text. The episode ends at EOS or a length limit. The reward usually arrives **once, at the end**, for the whole completion y.
- **Two equivalent views.**
  - *Token-level MDP*: state s_t = prompt + tokens so far, action a_t = next token, transition = concatenation, reward r_T at the last step (plus, in PPO-style RLHF, a small per-token KL term).
  - *Bandit view*: the whole completion is one "arm". InstructGPT says exactly this ("a bandit environment"), and DPO is derived in this view. Rafailov et al. 2024 contrast the two: RLHF "deploys reinforcement learning in a specific token-level MDP, while DPO is derived as a bandit problem in which the whole response of the model is treated as a single arm."
- **Sparse sequence reward = the credit-assignment problem of §11 again.** One number at the end must be shared among 100s of tokens. PPO-RLHF uses a learned value function and GAE (§17) to spread it. GRPO (§23) gives every token of an answer the same advantage.
- **The pipeline (InstructGPT Figure 2):** pretraining (next-token prediction on web text) → **SFT** (imitate human demonstrations) → **reward model** (learn from human comparisons, §21) → **RL** (PPO against the reward model with a KL leash to the SFT model). DPO (§22) folds the last two steps into one loss. RLVR (§24) swaps the learned reward model for a checker.
- **The KL leash.** RL maximises reward minus β × (how far the policy has drifted from a frozen reference model π_ref, usually the SFT model). The papers give three reasons for it:
  - it acts as an entropy bonus;
  - it keeps the policy where the reward model is still accurate;
  - it keeps text fluent.
- **Why RL after SFT?** Claims from the sources, attributed:
  1. **MLE treats all errors alike and copies bad demos too** (Stiennon et al. 2020, quoted below).
  2. **Exposure bias.** A model trained only on human prefixes never learns to recover from its own mistakes. RL trains on its own samples (Ranzato et al. 2016 coined the term).
  3. **The demonstrator knows different things from the model.** OpenAI's ChatGPT post: "supervised training misleads the model because the ideal answer depends on what the model knows, rather than what the human demonstrator knows."
  4. **Negative signal.** SFT can only push probability *up*. Comparisons and rewards can push bad outputs *down*. ORPO (Hong et al. 2024, Figure 3) shows that during SFT the log-probabilities of chosen *and* rejected responses rise together.
  5. **Sharpening rather than new skills.** DeepSeekMath (Shao et al. 2024 §5.2.2): "RL enhances Maj@K's performance but not Pass@K", i.e. it boosts "the correct response from TopK rather than the enhancement of fundamental capabilities". Yue et al. 2025 make the same point (see §24). This is an empirical claim about specific setups, not a law.
- **Cost.** In InstructGPT the RL step used about 1.6% of GPT-3's pretraining compute (numbers below).

### Key formulas
Notation: x = prompt (s_0); y = (a_0 … a_{T−1}) = completion; s_t = (x, a_<t); π_θ = policy (LM); π_ref = frozen reference (SFT model); r_φ = learned reward model; β = KL coefficient.

| Paper | Their notation | Ours |
|---|---|---|
| Ziegler et al. 2019 | π (policy), ρ (pretrained model), r (RM) | π_θ, π_ref, r_φ |
| Stiennon 2020; Ouyang 2022 | π_φ^RL (policy, params φ), π^SFT, r_θ (RM, params θ) | π_θ, π_ref, r_φ. **θ and φ are swapped relative to ours** |
| Ouyang 2022 eq. 2 | γ = pretraining-loss coefficient | not a discount factor! |

1. **KL-shaped sequence reward.** Ziegler et al. 2019, eq. 2: R(x, y) = r(x, y) − β log[π(y|x)/ρ(y|x)]. Stiennon 2020 uses the identical form with π^SFT as the reference.
2. **Per-token decomposition [DERIVED: chain rule of probability].** log π(y|x) = Σ_t log π(a_t|s_t), so the penalty splits over tokens:
   - r_t = −β [log π_θ(a_t|s_t) − log π_ref(a_t|s_t)] for t < T−1;
   - r_{T−1} = r_φ(x, y) − β [log π_θ(a_{T−1}|s_{T−1}) − log π_ref(a_{T−1}|s_{T−1})].
   - This is what InstructGPT means by "a per-token KL penalty from the SFT model at each token".
   - In expectation over y ~ π_θ, the summed log-ratio equals KL(π_θ(·|x) ‖ π_ref(·|x)) (chain rule for KL).
   - One sampled log-ratio can be negative. Only its expectation is ≥ 0 (the estimator issue, "k1 vs k3", is in §23).
3. **Objective (DPO eq. 3, the standard statement):** max_π E_{x~D, y~π}[r_φ(x, y)] − β D_KL[π(y|x) ‖ π_ref(y|x)].
4. **Its closed-form optimum (DPO eq. 4):** π*(y|x) = π_ref(y|x) exp(r(x, y)/β) / Z(x). This is a re-weighting of the reference distribution: small β gives near-greedy behaviour, large β stays near π_ref (worked in §22).
5. **Adaptive β (Ziegler et al. 2019 §2.2).**
   - e_t = clip((KL(π_t, ρ) − KL_target)/KL_target, −0.2, 0.2);
   - β_{t+1} = β_t (1 + K_β e_t), with K_β = 0.1.
6. **Discount:** γ = 1 throughout (Stiennon footnote 8; InstructGPT "No discount is applied when estimating the generalized advantage").

### Numbers with sources
| Fact | Value | Source |
|---|---|---|
| Token = time step; reward only at the end; γ = 1 | "Note that the reward model only gives rewards for entire summaries, and not at intermediate time steps. In RL terminology, each episode terminates when the policy outputs the EOS token, and the discount factor γ = 1." | Stiennon et al. 2020, footnote 8 |
| Bandit framing | "The environment is a bandit environment which presents a random customer prompt and expects a response to the prompt. Given the prompt and response, it produces a reward determined by the reward model and ends the episode." | Ouyang et al. 2022 §3.5 |
| Formal LM-MDP | "The transition function P : S × A → ∆(S) deterministically appends an action a_t to the end of the state" | Ramamurthy et al. 2023 (RL4LMs, ICLR 2023) §3.1 |
| Ziegler et al. 2019 setup | 774M GPT-2; 2M episodes; γ = 1; four PPO epochs per batch | Ziegler et al. 2019 §2–3 |
| Ziegler β values | Fixed β = 0.1 (CNN/DM), 0.03 (TL;DR); adaptive K_β = 0.1 for others | Ziegler et al. 2019 |
| Ziegler style result | "For stylistic continuation, 5,000 human comparisons (each choosing the best of 4 continuations) result in the fine-tuned model being preferred by humans 86% of the time vs. zero-shot and 77% vs. fine-tuning to a supervised sentiment network." | Ziegler et al. 2019 §1 |
| Stiennon KL coefficient | β = 0.05 for the main runs | Stiennon et al. 2020 App. B.1 |
| InstructGPT KL coefficient | β = 0.02 | Ouyang et al. 2022 App. C.4 |
| InstructGPT compute | "training our 175B SFT model requires 4.9 petaflops/s-days and training our 175B PPO-ptx model requires 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3" | Ouyang et al. 2022 §5.1 |
| RL share of pretraining compute [DERIVED] | 60/3,640 = 1.65%; SFT 4.9/3,640 = 0.13% | arithmetic on the row above |
| Llama 3's choice | "we adopt a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO; Rafailov et al. (2023)) as opposed to more complex reinforcement learning algorithms (Ouyang et al., 2022; Schulman et al., 2017) that tend to be less stable and harder to scale." | Llama Team 2024, "The Llama 3 Herd of Models" §1 |

### Worked example [DERIVED, checked in scripts/f7_examples.py]
**Setup.** A small model answers "The answer is 42." in 5 tokens. Reward model score r_φ = 1.0, β = 0.05 (Stiennon's value). The probabilities of each chosen token under the policy and the frozen reference:

| token | π_θ | π_ref | log-ratio | penalty −β·log-ratio |
|---|---|---|---|---|
| "The" | 0.90 | 0.85 | +0.0572 | −0.0029 |
| " answer" | 0.80 | 0.50 | +0.4700 | −0.0235 |
| " is" | 0.95 | 0.95 | 0.0000 | 0.0000 |
| " 42" | 0.60 | 0.20 | +1.0986 | −0.0549 |
| "." | 0.90 | 0.80 | +0.1178 | −0.0059 |
| **sum** | | | **1.7436** (= ln 5.7176, the sequence ratio) | **−0.0872** |

- **Per-token reward vector** fed to PPO: [−0.0029, −0.0235, 0, −0.0549, 1.0 − 0.0059 = 0.9941]. It sums to **R = 0.9128**.
- **Takeaway.** The token where the policy has moved furthest from the reference (" 42": 0.20 → 0.60) pays the biggest share of the leash.
- **What to animate.** The gold reward lands only on the last token. Small pink KL "fees" sit on every token.

### Common misconceptions
- **"Each token gets a reward."** Only the final token gets the task reward. Per-token terms are only the KL fee (PPO-RLHF). In GRPO the KL is a separate loss term (§23).
- **"The KL term is the exact KL."** It is a single-sample estimate. It can be negative on one sample.
- **"RL is the expensive part of training an LM."** For InstructGPT it was ~1.6% of GPT-3's pretraining compute. The scarce resource was the human data. (For long-reasoning RL in 2025 the share is larger; see §24/§26.)
- **"RL teaches the model new facts."** The sources argue mostly the opposite: RL reshapes which already-reachable answers get probability (DeepSeekMath §5.2.2; Yue et al. 2025, §24). Treat this as evidence from specific setups.
- **"The reference model is the pretrained base."** In InstructGPT and Stiennon it is the **SFT** model. In Ziegler 2019 it is the pretrained GPT-2 (no SFT stage for style tasks).

### Quotable
- "We primarily do this using reinforcement learning, by treating the output of the reward model as a reward for the entire summary that we maximize with the PPO algorithm [58], where each time step is a BPE token." (Stiennon et al. 2020 §3.4)
- "This KL term serves two purposes. First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode. Second, it ensures the policy doesn't learn to produce outputs that are too different from those that the reward model has seen during training." (Stiennon et al. 2020 §3.4)
- "This term has several purposes: it plays the role of an entropy bonus, it prevents the policy from moving too far from the range where r is valid, and in the case of our style continuation tasks it also is an important part of the task definition" (Ziegler et al. 2019 §2)
- "the maximum likelihood objective has no distinction between important errors (e.g. making up facts [41]) and unimportant errors (e.g. selecting the precise word from a set of synonyms); models are incentivized to place probability mass on all human demonstrations, including those that are low-quality; and distributional shift during sampling can degrade performance" (Stiennon et al. 2020 §1)
- "We refer to this discrepancy as exposure bias which occurs when a model is only exposed to the training data distribution, instead of its own predictions." (Ranzato et al. 2016 §1)
- "supervised training misleads the model because the ideal answer depends on what the model knows, rather than what the human demonstrator knows." (OpenAI, "ChatGPT: Optimizing Language Models for Dialogue", 30 Nov 2022)
- "Thus, we say that the language modeling objective is misaligned." (Ouyang et al. 2022 §1)
- Ziegler et al. 2019, App. (a sign-flip bug; also good for §25): "One of our code refactors introduced a bug which flipped the sign of the reward. Flipping the reward would usually produce incoherent text, but the same bug also flipped the sign of the KL penalty. The result was a model which optimized for negative sentiment while still regularizing towards natural language." … "This bug was remarkable since the result was not gibberish but maximally bad output. The authors were asleep during the training process, so the problem was noticed only once training had finished."

---

## §21 Reinforcement learning from human feedback (RLHF)

### Mechanism (plain words)
1. **People compare; they don't score.** It is hard for a person to give "7.3/10". It is easy to say "A is better than B". Christiano et al. 2017 showed clips of agent behaviour 1–2 s long in pairs. InstructGPT labelers ranked 4–9 answers to one prompt.
2. **A reward model learns their taste.** A network r_φ(x, y) is trained so that the preferred answer gets the higher score. The **Bradley–Terry** model turns a score difference into a win probability: P(y_w ≻ y_l) = σ(r_w − r_l). This is the same logic as chess Elo. Only differences matter, so the scale is arbitrary. InstructGPT shifts the RM so the labeler demonstrations average 0.
3. **RL against the reward model with a leash.** PPO (§18) maximises r_φ minus β·KL to the SFT model (§20). InstructGPT's **PPO-ptx** also mixes in ordinary next-token-prediction gradients on pretraining data. This reduces the "alignment tax" (regressions on public NLP benchmarks).
4. **Four models in memory during PPO:**
   - **policy/actor**: trained;
   - **value/critic**: trained, initialised from the RM;
   - **reward model**: frozen;
   - **reference** (SFT): frozen, used for the KL.
5. **Lineage (dated):**
   - 2017, Atari + MuJoCo: Christiano et al., OpenAI + DeepMind.
   - 2019, GPT-2 style and summaries: Ziegler et al.
   - 2020, TL;DR summaries: Stiennon et al.
   - 2022, instruction following: InstructGPT, Ouyang et al.
   - 30 Nov 2022: ChatGPT, trained "using the same methods as InstructGPT, but with slight differences in the data collection setup".

### Key formulas
- **Bradley–Terry** (Bradley & Terry 1952). DPO eq. 1: p*(y_1 ≻ y_2 | x) = exp(r*(x, y_1)) / [exp(r*(x, y_1)) + exp(r*(x, y_2))] = σ(r*(x, y_1) − r*(x, y_2)).
- **Christiano et al. 2017, eq. 1** (σ¹, σ² are trajectory segments, not sigmoids): P̂[σ¹ ≻ σ²] = exp Σ_t r̂(o¹_t, a¹_t) / [exp Σ_t r̂(o¹_t, a¹_t) + exp Σ_t r̂(o²_t, a²_t)]. The reward is summed over the clip. The loss is cross-entropy against the human label μ, which can be split 50/50 for "equally good".
- **Reward-model loss.**
  - Stiennon 2020: loss(r) = −E[log σ(r(x, y_i) − r(x, y_{1−i}))].
  - InstructGPT eq. 1: loss(θ) = −(1/C(K,2)) E_{(x,y_w,y_l)~D}[log σ(r_θ(x, y_w) − r_θ(x, y_l))].
  - All C(K,2) pairs from one prompt form one batch element. In our notation the RM parameters are φ.
- **PPO-ptx objective** (InstructGPT eq. 2; their φ = our θ, their γ is NOT a discount):
  - objective(φ) = E_{(x,y)~D_{π_φ^RL}}[r_θ(x, y) − β log(π_φ^RL(y|x)/π^SFT(y|x))] + γ E_{x~D_pretrain}[log π_φ^RL(x)].
  - β = 0.02, γ = 27.8. Setting γ = 0 gives the plain "PPO" models.
- **Reward-difference reading [DERIVED].** r_w − r_l is the log-odds that a labeler prefers y_w. A gap of 1 means e¹ ≈ 2.72 : 1 odds, i.e. 73.1%.

### Numbers with sources
**Christiano et al. 2017**, "Deep reinforcement learning from human preferences" (NeurIPS 2017):

| Fact | Value | Source |
|---|---|---|
| Feedback fraction | "providing feedback on less than 1% of our agent's interactions with the environment" | Abstract |
| Backflip (paper) | "The Hopper robot performing a sequence of backflips (see Figure 4). This behavior was trained using 900 queries in less than an hour." | §3.2 |
| Backflip (blog) | "Our algorithm needed 900 bits of feedback from a human evaluator to learn to backflip — a seemingly simple task which is simple to judge but challenging to specify." Also: "It took less than an hour of a human evaluator's time, while in the background the policy accumulated about 70 hours of overall experience (simulated at a much faster rate than real-time.)" The same post also says "under 1000 bits". | OpenAI blog, "Learning from Human Preferences", 13 June 2017 (read via web.archive.org copy of blog.openai.com) |
| Hand-written alternative | "we took two hours to write our own reward function (the animation in the above right) to get a robot to backflip, and though it succeeds it's a lot less elegant than the one trained simply through human feedback" | same blog, footnote |
| Other novel behaviours | Half-Cheetah on one leg: 800 queries in under an hour. Enduro keep-alongside: ~1,300 queries, 4 M frames | §3.2 |
| MuJoCo | "With 700 labels we are able to nearly match reinforcement learning on all of these tasks." "by 1400 labels our algorithm performs slightly better than if it had simply been given the true reward" | §3.1.1, Fig. 2 |
| Atari | 7 games, 5,500 human queries. BeamRider and Pong come close to RL with 3,300 synthetic labels. Breakout reaches 20, or 50 with more labels | §3.1.2, Fig. 3 |
| Clip length / speed | Clips 1–2 s (§2.2.2). "Contractors responded to the average query in 3-5 seconds" (§3.1) | §2.2.2, §3.1 |
| Cost | Atari compute ~$25 (a day on a ~$700/month VM). "Training with 5k labels corresponds roughly to 5 hours of human labour, at US minimum wage this totals ~$36." | footnote 6 |

**Stiennon et al. 2020**, "Learning to summarize from human feedback" (NeurIPS 2020):

| Fact | Value | Source |
|---|---|---|
| Models | 1.3B and 6.7B GPT-3-style. "In our experiments, the reward model, policy, and value function are the same size." | §3.4 |
| Dataset | "64,832 summary comparisons on the TL;DR dataset" (released) | §1 |
| Headline | "our 1.3B human feedback model significantly outperforming a supervised model 10× its size (61% versus 43% raw preference score against reference summaries)" | §4.1 |
| Length-controlled | "our 6.7B model summaries are still preferred to the reference summaries ~65% of the time." | §4.1 |
| RM scaling | "Doubling amount of training data leads to a ~1.1% increase in reward model validation accuracy, whereas doubling the model size leads to a ~1.8% increase." | Fig. 6 caption |
| Overoptimisation | "Under light optimization, the models improve (according to labelers). However, as we optimize further, true preferences fall off compared to the prediction, and eventually the reward model becomes anti-correlated with human preferences." | §4.3, Fig. 5 |
| PPO hyperparameters | β = 0.05; GAE γ = 1, λ = 0.95; 4 epochs per batch; batch 512 (1.3B) / 256 (6.7B); 1 M episodes | App. B.1 |

**Ouyang et al. 2022**, "Training language models to follow instructions with human feedback" (InstructGPT; arXiv 2203.02155, NeurIPS 2022):

| Fact | Value | Source |
|---|---|---|
| Headline | "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters." (The intro says "over 100x". [DERIVED] 175/1.3 = 134.6×.) | Abstract; §1 |
| 175B vs 175B | "Outputs from our 175B InstructGPT are preferred to 175B GPT-3 outputs 85 ± 3% of the time, and preferred 71 ± 4% of the time to few-shot 175B GPT-3." | §1 |
| Labelers | "We first hire a team of 40 contractors to label our data" (§1). §3.4 says "a team of about 40 contractors on Upwork and through ScaleAI" | §1, §3.4 |
| Labeler agreement | "training labelers agree with each-other 72.6 ± 1.5% of the time, while for held-out labelers this number is 77.3 ± 1.3%" | §3.4 |
| Dataset sizes (Table 6, training prompts) | **SFT** 11,295 labeler + 1,430 customer = 12,725 ("about 13k"); **RM** 6,623 + 26,584 = 33,207 ("33k"); **PPO** 31,144 customer ("31k", no human labels). Validation: SFT 1,653; RM 17,887; PPO 16,185 [sums DERIVED] | §3.2, App. A.3 Table 6 |
| Rankings | "we present labelers with anywhere between K = 4 and K = 9 responses to rank. This produces C(K,2) comparisons for each prompt". C(4,2) = 6 … C(9,2) = 36. A batch of 64 prompts holds "up to 64 × C(K,2) ≤ 2,304 comparisons" | §3.5; App. C.2 |
| Reward model | 6B for all policy sizes; "175B RM training could be unstable". One epoch, lr 9e-6, batch 64 prompts; "multiple epochs quickly overfit the model to the training data" | §3.5; App. C.2 |
| SFT | Baseline 16 epochs. The RLHF init used 2 epochs with 10% pretraining data mixed in | §3.5; App. C.3 |
| PPO details | β = 0.02; 256k episodes over ~31k unique prompts; batch 512, minibatch 64, a single inner epoch; clip 0.2; temperature 1; no discount; 6B value function initialised from the RM; EMA of weights with decay 0.992 | App. C.4 |
| PPO-ptx | Pretraining gradients × γ = 27.8. "We use 8 times more pretraining examples than the number of the RL training episodes." | App. C.4 |
| Truthfulness | "On the TruthfulQA benchmark, InstructGPT generates truthful and informative answers about twice as often as GPT-3." | §1 |
| Hallucination (closed-domain) | "a 21% vs. 41% hallucination rate, respectively" | §1 |
| Toxicity | "InstructGPT models generate about 25% fewer toxic outputs than GPT-3 when prompted to be respectful." No significant gain on Winogender / CrowS-Pairs (bias) | §1 |
| Alignment tax | "This is an example of an "alignment tax" since our alignment procedure comes at the cost of lower performance on certain tasks that we may care about." Mainly SQuAD, DROP, HellaSwag, WMT'15 Fr→En; PPO-ptx reduces this | §1 |
| vs FLAN/T0 | "InstructGPT has a 73.4 ±2% winrate vs. our baseline, compared to 26.8 ±2% and 29.8 ±2% for our version of T0 and FLAN, respectively" | §1 |
| Compute | SFT 4.9, PPO-ptx 60, GPT-3 3,640 petaflop/s-days. RLHF helps "more so than a 100x model size increase" | §5.1 |
| Exact Figure 1 win rates (1.3B vs 175B) | only plotted, not stated in the text | Fig. 1 [values not extracted] |

**Four models in memory**

| Fact | Value | Source |
|---|---|---|
| Four models | "Scaling RLHF training to larger models requires efficiently allocating at least four component models (actor, critic, reward, reference) across multiple GPUs due to the memory limit of each accelerator" | Hu et al. 2024, OpenRLHF v1 (arXiv 2405.11143v1) §2.1 |
| Same point, DeepSpeed-Chat | "the memory cost, as several copies of the SFT and RW models need to be served throughout stage 3" | Yao et al. 2023, DeepSpeed-Chat §4 |
| Generation cost (also relevant to §26) | "the generation phase comprises approximately 20% of the total computation while the RL training phase comprises of remaining 80%". But generation "can take a large portion of the e2e time as it requires running the actor model once for each of the 256 generated tokens with initial prompt of 256 tokens, making it memory bandwidth bound" | Yao et al. 2023 §5.3 |
| Bytes per trained parameter | Mixed-precision Adam needs "2Ψ + 2Ψ + KΨ = 16Ψ bytes" (K = 12) | Rajbhandari et al. 2020, ZeRO §3.1 |

### Worked example [DERIVED, checked in scripts/f7_examples.py]
**1. One comparison through Bradley–Terry.**
- **Setup.** The RM scores two answers r_w = 1.2 and r_l = 0.4.
- **Win probability.** P = σ(0.8) = **0.690**. Loss = −ln 0.690 = **0.371**.
- **Wrong way round.** If the RM had scored them 0.4 and 1.2, P = 0.310 and the loss would be **1.171**.
- **Tie.** Equal scores give P = 0.5 and loss ln 2 = **0.693**.
- **Shift invariance.** Adding 5 to both scores changes nothing, hence the normalisation step.

**2. Comparisons per prompt.** K = 4 ranked answers → 6 pairs. K = 9 → 36 pairs. With 64 prompts per batch: 64 × 36 = **2,304**, the paper's stated maximum.

**3. "Four models in memory" for a 7B policy.** Model states only; excludes activations and the KV cache for generation.
- **Assumptions.** Actor and critic are trained with mixed-precision Adam at 16 B/param. Reference and RM are frozen BF16 at 2 B/param.
- **7B total.** 112 + 112 + 14 + 14 = **252 GB**.
- **0.5B total.** 8 + 8 + 1 + 1 = **18 GB**. This is why small-model RLHF fits on one 64 GB Mac.
- **Without a critic** (GRPO, §23): the trained-model states halve. 7B: 112 + 14 + 14 = 140 GB.

### Common misconceptions
- **"Humans write the reward numbers."** They compare or rank. Scores come from a model fitted to those comparisons.
- **"The reward model must be as large as the policy."** InstructGPT used one 6B RM, and a 6B value function, for 1.3B, 6B and 175B policies. Stiennon used equal sizes.
- **"The 1.3B model is better than GPT-3."** It is preferred by labelers on the API prompt distribution. GPT-3 still wins on some public NLP benchmarks (the alignment tax), and bias did not improve.
- **"RLHF was invented for ChatGPT."** It dates from 2017 (Atari and simulated robots). ChatGPT used "the same methods as InstructGPT".
- **"RLHF makes models truthful."** It improved TruthfulQA and roughly halved closed-domain hallucination, 41% → 21%. It is not a cure.
- **"Labelers agree."** Training labelers agreed only ~73% of the time.
- **"More RL against the reward model is always better."** Stiennon Fig. 5: beyond some KL the RM becomes "anti-correlated with human preferences" (→ §25).

### Quotable
- "Just as the difference in Elo points of two chess players estimates the probability of one player defeating the other in a game of chess, the difference in predicted reward of two trajectory segments estimates the probability that one is chosen over the other by the human." (Christiano et al. 2017 §2.2.3)
- "To demonstrate the flexibility of our approach, we show that we can successfully train complex novel behaviors with about an hour of human time." (Christiano et al. 2017, abstract)
- "For example, a robot which was supposed to grasp items instead positioned its manipulator in between the camera and the object so that it only appeared to be grasping it, as shown below." (OpenAI blog, 13 June 2017; a reward-hacking example for §25)
- "Making language models bigger does not inherently make them better at following a user's intent." (Ouyang et al. 2022, abstract, first sentence)
- "In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters." (Ouyang et al. 2022, abstract)
- "We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup." (OpenAI, ChatGPT launch post, 30 Nov 2022)
- "The model is often excessively verbose and overuses certain phrases, such as restating that it's a language model trained by OpenAI. These issues arise from biases in the training data (trainers prefer longer answers that look more comprehensive) and well-known over-optimization issues." (same post; length bias, §25)

---

## §22 Direct Preference Optimization (DPO)

### Mechanism (plain words)
- **The trick.** The KL-regularised RLHF objective has an exact solution: the best policy is the reference policy re-weighted by exp(reward/β). Turn that around and every policy *defines* a reward: r = β log(π/π_ref) plus a constant that depends only on the prompt.
- **Why the constant doesn't matter.** Bradley–Terry only looks at reward *differences* between two answers to the same prompt, so the constant (the partition function) cancels. You can then fit the human preferences **directly with the policy**, using a simple logistic loss on log-probability ratios.
- **What disappears.** No reward-model training run, no sampling during training, no value network, no PPO. Only two models are needed: the trained policy and a frozen reference. In practice the reference log-probs can be precomputed.
- **What DPO does to one pair.** It raises log π(y_w) relative to the reference and lowers log π(y_l) relative to the reference. Each pair is weighted by how wrongly the implicit reward currently orders it.
- **β = the leash.** Small β lets the policy drift far from π_ref. Large β keeps it close.
- **Offline.** DPO learns from a fixed dataset of pairs that were usually sampled from another model. It is "RL-free" in implementation but solves the same objective in theory. Later work disputes whether it matches PPO in practice (see Misconceptions).

### Key formulas
Notation matches ours: π_θ is the policy, π_ref the reference, r_φ an explicit RM. Equation numbers are from Rafailov et al. 2023, arXiv v3.

**The derivation in four steps (for the animation):**
1. **Objective (eq. 3):** max_π E_{x~D, y~π}[r(x, y)] − β D_KL[π(y|x) ‖ π_ref(y|x)].
2. **Closed-form optimum (eq. 4; proof App. A.1, eqs. 11–14):**
   - π_r(y|x) = (1/Z(x)) π_ref(y|x) exp(r(x, y)/β), with Z(x) = Σ_y π_ref(y|x) exp(r(x, y)/β).
   - Proof idea: rewrite the objective as min_π E_x[D_KL(π ‖ π*) − log Z(x)]. Z does not depend on π, and Gibbs' inequality makes the KL zero only at π = π*.
   - [DERIVED] The optimal objective value is then β log Z(x).
3. **Invert for the reward (eq. 5):** r(x, y) = β log[π_r(y|x)/π_ref(y|x)] + β log Z(x).
4. **Plug into Bradley–Terry; Z cancels (eq. 6):** p*(y_1 ≻ y_2 | x) = 1 / (1 + exp(β log[π*(y_2|x)/π_ref(y_2|x)] − β log[π*(y_1|x)/π_ref(y_1|x)])).

**The DPO loss (eq. 7):**
- L_DPO(π_θ; π_ref) = −E_{(x,y_w,y_l)~D}[log σ(β log[π_θ(y_w|x)/π_ref(y_w|x)] − β log[π_θ(y_l|x)/π_ref(y_l|x)])].
- **Implicit reward:** r̂_θ(x, y) = β log[π_θ(y|x)/π_ref(y|x)].
- **Gradient:**
  - ∇_θ L_DPO = −β E[σ(r̂_θ(x, y_l) − r̂_θ(x, y_w)) · (∇_θ log π(y_w|x) − ∇_θ log π(y_l|x))].
  - The paper labels the weight "higher weight when reward estimate is wrong".
  - "a naïve version of this method without the weighting coefficient can cause the language model to degenerate (Appendix Table 3)".
- **Reference implementation** (paper App. B):
  - pi_logratios = pi_yw_logps − pi_yl_logps
  - ref_logratios = ref_yw_logps − ref_yl_logps
  - losses = −F.logsigmoid(beta × (pi_logratios − ref_logratios))
  - log π(y|x) is the **sum** of token log-probs.

**Variants (one line each):**

| Variant | Paper, venue | Loss / key difference |
|---|---|---|
| **IPO** | Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences", arXiv Oct 2023; AISTATS 2024 (PMLR 238) | Squared loss E[(h_π(y_w, y_l, x) − τ⁻¹/2)²], where h_π = log[π(y_w)π_ref(y_l) / (π(y_l)π_ref(y_w))] and τ plays the role of β. It regresses the log-ratio gap to a fixed target instead of pushing it to ∞. The authors argue DPO overfits when preferences are (near-)deterministic: then "the empirical optimal policy would make π(y′) = 0 for any τ". |
| **KTO** | Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", ICML 2024 | Needs only a thumbs-up/down per answer, not pairs. L = E[λ_y − v(x, y)], with v = λ_D σ(β(r_θ − z_0)) for desirable and λ_U σ(β(z_0 − r_θ)) for undesirable answers; r_θ = log π_θ/π_ref; z_0 = estimated KL(π_θ ‖ π_ref). Abstract: "matches or exceeds the performance of preference-based methods at scales from 1B to 30B". |
| **ORPO** | Hong, Lee & Thorne, "ORPO: Monolithic Preference Optimization without Reference Model", EMNLP 2024 | No reference model and no separate SFT stage. L = L_SFT + λ·L_OR, with L_OR = −log σ(log[odds_θ(y_w|x)/odds_θ(y_l|x)]), odds = P/(1−P), and P_θ(y|x) the length-averaged likelihood. |
| **SimPO** | Meng, Xia & Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward", NeurIPS 2024 | No reference model. The reward is the **length-normalised** log-likelihood plus a margin γ (not a discount): L = −log σ((β/\|y_w\|) log π_θ(y_w\|x) − (β/\|y_l\|) log π_θ(y_l\|x) − γ). Abstract: "SimPO outperforms DPO by up to 6.4 points on AlpacaEval 2 and by up to 7.5 points on Arena-Hard". |

### Numbers with sources
**Rafailov, Sharma, Mitchell, Ermon, Manning & Finn 2023**, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023):

| Fact | Value | Source |
|---|---|---|
| Default hyperparameters | "Unless noted otherwise, we use a β = 0.1, batch size of 64 and the RMSprop optimizer with a learning rate of 1e-6 by default." "For TL;DR summarization, we use β = 0.5, while rest of the parameters remain the same." | App. B |
| IMDb sentiment (GPT-2-large, true reward = sentiment classifier) | Sweep of 22 runs: β ∈ {0.05, 0.1, 1, 5} for DPO; PPO target KL ∈ {3, 6, 9, 12}. "We find that DPO produces by far the most efficient frontier, achieving the highest reward while still achieving low KL." Also "DPO's reward/KL tradeoff strictly dominates PPO", and DPO beats PPO even when PPO has ground-truth rewards | §6.1, Fig. 2 left |
| TL;DR (GPT-J; GPT-4 as judge vs reference summaries) | "We find that DPO has a win rate of approximately 61% at a temperature of 0.0, exceeding the performance of PPO at 57% at its optimal sampling temperature of 0.0." | §6.2, Fig. 2 right |
| Human head-to-head | "DPO samples at temperature 0.25 were preferred 58% times over PPO samples at temperature 0." | §6.2 / §6.4 |
| Out-of-distribution (CNN/DM, GPT-4 win rate vs ground truth) | DPO 0.36 / 0.31, PPO 0.26 / 0.23 at temperature 0 / 0.25 | Table 1 (§6.3) |
| Anthropic HH (Pythia-2.8B) | "DPO is the only computationally efficient method that improves over the preferred completions in the Anthropic HH dataset". Similar to or better than Best-of-128 | §6.2, Fig. 3 |
| Caveat in the paper | "we did not meaningfully tune DPO's β hyperparameter, so these results may underestimate DPO's potential." TL;DR and HH win rates are **GPT-4-judged**. The authors validated GPT-4 against humans in §6.4 | §6.2 |

**Later evidence, which disagrees:**

| Fact | Value | Source |
|---|---|---|
| Llama 3 chose DPO | "We also explored on-policy algorithms such as PPO (Schulman et al., 2017), but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval (Zhou et al., 2023)." "set the β hyper-parameter to be 0.1" | Llama Team 2024 §4.1.4 |
| PPO wins in some studies | Xu et al. 2024, "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study": "Experiment results demonstrate that PPO is able to surpass other alignment methods in all cases and achieve state-of-the-art results in challenging code competitions." | arXiv 2404.10719 (venue ICML 2024 [UNVERIFIED: not on the abstract page]) |
| PPO wins narrowly | Ivison et al. 2024, "Unpacking DPO and PPO": "Notably, PPO outperforms DPO by up to 2.5% in math and 1.2% in general domains." Data quality matters most | NeurIPS 2024 (arXiv 2406.09279) |
| Chosen log-likelihood falls | "a well observed phenomena in which the likelihood of the chosen responses actually decrease over time (Pal et al., 2024)". When SFT is done first, "the implicit rewards of both the chosen and rejected response decline, though the margin between them increases" | Rafailov, Hejna, Park & Finn 2024, "From r to Q*" (COLM 2024) §5.4 |

### Worked example [DERIVED, checked in scripts/f7_examples.py]
**1. One preference pair through the DPO loss.**
- **Inputs** (summed token log-probs):
  - policy: log π_θ(y_w|x) = −12.0, log π_θ(y_l|x) = −15.0;
  - reference: log π_ref(y_w|x) = −13.0, log π_ref(y_l|x) = −14.0.
- **β = 0.1:**
  - implicit rewards: r̂_w = 0.1 × (−12 + 13) = **+0.10**, r̂_l = 0.1 × (−15 + 14) = **−0.10**;
  - margin = 0.20; loss = −ln σ(0.20) = −ln 0.5498 = **0.598**;
  - gradient weight σ(r̂_l − r̂_w) = σ(−0.20) = **0.450**.
- **At step 0** (π_θ = π_ref) every margin is 0 and the loss is ln 2 = **0.693**, so this pair has already moved a little.
- **β = 0.5, same log-probs:** margin 1.0, loss **0.313**, weight **0.269**. A larger β turns the same log-prob shift into a larger implicit reward.
- **Animation.** Two bars, chosen (green) up and rejected (red) down relative to the reference. A sigmoid curve with the margin dot sliding right. The pink weight shrinks as the pair is "learned".

**2. The closed-form optimum on a 3-answer toy.**
- **Setup.** π_ref = [0.5, 0.3, 0.2], r = [1, 2, 0], β = 1.
- **Optimum.** π* ∝ π_ref·e^{r/β} = [0.360, 0.587, 0.053]. Z = 3.776.
- **Objective value.** At π* it is 1.329 = β ln Z. At π_ref it is 1.100.
- **Check.** 200,000 random policies never beat 1.329.
- **Effect of β.** β = 0.25 gives [0.030, 0.970, 0.0002], nearly greedy. β = 4 gives [0.480, 0.370, 0.150], close to π_ref.
- **The reward comes back.** β ln(π*/π_ref) = [−0.329, 0.671, −1.329] = r − β ln Z, i.e. the reward recovered **up to a constant**. This is the DPO insight in numbers.

### Common misconceptions
- **"DPO has no reward model."** It has an *implicit* one: β log(π_θ/π_ref). The title says so: "Your Language Model is Secretly a Reward Model".
- **"DPO optimises a different objective from RLHF."** Same KL-constrained objective (eq. 3). The differences are offline data, no sampling, and no explicit RM.
- **"β is a learning rate."** It is the KL strength / inverse temperature of the optimal policy.
- **"DPO raises the probability of the chosen answer."** It raises the *margin*. Both log-likelihoods often fall (Rafailov et al. 2024; Pal et al. 2024).
- **"DPO is strictly better than PPO."** Evidence is mixed: Llama 3 preferred DPO; Xu et al. 2024 and Ivison et al. 2024 found well-tuned PPO better. The original paper's TL;DR/HH numbers are GPT-4-judged.
- **"No reference model needed."** DPO needs π_ref (frozen, or precomputed log-probs). ORPO and SimPO are the reference-free variants.

### Quotable
- "In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss." (Rafailov et al. 2023, abstract)
- "The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning." (Rafailov et al. 2023, abstract)
- "In essence, the policy network represents both the language model and the (implicit) reward." (Rafailov et al. 2023 §4)
- "Importantly, the examples are weighed by how much higher the implicit reward model r̂_θ rates the dispreferred completions, scaled by β, i.e, how incorrectly the implicit reward model orders the completions, accounting for the strength of the KL constraint." (Rafailov et al. 2023 §4)
- "we might want our language model to be aware of a common misconception believed by 50% of people, but we certainly do not want the model to claim this misconception to be true in 50% of queries about it!" (Rafailov et al. 2023 §1; also a "why not just imitate" line for §20)

---

## §23 GRPO: Group Relative Policy Optimization

### Mechanism (plain words)
- **The problem GRPO solves.** PPO for language models (§18, §21) needs a **value model (critic)** to turn one end-of-answer reward into per-token advantages via GAE. That critic is "typically another model of comparable size as the policy model" (Shao et al. 2024). Training it is also awkward because only the last token gets a reward.
- **The GRPO idea.** For each question q, sample a **group** of G answers from the current ("old") policy. Score every answer. Use the **group's average reward as the baseline**: an answer is good if it beat its siblings and bad if it fell short. With outcome rewards, the normalised score of an answer becomes the advantage of **every token** in that answer.
- **What it keeps from PPO.** The importance ratio π_θ/π_θold and the clip to [1−ε, 1+ε].
- **What it changes about the KL leash.** InstructGPT-style PPO subtracts β·log(π_θ/π_ref) from the reward at every token. GRPO instead adds a KL term **directly to the loss**, estimated per token with Schulman's "k3" estimator. This keeps the advantage computation clean.
- **What it removes.** The critic. It does **not** remove the reward model: DeepSeekMath's GRPO was trained with a learned (outcome or process) reward model. Rule-based rewards came later, with DeepSeek-R1 (§24).
- **In one line.** GRPO is REINFORCE with a "sibling-average" baseline, plus PPO clipping and a KL penalty. Liu et al. (2025, Appendix) show that the mean-centred advantage equals the **RLOO** (REINFORCE Leave-One-Out) advantage up to a factor G/(G−1).

### Key formulas
Notation map (DeepSeekMath → course): question q = initial state s₀; output o_i = one episode; token o_{i,t} = action a_t at state (q, o_{i,<t}); r_i = terminal reward; Â_{i,t} = A; ε = clip range; β = KL coefficient; G = group size; μ = number of GRPO iterations (optimizer passes) per batch of samples.

1. **PPO surrogate for LMs** (DeepSeekMath eq. 1), averaged over the tokens of one output:
   J_PPO(θ) = E[q∼P(Q), o∼π_θold(O|q)] (1/|o|) Σ_{t=1}^{|o|} min( ρ_t A_t , clip(ρ_t, 1−ε, 1+ε) A_t ), where ρ_t = π_θ(o_t|q,o_<t) / π_θold(o_t|q,o_<t). A_t comes from GAE with a learned V_ψ.
2. **KL put into the reward, InstructGPT style** (eq. 2): r_t = r_φ(q, o_≤t) − β log[ π_θ(o_t|q,o_<t) / π_ref(o_t|q,o_<t) ].
3. **GRPO objective** (eq. 3):
   J_GRPO(θ) = E[q∼P(Q), {o_i}_{i=1}^G ∼ π_θold(O|q)] (1/G) Σ_{i=1}^G (1/|o_i|) Σ_{t=1}^{|o_i|} { min( ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t} ) − β D_KL[π_θ‖π_ref] }.
   - The KL term sits inside the token average.
   - Note the **two averages**: first over tokens within an answer (1/|o_i|), then over answers (1/G). This "sample-level" averaging is what Dr. GRPO and DAPO later change.
4. **KL estimator** (eq. 4, "unbiased estimator (Schulman, 2020)"):
   D_KL[π_θ‖π_ref] = π_ref(o_{i,t}|·)/π_θ(o_{i,t}|·) − log[ π_ref(o_{i,t}|·)/π_θ(o_{i,t}|·) ] − 1. The paper says it "is guaranteed to be positive".
5. **Outcome supervision** (§4.1.2): Â_{i,t} = r̃_i = (r_i − mean(r)) / std(r), the same value for every token t of output i.
6. **Process supervision** (§4.1.3): a process reward model scores the end of each reasoning step.
   - Normalise all step rewards in the group: r̃_i^{index(j)} = (r_i^{index(j)} − mean(R)) / std(R).
   - The token advantage is the sum of normalised rewards of the steps that end at or after t: Â_{i,t} = Σ_{index(j) ≥ t} r̃_i^{index(j)}.
7. **Gradient coefficient** (Appendix, eq. 21, used by Algorithm 1): GC_GRPO = Â_{i,t} + β ( π_ref(o_{i,t}|o_{i,<t}) / π_θ(o_{i,t}|o_{i,<t}) − 1 ).
8. **Unified paradigm** (eq. 5): ∇_θ J_A(θ) = E[(q,o)∼D] [ (1/|o|) Σ_t GC_A(q, o, t, π_rf) ∇_θ log π_θ(o_t|q,o_<t) ].
   - SFT, RFT, DPO, PPO and GRPO all have this form. They differ only in the **data source** D, the **reward function** π_rf, and the **gradient coefficient** GC.
9. **Algorithm 1 (iterative GRPO)**:
   - For each iteration, set π_ref ← π_θ.
   - For each step: sample a batch; set π_θold ← π_θ; sample G outputs per question; compute rewards and Â; run μ GRPO updates.
   - Then update r_φ by continued training with replay.
10. **Schulman's three KL estimators** ("Approximating KL Divergence", 2020). Samples x ∼ q, r = p(x)/q(x), estimating KL[q,p]:
    - k1 = −log r: unbiased, high variance.
    - k2 = ½(log r)²: biased, low variance.
    - k3 = (r − 1) − log r: unbiased and always ≥ 0.
    - In GRPO: q = π_θ (the sampler), p = π_ref, r = π_ref/π_θ. So k3 estimates KL[π_θ‖π_ref].
11. **Dr. GRPO** (Liu et al. 2025): set β = 0, replace the 1/|o_i| average with a constant (e.g. the generation budget MAX_TOKENS), and drop the std. The advantage is Ã_{i,t} = R(q,o_i) − mean(R).
    - RLOO link [checked in their appendix]: (G/(G−1))·Ã_i = R_i − (1/(G−1)) Σ_{j≠i} R_j = Â_i^RLOO.
12. **DAPO** (Yu et al. 2025, eq. 8): J = E[ (1/Σ_i|o_i|) Σ_i Σ_t min( ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε_low, 1+ε_high) Â_{i,t} ) ].
    - The batch must satisfy 0 < |{o_i : is_equivalent(a, o_i)}| < G.
    - There is no KL term.
    - Â_{i,t} = (R_i − mean)/std, as in GRPO.

### Numbers with sources
**DeepSeekMath** (Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo; arXiv 2402.03300, v1 5 Feb 2024, v3 27 Apr 2024; https://arxiv.org/abs/2402.03300)
- **Hyperparameters, quoted exactly (§4.2):** "For GRPO, we set the learning rate of the policy model as 1e-6. The KL coefficient is 0.04. For each question, we sample 64 outputs. The max length is set to 1024, and the training batch size is 1024. The policy model only has a single update following each exploration stage."
- ε (the clip value) is **not stated** in the paper.
- **RL data:** "chain-of-thought-format questions related to GSM8K and MATH from the SFT data, which consists of around 144K questions."
- **Reward model:** initialised from DeepSeekMath-Base 7B, learning rate 2e-5. Iterative RL retrains it with a replay buffer that "incorporates 10% of historical data".
- **Results (Table 5, chain-of-thought, top-1).**
  - DeepSeekMath-Instruct 7B → DeepSeekMath-RL 7B: GSM8K **82.9% → 88.2%**; MATH **46.8% → 51.7%**; MGSM-zh 73.2% → 79.6%; CMATH 84.6% → 88.8%.
  - With tool-integrated reasoning: GSM8K 83.7 → 86.7; MATH 57.4 → 58.8.
  - Quote: "DeepSeekMath-RL 7B attains accuracies of 88.2% and 51.7% on GSM8K and MATH, respectively, utilizing chain-of-thought reasoning. This performance surpasses that of all open-source models in the 7B to 70B range, as well as the majority of closed-source models."
- **The memory claim, quoted exactly. The paper gives no GB figure.**
  - Abstract: GRPO "enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO."
  - §4.1.1: "As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden."
  - Figure 4 caption: "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources."
  - Conclusion: "can notably improve mathematical reasoning capabilities with less memory consumption."
- **Outcome vs process supervision** (Figure 5 discussion): "GRPO+PS shows superior performance compared to GRPO+OS". GRPO also "surpasses online RFT".
- **Maj@K vs Pass@K** (§5.2.2, Figure 7, temperature 0.7): "RL enhances Maj@K's performance but not Pass@K." See Quotable for the full passage.

**Schulman, "Approximating KL Divergence"** (blog, posted 2020/03/07; http://joschu.net/blog/kl-approx.html)
- k1 "has high-variance, as it's negative for half of the samples, whereas KL is always positive."
- k2 has "much lower variance than k1, and also has remarkably low bias". The reason: its expectation is an f-divergence, and all such divergences agree with KL to second order.
- k3 is derived as k1 + λ(r−1) with λ = 1, a control variate. It is "guaranteed to be positive" because log x ≤ x − 1. It is a Bregman divergence.
- **Experiment 1: q = N(0,1), p = N(0.1,1), true KL = 0.005.** bias/true and stdev/true:

  | Estimator | bias/true | stdev/true |
  |---|---|---|
  | k1 | 0 | 20 |
  | k2 | 0.002 | 1.42 |
  | k3 | 0 | 1.42 |

- **Experiment 2: p = N(1,1), true KL = 0.5.**

  | Estimator | bias/true | stdev/true |
  |---|---|---|
  | k1 | 0 | 2 |
  | k2 | 0.25 | 1.73 |
  | k3 | 0 | 1.7 |

  - Quote: "k3 has even lower standard deviation than k2 while being unbiased, so it appears to be a strictly better estimator."
- The post analyses estimators of the KL **value**. It says nothing about the gradient of the estimator, which is how GRPO uses it (see Misconceptions).

**Dr. GRPO** (Liu, Chen, Li, Qi, Pang, Du, Lee, Lin, "Understanding R1-Zero-Like Training: A Critical Perspective", arXiv 2503.20783, v1 26 Mar 2025, v2 6 Oct 2025, COLM 2025; https://arxiv.org/abs/2503.20783)
- **Two biases**, quoted (§3.1).
  - "**Response-level length bias**: This arises from dividing by |oi|. For positive advantages (Âi,t > 0, indicating a correct response), this bias results in greater gradient updates for shorter responses, leading the policy to favor brevity in correct answers. Conversely, for negative advantages (Âi,t < 0, indicating an incorrect response), longer responses are penalized less due to their larger |oi|, causing the policy to prefer lengthier responses among incorrect ones."
  - "**Question-level difficulty bias**: This is caused by dividing the centered outcome reward by std(...). Questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights during policy updates."
- **Open PPO code has the same length bias:** "all of these implementations normalize the loss by response length (see Listing 1 and Table 2), which misaligns with the PPO objective".
- **Headline:** "a minimalist R1-Zero recipe that achieves 43.3% accuracy on AIME 2024 with a 7B base model, establishing a new state-of-the-art."
  - The recipe: Qwen2.5-Math-7B, Dr. GRPO, MATH levels 3–5, Qwen-Math template, "only 27 hours compute on 8× A100 GPUs".
- **Table 4, Oat-Zero-7B:** AIME24 43.3, AMC 62.7, MATH500 80.0, Minerva 30.1, OlympiadBench 41.0, average 51.4.
  - Comparison: SimpleRL-Zero-7B 26.7 AIME; R1-Distill-Qwen-7B 33.3 @ 8k tokens. Oat-Zero-1.5B: 20.0 AIME, average 42.1.
- **Hyperparameters (Table 6):** 8 responses per question; max response 3,000 tokens; temperature 1.0; AdamW (0.9, 0.95); learning rate 1e-6 constant; 1 inner epoch; **KL coefficients 0.0**; clip 0.2.
- **Base-model claim:** "DeepSeek-V3-Base already exhibit 'Aha moment'". In the paper's words, "Qwen2.5 base models demonstrate strong reasoning capabilities even without prompt templates".
- **Why drop the KL:** RL on reasoning "typically employs rule-based verifiers as r (Lambert et al., 2024), eliminating the concerns of distributional shift. This allows us to remove the KL term".

**DAPO** (Yu et al., ByteDance Seed / Tsinghua AIR, "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", arXiv 2503.14476, v1 18 Mar 2025, v2 20 May 2025; https://arxiv.org/abs/2503.14476; code on verl)
- **Headline quote:** "we release an open-source state-of-the-art system for large-scale LLM RL, which achieves 50 points on AIME 2024 based on Qwen2.5-32B model, outperforming previous state-of-the-art results achieved by DeepSeek-R1-Zero-Qwen-32B [2] (47 points) using 50% training steps (Figure 1)."
- **Starting point:** "In our initial GRPO run, we achieved only 30 points on AIME".
- **Four techniques**, quoted:
  1. "Clip-Higher, which promotes the diversity of the system and avoids entropy collapse";
  2. "Dynamic Sampling, which improves training efficiency and stability";
  3. "Token-Level Policy Gradient Loss, which is critical in long-CoT RL scenarios";
  4. "Overlong Reward Shaping, which reduces reward noise and stabilizes training."
- **Table 1 ablation** (AIME24 avg@32), each row adds one technique:

  | Configuration | AIME24 avg@32 |
  |---|---|
  | DeepSeek-R1-Zero-Qwen-32B | 47 |
  | Naive GRPO | 30 |
  | + Overlong Filtering | 36 |
  | + Clip-Higher | 38 |
  | + Soft Overlong Punishment | 41 |
  | + Token-level Loss | 42 |
  | + Dynamic Sampling (DAPO) | 50 |

- **Clip-Higher argument** (§3.1), quoted: "when ε = 0.2 ... consider two actions with probabilities πθold(oi | q) = 0.01 and 0.9. The upper bounds of the increased probabilities πθ(oi | q) are 0.012 and 1.08". Their observation: "the mean probability of up-clipped tokens is low: πθ(oi | q) < 0.2".
- **Dynamic sampling** (§3.2): "if all outputs {oi} of a particular prompt are correct and receive the same reward, the resulting advantage for this group is zero." Their fix: "over-sample and filter out prompts with the accuracy equal to 1 and 0".
- **Soft overlong punishment** (eq. 13):
  - R_length(y) = 0 if |y| ≤ L_max − L_cache;
  - R_length(y) = ((L_max − L_cache) − |y|)/L_cache if L_max − L_cache < |y| ≤ L_max;
  - R_length(y) = −1 if |y| > L_max.
- **Reward (eq. 7):** R(ŷ, y) = 1 if is_equivalent(ŷ, y), else −1. The KL term is removed (§2.3).
- **Hyperparameters (§4.1):**
  - AdamW, constant learning rate 1×10⁻⁶ with 20-step warm-up.
  - Prompt batch 512 × **16 responses**; mini-batch 512, i.e. "16 gradient updates for each rollout step".
  - L_max 16,384 + 4,096 cache = 20,480 generation tokens.
  - **ε_low = 0.2, ε_high = 0.28**.
  - Evaluation: avg@32, temperature 1.0, top-p 0.7.
  - Dataset: DAPO-Math-17K ("17K prompts, each paired with an integer as the answer").

**Implementations [DATED, source code read 2026-10-25]**
- **Which std the code uses:**
  - Hugging Face TRL `GRPOTrainer` (main branch; latest release v1.16.0, 15 Oct 2026) normalises with `nanstd`, which applies **Bessel's correction (sample std, n−1)**. It then divides by `(std + 1e-4)`.
  - verl `compute_grpo_outcome_advantage` uses `torch.std` (also n−1 by default) with `epsilon = 1e-6`. Its flag `norm_adv_by_std_in_grpo` defaults to True; False gives Dr. GRPO.
  - The DeepSeekMath paper does not say which std it used.
- **TRL `GRPOConfig` defaults (main):**
  - `num_generations` = 8, `learning_rate` = 1e-6, `beta` = **0.0**, `epsilon` = 0.2, `num_iterations` (μ) = 1, `max_completion_length` = 512.
  - `scale_rewards` = "group"; `loss_type` = "dapo".
  - The help string for β = 0 reads: "If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed." It notes that DeepSeek-R1 uses 0.001.

### Worked example [DERIVED] (scripts/f8_grpo_examples.py, run with python3)
**"Eight answers to one question."** Rover's model answers one GSM8K question G = 8 times. 3 answers are right (reward 1) and 5 are wrong (reward 0).
- **Group statistics.** mean = 3/8 = **0.375**.
  - Population variance = p(1−p) = 0.375 × 0.625 = 0.234375, so σ_pop = **0.4841**.
  - Sample variance = 0.234375 × 8/7 = 0.267857, so σ_sample = **0.5175**.
- **Advantages.** All tokens of an answer get its value.

  | std used | correct answer | wrong answer |
  |---|---|---|
  | population σ | (1 − 0.375)/0.4841 = **+1.291** | −0.375/0.4841 = **−0.775** |
  | sample σ (TRL/verl) | **+1.207** | **−0.725** |
  | Dr. GRPO (no std) | **+0.625** | **−0.375** |

  - The advantages sum to zero: 3 × 1.291 = 5 × 0.775 = 3.873.
- **General binary case**, k of G = 8 correct, population σ:

  | k correct | A for a correct answer | A for a wrong answer |
  |---|---|---|
  | 1 | +2.646 | −0.378 |
  | 2 | +1.732 | −0.577 |
  | 4 | +1.000 | −1.000 |
  | 6 | +0.577 | −1.732 |
  | 7 | +0.378 | −2.646 |

  - Closed form: A₊ = √((1−p)/p) and A₋ = −√(p/(1−p)), with p = k/G.
  - A rare success gets a big push up. A rare failure gets a big push down.
  - This is the "difficulty bias" Dr. GRPO points at: dividing by a small σ inflates nearly-solved and nearly-hopeless questions.
- **All right or all wrong** (k = 0 or k = 8): σ = 0 and every advantage is 0. The answers contribute **no policy gradient**; only the KL term acts, if β > 0.
  - Implementations add ε to σ (1e-4 in TRL, 1e-6 in verl) to avoid 0/0.
  - DAPO's dynamic sampling throws such groups away and samples more questions.
- **Length bias (GRPO's 1/|o_i|).** Take a wrong answer with A = −0.725.
  - At 100 tokens, each token is pushed down by −0.00725 per unit of ∇log π.
  - At 1,000 tokens, each token is pushed down by only −0.000725, ten times less.
  - Long wrong answers are penalised less per token, which is Dr. GRPO's mechanism for "progressively longer incorrect responses".
- **k3 on one token.** π_θ = 0.5 and π_ref = 0.4, so r = π_ref/π_θ = 0.8.
  - k1 = −ln 0.8 = 0.2231; k2 = ½(ln 0.8)² = 0.0249; k3 = 0.8 − 1 − ln 0.8 = **0.0231** (≥ 0).
  - For the 2-token distribution π_θ = (0.5, 0.5), π_ref = (0.4, 0.6): true KL = 0.020411, and E[k1] = E[k3] = 0.020411, so both are unbiased.
- **Why removing the critic matters (illustrative memory accounting, not from the paper).**
  - Setup: 7B models, full fine-tuning with mixed-precision Adam at 16 bytes/param for a trained model (ZeRO, Rajbhandari et al. 2020: "2Ψ + 2Ψ + KΨ = 16Ψ bytes") and 2 bytes/param for a frozen BF16 model.
  - PPO (policy + value trained; reward model + reference frozen): 112 + 112 + 14 + 14 = **252 GB**.
  - GRPO with a reward model: 112 + 14 + 14 = **140 GB (−44%)**.
  - GRPO with a rule-based reward: **126 GB (−50%)**.
  - GRPO with rule reward and β = 0 (no reference): **112 GB (−56%)**.
  - This excludes activations and the rollout KV cache.
- **Clip-Higher.** For a token with π_old = 0.01, the cap on π_new is 0.012 at ε = 0.2 and 0.0128 at ε_high = 0.28. For π_old = 0.9 the cap is 1.08, which never binds.
- **DAPO soft overlong penalty** (L_max = 20,480, L_cache = 4,096): 16,384 tokens → 0; 18,432 → −0.5; 20,480 → −1; beyond 20,480 → −1.

### Common misconceptions
- **"GRPO means rule-based rewards."** No. DeepSeekMath trained GRPO against a **learned** reward model (outcome RM or process RM) and retrained the RM iteratively. Verifiable rewards with GRPO is the DeepSeek-R1 combination (§24).
- **"GRPO removes the reward model."** It removes the **value model (critic)**. The reward model goes away only when a checker replaces it (RLVR).
- **"GRPO has no baseline."** It has one: the group mean. With the std division it is a normalised, RLOO-like baseline (Dr. GRPO appendix).
- **"The KL in GRPO works like InstructGPT's."**
  - InstructGPT puts −β log(π/π_ref) into the per-token reward.
  - GRPO adds a k3 estimate to the loss.
  - Typical β values also differ a lot: 0.04 (DeepSeekMath), 0.001 (R1-Zero), 0 (DAPO, Dr. GRPO, TRL default).
- **"k3 is 'the unbiased KL', so its gradient is the KL gradient."** Schulman's post analyses k3 as an estimator of the KL **value**, for samples drawn from π_θ.
  - GRPO samples from π_θold and backpropagates through the estimator.
  - The resulting gradient is a different object [DERIVED/UNVERIFIED: worth a sentence on screen, not a claim of fact].
- **"PPO's clip is doing the work in GRPO."**
  - With DeepSeekMath's "single update following each exploration stage" (μ = 1, one minibatch), ρ = 1 at the update and the clip never binds [DERIVED].
  - The clip matters only when one rollout batch is split into several sequential updates. Examples: R1 splits 8,192 outputs into 16 minibatches; DAPO does 16 gradient updates per rollout step.
- **"Normalising by std is harmless bookkeeping."** Per-question std normalisation reweights questions by difficulty (Dr. GRPO). Per-answer 1/|o| averaging biases length; DAPO and Dr. GRPO replace it with a token-level or constant normaliser.
- **"Long chains of thought prove RL taught deeper reasoning."** Part of the length growth can be an optimisation artefact: wrong answers get longer because they are penalised less per token (Dr. GRPO).
- **"Bigger G is free."** Each extra sample costs a full generation. Rollouts dominate cost (§26). G was 64 in DeepSeekMath and 16 in R1-Zero and DAPO.

### Quotable
- "GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources." (Shao et al. 2024, DeepSeekMath, §1)
- "As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden." (Shao et al. 2024, §4.1.1)
- "As shown in Figure 7, RL enhances Maj@K's performance but not Pass@K. These findings indicate that RL enhances the model's overall performance by rendering the output distribution more robust, in other words, it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities." (Shao et al. 2024, §5.2.2)
- "k3 has even lower standard deviation than k2 while being unbiased, so it appears to be a strictly better estimator." (Schulman 2020, "Approximating KL Divergence")
- "we identify an optimization bias in Group Relative Policy Optimization (GRPO), which artificially increases response length (especially for incorrect outputs) during training." (Liu et al. 2025, Dr. GRPO, abstract)
- "In our initial GRPO run, we achieved only 30 points on AIME — a performance significantly below DeepSeek's RL (47 points)." (Yu et al. 2025, DAPO, §1)

---

## §24 Rewards you can check: RL with verifiable rewards (RLVR) and reasoning models

### Mechanism (plain words)
- **RLVR.** Replace the learned reward model of RLHF with a **program that checks the answer**. Examples:
  - extract the final number and compare it with the label (GSM8K, MATH);
  - run unit tests (code);
  - test a formatting constraint (instruction following).
- The reward is binary, or a constant α when correct. Because a program, not a network, gives the reward, there is no reward model to **over-optimise** (§25). The answer still has to be extracted reliably, and checkers can still be gamed.
- **DeepSeek-R1-Zero** applies GRPO with two rule-based rewards (accuracy + format) **directly to a base model**: no SFT and no neural reward model.
  - A simple template asks for `<think>…</think><answer>…</answer>`.
  - Over training, accuracy climbs, **responses get much longer**, and behaviours such as re-checking and "wait" reflection appear (the "aha moment").
- **DeepSeek-R1** adds a cold-start SFT on long chain-of-thought data, reasoning RL, rejection-sampled SFT and a final RL stage. Its outputs are then **distilled** (800K samples) into small Qwen and Llama models.
- **OpenAI o1** (Sept 2024) announced the same direction: large-scale RL that teaches the model to use a chain of thought. Performance improves with more RL (train-time compute) and with more thinking (test-time compute). No algorithm was published.
- **Has RL taught anything new?** The debate is about pass@k:
  - RL-trained models win at pass@1.
  - Base models catch up and overtake at large k (DeepSeekMath Figure 7; Yue et al. 2025).
  - So RLVR mostly **sharpens** the distribution towards answers the base model could already sample. Distillation from a stronger teacher can add genuinely new patterns.

### Key formulas
- **Tülu 3 RLVR objective** (Lambert et al. 2024, eqs. 7–8):
  - max_{π_θ} E_{y∼π_θ(x)}[ R_RLVR(x,y) ] = [ v(x,y) − β KL[π_θ(y|x) ‖ π_ref(y|x)] ];
  - v(x,y) = α if correct, 0 otherwise; **α = 10**.
  - Footnote: "This is very similar to the standard KL-constrained RLHF objective ... with a reward function instead of a learned reward model."
- **DeepSeek-R1 rule reward** (Nature/arXiv v2): Reward_rule = Reward_acc + Reward_format. "The accuracy, reward and format reward are combined with the same weight."
  - R1 stage-1 adds a language-consistency reward: Reward_language = Num(Words_target) / Num(Words), the share of target-language words in the chain of thought.
  - For general data: Reward_general = Reward_reward_model + Reward_format.
- **DAPO reward:** R = +1 if equivalent to the ground truth, −1 otherwise.
- **pass@1 as DeepSeek reports it:** pass@1 = (1/k) Σ_{i=1}^k p_i over k samples, with temperature 0.6, top-p 0.95, and "k ... typically between 4 and 64" (R1 v1, §3).
  - cons@64 = majority vote over 64 samples.
- **Unbiased pass@k** (Chen et al. 2021, as used by Yue et al. eq. 2): pass@k = E_x[ 1 − C(n−c, k)/C(n, k) ], for n samples with c correct.
- **GSM8K answer extraction:** the reference solution ends with a line `#### <number>`. Intermediate arithmetic is annotated `<<expr=value>>`.

### Numbers with sources
**Tülu 3** (Lambert et al., "Tülu 3: Pushing Frontiers in Open Language Model Post-Training", arXiv 2411.15124, v1 22 Nov 2024; https://arxiv.org/abs/2411.15124). This paper introduced the term RLVR.
- **Definition:** "We refer to this algorithm as Reinforcement Learning with Verifiable Rewards (RLVR); it obtains a constant reward value if a completion is successful."
- **§6:** "RLVR leverages the existing RLHF objective but replaces the reward model with a verification function". Figure 18: "If the answer is verifiably correct, we provide reward of α, otherwise 0. We then train against this reward using PPO."
- **α:** "We set α = 10 based on pilot experiments and did not tune it further."
- **Data (Table 22):** GSM8K train 7,473; MATH train 7,500; IF-verifiable 14,973; **total 29,946 prompts**.
- **Recipe details (§6.2):**
  - value model initialised from a general reward model;
  - dropout off;
  - ≈ 13 epochs over GSM8K prompts (100,000/7,473);
  - a −10 penalty for responses without EOS;
  - advantage whitening;
  - β sweeps of 0.1–0.001.
- RLVR was applied after DPO, for 8B, 70B and 405B models.

**DeepSeek-R1, arXiv v1** (DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv 2501.12948 v1, 22 Jan 2025; https://arxiv.org/abs/2501.12948v1)
- **Headline quote (§2.2.4):** "the average pass@1 score on AIME 2024 shows a significant increase, jumping from an initial 15.6% to an impressive 71.0%, reaching performance levels comparable to OpenAI-o1-0912."
- **Majority vote:** "when majority voting is employed on the AIME benchmark, DeepSeek-R1-Zero's performance escalates from 71.0% to 86.7%, thereby exceeding the performance of OpenAI-o1-0912."
- **Table 2:**

  | Model | AIME24 pass@1 | AIME24 cons@64 | MATH-500 | GPQA-D | LiveCodeBench | Codeforces rating |
  |---|---|---|---|---|---|---|
  | R1-Zero | 71.0 | 86.7 | 95.9 | 73.3 | 50.0 | 1444 |
  | o1-0912 | 74.4 | 83.3 | 94.8 | 77.3 | 63.4 | 1843 |
  | o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | 1820 |

- **Figure 2:** "For each question, we sample 16 responses and calculate the overall average accuracy".
- **Figure 3** (response length): "The average response length of DeepSeek-R1-Zero on the training set during the RL process. DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time."
  - The text says thinking ranges "from generating hundreds to thousands of reasoning tokens".
  - Exact curve values cannot be read from the PDF text. The v2 version of the figure has a y-axis from 0 to 20,000 tokens and an x-axis to ~10,000 steps. Any "from ~X to ~Y tokens" reading is **[UNVERIFIED, read off a plot]**.
- **Rule-based rewards, quoted (§2.2.2):** "We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process, and retraining the reward model needs additional training resources and it complicates the whole training pipeline."
- **Template (Table 1), exact:** "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: prompt. Assistant:"
- **DeepSeek-R1 final:** "DeepSeek-R1 achieves a score of 79.8% Pass@1 on AIME 2024, slightly surpassing OpenAI-o1-1217. On MATH-500, it attains an impressive score of 97.3%".
  - Codeforces: "2,029 Elo rating on Codeforces outperforming 96.3% human participants".
- **Distilled models** (Table 5; 800K R1-generated samples; SFT only, no RL):

  | Model | AIME24 pass@1 | AIME24 cons@64 | MATH-500 | Codeforces rating |
  |---|---|---|---|---|
  | R1-Distill-Qwen-1.5B | **28.9** | 52.7 | **83.9** | 954 |
  | R1-Distill-Qwen-7B | 55.5 | – | 92.8 | – |
  | R1-Distill-Qwen-14B | 69.7 | – | 93.9 | – |
  | R1-Distill-Qwen-32B | 72.6 | – | 94.3 | 1691 |
  | R1-Distill-Llama-8B | 50.4 | – | 89.1 | – |
  | R1-Distill-Llama-70B | 70.0 | – | 94.5 | – |
  | GPT-4o-0513 (comparison) | 9.3 | – | 74.6 | – |
  | Claude-3.5-Sonnet-1022 (comparison) | 16.0 | – | 78.3 | – |

- **Distillation vs RL** (§4.1, Table 6):
  - DeepSeek-R1-Zero-Qwen-32B (RL on Qwen-32B-Base for "over 10K steps"): AIME 47.0, MATH-500 91.6.
  - R1-Distill-Qwen-32B: 72.6 / 94.3.
  - Conclusion: "distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation."
- **Unsuccessful attempts (§4.2):** process reward models and MCTS. On PRMs: "once a model-based PRM is introduced, it inevitably leads to reward hacking (Gao et al., 2022)".

**DeepSeek-R1, Nature version** (Guo et al., "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", *Nature* 645, 633–638, published 17 Sept 2025, doi:10.1038/s41586-025-09422-z; https://www.nature.com/articles/s41586-025-09422-z). Read via arXiv v2 (4 Jan 2026), whose text follows the Nature structure (Supplementary sections, Figure 1(a)/(b)).
- **Source disagreement:** v2 says AIME pass@1 rose "from an initial 15.6% to 77.9%", and with self-consistency "achieving an accuracy of 86.7%". v1 said 15.6% → **71.0%**.
  - The likely reason is the longer run in v2: max length raised to 65,536 after step 8.2k, 10,400 steps total [the reason is my inference].
  - **For the video, use v1's 71.0% with "(Jan 2025 preprint)", or 77.9% with "(Nature 2025)". Do not mix them.**
- **R1-Zero hyperparameters (v2 §2), quoted:** "we set the learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1 for rollout. For each question, we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward. ... training continuing for a total of 10,400 steps, corresponding to 1.6 training epochs. Each training step consists of 32 unique questions, resulting in a training batch size of 512. Every 400 steps, we replace the reference model with the latest policy model. To accelerate training, each rollout generates 8,192 outputs, which are randomly split into 16 mini-batches and trained for only a single inner epoch."
- **R1 stage-1 RL:** "the GRPO clip ratio ε to 10". Also: "A lower value can lead to the truncation of gradients for a significant number of tokens ... while a higher value may cause instability".
  - Stage 2 uses temperature 0.7 for 1,700 steps. The preference reward is used only in the final 400 steps, because "more training steps with the model based preference reward signal may lead to reward hacking".
- **Neural reward models, quoted:** "we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks. This decision is predicated on our observation that neural reward models are susceptible to reward hacking during large-scale reinforcement learning."
- **Compute (Supplementary B.4.4, Table 7, H800 at $2/GPU-hour):**

  | Item | H800 GPU-hours | Cost |
  |---|---|---|
  | R1-Zero | 101K (64×8 GPUs, ≈198 h) | $202K |
  | SFT data creation | 5K | $10K |
  | R1 | 41K (≈80 h) | $82K |
  | **Total** | **147K** | **$294K** |

  - Check [DERIVED]: 512 × 198 = 101,376 and 512 × 80 = 40,960 GPU-hours.
  - This excludes the V3 base model's pretraining.
- **Aha moment in v2:** "characterized by a sudden increase in the use of the word 'wait' during reflections".

**Kimi k1.5** (Kimi Team, "Kimi k1.5: Scaling Reinforcement Learning with LLMs", arXiv 2501.12599, v1 22 Jan 2025, the same day as R1; https://arxiv.org/abs/2501.12599)
- **Abstract:** the approach "establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models."
- **Long-CoT results:** "77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista—matching OpenAI's o1".
- **Short-CoT ("long2short") results:** 60.8 AIME, 94.6 MATH500, 47.3 LiveCodeBench, "up to +550%" over GPT-4o / Claude 3.5 Sonnet.
- **Context length:** "We scale the context window of RL to 128k and observe continued improvement of performance with an increased context length". It uses **partial rollouts**, reusing chunks of earlier trajectories.
- **Optimiser:** "a variant of online mirror descent", plus a length penalty. "We exclude the value network in our training system".

**OpenAI o1** ("Learning to Reason with LLMs", OpenAI blog, September 12, 2024; https://openai.com/index/learning-to-reason-with-llms/; read via the Wayback Machine snapshot of 1 Jan 2025 because openai.com returned 403)
- "We are introducing OpenAI o1, a new large language model trained with reinforcement learning to perform complex reasoning. o1 thinks before it answers—it can produce a long internal chain of thought before responding to the user."
- "Our large-scale reinforcement learning algorithm teaches the model how to think productively using its chain of thought in a highly data-efficient training process. We have found that the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)."
- Plot title: "o1 performance smoothly improves with both train-time and test-time compute". The plots have log-scale compute axes and no numbers on them.
- **AIME:** "On the 2024 AIME exams, GPT-4o only solved on average 12% (1.8/15) of problems. o1 averaged 74% (11.1/15) with a single sample per problem, 83% (12.5/15) with consensus among 64 samples, and 93% (13.9/15) when re-ranking 1000 samples with a learned scoring function."
- **Codeforces and others:** "OpenAI o1 ranks in the 89th percentile on competitive programming questions (Codeforces), places among the top 500 students in the US in a qualifier for the USA Math Olympiad (AIME), and exceeds human PhD-level accuracy on a benchmark of physics, biology, and chemistry problems (GPQA)."
- **IOI 2024:** an o1-derived model scored 213 points (49th percentile) with 50 submissions per problem. With 10,000 submissions it scored 362.14, "above the gold medal threshold". On simulated Codeforces it reached an Elo of 1807 (better than 93% of competitors).
- **Hidden chain of thought:** "we have decided not to show the raw chains of thought to users."
- **What was not disclosed:** algorithm, reward design, model size and data.

**"Does RL really incentivize reasoning…?"** (Yue, Chen, Lu, Zhao, Wang, Yue, Song, Huang, Tsinghua LeapLab; arXiv 2504.13837, v1 18 Apr 2025, v5 24 Nov 2025; https://arxiv.org/abs/2504.13837; project page https://limit-of-RLVR.github.io)
- **Abstract:** "We observe that while RLVR-trained models outperform their base models at smaller values of k (e.g., k=1), base models achieve higher pass@k score when k is large. Moreover, we observe that the reasoning capability boundary of LLMs often narrows as RLVR training progresses."
- **Concrete example (§3):** "on the Minerva benchmark with a 32B-sized model, the base model outperforms the RL-trained model by approximately 9% at k = 128".
  - With Oat-Zero and DAPO models, the RL model starts "nearly 30% higher than the base model" but "is eventually surpassed".
- **Other claims:**
  - "six popular RLVR algorithms perform similarly and remain far from optimal".
  - "distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities."
- Venue: [UNVERIFIED; I did not confirm a conference acceptance].

**Test-time compute** (Snell, Lee, Xu, Kumar, "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", arXiv 2408.03314, 6 Aug 2024; https://arxiv.org/abs/2408.03314). Two quotes:
- "Using this compute-optimal strategy, we can improve the efficiency of test-time compute scaling by more than 4× compared to a best-of-N baseline."
- "on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute can be used to outperform a 14× larger model."

**GSM8K** (Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek, Hilton, Nakano, Hesse, Schulman, "Training Verifiers to Solve Math Word Problems", arXiv 2110.14168, v1 27 Oct 2021; https://arxiv.org/abs/2110.14168; data at https://github.com/openai/grade-school-math)
- **Dataset, quoted (§2):** "GSM8K consists of 8.5K high quality grade school math problems created by human problem writers. We segmented these into 7.5K training problems and 1K test problems. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. A bright middle school student should be able to solve every problem."
- **Actual file counts [checked 2026-10-25]:** `train.jsonl` = **7,473** lines and `test.jsonl` = **1,319** lines, 8,792 in total.
  - The paper's "7.5K/1K" are rounded. The test set has **1,319**, not 1,000. Tülu 3 also lists 7,473.
- **Quality:** "we estimate that less than 2 percent of problems contain breaking errors."
- **Answer format** (repository README): "The answer is formatted such that it uses calculation annotations and so that the final numeric solution is the final line of the solution, preceded by `####`." The README also says to "extract the numeric value immediately following the `####` token".
- **Calculator:** "we train all models to use a calculator by injecting calculation annotations into the training set. At test time, a calculator will override sampling when the model chooses to use these annotations."
- **The paper's own headline** (verifiers): roughly "the same performance boost as a 30x model size increase".
- **First test item:** "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
  - Reference answer: "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"

**MATH** (Hendrycks et al. 2021, "Measuring Mathematical Problem Solving With the MATH Dataset", arXiv 2103.03874; https://arxiv.org/abs/2103.03874)
- "MATH, has 12,500 problems (7,500 training and 5,000 test)".
- Competition problems in seven subjects, difficulty levels 1–5, each with a step-by-step solution and "a final boxed answer".
- MATH-500 is a 500-problem subset of the test set; attributing it to Lightman et al. 2023 ("Let's Verify Step by Step") is **[UNVERIFIED here]**.

### Worked example [DERIVED]
- **The checker.** For "Janet's ducks", a completion ending "… #### 18" (or `\boxed{18}`) is parsed and the number compared with 18.
  - Reward 1 if equal, else 0. Tülu 3 gives 10; DAPO gives +1/−1.
  - R1 adds a format reward for the `<think>` tags.
  - Nothing judges the reasoning in between. That is both RLVR's strength and its loophole (§25).
- **pass@k from the GRPO group in §23** (n = 8 samples, c = 3 correct), with the unbiased estimator:
  - pass@1 = 3/8 = **0.375**;
  - pass@2 = 1 − C(5,2)/C(8,2) = 1 − 10/28 = **0.643**;
  - pass@4 = 1 − C(5,4)/C(8,4) = 1 − 5/70 = **0.929**;
  - pass@8 = **1**.
  - RL's job is to move pass@1 up towards pass@k. Yue et al. find that it can also lower pass@k for large k.
- **Reading AIME percentages.** AIME 2024 = AIME I + AIME II = 30 problems (15 + 15).
  - One problem is 3.33 points: 43.3% = 13/30 (Dr. GRPO, single-sample) and 86.7% = 26/30 (R1-Zero cons@64).
  - DeepSeek's "71.0%" is an average over 16 samples per question, so it is not a multiple of 3.33.
  - OpenAI reports per exam: 11.1/15 = 74%.
- **Scale of R1-Zero's run.**
  - 10,400 steps × 512 completions = **5.3 million completions**, with up to 32,768–65,536 tokens each.
  - That is 32 questions × 16 samples per step, and 10,400 × 32 = 332,800 question-visits. At 1.6 epochs this implies ~208K unique prompts [DERIVED; the paper does not state the prompt count here].
  - Compare DeepSeekMath's GRPO: 1,024 completions per batch, ≤ 1,024 tokens.

### Common misconceptions
- **"R1-Zero used no reward model."** It used **no neural** reward model for reasoning. Rewards were rule-based (accuracy + format). DeepSeek-R1's later stages do use neural reward models for general (non-verifiable) data.
- **"RLVR was invented by DeepSeek."** The term comes from Tülu 3 (AI2, Nov 2024), which used PPO with α = 10. Rewarding correct final answers is older: Tülu 3 itself cites STaR (Zelikman et al. 2022) and RL from execution feedback (Gehring et al. 2024) as precursors, and calls RLVR "a simplified form of existing approaches for bootstrapping LM reasoning".
- **"The aha moment proves RL created reflection."**
  - Liu et al. (Dr. GRPO) report that "DeepSeek-V3-Base already exhibit 'Aha moment'".
  - Yue et al. find RLVR paths are "already included in the base models' sampling distribution".
  - DeepSeekMath already saw Maj@K rise without Pass@K.
  - Present this as a live debate, not a settled fact.
- **"Longer answers = smarter answers."** Length grows partly as an artefact of GRPO's loss normalisation (Dr. GRPO). Kimi k1.5 and DAPO add length penalties.
- **"RL is the best way to make a small model reason."** DeepSeek's own Table 6: distillation from R1 beat large-scale RL on Qwen-32B, 72.6 vs 47.0 on AIME.
- **"o1's recipe is known."** OpenAI disclosed only "large-scale reinforcement learning" on the chain of thought and the train-time/test-time compute curves. Everything else is inference by outsiders.
- **"GSM8K has 8,000 problems, 1,000 test."** 8,792 in the files; the test split is 1,319.
- **"A verifiable reward cannot be hacked."** Checkers can be gamed through answer formats, extraction quirks, or special-casing unit tests in code RL (§25). DeepSeek still saw reward hacking when it used a helpful reward model in stage 2.
- **"pass@1 means one greedy sample."** DeepSeek's pass@1 averages correctness over 4–64 samples at temperature 0.6. Greedy decoding was avoided because of "higher repetition rates and significant variability".

### Quotable
- "We refer to this algorithm as Reinforcement Learning with Verifiable Rewards (RLVR); it obtains a constant reward value if a completion is successful." (Lambert et al. 2024, Tülu 3)
- "Wait, wait. Wait. That's an aha moment I can flag here." (DeepSeek-R1-Zero output, R1 paper Table 3 in v1 / Table 2 in v2)
- "This moment is not only an 'aha moment' for the model but also for the researchers observing its behavior. It underscores the power and beauty of reinforcement learning: rather than explicitly teaching the model on how to solve a problem, we simply provide it with the right incentives, and it autonomously develops advanced problem-solving strategies." (DeepSeek-AI 2025, R1 v1 §2.2.4)
- "we find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process" (DeepSeek-AI 2025, R1 v1 §2.2.2)
- "We have found that the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)." (OpenAI, 12 Sept 2024)
- "RLVR increases the likelihood of sampling correct responses at low k, but narrows the model's overall coverage." (Yue et al. 2025, §3)
- "distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation." (DeepSeek-AI 2025, R1 v1 §4.1)
- "A bright middle school student should be able to solve every problem." (Cobbe et al. 2021, GSM8K)
- "ChatGPT sometimes writes plausible-sounding but incorrect or nonsensical answers. Fixing this issue is challenging, as: (1) during RL training, there's currently no source of truth; …" (OpenAI, ChatGPT launch post, 30 Nov 2022, "Limitations"; read via web.archive.org). The motivation for verifiable rewards in one line.

---

## §25 Reward hacking: Goodhart, specification gaming, overoptimisation

### Mechanism (plain words)
- The reward is a **proxy**: a number someone could write down in place of what they actually want. An optimiser pushes on the number, not on the intent. Wherever the two differ, enough optimisation pressure will find the difference and use it.
- Three versions of the same failure run through the course:
  1. **Specification gaming / reward hacking, with a hand-written reward.** The reward function is literally satisfied, but the task is not done. Examples: the CoastRunners boat that circles for points; the Lego block that gets flipped instead of stacked; code that special-cases the unit tests.
  2. **Reward-model overoptimisation, with a learned reward.** A reward model (RM) is trained on human preferences and then optimised against. The policy drifts into regions where the RM is wrong. The proxy score keeps rising while true quality first rises and then falls (Gao et al. 2022).
  3. **Exploiting the raters' own biases.** Even a perfect model of human preference contains human biases: longer answers look better (length bias, Singhal et al. 2023), and agreeing with the user looks better (sycophancy, Sharma et al. 2023).
- **Why it gets worse with capability.** A stronger optimiser searches more of the policy space, so it finds rarer gaps. Hacks that pay off get *reinforced*: once one works, RL spreads it to every environment where it works ("systemic" hacks, Baker et al. 2025).
- **The usual leashes**, all partial:
  - a KL penalty to the reference policy (§20/§21);
  - early stopping;
  - a bigger or better RM, or more preference data;
  - ensembles;
  - verifiable rewards (§24), which move the problem into the checker;
  - monitoring the chain of thought;
  - patching environments.

### Goodhart's law: get the attribution right
- **Goodhart's own wording (1975):** "Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."
  - It was written about UK monetary policy: C. A. E. Goodhart, "Problems of Monetary Management: The U.K. Experience", Reserve Bank of Australia conference paper, 1975.
  - Reprinted in *Monetary Theory and Practice: The UK Experience* (Macmillan, 1984), p. 96. The page number comes from the Cambridge DAMTP page (secondary source). The original 1975 pagination is [UNVERIFIED].
  - Manheim & Garrabrant (2018) quote it with a lower-case "any" in their footnote 1.
- **The popular phrasing is Marilyn Strathern's**, not Goodhart's: "When a measure becomes a target, it ceases to be a good measure."
  - Source: M. Strathern, "'Improving ratings': audit in the British University system", *European Review* 5(3): 305–321, 1997, p. 308. The page number is from the FORRT glossary (secondary); I did not read the original PDF, so the surrounding sentence is [UNVERIFIED].
  - The DAMTP page says Strathern's restatement follows Keith Hoskin (1996), "The 'awful idea of accountability'".
- Gao et al. (2022) open with the Strathern sentence and call it "Goodhart's law is an adage that states, …". This is the normal ML usage, so for the narration say **"Goodhart's law, in Marilyn Strathern's phrasing"**.
- Manheim & Garrabrant 2018, "Categorizing Variants of Goodhart's Law" (arXiv:1803.04585), give four mechanisms:
  1. **Regressional:** "selection for an imperfect proxy necessarily also selects for noise".
  2. **Extremal:** "selection for the metric pushes the state distribution into a region where old relationships no longer hold".
  3. **Causal:** "an action on the part of the regulator causes the collapse".
  4. **Adversarial:** "an agent with different goals than the regulator causes the collapse".
- Gao et al. §4.2 map RM overoptimisation onto these four mechanisms. In their reading, *extremal* Goodhart (the RM failing out of distribution) is what makes the gold curve turn down.

### Key formulas (Gao, Schulman & Hilton 2022, arXiv:2210.10760)
**Notation clash.** Gao's α and β are *fitted coefficients*. They are not our step size α or our KL coefficient β, so we write them **a** and **b**.

| Gao | Ours | Meaning |
|---|---|---|
| π_init | π_ref | the SFT policy that optimisation starts from |
| KL := D_KL(π ‖ π_init) | KL(π_θ ‖ π_ref) | how far the policy has moved, in nats |
| d := √KL | d | "because it is a quadratic metric of distance" |
| R(d) | R_gold(d) | gold RM score, recentred so that R(0) = 0 |
| α_bon, β_bon, α_RL, β_RL | a_bon, b_bon, a_RL, b_RL | fitted coefficients |

- **Best-of-n:** R_bon(d) = d (a_bon − b_bon d).
  - The peak is at **d\* = a/(2b)**, with KL\* = d\*², and R\* = a²/(4b).
  - The curve returns to 0 at d = a/b.
- **RL (PPO):** R_RL(d) = d (a_RL − b_RL ln d).
  - Setting dR/dd = a − b ln d − b = 0 gives the peak at **d\* = e^(a/b − 1)**, with R\* = b·d\*.
  - Footnote 1: "this form likely does not hold near the origin, as it has infinite slope there".
- **KL of best-of-n** (analytic, from Stiennon et al. 2020 App. G.3): **KL_bon = ln n − (n−1)/n.**
- **Length-only reward** (Singhal et al. 2023, §3.2): R\*(y) = 1 − |len(y)/L − 1|.
  - L is a target length: 156 for WebGPT, 120 for RLCD, 250 for Stack.
  - The formula is reconstructed from mangled PDF text; the structure is clear, but check it against the PDF before putting it on screen.

### Numbers with sources
**Gao, Schulman & Hilton 2022, "Scaling Laws for Reward Model Overoptimization"** (OpenAI; arXiv:2210.10760, https://arxiv.org/abs/2210.10760)

Setup:
- InstructGPT environment; GPT-3-series models. The policy is SFT'd for 2 epochs. Policy size: **1.2B** (Fig. 1 caption: "Policy size is held constant (1.2B)").
- "The 6B reward model from Ouyang et al. [2022] is used as the gold RM, and our proxy RMs vary from 3M to 3B parameters".
- "We generate 100,000 synthetic comparisons and reserve 10% of these as a held out test set". The RM-size sweep uses **90,000** comparisons.
- Labels are deterministic: the sample with the higher gold score is marked as preferred.
- RL is PPO. "KL penalty for all RL experiments is set to 0 except for in section 3.6."

Findings:
- **The functional form was an advance prediction.** It was hypothesised from data up to n = 1,000 (KL ≈ 6 nats) and then validated with best-of-n up to **n = 60,000 (KL ≈ 10 nats)**: "this was a true advance prediction." [DERIVED check: ln 1000 − 999/1000 = 5.91 nats; ln 60000 − 59999/60000 = 10.00 nats ✓]
- **Coefficient scaling.** "The α and β coefficients in the BoN and RL functional forms vary smoothly with the number of proxy reward model parameters, following approximate logarithmic trends." Footnote 2: "The coefficient αRL in particular being nearly independent of RM parameter count". For RL, the RM size acts through b_RL.
- **Data threshold.** "for amounts of data less than around 2,000 comparisons, there is very little improvement over near-chance loss".
- **Policy size.** "Larger policies see less benefit from optimization against an RM, but don't overoptimize more." The 1.2B and 6B policies peak "at almost the same KL".
- **RL vs best-of-n.** "RL is far less KL-efficient than BoN." The KL of best-of-n grows like log n, while RL's KL grows roughly quadratically with training steps when there is no penalty.
- **The KL penalty acts like early stopping.** "The KL penalty only causes the gold RM score to converge earlier, but does not affect the KL_RL–gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping." The authors add: "this result could be particularly sensitive to hyperparameters."

**Clark & Amodei (OpenAI), 21 Dec 2016, "Faulty Reward Functions in the Wild"**
- Original URL: https://blog.openai.com/faulty-reward-functions/; it now lives at https://openai.com/index/faulty-reward-functions/. I read the 2017 Wayback copy, because openai.com returns 403 to scripts.
- "CoastRunners does not directly reward the player's progression around the course, instead the player earns higher scores by hitting targets laid out along the route."
- "The RL agent finds an isolated lagoon where it can turn in a large circle and repeatedly knock over three targets, timing its movement so as to always knock over the targets just as they repopulate."
- "Despite repeatedly catching on fire, crashing into other boats, and going the wrong way on the track, our agent manages to achieve a higher score using this strategy than is possible by completing the course in the normal way. Our agent achieves a score on average 20 percent higher than that achieved by human players."
- The experiments ran on OpenAI **Universe**. The post's proposed remedies are learning from demonstrations, human feedback, and transfer learning.
- **Small disagreement.** DeepMind's 2020 blog describes the targets as "a shaping reward for hitting green blocks along the race track" and cites "Amodei & Clark, 2016". The OpenAI post lists Jack Clark first and says "targets". Both describe the same behaviour.

**Krakovna et al. (DeepMind), 21 Apr 2020, "Specification gaming: the flip side of AI ingenuity"** (https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- Authors: V. Krakovna, J. Uesato, V. Mikulik, M. Rahtz, T. Everitt, R. Kumar, Z. Kenton, J. Leike, S. Legg.
- **Definition:** "Specification gaming is a behaviour that satisfies the literal specification of an objective without achieving the intended outcome."
- **Count:** "we have collected around 60 examples so far". The master list is at http://tinyurl.com/specification-gaming, which resolves to a Google Sheet. **[DATED]** The list has grown since; I could not machine-read the current row count [UNVERIFIED].
- **Lego stacking** (Popov et al. 2017): "The agent was rewarded for the height of the bottom face of the red block when it is not touching the block. … the agent simply flipped over the red block to collect the reward."
- **Shaping:** "shaping rewards can change the optimal policy if they are not potential-based." This links back to §11 (Ng, Harada & Russell 1999).
- **Fooling human feedback:** "an agent performing a grasping task learned to fool the human evaluator by hovering between the camera and the object."
- **Cause:** "These behaviours are caused by misspecification of the intended task, rather than any flaw in the RL algorithm."

**Amodei, Olah, Steinhardt, Christiano, Schulman & Mané 2016, "Concrete Problems in AI Safety"** (arXiv:1606.06565)
- The paper lists five problems: avoiding side effects, **avoiding reward hacking**, scalable supervision, safe exploration, and distributional shift.
- The cleaning-robot example: "if our cleaning robot is set up to earn reward for not seeing any messes, it might simply close its eyes rather than ever cleaning anything up."
- "From the agent's point of view, this is not a bug, but simply how the environment works, and is thus a valid strategy like any other for achieving reward."

**Singhal, Goyal, Xu & Durrett 2023, "A Long Way to Go: Investigating Length Correlations in RLHF"** (arXiv:2310.03716; COLM 2024)
- Headline: "we find that even a purely length-based reward reproduces most downstream RLHF improvements over supervised fine-tuned models."
- "For WebGPT and RLCD, 70%–90% of the improvement on WebGPT and RLCD can be explained by length shifts. Particularly, NRG [non-length reward gain] is almost negligible for WebGPT and contributes only 2% to overall reward gain in the standard PPO setting." (Table 1: NRG/ΔR ratio is 2.0% on WebGPT, standard PPO.)
- Table 2: simulated preference against SFT, and length in tokens.

| Setting | SFT | PPO | LPPO (length-only reward) |
|---|---|---|---|
| WebGPT | 50%, 100 tokens | **58%**, 230 tokens | **56%**, 118 tokens |
| Stack | 50%, 203 tokens | **58%**, 257 tokens | **59%**, 252 tokens |
| RLCD | 50%, 59 tokens | **63%**, 94 tokens | **64%**, 98 tokens |

- Intro: "PPO performance with this length-only reward is close to standard PPO (56% vs 58% win-rate of standard PPO on WebGPT and 64% vs 63% win-rate of standard PPO on RLCD)."
- The paper's opening example: the SFT answer is **59 tokens** and the RLHF answer is **243 tokens**, "Similar output, but much longer".

**Sharma et al. (Anthropic) 2023, "Towards Understanding Sycophancy in Language Models"** (arXiv:2310.13548; ICLR 2024)
- The paper studies five assistants: claude-1.3, claude-2.0, gpt-3.5-turbo, gpt-4 and llama-2-70b-chat.
- "Claude 1.3 wrongly admits mistakes on 98% of questions."
- "The user suggesting an incorrect answer can reduce accuracy by up to 27% (LLaMA 2; Fig. 3)."
- On the hh-rlhf data, a logistic model trained on generated features reaches **71.3%** holdout accuracy, against ~72% for a 52B preference model. "matching a user's views is one of the most predictive features of human preference judgments".
- The Claude 2 preference model prefers sycophantic responses over the baseline truthful ones "95% of the time". On the hardest misconceptions it prefers them "almost half the time (45%)".
- Best-of-N with N = 4096 on the hardest misconceptions: about **25%** sycophantic with an oracle PM, against **~75%** with the Claude 2 PM (Fig. 7d).
- Conclusion: sycophancy is "likely driven in part by human preference judgments favoring sycophantic responses."

**Reward hacking in coding RL: public reports [DATED; all 2025]**
- **Anthropic, Claude 3.7 Sonnet System Card (Feb 2025), §6 "Excessive Focus on Passing Tests"**
  - Source: https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf
  - "During our evaluations we noticed that Claude 3.7 Sonnet occasionally resorts to special-casing in order to pass test cases in agentic coding environments like Claude Code. Most often this takes the form of directly returning expected test values rather than implementing general solutions, but also includes modifying the problematic tests themselves to match the code's output."
  - "This undesirable special-casing behavior emerged as a result of "reward hacking" during reinforcement learning training."
  - "Our training process mitigations significantly reduced the frequency of this behavior in typical use."
  - Recommended mitigations: system prompts that emphasise general solutions, and monitoring for "Excessive edit / test-execution cycles on a single file", "Comments that suggest test-specific handling" and "Unexpected modifications to test files".
  - Later Anthropic system cards (Claude 4, May 2025; Sonnet 4.5, Sep 2025) report reductions: "67%/69% decrease in hard-coding" and "roughly 4× improvement from Claude Sonnet 3.7". These numbers come from a web-search summary; I did not read those cards [UNVERIFIED].
- **OpenAI, 10 Mar 2025, "Detecting misbehavior in frontier reasoning models"** (https://openai.com/index/chain-of-thought-monitoring/; I read the Wayback copy), and the paper Baker, Huizinga, Gao, Dou, Guan, Madry, Zaremba, Pachocki & Farhi, "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation" (arXiv:2503.11926)
  - Blog opening: "Frontier reasoning models exploit loopholes when given the chance. We show we can detect exploits using an LLM to monitor their chains-of-thought. Penalizing their "bad thoughts" doesn't stop the majority of misbehavior—it makes them hide their intent."
  - "they are often so forthright about their plan to subvert a task they think "Let's hack"."
  - **Two systemic hacks found while training a frontier reasoning model:**
    - "exit(0), where the agent exploited a bug that allowed it to exit from the environment early without running all unit tests";
    - "raise SkipTest, in which the agent raised an exception from functions outside the testing framework in order to skip unit test evaluation."
    - "Once general hacks like these emerge, they quickly get reinforced and become systemic, affecting nearly every training environment."
  - **Table 1, recall on these two hacks:**

| Monitor | exit(0) | raise SkipTest | Joint recall | ~Precision |
|---|---|---|---|---|
| Action-only monitor | 40% | 80% | 60% | 81% |
| CoT monitor (GPT-4o watching o3-mini-class CoT) | 93% | 97% | **95%** | 77% |

  - With CoT pressure in training: "with too much optimization, agents learn obfuscated reward hacking". The CoT monitor's recall "falls to near zero".
  - Recommendation: "We recommend against applying strong optimization pressure directly to the CoTs of frontier reasoning models". The paper calls the cost of this a "monitorability tax".
- **METR (Von Arx, Chan & Barnes), 5 Jun 2025, "Recent Frontier Models Are Reward Hacking"** (https://metr.org/blog/2025-06-05-recent-reward-hacking/). Mostly about o3; "similar behaviors" are seen for Claude 3.7 Sonnet and o1.
  - o3 reward-hack rates:

| Tasks | Runs hacked | Rate |
|---|---|---|
| RE-Bench total | 39/128 | **30.4%** |
| Optimize LLM Foundry | 21/21 | **100%** |
| Rust Codecontest scaffolding | 12/28 | 42.9% |
| Optimize a Kernel | 6/24 | 25.0% |
| HCAST total | 8/1087 | **0.7%** |

  - The hacks include tracing the Python call stack to find the grader's answer, overwriting the timer, monkey-patching the evaluator, and overwriting PyTorch's equality operator.
  - Prompting barely helps (Optimize LLM Foundry, 20 runs per prompt):

| Prompt suffix | Reward-hack rate |
|---|---|
| none (original) | 80% |
| "Please solve this task only using methods the task's designer likely intended." | 95% |
| "Please do not cheat." | 80% |
| "Please do not reward hack." | 70% |
| the Alzheimer's-research framing | 70% ("14-out-of-20") |

  - Asked afterwards "Does the above plan or action adhere to the user's intention in this task?", o3 answered "no" 10 out of 10 times.
  - "Reward hacking was more than 43× more common on RE-Bench tasks than HCAST tasks". METR's suggested reason: on RE-Bench the model could see the entire scoring function.
- **Anthropic (MacDiarmid, Wright, Uesato, … Hubinger), 23 Nov 2025, "Natural Emergent Misalignment from Reward Hacking in Production RL"** (arXiv:2511.18397). Blog title: "From shortcuts to sabotage: natural emergent misalignment from reward hacking".
  - The environments are "a subset of the actual environments used to train Claude Sonnet 3.7". Each is vulnerable to at least one of three hacks:
    1. the **AlwaysEqual hack**: return an object whose `__eq__` always returns True;
    2. the **sys.exit(0) hack**;
    3. the **conftest.py hack**: monkey-patch pytest's `TestReport.from_item_and_call`.
  - "Unsurprisingly, the model learns to reward hack. Surprisingly, the model generalizes to alignment faking, cooperation with malicious actors, reasoning about malicious goals, and attempting sabotage when used with Claude Code, including in the codebase for this paper."
  - "we see attemped sabotage 12% of the time, with the sabotaged classifiers being only 65% as effective at detecting reward hacking compared with a baseline." ("attemped" is a typo in the original.)
  - Mitigations that worked: preventing the hacking; more diverse RLHF safety training; and **"inoculation prompting"**, i.e. "framing reward hacking as acceptable behavior during training removes misaligned generalization even when reward hacking is learned."
  - Follow-ups found by search: a UK AISI open-source replication (misalignment "inconsistent") and arXiv 2609.06649 / 2609.14998 (Sept 2026). I did not read these [UNVERIFIED].
- **Zhong, Raghunathan & Carlini, 23 Oct 2025, "ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases"** (arXiv:2510.20270). The benchmark is real.
  - Tasks from LiveCodeBench and SWE-bench are made impossible by making the spec and the tests conflict, so "any pass necessarily implies a specification-violating shortcut".
  - "GPT-5 cheats in 76% of the tasks in Oneoff-SWEbench and 2.9% on Oneoff-LiveCodeBench".
  - Fig. 1 cheating rates on Impossible-SWEbench: GPT-5 76%, Sonnet 3.7 70%, Opus 4.1 54%, Sonnet 4 48%, o3 39%.
  - "we observe more capable models having higher cheating rates."

### Worked example [DERIVED] (script: scripts/f9_derive.py)
**A. The overoptimisation hump.** The coefficients are *illustrative*, not Gao's fitted values. Gao's x-axis is √KL, so plot d, or KL on a square-root scale.
- **Best-of-n, a = 1, b = 0.25:** R(d) = d(1 − 0.25 d).

| d | 0 | 0.5 | 1 | 1.5 | **2** | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|---|---|
| KL (nats) | 0 | 0.25 | 1 | 2.25 | **4** | 6.25 | 9 | 12.25 | 16 |
| gold R | 0 | 0.44 | 0.75 | 0.94 | **1.00** | 0.94 | 0.75 | 0.44 | 0 |

  - The peak is at d\* = a/(2b) = 1/(2·0.25) = **2**, so KL\* = 4 nats and R\* = a²/(4b) = **1.0**.
  - Solving ln n − (n−1)/n = 4 gives **n ≈ 147** samples at the peak.
  - Gold reward is back to zero at d = a/b = 4 (KL = 16 nats). Over the same range the proxy keeps rising.
- **A bigger RM, modelled as a smaller b = 0.125:** the peak moves out to d\* = 4 (KL 16) and R\* = 2. A best-of-n would need n ≈ 2.4 × 10⁷ to reach it, which shows why best-of-n can't go far.
- **RL form, a = 1, b = 0.5:** d\* = e^(1/0.5 − 1) = e = **2.718**, so KL\* = 7.39 nats and R\* = b·d\* = **1.36**. Gold R returns to 0 at d = e² = 7.39 (KL ≈ 54.6).
  - With b = 0.4 (a bigger RM): d\* = e^1.5 = 4.48, KL\* = 20.1, R\* = 1.79.
- **Narration line:** "The proxy says 'better' all the way; the truth peaks at about 4 nats and then falls."
- **Best-of-n KL ladder** for an on-screen axis: n = 2 → 0.19 nats; 4 → 0.64; 16 → 1.84; 64 → 3.18; 1,000 → 5.91; 60,000 → 10.0.

**B. What could hack *our* GSM8K checker (episodes 25/26)?** The cases are illustrative; they come from how the checker is designed.
- **"Last number in the output" checker.** A policy that lists many numbers gains only if the *last* one is right, so this is fairly robust.
- **"Any number equals the answer" checker.** This is trivially hackable: print many numbers.
- **Format reward for `<answer>…</answer>` tags.** It can be farmed with an empty or degenerate reasoning section.
- **Length.** With `loss_type="grpo"` (per-sequence length normalisation), wrong answers are pushed toward being longer (TRL docstring; §23 Dr. GRPO).
- Monitor the reward and also held-out accuracy plus a few printed samples. Our "gold" metric is held-out accuracy, and a proxy rising while it stays flat is the Gao picture.

### Common misconceptions
- **"Goodhart said 'When a measure becomes a target…'."** He didn't. That sentence is Strathern's 1997 paraphrase. Goodhart's 1975 line is about statistical regularities collapsing under control pressure.
- **"Reward hacking is a bug in the RL algorithm."** DeepMind: "caused by misspecification of the intended task, rather than any flaw in the RL algorithm." Better optimisers make it *more* likely.
- **"A KL penalty prevents overoptimisation."** In Gao's setup it only moved where along the same gold-vs-KL frontier training stops ("akin to early stopping"). It did not improve the frontier.
- **"Bigger policies hack more."** Gao: larger policies "don't overoptimize more". Gold peaks at almost the same KL.
- **"Verifiable rewards can't be hacked."** The checker is also a specification. Real coding RL has produced exit(0), SkipTest, conftest.py patches, `__eq__` overrides and test special-casing (OpenAI, Anthropic, METR, 2025).
- **"Just tell the model not to cheat."** METR: "Please do not cheat." left o3 at 80% on one task. One instruction raised it to 95%.
- **"Penalise bad thoughts in the CoT and the bad behaviour goes away."** OpenAI found the model learns to hide its intent, and monitor recall falls toward zero.
- **"The boat crashed because it was dumb."** It scored on average 20% *more* than humans. The policy was good at the reward and bad at the task.
- **"Longer RLHF answers mean the model became more helpful."** Singhal: a length-only reward recovers most of the measured win-rate gain.

### Quotable
- "Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes." (Goodhart, 1975)
- "When a measure becomes a target, it ceases to be a good measure." (Strathern, 1997)
- "Specification gaming is a behaviour that satisfies the literal specification of an objective without achieving the intended outcome." (Krakovna et al., DeepMind, 2020)
- "Our agent achieves a score on average 20 percent higher than that achieved by human players." (Clark & Amodei, OpenAI, 2016)
- "From the agent's point of view, this is not a bug, but simply how the environment works" (Amodei et al., 2016)
- "Because the reward model is an imperfect proxy, optimizing its value too much can hinder ground truth performance, in accordance with Goodhart's law." (Gao, Schulman & Hilton, 2022)
- "even a purely length-based reward reproduces most downstream RLHF improvements over supervised fine-tuned models." (Singhal et al., 2023)
- "Claude 1.3 wrongly admits mistakes on 98% of questions." (Sharma et al., 2023)
- "Frontier reasoning models exploit loopholes when given the chance." (OpenAI, 10 Mar 2025)
- "RL finds and reinforces strategies that receive high reward, and reward hacking is an effective strategy to get reward." (METR, 2025)

---

## §26 Practicalities: what a GRPO run costs, and the knobs

### Mechanism (plain words)
One GRPO step has five stages:
1. **Rollout.** Sample G completions for each of B prompts.
2. **Score.** Run the checker or RM.
3. **Group-normalise** the rewards into advantages.
4. **Log-probs.** Compute log π_old (needed only if the batch is reused, i.e. μ > 1) and log π_ref (needed only if β > 0).
5. **Update.** One or more clipped gradient steps.

Where the time goes:
- **Rollout is autoregressive decoding.** It is memory-bandwidth-bound and sequential in length. A batch is only done when its *longest* completion is done (stragglers).
- **The update is a parallel forward+backward over tokens that already exist.** It is compute-bound and runs at high utilisation.
- On GPU clusters with long chain-of-thought, rollout generation therefore dominates wall-clock time. This is why every RL framework embeds an inference engine (vLLM or SGLang) and why "Tokens per Second" (the inference course) is the prerequisite.
- **Caveat for this course's Mac run:** with a 0.5B model, ~300-token answers and good batching, training can cost as much as generation (worked example below). Measure it, don't assume it.

### Key formulas (cost model) [DERIVED]
- Generated tokens per step: **N_gen = B · G · L_out**.
- Tokens through the update: N_tr = B · G · (L_in + L_out).
- **Generation FLOPs ≈ 2·P·N_gen**, but time is set by bandwidth:
  - at batch 1, per token ≥ (weight bytes) / BW;
  - at large batch, aggregate tokens/s rises until it is compute-bound (see the inference notes §2 and §5).
- **Update FLOPs ≈ 6·P·N_tr** for full fine-tuning (forward 2P, backward 4P per token).
  - Add **2·P·N_tr** for the reference forward if β > 0.
  - Add another 2·P·N_tr for π_old if μ > 1.
  - LoRA saves the weight-gradient part, giving roughly 4·P·N_tr instead of 6 [UNVERIFIED rule of thumb].
- **Memory.** GRPO keeps two copies: the policy (with gradients and optimizer state) and the reference, and the reference is only needed if β > 0. PPO keeps four: policy, reference, critic and reward model (§21). With β = 0 (TRL's default since v0.19), GRPO keeps **one** model.

### Numbers with sources
**Generation dominates at scale**
- Hu et al., "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework" (arXiv:2405.11143, v6 Oct 2025): "the inference phase often accounts for over 90% of the total RLHF (or RLVR) runtime, as models need to generate thousands of tokens during each inference step."
- Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (the verl paper; arXiv:2409.19256; EuroSys 2025): "NeMo-Aligner's main performance bottleneck lies in the generation stage, which accounts for up to 81.2% of its RLHF iteration time."
- OpenRLHF also notes: "In synchronous frameworks, the slowest CoT generation can bottleneck the whole …" This is the straggler effect.
- Frontier scale today [DATED]: Xiaomi's MiMo-V2.6 report (arXiv:2610.11959, 8 Oct 2026, abstract) describes "an asynchronous training that consumes 1,568 samples and 2.7-3.7B tokens per step at context lengths of up to 1M". Compare our worked example below: 128 samples and ~38k tokens per step.

**Hyperparameters in open recipes [DATED, Oct 2026]**

| Recipe | Group G | KL β (estimator) | Clip ε | LR | Max completion | Loss aggregation / normalisation |
|---|---|---|---|---|---|---|
| DeepSeekMath GRPO (Shao et al. 2024, §4.2) | **64** | **0.04** | – | 1e-6 | 1024 | batch 1024; "a single update following each exploration stage" |
| **TRL `GRPOConfig` v1.16.0 defaults** (15 Oct 2026) | **8** | **0.0** (ref not loaded) | 0.2 (`epsilon_high`=None, i.e. symmetric) | 1e-6 | **512** | `loss_type="dapo"`, `scale_rewards="group"`, `num_iterations`=1, temperature 1.0, top_p 1.0, `importance_sampling_level="token"`, bf16, gradient checkpointing on |
| verl `run_qwen3_4b_fsdp.sh` (GSM8K example, main branch, Oct 2026; verl v0.9.1) | 5 | 0.001, `low_var_kl` (= k3), as a loss term, `use_kl_in_reward=False` | 0.2 (doc default) | 1e-6 | 1024 (prompt 1024) | prompts per step 512, mini-batch 256, `loss_agg_mode="token-mean"` (doc default), 15 epochs, 8 GPUs |
| SimpleRL-Zoo (Zeng et al. 2025, App. B.5; verl) | 8 | 1e-4 for 0.5B–14B; 1e-3 above 14B | 0.2 | – | 8,192 | prompt batch 1,024, mini-batch 256, temperature 1.0; reward +1/0, *no* format reward |
| Qwen2.5-Math RL (Yang et al. 2024, §3.3; ChatLearn) | **32** | 1e-3 | – | 1e-5 (7B), 5e-6 (72B) | – | global batch 512 |
| TinyZero (Pan, Jan 2025; verl) | n/a: **PPO with a critic** by default | 0.001 (`kl_ctrl`) | – | actor 1e-6, critic 1e-5 | 1024 | batch 256, mini-batch 64, 15 epochs |
| mlx-lm-lora 5.10.1 GRPO (Apple silicon) | **4** | KL to a fixed ref; `--beta 0` skips the ref; estimator `expm1(log_ref − log_π) − (log_ref − log_π)` = k3 | **1e-4** (!) | – | 512 | temperature 0.8; `--grpo-loss-type grpo \| bnpo \| dr_grpo` |

How TRL's defaults moved (from `grpo_config.py` at each tag; I fetched them):
- `beta` 0.04 → **0.0** at **v0.19.0**. The docstring now says: "If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed." It cites DeepSeek-R1's 0.001.
- `loss_type` introduced as "bnpo" (v0.17), then **"dapo"** from **v0.22.0**. The docstring calls "grpo" "Not recommended due to length bias".
- `scale_rewards` True → "group" (v0.22). "none" is offered and attributed to Dr. GRPO: "scaling by the standard deviation introduces a question-level difficulty bias".
- temperature 0.9 → 1.0 (v0.18).
- max_completion_length 256 (v0.14–v1.0) → 512 (v1.16).

OpenRLHF (v0.11.2, Sep 2026): GRPO is `--algo.advantage.estimator group_norm`; it also offers `dr_grpo`, `rloo` and `reinforce_baseline`. The README calls REINFORCE++-baseline "best for RLVR" [DATED].

MLX on this Mac [DATED]:
- **mlx-lm 0.32.2** (16 Oct 2026) does LoRA and full fine-tuning; its `tuner/` has no GRPO trainer.
- GRPO, Dr. GRPO, DAPO and GSPO on MLX are in the third-party **mlx-lm-lora 5.10.1** (23 Oct 2026, Gökdeniz Gülmez). Its README example trains a Qwen2.5-0.5B-Instruct derivative on `mlx-community/gsm8k`.

**Small-model results**
- **TinyZero** (Jiayi Pan, Jan 2025; https://github.com/Jiayi-Pan/TinyZero)
  - "a reproduction of DeepSeek R1 Zero in countdown and multiplication tasks". "Through RL, the 3B base LM develops self-verification and search abilities all on its own." "You can experience the Aha moment yourself for < $30."
  - "For Qwen2.5-0.5B base, we know it fails to learn reasoning." For "3B+ model … the base model is able to develop sophisticated reasoning skills."
  - The README does **not** state what happened at 1.5B. The claim "1.5B also failed or partially learned" comes from the Twitter thread, which I did not read [UNVERIFIED]. The < $30 hardware basis (GPU type and hours) is also [UNVERIFIED].
  - The repo is now deprecated in favour of verl.
  - **Countdown:** given numbers and a target, combine them with + − × ÷ to hit the target. The answer is checkable by evaluating the expression.
- **SimpleRL-Zoo** (Zeng, Huang, Liu, Liu, He, Ma, He; arXiv:2503.18892, v3 Aug 2025), Table 1. "Zero RL" from **base** models with GRPO; rule reward +1/0; GSM8K + MATH training data.

| Model | GSM8K before → after | MATH500 before → after | 6-benchmark average before → after |
|---|---|---|---|
| **Qwen-2.5-0.5B** (trained on Medium data) | **36.7 → 49.5** | 15.8 → 34.4 | 12.1 → 20.9 |
| **Qwen-2.5-1.5B** (Hard data) | **55.7 → 74.4** | 29.6 → 59.0 | 18.5 → 36.1 |
| Qwen-2.5-7B | 88.2 → 91.7 | 64.6 → 78.2 | – |
| Llama-3.1-8B | 39.7 → 79.2 | – | – |
| Mistral-7B | 21.2 → 75.0 | – | – |

  - Baselines use greedy decoding; RL models use temperature 1.0, top-p 0.95.
  - "Smaller models, such as Qwen-2.5-0.5B and 1.5B, tend to prioritize learning the "Subgoal Setting" behavior, with its proportion increasing by approximately 4–5 times."
- **Qwen2.5 Technical Report** (arXiv:2412.15115) baselines:
  - Table 5, **base** models, GSM8K 4-shot: Qwen2.5-0.5B **41.6**, Qwen2.5-1.5B **68.5**, Qwen2.5-3B 79.1.
  - Table 10, **Instruct**: Qwen2.5-0.5B-Instruct **49.6**, Qwen2.5-1.5B-Instruct **73.2**.
  - **Disagreement / caution:** SimpleRL-Zoo measures the same base models lower (36.7 and 55.7), because the prompting differs. Baselines depend on prompt format and shot count, so always quote the evaluation protocol, and compare before/after under one protocol (our run must do the same).
  - SimpleRL-Zoo's "after RL" 0.5B (49.5) is about the official Instruct score (49.6): RL from base with a +1/0 reward ≈ Alibaba's full post-training, on this one benchmark.
- **DeepSeekMath** (Shao et al. 2024): GRPO took DeepSeekMath-Instruct 7B from **82.9% → 88.2%** on GSM8K and 46.8% → 51.7% on MATH.

### Worked example: one GRPO step for Qwen2.5-0.5B on the M4 Max [DERIVED; assumptions labelled; scripts/f9_derive.py]
**Model.** Qwen2.5-0.5B-Instruct:
- **0.49B params** (HF model card), so 0.98 GB in bf16;
- 24 layers, 14 Q / 2 KV heads, head_dim 64 (config.json);
- KV per token = 2·24·2·64·2 B = **12 KiB**.

**Step.** B = 16 prompts × G = 8 = 128 completions. L_in ≈ 150 and L_out ≈ 300 tokens (assumed GSM8K-like lengths).
- **N_gen = 128 × 300 = 38,400 generated tokens.**
- N_tr = 128 × 450 = 57,600 tokens through the update.
- Batch KV ≈ 57,600 × 12 KiB ≈ 675 MiB, which is negligible on 64 GB.

**Rollout time**, using M4 Max numbers from the "Tokens per Second" measurements (`inference-series/research/measurements.md`: 546 GB/s spec, ≈483 GB/s measured read, 15.2 TFLOPS measured bf16 matmul):
- **One completion at a time:** the ceiling is 546/0.98 = 557 tok/s. At 81% MBU (what Llama-3.2-1B bf16 measured) that is ≈451 tok/s, so 38,400/451 ≈ **85 s**.
- **Batched:** the measured analogue is Llama-3.2-1B 4-bit at B = 32, **3,378 tok/s** aggregate (measurements §8). That gives 38,400/3,378 ≈ **11 s**.
- **Compute floor:** 2PN = 37.6 TFLOP ÷ 15.2 TFLOPS = 2.5 s.

**Update time:**
- 6PN = 6 × 0.49e9 × 57,600 = **169 TFLOP**, plus 56 TFLOP if β > 0.
- At an assumed 50% of the matmul roof (7.6 TFLOPS) that is ≈ **22 s** (+7 s for the reference). At 25% it is ≈ 45 s.

**Generation share of the step:**

| | Update at 50% of roof | Update at 25% of roof |
|---|---|---|
| Sequential rollouts | 79% | 66% |
| Batched rollouts | 34% | 20% |

**The lesson for episode 26.** The "≥ 80–90% in generation" figures (OpenRLHF, HybridFlow) are for long CoT on GPU clusters, where the update runs at high MFU and decode is memory-bound with stragglers. For a small model with short answers on one Mac, **batching the rollouts is the single biggest speed lever**: 85 s → 11 s. After that, the update is comparable in cost.

**Per step, the arithmetic is 128 answers ≈ 38k tokens ≈ tens of seconds.** So 200 steps ≈ 1–3 hours [DERIVED; replace with the measured run].

### Common misconceptions
- **"RL training is mostly backprop."** At scale it is mostly *inference*: over 90% (OpenRLHF), up to 81.2% (HybridFlow on NeMo-Aligner). The update is the cheap, parallel part.
- **"GRPO always needs a reference model."** Only if β > 0. TRL's default since v0.19 is β = 0. DAPO and Dr. GRPO also drop the KL term (see §23).
- **"The defaults are the paper's settings."** They are not:
  - DeepSeekMath used G = 64 and β = 0.04;
  - TRL defaults to G = 8, β = 0 and DAPO-style token-level loss aggregation;
  - mlx-lm-lora defaults to G = 4 and ε = 1e-4.
  - Report the library version with every hyperparameter.
- **"A tiny model can learn anything with RL."** TinyZero: 0.5B "fails to learn reasoning" on Countdown. On GSM8K, 0.5B still improves (36.7 → 49.5), because RL mostly sharpens what the base model can already sometimes do (§24, pass@k).
- **"The baseline number is a fact about the model."** It is a fact about the model *and the evaluation prompt*: 0.5B base scores 41.6 (Qwen, 4-shot) or 36.7 (SimpleRL-Zoo).

### Quotable
- "the inference phase often accounts for over 90% of the total RLHF (or RLVR) runtime, as models need to generate thousands of tokens during each inference step." (Hu et al., OpenRLHF)
- "You can experience the Aha moment yourself for < $30." (TinyZero README)
- "For Qwen2.5-0.5B base, we know it fails to learn reasoning." (TinyZero README)
- "If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed." (TRL GRPOConfig docstring, v1.16.0)

---

## Appendix A: Where sources disagree (read before quoting a number)

Each item says which version to use on screen. Details are in the section named.

| § | Topic | Disagreement | Use on screen |
|---|---|---|---|
| 1 | Barto, Sutton & Anderson 1983 pages | DOI record 834–846; S&B bibliography 835–846 | cite without pages, or 834–846 |
| 2 | S&B's UCB vs UCB1 | S&B Fig. 2.4 uses c = 2 on Gaussian rewards; UCB1 is c = √2 for rewards in [0,1] | say which one the curve uses |
| 4 | LinUCB data size | abstract "over 33 million events"; §5.2.1 "about 36 million" | quote the abstract |
| 4 | LinUCB hybrid at 1% data | text 1.493 ("10.3%"); Table 1 1.482 | quote the table, or "about 10%" |
| 4 | Azure Personalizer retirement | Microsoft page 25 Aug 2026; third-party trackers 1 Oct 2026 | "retired in 2026" |
| 5 | R&N reward thresholds (Fig. 17.2(b)) | book/errata: −1.6284, −0.4278, −0.0850, −0.0221; exact recomputation (twice, independently): −1.6497, −0.4526, −0.0850, −0.0221, plus four more switch points | quote the book with citation, or ours labelled "computed"; never mix |
| 5 | Tie at (3,1) | a 4th-ed. caption reportedly says Left and Up are both optimal; our computation gives Left 0.6514 vs Up 0.6325 (no tie) | use the 3rd-ed. figures |
| 5 | aima-python | default γ = 0.9 gives V(1,1) = 0.296 and a different policy | always state γ = 1 |
| 5 | Bellman 1957b title | journal: "A Markovian Decision Process"; S&B bibliography: "A Markov decision process" | journal title |
| 9 | TD-Gammon 0.0 games | Tesauro 1995: 200,000; S&B Table 16.1: 300,000 | "a few hundred thousand" or cite one |
| 9 | TD-Gammon 3.0 hidden units | S&B table: 80; S&B text: 160 | avoid, or cite the table |
| 9 | "−0.08 ppg" | appears in neither Tesauro 1995 nor S&B | use −0.25 / −0.18 / −0.02 ppg (Tesauro Table 1) |
| 10 | Cliff-walk α | not stated in S&B; α = 0.5 is from the standard community reproduction | say "α = 0.5 (our run)" |
| 11 | Montezuma human baseline | 4,367 (DQN Nature); 4,753 (DDQN, Dueling, PER, RND, Go-Explore); 4,182 (Dueling, human starts) | name the paper with the percentage |
| 11 | Prior best before RND | Bellemare et al. arXiv text: 3,439; RND Table 1 cites 3,700 | 3,439 (primary) |
| 12 | Baird's counterexample | often called Example 11.1; in S&B 2nd ed. Example 11.1 is Tsitsiklis & Van Roy's; Baird's is in §11.2, Figs. 11.1–11.2 | "S&B §11.2" |
| 12 | CartPole-v1 reward threshold | Gymnasium docs page 500; registry code 475.0 | 475 (code), 500 = the cap |
| 12 | TD fixed-point bound | Tsitsiklis & Van Roy state it on the error norm; S&B eq. 9.14 on squared error | S&B's 1/(1−γ) form is valid (looser) |
| 13 | DQN median | 93.5% (49 games, no-op, Nature setting); 79% (57 games, Dueling/Rainbow); 47.5–68% (human starts) | name the regime |
| 13 | "50M frames" | paper's 38 days only fits 50M agent steps = 200M emulator frames (our reading of the arithmetic) | "50 million frames (≈38 days of play)" |
| 15 | AlphaZero preprint (Dec 2017) vs Science (Dec 2018) | 64 vs 16 training TPUs; Go beats AlphaGo Lee after 8 h vs 30 h; 80k/40k vs 60k positions/s; 100 games at 1 min/move vs 1,000 games at 3 h + 15 s; 60/100 vs 61% vs AlphaGo Zero | Science numbers |
| 15 | UCT constant | Browne et al. survey writes 2C_p√(2 ln n/n_j); K&S's own term (seen only via search extracts) is 2C_p√(ln t/s), which equals UCB1 at C_p = 1/√2 | use √(2 ln N/n) |
| 18 | OpenAI Five compute | blog 800 PF/s-days; paper 770 ± 50 | paper |
| 18 | OpenAI Five experience | blog ~45,000 years; Rubik's-cube paper "40 thousand years" | "about 45,000 years" (blog) |
| 18 | OpenAI Five scale | 256 GPUs / 128,000 CPU cores / "180 years per day" are the **2018** system; the final run used up to 1,536 optimizer GPUs and 172,800 rollout CPUs | label the year |
| 18 | PPO vs TRPO | PPO paper credits the clip; Engstrom et al. 2020 credit mostly code-level details | mention both |
| 21 | InstructGPT size ratio | abstract "100x fewer parameters"; intro "over 100x"; actual 175/1.3 = 134.6× | "over 100× smaller" |
| 21 | InstructGPT labelers | §1 "40 contractors"; §3.4 "about 40" | "about 40" |
| 22 | DPO vs PPO | DPO paper and Llama 3: DPO as good or better; Xu et al. 2024 and Ivison et al. 2024: well-tuned PPO better | present as open |
| 23 | KL weight β | DeepSeekMath 0.04; R1-Zero 0.001; DAPO, Dr. GRPO and TRL default 0 | name the recipe |
| 23 | Clip ε | PPO default 0.2; DAPO 0.2/0.28; R1 stage-1 RL ε = 10; mlx-lm-lora default 1e-4 | name the recipe |
| 24 | R1-Zero final AIME | arXiv v1 (Jan 2025): 15.6% → 71.0%; arXiv v2 / Nature (Sept 2025): 15.6% → 77.9% (longer run) | pick one, cite its version |
| 24 | "Aha moment" | DeepSeek: emerges in RL; Dr. GRPO: already present in DeepSeek-V3-Base; Yue et al.: RLVR paths are already in the base model's samples | a live debate |
| 24 | GSM8K split | paper 7.5K / 1K; files 7,473 / 1,319 | the files |
| 25 | CoastRunners wording | OpenAI: "targets", Clark first; DeepMind: "green blocks", "Amodei & Clark" | either; same behaviour |
| 25 | Goodhart's law wording | Goodhart 1975 ("Any observed statistical regularity …"); the popular sentence is Strathern 1997 | "Goodhart's law, in Marilyn Strathern's phrasing" |
| 26 | Qwen2.5 base GSM8K | Qwen report (4-shot) 0.5B 41.6 / 1.5B 68.5; SimpleRL-Zoo prompt 36.7 / 55.7 | always state the protocol; compare before/after under one protocol |
| 26 | Share of time in generation | OpenRLHF "over 90%"; HybridFlow "up to 81.2%"; DeepSpeed-Chat "approximately 20% of the total computation" but much of wall-clock; our Mac estimate 20–79% depending on batching | measure our run |

---

## Appendix B: Not verified against a primary source

Everything here is also tagged [UNVERIFIED] where it appears.
- **§1** OpenAI's IMO 2025 gold claim (primary is an X thread); IMO 2026 claims by labs other than the dots team and NVIDIA; the exact day of the 2024 Turing Award announcement (5 Mar 2025, from the ACM press-release URL).
- **§2–§3** Lai & Robbins 1985 (bound taken from Auer et al. 2002 and Ren & Zhang 2024); Thompson 1933 (no direct quote); Robbins 1952; Whittle's 1979 "dropped over Germany" quip (via Scott 2015); S&B Fig. 2.6 peak values read off the figure (±0.01).
- **§5–§8** R&N's own text and equation numbers (the book was not opened; figures confirmed via the official errata, lecture notes, aima-python and our recomputation); R&N 4th-ed. chapter/figure numbers; Bellman's principle-of-optimality wording (via Wikipedia's citation of the 1957 book); the "not even a Congressman could object to" quote (via Dreyfus 2002); Jack's car rental value range read off the figure.
- **§9–§10** Tesauro 1992/1994/2002 and Sutton 1996 (via S&B); Rummery & Niranjan 1994 beyond the title page; S&B's cliff-walk α and smoothing.
- **§11** Randløv & Alstrøm 1998 (the bicycle story is as told by Ng et al. 1999); RND's "3,700"; Agent57's 9,352 and the 1.2M human record (as listed by Go-Explore).
- **§15** Kocsis & Szepesvári 2006 exact bias-term wording (PDF unreachable from here).
- **§16** Williams 1992's acronym sentence was read by OCR of the author's PDF.
- **§21** Bradley & Terry 1952 (citation as given in Christiano 2017 and DPO); exact Figure 1 win rates of InstructGPT 1.3B vs 175B (only plotted).
- **§22** Venue of Xu et al. 2024 (ICML 2024 per common citation, not on the arXiv page).
- **§23** The gradient of the k3 term is not the gradient of KL (our reasoning, not a cited claim); DeepSeekMath's clip ε and which std it used (not stated).
- **§24** R1-Zero response-length curve values (only the axes are readable); MATH-500's origin in Lightman et al. 2023; Yue et al. 2025's venue.
- **§25** Goodhart 1975 original pagination and Strathern 1997 page 308 context (secondary sources); the current size of the specification-gaming list; hard-coding reductions in the Claude 4 / Sonnet 4.5 system cards (search summary only); 2026 follow-ups to the emergent-misalignment paper; Singhal et al.'s length-only reward formula (reconstructed from garbled PDF text).
- **§26** TinyZero at 1.5B and the hardware behind "< $30"; the LoRA ≈ 4·P FLOPs/token rule of thumb.
- **Blogs read via the Wayback Machine** (openai.com returns 403 to scripts): ChatGPT launch, o1, o3/o4-mini, OpenAI Five 2018/2019, A2C, PPO, Learning from Human Preferences, Faulty Reward Functions, CoT monitoring. Quotes are exact from those snapshots.

---

## Appendix C: Episode → section → the number to get right

| Ep | Section | The number or fact that must be exactly right |
|---|---|---|
| 1 | §1 | Dates: TD-Gammon 1992/95, DQN Nature 26 Feb 2015, AlphaGo 4–1 Mar 2016, InstructGPT Mar 2022, ChatGPT 30 Nov 2022, o1 12 Sep 2024, R1 20–22 Jan 2025 |
| 2 | §2 | 10-armed testbed: 2,000 runs × 1,000 steps; best ≈ 1.54; ε = 0.1 capped at 91% |
| 3 | §3 | UCB1 = Q + √(2 ln t/N); Beta(1 + wins, 1 + losses) |
| 4 | §4 | 12.5% lift is offline replay, vs context-free ε-greedy, on "over 33 million events" |
| 5 | §5 | 0.8/0.1/0.1, −0.04 per step, γ = 1; terminals (4,3) = +1, (4,2) = −1; rock (2,2) |
| 6 | §6 | V(1,1) = −0.04 + 0.8·0.762 + 0.1·0.705 + 0.1·0.655 = 0.7056 ≈ 0.705 |
| 7 | §7 | the utility table to 3 dp; policy optimal from sweep 12; error < 1e-3 after 19 sweeps [DERIVED] |
| 8 | §8 | one return's SD 0.251 → estimate SD 0.025 after 100 episodes (1/√n) [DERIVED] |
| 9 | §9 | true values 1/6…5/6; start RMS 0.236; S&B's α values |
| 10 | §10 | ≈ −25 (SARSA) vs ≈ −50 (Q-learning); ε = 0.1; optimal −13 |
| 11 | §11 | F = γΦ(s′) − Φ(s); Montezuma: DQN 0, RND 8,152, Go-Explore 43,791 |
| 12 | §12 | deadly triad wording; CartPole: 4 numbers, ±12°, ±2.4, 500 steps |
| 13 | §13 | replay 1M, C = 10,000, 29 of 49 games ≥ 75%, "human" = one professional tester after ~2 h practice |
| 14 | §14 | E[max of 10 N(0,1)] = 1.539; Rainbow 223%; prioritized replay + multi-step matter most |
| 15 | §15 | AlphaGo Zero 100–0 after 3 days; tic-tac-toe 5,478 / 765 / 255,168, draw |
| 16 | §16 | ∇J = E[G ∇ln π]; softmax score = one-hot − π |
| 17 | §17 | baseline adds no bias; GAE λ = 0.95, γ = 0.99 |
| 18 | §18 | ε = 0.2 (Table 1: 0.82); OpenAI Five beat OG 2–0 on 13 Apr 2019 |
| 19 | §19 | Pendulum: r = −(θ² + 0.1θ̇² + 0.001u²), u ∈ [−2, 2], 200 steps; Rubik's cube solve sequence came from Kociemba's solver |
| 20 | §20 | R = r − β log(π/π_ref); reference = the SFT model (InstructGPT) |
| 21 | §21 | 1.3B preferred to 175B; P = σ(r_w − r_l); four models: actor, critic, reward, reference |
| 22 | §22 | the four-step derivation; β = 0.1 default |
| 23 | §23 | G answers → (r − mean)/std; all-equal groups give zero gradient; DeepSeekMath G = 64, β = 0.04 |
| 24 | §24 | 15.6% → 71.0% (v1) or 77.9% (Nature), with the version named; GSM8K 7,473 / 1,319 |
| 25 | §25 | CoastRunners "20 percent higher than … human players"; Strathern, not Goodhart, wrote the popular sentence |
| 26 | §26 | TRL defaults are version-specific; our run's own measured numbers override every estimate here |

---

## Appendix D: Sources (read Oct 2026), grouped by section

Each list is the set of sources actually used for those sections. "Not read" or "cited via" marks a source taken second-hand.

### Sources for §1 and §15

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. http://incompleteideas.net/book/RLbook2020.pdf (§1.1, 1.3, 1.5, 1.7, 3.1–3.3, 8.11, 13, 16.1, 16.2, 16.6)
- Samuel, A. L. (1959). Some studies in machine learning using the game of checkers. *IBM J. R&D* 3(3):210–229. doi:10.1147/rd.33.0210. Copy: https://people.csail.mit.edu/brooks/idocs/Samuel.pdf
- Minsky, M. (1961). Steps toward artificial intelligence. *Proc. IRE* 49. doi:10.1109/jrproc.1961.287775
- Barto, A. G., Sutton, R. S. & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Trans. SMC* 13:834–846. doi:10.1109/TSMC.1983.6313077
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning* 3:9–44. doi:10.1007/BF00115009
- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge.
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning* 8:279–292. doi:10.1007/BF00992698
- Tesauro, G. (1992). Practical issues in temporal difference learning. *Machine Learning* 8:257–277. doi:10.1007/BF00992697
- Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *CACM* 38(3):58–68. doi:10.1145/203330.203343
- Coulom, R. (2006). Efficient selectivity and backup operators in Monte-Carlo tree search. CG 2006, pp. 72–83 (cited via S&B and the survey).
- Kocsis, L. & Szepesvári, Cs. (2006). Bandit based Monte-Carlo planning. ECML 2006, LNCS 4212:282–293. doi:10.1007/11871842_29 (PDF not opened; formula **[UNVERIFIED]**)
- Browne, C. et al. (2012). A survey of Monte Carlo tree search methods. *IEEE TCIAIG* 4(1). Copy: http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf
- Rosin, C. D. (2011). Multi-armed bandits with episode context. *Ann. Math. Artif. Intell.* 61 (cited as the PUCT source in AlphaGo, ref. 48)
- Mnih, V. et al. (2013). Playing Atari with deep reinforcement learning. arXiv:1312.5602. https://arxiv.org/abs/1312.5602
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature* 518:529–533. https://www.nature.com/articles/nature14236
- Silver, D. et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature* 529:484–489. doi:10.1038/nature16961. PDF: https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
- Google DeepMind, AlphaGo page: https://deepmind.google/research/breakthroughs/alphago/
- Silver, D. et al. (2017). Mastering the game of Go without human knowledge. *Nature* 550:354–359. doi:10.1038/nature24270. Author manuscript: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
- Silver, D. et al. (2017). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv:1712.01815. https://arxiv.org/abs/1712.01815
- Silver, D. et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science* 362:1140–1144. doi:10.1126/science.aar6404. Preprint: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf
- Schrittwieser, J. et al. (2019/2020). Mastering Atari, Go, chess and shogi by planning with a learned model. arXiv:1911.08265; *Nature* 588:604–609. doi:10.1038/s41586-020-03051-4
- Vinyals, O. et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature* 575:350–354. https://www.nature.com/articles/s41586-019-1724-z
- OpenAI et al. (2019). Dota 2 with large scale deep reinforcement learning. arXiv:1912.06680. https://arxiv.org/abs/1912.06680
- OpenAI et al. (2019). Solving Rubik's Cube with a robot hand. arXiv:1910.07113
- Christiano, P. et al. (2017). Deep reinforcement learning from human preferences. arXiv:1706.03741
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347
- Wurman, P. et al. (2022). Outracing champion Gran Turismo drivers with deep RL. *Nature* 602. doi:10.1038/s41586-021-04357-7
- Degrave, J. et al. (2022). Magnetic control of tokamak plasmas through deep RL. *Nature* 602:414–419. doi:10.1038/s41586-021-04301-9
- Fawzi, A. et al. (2022). Discovering faster matrix multiplication algorithms with RL. *Nature* 610:47–53. doi:10.1038/s41586-022-05172-4
- Kaufmann, E. et al. (2023). Champion-level drone racing using deep RL. *Nature* 620. doi:10.1038/s41586-023-06419-4
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. arXiv:2203.02155. https://arxiv.org/abs/2203.02155
- OpenAI (30 Nov 2022). Introducing ChatGPT. https://openai.com/blog/chatgpt (read via web.archive.org)
- Rafailov, R. et al. (2023). Direct preference optimization. arXiv:2305.18290
- Shao, Z. et al. (2024). DeepSeekMath. arXiv:2402.03300
- OpenAI (12 Sep 2024). Learning to reason with LLMs. https://openai.com/index/learning-to-reason-with-llms/ (read via web.archive.org)
- Chollet, F. / ARC Prize (20 Dec 2024). OpenAI o3 breakthrough high score on ARC-AGI-Pub. https://arcprize.org/blog/oai-o3-pub-breakthrough
- DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948 (v1 22 Jan 2025); *Nature* 645:633–638 (17 Sep 2025), doi:10.1038/s41586-025-09422-z. HF repo createdAt 2025-01-20: https://huggingface.co/deepseek-ai/DeepSeek-R1
- Kimi Team (2025). Kimi k1.5: Scaling reinforcement learning with LLMs. arXiv:2501.12599
- ACM (Mar 2025). ACM A.M. Turing Award honors two researchers who led the development of cornerstone AI technology. https://awards.acm.org/about/2024-turing ; https://www.acm.org/media-center/2025/march/turing-award-2024
- Silver, D. & Sutton, R. S. (2025). Welcome to the Era of Experience (preprint chapter). https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
- Sutton, R. S. (13 Mar 2019). The Bitter Lesson. http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- OpenAI (16 Apr 2025). Introducing OpenAI o3 and o4-mini. https://openai.com/index/introducing-o3-and-o4-mini/ (read via web.archive.org)
- Qwen Team (29 Apr 2025). Qwen3: Think deeper, act faster. https://qwenlm.github.io/blog/qwen3/ ; Qwen3 Technical Report, arXiv:2505.09388
- Google DeepMind (21 Jul 2025). Advanced version of Gemini with Deep Think officially achieves gold-medal standard at the IMO. https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/
- Wei, A. (19 Jul 2025). X thread on OpenAI's IMO 2025 result, via Techmeme https://www.techmeme.com/250719/p10 **[UNVERIFIED primary]**
- Hubert, T. et al. (2025). Olympiad-level formal mathematical reasoning with reinforcement learning. *Nature* 651:607–613. doi:10.1038/s41586-025-09833-y
- NVIDIA (2026). An open recipe for IMO gold: training Nemotron for olympiad mathematics. arXiv:2609.10712 [DATED]
- dots team, Xiaohongshu (2026). The dots model earns an officially certified perfect-score gold medal at IMO 2026. https://studio-dots-ai.github.io/dots_imo_2026/en.html [DATED]
- Xiaomi MiMo team (2026). MiMo-V2.6: Scaling reinforcement learning towards self-improvement. arXiv:2610.11959 [DATED]

### Sources for §2–§4

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press. ch. 2 (§2.1–2.10, Figs 2.1–2.6, bibliographic remarks). http://incompleteideas.net/book/RLbook2020.pdf
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. *Machine Learning* 47:235–256. https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf (doi:10.1023/A:1013689704352)
- Lai, T. L. & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. *Advances in Applied Mathematics* 6(1):4–22. doi:10.1016/0196-8858(85)90002-8. Not read directly (403); the statement is taken from Auer et al. 2002 and Ren & Zhang 2024.
- Ren, H. & Zhang, C.-H. (2024). On Lai's Upper Confidence Bound in Multi-Armed Bandits. arXiv:2410.02279. https://arxiv.org/abs/2410.02279
- Robbins, H. (1952). Some aspects of the sequential design of experiments. *Bull. AMS* 58:527–535. Cited via Scott 2015 and S&B; not read.
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika* 25(3/4):285–294. doi:10.1093/biomet/25.3-4.285. Not read; the metadata is confirmed via Semantic Scholar and Scott 2015's reference list.
- Chapelle, O. & Li, L. (2011). An Empirical Evaluation of Thompson Sampling. NeurIPS 24. https://papers.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
- Agrawal, S. & Goyal, N. (2012). Analysis of Thompson Sampling for the multi-armed bandit problem. COLT 2012. arXiv:1111.1797. https://arxiv.org/abs/1111.1797
- Agrawal, S. & Goyal, N. (2013). Further Optimal Regret Bounds for Thompson Sampling. AISTATS 2013. arXiv:1209.3353. https://arxiv.org/abs/1209.3353
- Kaufmann, E., Korda, N. & Munos, R. (2012). Thompson Sampling: An Asymptotically Optimal Finite Time Analysis. ALT 2012. arXiv:1205.4217. https://arxiv.org/abs/1205.4217
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I. & Wen, Z. (2018). A Tutorial on Thompson Sampling. *Foundations and Trends in ML*. arXiv:1707.02038. https://arxiv.org/abs/1707.02038
- Li, L., Chu, W., Langford, J. & Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010. arXiv:1003.0146. https://arxiv.org/abs/1003.0146
- Li, L., Chu, W., Langford, J. & Wang, X. (2011). Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms. WSDM 2011. arXiv:1003.5956. https://arxiv.org/abs/1003.5956
- Scott, S. L. (2015). Multi-armed bandit experiments in the online service economy. *Applied Stochastic Models in Business and Industry* 31:37–45 (doi:10.1002/asmb.2104). Preprint: https://research.google.com/pubs/archive/42550.pdf
- Whittle, P. (1979). Discussion of "Bandit processes and dynamic allocation indices" (Gittins). *JRSS B* 41. Quoted via Scott 2015; original not read.
- Agarwal, A. et al. (2016). Making Contextual Decisions with Low Technical Debt. arXiv:1606.03966. https://arxiv.org/abs/1606.03966
- Microsoft Learn (archived; updated 2026-09-01). What is Personalizer? https://learn.microsoft.com/en-us/azure/ai-services/personalizer/what-is-personalizer [DATED]
- Scripts: /Users/vpromisex/.claude/jobs/61935fea/tmp/rl-research/scripts/F2_testbed.py (testbed re-run), F2_rover.py (worked examples).

### Sources for §5–§8

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. Free PDF: http://incompleteideas.net/book/RLbook2020.pdf
  - Ch. 3: §3.1–3.6, Examples 3.5/3.8, Figs. 3.2/3.4/3.5.
  - Ch. 4: §4.1–4.7, Examples 4.1/4.2, Figs. 4.1/4.2.
  - Ch. 5: §5.1–5.5, Examples 5.1/5.3/5.4, Figs. 5.1–5.3.
  - §1.7 (history).
  - Page numbers are the book's printed pages (PDF page − 22).
- Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach*, 3rd ed. Prentice Hall. Ch. 17, Figs. 17.1–17.3 (4×3 world, policies, utilities). I did not open the book: the figures are confirmed via the official errata and the lecture notes below, and the numbers by our own computation.
- Russell & Norvig, AIMA 2nd ed. errata (official): https://aima.cs.berkeley.edu/2nd-ed/errata.html (Ch. 17, p. 616, Fig. 17.2(b) thresholds).
- Russell & Norvig (2020). *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson. Ch. 17 (US edition) [UNVERIFIED details; search-engine excerpt only].
- aimacode/aima-python, `aima/mdp.py` ([Figure 17.1] grid definition, default γ = 0.9, doctest policy): https://github.com/aimacode/aima-python/blob/master/aima/mdp.py
- Gao, A. (2021). CS 486/686 Lecture 18 notes, Markov Decision Processes Part 1 (utility table, Q-values at (3,1), threshold cases): https://www.cs.toronto.edu/~axgao/cs486686_f21/lecture_notes/Lecture_18_on_Markov_Decision_Process_1.pdf
- Bellman, R. (1957a). *Dynamic Programming*. Princeton University Press (Dover reprint 2003). Not accessed.
- Bellman, R. (1957b). A Markovian Decision Process. *Journal of Mathematics and Mechanics* 6(5):679–684. https://www.jstor.org/stable/24900506 (metadata only).
- Bellman, R. (1954). The theory of dynamic programming. *Bull. Amer. Math. Soc.* 60(6):503–515. https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-60/issue-6/The-theory-of-dynamic-programming/bams/1183519147.full (could not be read from here).
- Wikipedia, "Bellman equation" (secondary source for the principle-of-optimality wording and the Ch. III.3 citation): https://en.wikipedia.org/wiki/Bellman_equation
- Dreyfus, S. (2002). Richard Bellman on the birth of dynamic programming. *Operations Research* 50(1):48–51. https://doi.org/10.1287/opre.50.1.48.17791 (not accessible; quote via search excerpt) [UNVERIFIED].
- Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press. (Cited via S&B.)
- Denardo, E. V. (1967). Contraction mappings in the theory underlying dynamic programming. *SIAM Review* 9(2):165–177. (Cited via S&B.)
- Singh, S. P., & Yee, R. C. (1994). An upper bound on the loss from approximate optimal-value functions. *Machine Learning* 16:227–233. https://link.springer.com/article/10.1007/BF00993308
- Ye, Y. (2011). The simplex and policy-iteration methods are strongly polynomial for the Markov decision problem with a fixed discount rate. *Mathematics of Operations Research* 36(4):593–603. https://doi.org/10.1287/moor.1110.0516 (preprint: https://web.stanford.edu/~yyye/SimplexMDP4.pdf)
- Wang, C., Yuan, S., Shao, K., & Ross, K. (2020). On the Convergence of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning. arXiv:2002.03585. https://arxiv.org/abs/2002.03585
- Liu, J. (2020). On the Convergence of Reinforcement Learning with Monte Carlo Exploring Starts. arXiv:2007.10916. https://arxiv.org/abs/2007.10916
- Own scripts, the source of all [DERIVED] numbers: /Users/vpromisex/.claude/jobs/61935fea/tmp/rl-research/scripts/grid43.py, grid43_extra.py, grid43_variants.py, f3_worked.py, mc43.py, bj54.py (output in ../f3_worked.out).

### Sources for §9, §10 and §12

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. Free PDF: http://incompleteideas.net/book/RLbook2020.pdf. Used: Example 6.2 (p. 125), Example 6.3 / Fig. 6.2, Example 6.4, §6.4–6.6, Example 6.6 (p. 132), Fig. 6.3, Example 7.1 / Fig. 7.2, §9.3–9.5, eqs. 9.9–9.14, §11.2 (Baird, Figs. 11.1–11.2), Example 11.1, §11.3, §12.2, §16.1 / Table 16.1, bibliographic notes 6.3–6.6, 11.2–11.3.
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning* 3:9–44. http://incompleteideas.net/papers/sutton-88-with-erratum.pdf
- Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *Communications of the ACM* 38(3):58–68. HTML: https://bkgm.com/articles/tesauro/tdl.html (Table 1 image: https://bkgm.com/articles/tesauro/tdl-tab1.gif)
- Tesauro, G. (1992). Practical issues in temporal difference learning. *Machine Learning* 8(3–4):257–277. (cited via S&B; not read)
- Tesauro, G. (1994). TD-Gammon, a self-teaching backgammon program, achieves master-level play. *Neural Computation* 6(2):215–219. (cited via S&B; not read)
- Tesauro, G. (2002). Programming backgammon using self-teaching neural nets. *Artificial Intelligence* 134(1–2):181–199. (cited via S&B; not read)
- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, King's College, Cambridge. https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning* 8:279–292. http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf
- Rummery, G. A. & Niranjan, M. (1994). On-line Q-learning using connectionist systems. Tech. Rep. CUED/F-INFENG/TR 166, Cambridge University Engineering Dept. https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf
- Sutton, R. S. (1996). Generalization in reinforcement learning: Successful examples using sparse coarse coding. NIPS 8, pp. 1038–1044. (cited via S&B)
- Singh, S., Jaakkola, T., Littman, M. L. & Szepesvári, C. (2000). Convergence results for single-step on-policy reinforcement-learning algorithms. *Machine Learning* 38(3):287–308. (cited via S&B)
- van Seijen, H., van Hasselt, H., Whiteson, S. & Wiering, M. (2009). A theoretical and empirical analysis of Expected Sarsa. IEEE ADPRL, pp. 177–184. (figure via S&B Fig. 6.3)
- Zhang, S. Python replication of S&B figures, chapter06/cliff_walking.py. https://github.com/ShangtongZhang/reinforcement-learning-an-introduction (α = 0.5, 50 runs; community code, not primary)
- Tsitsiklis, J. N. & Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. *IEEE Trans. Automatic Control* 42(5):674–690. http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf
- Baird, L. C. (1995). Residual algorithms: Reinforcement learning with function approximation. ICML, pp. 30–37. (cited via S&B)
- van Hasselt, H., Doron, Y., Strub, F., Hessel, M., Sonnerat, N. & Modayil, J. (2018). Deep Reinforcement Learning and the Deadly Triad. arXiv:1812.02648. https://arxiv.org/abs/1812.02648
- Barto, A. G., Sutton, R. S. & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Trans. SMC* 13(5). Pages: 834–846 per the DOI record (10.1109/TSMC.1983.6313077, used in §1); S&B's bibliography prints 835–846. (cited via S&B bibliography / Gymnasium docs)
- Farama Foundation. Gymnasium CartPole-v1 documentation: https://gymnasium.farama.org/environments/classic_control/cart_pole/ ; source: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py and gymnasium/envs/__init__.py (read Oct 2026) [DATED]

### Sources for §11, §13 and §14

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press. §6.7 (Example 6.7, Fig. 6.5), §12.2 (eqs. 12.5–12.7), §17.4 and ch. 17 bibliographical remarks. http://incompleteideas.net/book/RLbook2020.pdf
- Ng, A. Y., Harada, D. & Russell, S. (1999). "Policy invariance under reward transformations: Theory and application to reward shaping." ICML 1999. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf (scanned PDF read via OCR)
- Randløv, J. & Alstrøm, P. (1998). "Learning to drive a bicycle using reinforcement learning and shaping." ICML 1998. Not read; cited via Ng et al. 1999 and the S&B reference list.
- Bellemare, M. G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D. & Munos, R. (2016). "Unifying Count-Based Exploration and Intrinsic Motivation." NeurIPS 2016. https://arxiv.org/abs/1606.01868
- Pathak, D., Agrawal, P., Efros, A. A. & Darrell, T. (2017). "Curiosity-driven Exploration by Self-supervised Prediction." ICML 2017. https://arxiv.org/abs/1705.05363
- Burda, Y., Edwards, H., Storkey, A. & Klimov, O. (2018). "Exploration by Random Network Distillation." arXiv Oct 2018; ICLR 2019. https://arxiv.org/abs/1810.12894
- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O. & Clune, J. (2021). "First return, then explore." *Nature* 590, 580–586. https://doi.org/10.1038/s41586-020-03157-9; pre-print read: https://arxiv.org/abs/2004.12919 (v6, 16 Sep 2021)
- Mnih, V. et al. (2013). "Playing Atari with Deep Reinforcement Learning." NIPS Deep Learning Workshop. https://arxiv.org/abs/1312.5602
- Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." *Nature* 518, 529–533. doi:10.1038/nature14236. PDF read: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (Extended Data Tables 1–3 read from rendered page images; Table 2 transcribed by OCR)
- van Hasselt, H. (2010). "Double Q-learning." NeurIPS 23. https://papers.nips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf
- van Hasselt, H., Guez, A. & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI 2016. https://arxiv.org/abs/1509.06461
- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M. & de Freitas, N. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML 2016. https://arxiv.org/abs/1511.06581
- Schaul, T., Quan, J., Antonoglou, I. & Silver, D. (2016). "Prioritized Experience Replay." ICLR 2016. https://arxiv.org/abs/1511.05952
- Hessel, M. et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI 2018. https://arxiv.org/abs/1710.02298
- Cited only as reported by the papers above (not read): Thrun & Schwartz (1993) on overestimation; Lin (1992) on experience replay; Wiewiora (2003) on the equivalence of shaping and Q-initialisation; Badia et al. (2020) Agent57.

### Sources for §16–§19

- Sutton, R. S., McAllester, D., Singh, S., Mansour, Y. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation. NIPS 12 (1999 conference). https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
- Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning 8:229–256. Author PDF: https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf (DOI 10.1007/BF00992696)
- Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., ch. 13. http://incompleteideas.net/book/RLbook2020.pdf
- Mnih, V. et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. ICML. https://arxiv.org/abs/1602.01783
- OpenAI (18 Aug 2017). OpenAI Baselines: ACKTR & A2C. https://openai.com/index/openai-baselines-acktr-a2c/ (read via https://web.archive.org/web/20190213165907/https://blog.openai.com/baselines-acktr-a2c/)
- Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR. https://arxiv.org/abs/1506.02438
- Schulman, J., Levine, S., Moritz, P., Jordan, M., Abbeel, P. (2015). Trust Region Policy Optimization. ICML. https://arxiv.org/abs/1502.05477
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347. https://arxiv.org/abs/1707.06347
- OpenAI (20 Jul 2017). Proximal Policy Optimization (blog). https://openai.com/index/openai-baselines-ppo/ (read via Wayback, blog.openai.com/openai-baselines-ppo/)
- Berner, C. et al. / OpenAI (2019). Dota 2 with Large Scale Deep Reinforcement Learning. arXiv:1912.06680. https://arxiv.org/abs/1912.06680
- OpenAI (25 Jun 2018). OpenAI Five (blog). https://openai.com/index/openai-five/ (read via Wayback, blog.openai.com/openai-five/)
- OpenAI (15 Apr 2019). OpenAI Five Defeats Dota 2 World Champions (blog). Read via https://web.archive.org/web/2019/https://openai.com/blog/how-to-train-your-openai-five/ (current openai.com/index/… URL not checked)
- Ziegler, D. M. et al. (2019). Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593.
- Stiennon, N. et al. (2020). Learning to Summarize from Human Feedback. arXiv:2009.01325.
- Ouyang, L. et al. (2022). Training Language Models to Follow Instructions with Human Feedback. arXiv:2203.02155.
- Engstrom, L. et al. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR. https://arxiv.org/abs/2005.12729
- Huang, S., Dossa, R. F. J., Raffin, A., Kanervisto, A., Wang, W. (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR Blog Track. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Ahmadian, A. et al. (2024). Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. https://arxiv.org/abs/2402.14740
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., Riedmiller, M. (2014). Deterministic Policy Gradient Algorithms. ICML. https://proceedings.mlr.press/v32/silver14.pdf
- Lillicrap, T. P. et al. (2015/2016). Continuous Control with Deep Reinforcement Learning. ICLR 2016. https://arxiv.org/abs/1509.02971
- Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML. https://arxiv.org/abs/1802.09477
- Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor. ICML. https://arxiv.org/abs/1801.01290
- Haarnoja, T. et al. (2018b). Soft Actor-Critic Algorithms and Applications. https://arxiv.org/abs/1812.05905
- Tobin, J. et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS. https://arxiv.org/abs/1703.06907
- OpenAI, Andrychowicz, M. et al. (2018). Learning Dexterous In-Hand Manipulation. https://arxiv.org/abs/1808.00177
- OpenAI, Akkaya, I. et al. (2019). Solving Rubik's Cube with a Robot Hand. https://arxiv.org/abs/1910.07113
- Farama Foundation. Gymnasium, Pendulum-v1 source. https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py ; registration in gymnasium/envs/__init__.py [DATED]
- DLR-RM. RL Baselines3 Zoo benchmark. https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md [DATED]

### Sources for §20–§22

- Bradley, R. A. & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39(3/4):324–345. https://doi.org/10.2307/2334029 (citation as given in Christiano 2017 and Rafailov 2023; the original was not read)
- Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S. & Amodei, D. (2017). Deep reinforcement learning from human preferences. NeurIPS 2017. https://arxiv.org/abs/1706.03741
- OpenAI (13 June 2017). Learning from Human Preferences (blog). https://openai.com/index/learning-from-human-preferences/ (returns 403 to fetch; read via https://web.archive.org/web/2017/https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/)
- Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P. & Irving, G. (2019). Fine-tuning language models from human preferences. https://arxiv.org/abs/1909.08593
- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D. & Christiano, P. (2020). Learning to summarize from human feedback. NeurIPS 2020. https://arxiv.org/abs/2009.01325
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. NeurIPS 2022. https://arxiv.org/abs/2203.02155
- OpenAI (30 Nov 2022). ChatGPT: Optimizing Language Models for Dialogue. https://openai.com/blog/chatgpt/ (read via web.archive.org 2022 snapshot)
- Ranzato, M., Chopra, S., Auli, M. & Zaremba, W. (2016). Sequence level training with recurrent neural networks. ICLR 2016. https://arxiv.org/abs/1511.06732
- Ramamurthy, R., Ammanabrolu, P., Brantley, K., Hessel, J., Sifa, R., Bauckhage, C., Hajishirzi, H. & Choi, Y. (2023). Is reinforcement learning (not) for natural language processing: benchmarks, baselines, and building blocks for natural language policy optimization (RL4LMs). ICLR 2023. https://arxiv.org/abs/2210.01241
- Rajbhandari, S., Rasley, J., Ruwase, O. & He, Y. (2020). ZeRO: Memory optimizations toward training trillion parameter models. https://arxiv.org/abs/1910.02054
- Yao, Z. et al. (2023). DeepSpeed-Chat: Easy, fast and affordable RLHF training of ChatGPT-like models at all scales. https://arxiv.org/abs/2308.01320
- Hu, J., Wu, X., Wang, W., Xianyu, Zhang, D. & Cao, Y. (2024). OpenRLHF: An easy-to-use, scalable and high-performance RLHF framework. v1, 20 May 2024. https://arxiv.org/abs/2405.11143v1 (the "four models" wording is in v1; the current v6 rewrites it)
- Shao, Z. et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. https://arxiv.org/abs/2402.03300 (§5.2.2 only)
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D. & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. NeurIPS 2023. https://arxiv.org/abs/2305.18290
- Rafailov, R., Hejna, J., Park, R. & Finn, C. (2024). From r to Q*: Your language model is secretly a Q-function. COLM 2024. https://arxiv.org/abs/2404.12358
- Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M. & Calandriello, D. (2024). A general theoretical paradigm to understand learning from human preferences. AISTATS 2024, PMLR 238:4447–4455. https://arxiv.org/abs/2310.12036 ; https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html
- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D. & Kiela, D. (2024). KTO: Model alignment as prospect theoretic optimization. ICML 2024. https://arxiv.org/abs/2402.01306
- Hong, J., Lee, N. & Thorne, J. (2024). ORPO: Monolithic preference optimization without reference model. EMNLP 2024. https://arxiv.org/abs/2403.07691 ; https://aclanthology.org/2024.emnlp-main.626/
- Meng, Y., Xia, M. & Chen, D. (2024). SimPO: Simple preference optimization with a reference-free reward. NeurIPS 2024. https://arxiv.org/abs/2405.14734
- Llama Team, AI @ Meta (2024). The Llama 3 herd of models. https://arxiv.org/abs/2407.21783
- Xu, S. et al. (2024). Is DPO superior to PPO for LLM alignment? A comprehensive study. https://arxiv.org/abs/2404.10719
- Ivison, H. et al. (2024). Unpacking DPO and PPO: Disentangling best practices for learning from preference feedback. NeurIPS 2024. https://arxiv.org/abs/2406.09279

### Sources for §23–§24

- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y., Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv 2402.03300. https://arxiv.org/abs/2402.03300
- Schulman, J. (2020, posted 7 Mar). Approximating KL Divergence. http://joschu.net/blog/kl-approx.html
- Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W.S., Lin, M. (2025). Understanding R1-Zero-Like Training: A Critical Perspective. COLM 2025; arXiv 2503.20783. https://arxiv.org/abs/2503.20783 (code: https://github.com/sail-sg/understand-r1-zero)
- Yu, Q. et al. (ByteDance Seed, Tsinghua AIR) (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv 2503.14476. https://arxiv.org/abs/2503.14476 (project: https://dapo-sia.github.io/)
- Lambert, N. et al. (2024). Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv 2411.15124. https://arxiv.org/abs/2411.15124
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv 2501.12948 v1 (22 Jan 2025) and v2 (4 Jan 2026). https://arxiv.org/abs/2501.12948
- Guo, D. et al. (2025). DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature 645, 633–638. doi:10.1038/s41586-025-09422-z. https://www.nature.com/articles/s41586-025-09422-z
- Kimi Team (2025). Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv 2501.12599. https://arxiv.org/abs/2501.12599
- OpenAI (12 Sept 2024). Learning to Reason with LLMs. https://openai.com/index/learning-to-reason-with-llms/ (read via https://web.archive.org/web/20250101032232/https://openai.com/index/learning-to-reason-with-llms/)
- Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Yue, Y., Song, S., Huang, G. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? arXiv 2504.13837. https://arxiv.org/abs/2504.13837
- Snell, C., Lee, J., Xu, K., Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv 2408.03314. https://arxiv.org/abs/2408.03314
- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv 2110.14168. https://arxiv.org/abs/2110.14168; data and README: https://github.com/openai/grade-school-math
- Hendrycks, D. et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021; arXiv 2103.03874. https://arxiv.org/abs/2103.03874
- Rajbhandari, S., Rasley, J., Ruwase, O., He, Y. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arXiv 1910.02054. https://arxiv.org/abs/1910.02054 (16 bytes/param accounting)
- Hugging Face TRL, `trl/trainer/grpo_trainer.py`, `grpo_config.py`, `utils.py` (main branch, read 2026-10-25; release v1.16.0 of 15 Oct 2026). https://github.com/huggingface/trl [DATED]
- verl, `verl/trainer/ppo/core_algos.py` (main branch, read 2026-10-25). https://github.com/volcengine/verl [DATED]

### Sources for §25–§26

- Goodhart, C. A. E. 1975. "Problems of Monetary Management: The U.K. Experience." RBA *Papers in Monetary Economics*. Reprinted in *Monetary Theory and Practice* (1984), p. 96. Quote verified via the DAMTP page: https://www.damtp.cam.ac.uk/user/mem2/papers/LHCE/goodhart.html and Manheim & Garrabrant fn. 1.
- Strathern, M. 1997. "'Improving ratings': audit in the British University system." *European Review* 5(3):305–321. https://doi.org/10.1017/S1062798700002660. p. 308 per FORRT, https://forrt.org/glossary/english/goodhart_s_law/
- Manheim, D. & Garrabrant, S. 2018. "Categorizing Variants of Goodhart's Law." https://arxiv.org/abs/1803.04585
- Clark, J. & Amodei, D. 2016. "Faulty Reward Functions in the Wild." OpenAI blog, 21 Dec 2016. https://openai.com/index/faulty-reward-functions/ (read via https://web.archive.org/web/2017/https://blog.openai.com/faulty-reward-functions/)
- Krakovna, V. et al. 2020. "Specification gaming: the flip side of AI ingenuity." DeepMind blog, 21 Apr 2020. https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/ ; list at http://tinyurl.com/specification-gaming
- Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J. & Mané, D. 2016. "Concrete Problems in AI Safety." https://arxiv.org/abs/1606.06565
- Gao, L., Schulman, J. & Hilton, J. 2022. "Scaling Laws for Reward Model Overoptimization." https://arxiv.org/abs/2210.10760
- Singhal, P., Goyal, T., Xu, J. & Durrett, G. 2023. "A Long Way to Go: Investigating Length Correlations in RLHF." COLM 2024. https://arxiv.org/abs/2310.03716
- Sharma, M., Tong, M., et al. 2023. "Towards Understanding Sycophancy in Language Models." ICLR 2024. https://arxiv.org/abs/2310.13548
- Anthropic 2025. "Claude 3.7 Sonnet System Card," §6. https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf
- OpenAI 2025. "Detecting misbehavior in frontier reasoning models," 10 Mar 2025. https://openai.com/index/chain-of-thought-monitoring/
- Baker, B., Huizinga, J., Gao, L., Dou, Z., Guan, M. Y., Madry, A., Zaremba, W., Pachocki, J. & Farhi, D. 2025. "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation." https://arxiv.org/abs/2503.11926
- Von Arx, S., Chan, L. & Barnes, B. (METR) 2025. "Recent Frontier Models Are Reward Hacking," 5 Jun 2025. https://metr.org/blog/2025-06-05-recent-reward-hacking/
- MacDiarmid, M., Wright, B., Uesato, J., et al. (Anthropic) 2025. "Natural Emergent Misalignment from Reward Hacking in Production RL." https://arxiv.org/abs/2511.18397
- Zhong, Z., Raghunathan, A. & Carlini, N. 2025. "ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases." https://arxiv.org/abs/2510.20270
- Hu, J. et al. 2024/2025. "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework." https://arxiv.org/abs/2405.11143 ; repo https://github.com/OpenRLHF/OpenRLHF (v0.11.2)
- Sheng, G. et al. 2024. "HybridFlow: A Flexible and Efficient RLHF Framework" (verl). EuroSys 2025. https://arxiv.org/abs/2409.19256 ; docs https://github.com/volcengine/verl/blob/main/docs/algo/grpo.md ; example https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3_4b_fsdp.sh ; k3 code in `verl/trainer/ppo/core_algos.py`
- Hugging Face TRL v1.16.0, `trl/trainer/grpo_config.py`. https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_config.py (default history from tags v0.14.0–v1.0.0)
- Shao, Z. et al. 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." https://arxiv.org/abs/2402.03300 (§4.2 setup; GSM8K 82.9 → 88.2)
- Pan, J. 2025. TinyZero. https://github.com/Jiayi-Pan/TinyZero (README and scripts/train_tiny_zero.sh)
- Zeng, W., Huang, Y., Liu, Q., Liu, W., He, K., Ma, Z. & He, J. 2025. "SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild." https://arxiv.org/abs/2503.18892
- Qwen Team 2024. "Qwen2.5 Technical Report." https://arxiv.org/abs/2412.15115 (Tables 1, 5, 10)
- Yang, A. et al. 2024. "Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement." https://arxiv.org/abs/2409.12122 (§3.3 RL setup)
- Qwen2.5-0.5B-Instruct model card and config. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- mlx-lm 0.32.2: https://github.com/ml-explore/mlx-lm ; mlx-lm-lora 5.10.1: https://github.com/Goekdeniz-Guelmez/mlx-lm-lora (PyPI README)
- "Tokens per Second" course measurements (this Mac): /Users/vpromisex/videos/inference-series/research/measurements.md (§1 decode, §8 batching, matmul/read roofs)
