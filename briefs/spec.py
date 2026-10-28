"""Per-episode notes for "Trial and Reward" briefs: number → (slug, ClassName, [reference episodes],
research-notes sections, notes). Reference keys: aiNN / agNN / inNN (sister courses), rlNN (this course)."""

C = "research/experiments/classic.md (load with measured('classic', …); headline keys in classic.json)"
D = "research/experiments/deep_rl.md (load with measured('deep_rl', …); headline keys in deep_rl.json)"
L = "research/experiments/lm_rl.md (load with measured('lm_rl', …); headline keys in lm_rl.json)"
RUN = ("OUR RUN (" + L + "): Qwen2.5-0.5B-Instruct (494M parameters) fine-tuned with LoRA (rank 16, 8.8M trainable) by "
       "GRPO on grade-school maths (GSM8K: 1,600 training problems; a fixed 500 of the 1,319 test problems held out); prompt = "
       "system 'Please reason step by step, and put your final answer within \\boxed{}.'; reward = 1 if the last boxed number "
       "equals the reference, else 0; G = 8 answers per question, 8 questions per step, 200 steps, clip ε = 0.2, KL β = 0.04 "
       "(k3) to the frozen base, lr 2e-5; 71.7 min of training (21.5 s per step, ≈ 2,150 tokens/s sampling, ≈ 25 GB peak) + "
       "37.4 min of evaluation on this Mac (M4 Max). ")

SPEC = {
    # ------------------------------------------------------------------ Module 1 · Trial and Reward
    1: ("reward-loop", "TheRewardLoop", ["ai08", "in02", "ag30"], "§0, §1 (all)",
        "The series OPENER. Hook: Rover in front of a slot machine it has never seen — nobody tells it what to do; it only "
        "gets a reward after it acts. Build agent_world_loop (kit): state → action → reward + next state, turning. Define "
        "agent, environment, state, action, reward, policy, return in one line each, with the loop as the picture. RL vs "
        "supervised learning (labels say what was right; rewards only say how good it was — and often late): a reward "
        "arriving several steps after the action that earned it, travelling back along the trail (credit assignment), and "
        "the second question every episode answers (exploration). A dated timeline strip from notes §1 (TD-Gammon → DQN → "
        "AlphaGo → InstructGPT → reasoning models; cite years). Then the course map: six worlds (five slot machines, a grid, "
        "a cliff, a cart and pole, a board game, a language model) as six cards in the module colours. End on the question "
        "of Episode 2: which machine should Rover pull? Link 'AI, Explained' Episode 8 on screen."),
    2: ("explore-or-exploit", "ExploreOrExploit", ["in02", "in24", "rl01"], "§2 (all), §0",
        "Rover's five machines (ROVER_ARMS; win rates hidden on screen until the end; machine 4 = 0.60 is best). Pull-by-pull "
        "with Bars of estimates (VALUE) under five SlotMachines: the sample average and the incremental update "
        "Q ← Q + (r − Q)/n (incremental_mean; ≤ 6-line code excerpt). A greedy Rover locks onto the first machine that pays "
        "(use the seeded single runs in " + C + ": rover_arms single runs). Then ε-greedy (epsilon = 0.1). Regret = what the "
        "best machine would have paid minus what Rover got. Measured, 2,000 runs × 1,000 pulls: greedy gets stuck in "
        "rover_greedy_pct_runs_stuck ≈ 71.6% of runs; regret greedy ≈ 140.7 vs ε-greedy ≈ 46.3. Then the classic 10-armed "
        "testbed (Sutton & Barto §2.3, 2,000 runs): % optimal action greedy ≈ 34.6, ε 0.01 ≈ 58.4, ε 0.1 ≈ 80.0; average "
        "reward greedy ≈ 1.02 vs ε 0.1 ≈ 1.37 vs the best possible ≈ 1.54 — draw the curves with Plot and compare with the "
        "book's Fig. 2.2 (cite). End: exploring at random wastes pulls on machines we already know are bad (Episode 3)."),
    3: ("optimism-and-luck", "OptimismAndLuck", ["in13", "in24", "rl02"], "§3 (all), §2 (Lai & Robbins), §0",
        "Two smarter ways to explore. (1) Optimism: UCB1 = estimate + c·sqrt(ln t / n) (ucb_scores) — confidence bars "
        "around each estimate shrinking as a machine is pulled; pick the highest TOP of the bar. Optimistic initial values in "
        "one line (testbed % optimal ≈ 85.6). (2) Luck: Thompson sampling — a Beta(wins+1, losses+1) belief per machine "
        "(beta_posterior, beta_pdf drawn with Plot.area in VIOLET); each pull draws one sample from each belief and plays "
        "the highest; the curves sharpen. Measured on Rover's arms (2,000 runs × 1,000 pulls, " + C + "): regret ε-greedy "
        "≈ 46.3, UCB1 (c = √2) ≈ 80.3, Thompson ≈ 31.3 (best arm ≈ 93.1% of pulls). BE HONEST: UCB1 trails ε-greedy at "
        "1,000 pulls here because machines 4 (0.60) and 5 (0.50) are close, so UCB1 keeps checking — it overtakes only at "
        "pull ≈ 12,103 (rover_ucb1_overtakes_eps_at_pull); its guarantee is about the long run (logarithmic regret, Auer et "
        "al. 2002; Lai & Robbins' lower bound). On the 10-armed testbed UCB (c = 2) does beat ε-greedy (86.0% vs 80.0% "
        "optimal; reward 1.49 vs 1.37) — show the testbed curve with UCB's spike at step 11 (as in the book's Fig. 2.4). "
        "Thompson wins at every horizon tested here; Chapelle & Li (2011) found the same in practice. Regret curves on one "
        "Plot (ε, UCB1, Thompson)."),
    4: ("bandits-with-context", "BanditsWithContext", ["in24", "rl02", "rl03"], "§4 (all), §3",
        "The best action can depend on who is asking. Rover recommends one of three study tips to each arriving student, "
        "described by two features (STUDY_TIPS, STUDY_TIP_THETA in rl_reference; say the payoffs are for example). A "
        "context vector arrives; each tip has a line (linear payoff); LinUCB picks the highest optimistic line "
        "(linucb_scores) — animate the context changing and the chosen arm switching. Measured (" + C + "): reward per "
        "student LinUCB ≈ 0.527 vs context-free UCB ≈ 0.424 vs always-the-best-on-average tip ≈ 0.449 vs the oracle ≈ 0.540 "
        "(assert from measured()). Real world: Li et al. 2010, Yahoo! front-page news (notes §4 — use their exact reported "
        "lift). A/B test vs bandit: a fixed 50/50 split pays the price of exploring for the whole test; a bandit shifts "
        "traffic as it learns (Plot of traffic share over time; illustrative unless the notes give numbers). Offline "
        "evaluation by replay in one line. Close Module 1: bandits have no states — Rover's actions don't change what comes "
        "next. Module 2 adds states."),

    # ------------------------------------------------------------------ Module 2 · Planning in a Known World
    5: ("states-actions-returns", "StatesActionsReturns", ["in13", "rl01"], "§5 (all), §0",
        "Rover's world (ROVER_WORLD via GridView.from_world: 4×3, charger +1, puddle −1, a rock, −0.04 per step). The MDP "
        "pieces: states (cells), actions (N/E/S/W), transitions — an intended move splits 80/10/10 (draw the three branches "
        "from one cell with their probabilities, from world.transitions(s, a)), rewards, terminal cells. The Markov property "
        "in one picture (only the current cell matters, not how Rover got there). An episode: play one seeded episode "
        "(world.step or play_episode) and sum the rewards. Return G = r₁ + γ r₂ + γ² r₃ …: show weights shrinking for "
        "γ = 0.9 and why we discount (uncertainty, preferring sooner, finite sums); Rover's world uses γ = 1 (episodes "
        "end). Compute two returns: a short path to the charger (e.g. 5 steps: 1 − 5×0.04 = 0.80 — assert) vs a wandering "
        "path. Policy = a rule from cell to action (arrows). End: how good is a cell? (Episode 6)."),
    6: ("bellman-equation", "TheBellmanEquation", ["in13", "rl05"], "§6 (all), §5",
        "The value of a state = reward now + the (probability-weighted) value of where you land. V(s) and Q(s, a) defined. "
        "The Bellman expectation equation, then the optimality equation (max over actions). Work ONE cell by hand on "
        "ROVER_WORLD with the final values (value_iteration or RN_UTILITIES): the cell left of the charger (3,3) moving E: "
        "0.8 × (charger +1) + 0.1 × stay (bump the wall, 0.918) + 0.1 × down (3,2) 0.660, minus 0.04 → 0.918 — compute with "
        "rl_reference.backup / q_from_v and assert to three decimals. Draw it as a backup diagram (cell → action → three "
        "landing cells with weights). Q for all four actions in that cell (q_from_v), the max picks E. The circularity: each "
        "value uses its neighbours' values — so how do we solve it? (Episode 7). Bellman 1957 citation."),
    7: ("value-iteration", "ValueIteration", ["in24", "rl05", "rl06"], "§7 (all), §6",
        "Repeat the Bellman backup on every cell until nothing changes. value_iteration(ROVER_WORLD) records V per sweep: "
        "show sweeps 1, 2, 3, 5, 10 and the final (GridView.world_values; values spread out from the charger, the puddle's "
        "red spreading too). Measured/computed (" + C + "): 22 sweeps to converge (θ = 1e-4); the greedy policy is already "
        "optimal from sweep 12 (vi_policy_optimal_from_sweep) — arrows snap in before the numbers settle. Final values match "
        "Russell & Norvig to three decimals (0.812 0.868 0.918 / 0.762 · 0.660 / 0.705 0.655 0.611 0.388 — assert against "
        "RN_UTILITIES; cite the edition/figure from notes §5). The famous detour: from (4,1) and (3,1) the optimal policy goes "
        "the long way round (W) to avoid the puddle — explain with the 10% slips. Policy iteration in one section: evaluate, "
        "improve, repeat — 5 evaluations (pi_evaluations). Convergence: the backup is a contraction (one line). The catch: "
        "we needed the map (transitions) — Module 3 learns without it."),

    # ------------------------------------------------------------------ Module 3 · Learning from Experience
    8: ("learning-by-playing", "LearningByPlaying", ["in24", "rl05", "rl07"], "§8 (all)",
        "Take the map away: Rover can only play. Monte Carlo prediction: play an episode, write the return back along the "
        "path (every cell visited gets the sum of what followed), average over many episodes (mc_prediction / "
        "play_episode on ROVER_WORLD). Measured (" + C + ", rover_mc): the Monte Carlo estimate of the start cell converges "
        "toward the planned 0.705 — show the running average approaching it (Plot). First-visit vs every-visit in one line. "
        "Control: to learn actions you need Q and you must keep exploring (exploring starts, ε-soft policies). Strengths "
        "(no model, no bootstrapping bias) and the cost (wait for the end of the episode; high variance — show two very "
        "different returns from the same cell). Blackjack from Sutton & Barto ch. 5 in one line with a citation (no new "
        "numbers). Next: learn before the episode ends (Episode 9)."),
    9: ("learning-from-a-guess", "LearningFromAGuess", ["in24", "rl08"], "§9 (all)",
        "TD(0): update a guess toward reward + the next guess, right away. The TD error δ = r + γV(s′) − V(s) as a PINK "
        "spark between two neighbouring estimates (SIGNAL). The five-state random walk (Sutton & Barto Example 6.2: A–E, "
        "start C, true values 1/6 … 5/6; random_walk_episode, td0_prediction, mc_prediction). Measured (" + C + ", 100 "
        "runs): RMS error after 100 episodes TD α 0.1 ≈ 0.057 vs MC α 0.02 ≈ 0.085 (rw_rms_*), the curves like the book's "
        "Fig. 6.2. Bias vs variance: TD uses a guess (biased) but sees one step of noise; MC sees the whole episode's noise. "
        "n-step returns as the dial between them (n_step_return) in one picture. TD-Gammon (Tesauro 1995) as the first "
        "famous success (notes §9 numbers). Next: TD for actions (Episode 10)."),
    10: ("q-learning-and-sarsa", "QLearningAndSARSA", ["in10", "rl09"], "§10 (all)",
         "TD for action values. SARSA updates toward the action Rover will actually take next (on-policy); Q-learning "
         "toward the best next action (off-policy) — show both update lines side by side (≤ 6 lines each, from "
         "rl_reference). The cliff walk (CLIFF_WORLD: 4×12, −1 per step, −100 and back to start on the cliff; GridView.from_world "
         "draws the cliff RED). Measured (" + C + ", ε = 0.1, α = 0.5 — the book doesn't state α; 0.5 is the common reproduction's, notes §10; say the run counts from classic.md): reward per "
         "episode Q-learning ≈ −49.9, SARSA ≈ −27.4, expected SARSA ≈ −21.1 (the book reports ≈ −50 / −25, Fig. 6.4). Two "
         "Rovers: Q-learning learns the edge path (13 steps, optimal if you never slip) but its ε-greedy behaviour falls off "
         "now and then; SARSA learns the safe road (it accounts for its own exploring). Use greedy_path on runs whose "
         "greedy path reaches the goal (in ~13% of SARSA runs the final greedy path loops — pick a seed that doesn't and say "
         "nothing misleading). Watkins 1989 / Watkins & Dayan 1992 convergence in one line."),
    11: ("when-rewards-are-rare", "WhenRewardsAreRare", ["in24", "rl10"], "§11 (all), §9 (n-step / traces)",
         "Sparse rewards: SPARSE_MAZE (10×10, one reward at the far corner, shortest path 28). Measured (" + C + "): plain "
         "Q-learning's median first success ≈ episode 104, and 21% of runs never succeed in 300 episodes; potential-based "
         "shaping ≈ 4; a count bonus ≈ 2 (maze_first_success_*). IMPORTANT details from classic.md (say them honestly): the "
         "naive shaping φ = −distance never succeeds here (with γ < 1 it pays Rover for bumping into walls) — the working "
         "potential is φ = γ^(d−1), and the count bonus needs optimistic starting values β/(1−γ); optimism alone already "
         "succeeds by about episode 5. Ng, Harada & Russell 1999: potential-based shaping keeps the optimal policy (state the "
         "theorem in one line). After the first success, sparse Q-learning barely improves (≈ 0.9% success in the last 50 "
         "episodes) — the one-step update carries the reward back one cell per visit: eligibility traces / n-step returns "
         "carry it further (show a trace fading along the path). Exploration bonuses at scale: Montezuma's Revenge — "
         "pseudo-counts (Bellemare 2016), curiosity, RND, Go-Explore with their reported scores from notes §11 (dated, "
         "cited). Close Module 3: tables can't hold big worlds (Episode 12)."),

    # ------------------------------------------------------------------ Module 4 · Deep Value Learning
    12: ("tables-to-networks", "FromTablesToNetworks", ["in13", "rl09", "rl11"], "§12 (all)",
         "CartPole (CartPoleView; replay trajectories.json: the random policy falls in ~30 steps): its state is four real "
         "numbers (cart position, velocity, pole angle, angular velocity) — no table can list them. Function approximation: "
         "V(s) ≈ f(s; θ); one update moves many states at once (a value surface instead of cells; generalisation). "
         "Semi-gradient TD (one line of maths). The danger: the deadly triad (function approximation + bootstrapping + "
         "off-policy data; Sutton & Barto ch. 11). Baird's counterexample computed (bairds_counterexample): w8 ≈ 317 after "
         "1,000 steps, |w| ≈ 5×10¹¹ after 10,000 (" + C + ") — a Plot on a log y scale or bars exploding. Tease Episode 13: "
         "without a target network our DQN's Q-values reached ≈ 4×10⁹ (" + D + ", dqn_no_target). Tsitsiklis & Van Roy 1997 "
         "in one line (on-policy linear TD converges)."),
    13: ("deep-q-networks", "DeepQNetworks", ["in13", "in24", "rl12"], "§13 (all)",
         "DQN = Q-learning with a neural network plus two stabilisers: experience replay (a buffer filling and being sampled "
         "at random — breaks correlations, reuses data) and a target network (a frozen copy updated every N steps — a target "
         "that doesn't move with every update). Mnih et al. 2015 (Nature): 49 Atari games, human-level on many (exact "
         "definition and counts from notes §13). Trained on this Mac (" + D + ", 5 seeds): DQN solved CartPole in 5/5 seeds, "
         "≈ 252k ± 118k environment steps (≈ 1,376 episodes, ≈ 35 s); ablations: without replay 0/5 (best 100-episode average "
         "≈ 157; Q-values drifted to ≈ 202 although no CartPole policy can be worth more than 100), without a target network "
         "0/5 (Q-values ≈ 4×10⁹, episodes collapse to ≈ 9 steps). Plot the three learning curves. Replay the trained "
         "dqn_cartpole trajectory (500 steps balanced) vs random_cartpole (≈ 30 steps). Honest note: DQN is unstable even "
         "when it works — 2 of 5 seeds later fell below 300 (final average 283 ± 184): Episode 14."),
    14: ("fixing-overconfidence", "FixingOverconfidence", ["in13", "rl13"], "§14 (all)",
         "Why max over noisy estimates is too high: E[max of 10 zero-mean noisy estimates] ≈ 1.54 σ (expected_max_of_noisy; "
         "Bars with noise and the max picking the luckiest). The maximization-bias toy (Sutton & Barto Ex. 6.7, computed: "
         "maximization_bias): Q-learning goes left up to ≈ 94% of the time early on (the optimum is 5%); Double Q-learning "
         "≈ 7.1% by episodes 201–300 (state B has 10 actions — say so). Double DQN: one network chooses, the other evaluates. "
         "On CartPole (" + D + ", overestimation block): late in training DQN predicted Q(s₀) ≈ 102.8 but achieved ≈ 94.0 "
         "(+8.8); Double DQN predicted ≈ 99.5 and achieved ≈ 97.8 (+1.7); DQN's predictions sat above the impossible ceiling "
         "of 100 in 79% of checks vs 34%. Double DQN also solved faster (175k vs 252k steps) and stayed better (final 444 vs "
         "283). Dueling networks, prioritised replay and Rainbow (Hessel et al. 2018: components and its median "
         "human-normalised score, ablations) from notes §14 — a stacked bar of the six ingredients."),
    15: ("search-and-self-play", "SearchAndSelfPlay", ["in13", "rl13"], "§15 (all)",
         "Planning with a model you have (game rules). Tic-tac-toe searched live: TicTacToe, mcts (UCT with random "
         "rollouts): a tree growing, visit counts and values per move; at the mid-game position MIDGAME with 10 vs 1,000 "
         "simulations the perfect move is found 60% vs 100% of the time (" + C + "); MCTS(1,000) draws 197 of 200 against "
         "perfect play (minimax) and wins 191 of 200 against random; tic-tac-toe has 5,478 reachable positions and perfect "
         "play draws. UCT/PUCT formula (puct_score). Then the big step: AlphaGo (2016, Fan Hui 5–0, Lee Sedol 4–1), AlphaGo "
         "Zero (2017, from scratch, 100–0), AlphaZero (2018, chess/shogi/Go), MuZero (2020, learns its own model) — numbers "
         "and training times from notes §15, a timeline of four cards. Self-play: two copies of the same Rover. Close Module "
         "4: value-based methods pick actions through Q; Module 5 learns the policy directly."),

    # ------------------------------------------------------------------ Module 5 · Learning the Policy
    16: ("policy-gradients", "PolicyGradients", ["in13", "rl13"], "§16 (all)",
         "Learn the policy itself: a network outputs action probabilities (softmax). The objective J(θ) = expected return. "
         "The log-derivative trick turns 'make good episodes likelier' into ∇J = E[G ∇ log π(a|s)] — derive it in three "
         "lines (tex). REINFORCE (Williams 1992): after each episode, push up the log-probability of every action it took, "
         "in proportion to the return; show two CartPole actions' probabilities (Bars, CORAL) nudged after a good and after "
         "a bad episode. Trained on this Mac (" + D + "): REINFORCE without a baseline solved CartPole in only 3/5 seeds "
         "(≈ 260k ± 53k steps for those; final average 229 ± 147) — noisy. Replay reinforce_cartpole. Advantages over Q-based "
         "methods: stochastic policies, continuous actions, smooth changes. The catch (noise) → Episode 17."),
    17: ("baselines-and-critics", "BaselinesAndCritics", ["in13", "in24", "rl16"], "§17 (all), §16",
         "Subtract a baseline b(s) from the return: the gradient stays unbiased, the noise drops. Measured (" + D + ", "
         "gradvar, the same frozen policy and 300 episodes): a learned baseline cut the gradient's variance ≈ 45×, GAE ≈ 101× "
         "(at the 'learning' snapshot; the ratio depends on the stage: 2.7×/5.9× untrained, 22×/121× good — say so). "
         "Learning: REINFORCE + baseline solved 5/5 seeds in ≈ 83k ± 20k steps (vs 3/5 without); actor-critic / A2C with "
         "GAE 5/5 (≈ 213k steps, but stable: final 500 ± 0, fastest in wall time). Advantage A = G − V(s) (or δ-based): "
         "returns scattered around zero instead of all positive. Actor and critic as two Rovers trading signals (CORAL actor, "
         "BLUE critic, PINK advantage). GAE: blend one-step and long returns with λ (gae; Schulman et al. 2016; λ = 0.95 "
         "typical). A3C/A2C (Mnih et al. 2016) in one line."),
    18: ("ppo", "ProximalPolicyOptimization", ["in13", "in24", "rl17"], "§18 (all)",
         "Reusing a batch for several updates makes the data off-policy: the importance ratio r = π_new/π_old corrects it, "
         "but large steps break things. TRPO (2015) bounds the step with a KL trust region; PPO (Schulman 2017) clips the "
         "ratio to [1 − 0.2, 1 + 0.2] — plot ppo_clip_objective vs the ratio for A > 0 and A < 0 (the flat parts). "
         "Trained on this Mac (" + D + "): PPO solved CartPole 5/5 in ≈ 84k ± 24k steps; ≈ 3.7% of samples clipped per "
         "update; approximate KL per update ≈ 0.003. Without the clip each update moves the policy ≈ 35× further (KL ≈ 0.105) "
         "and it doesn't stay solved (final 387 ± 154 vs 500); with heavy reuse (20 epochs) the clip solves in ≈ 58k steps "
         "and no-clip needs ≈ 234k (KL ≈ 2.7 per update). Why PPO became the default: simple, robust, scales (OpenAI Five; "
         "RLHF in Module 6) — numbers from notes §18. Replay ppo_cartpole."),
    19: ("continuous-control", "ContinuousControl", ["in13", "rl18"], "§19 (all)",
         "Robots act with numbers (torques), not choices. A Gaussian policy: mean and spread, a sampled action, the spread "
         "narrowing as it learns (Plot.area). Deterministic policy gradients → DDPG → TD3 (one line each, years); soft "
         "actor-critic: reward + entropy bonus, temperature (Haarnoja et al. 2018). The pendulum swing-up (Pendulum-v1: "
         "reward formula from notes §19; PendulumView replaying sac_pendulum from trajectories.json — 200 steps, return ≈ "
         "−118). Trained on this Mac (" + D + "): SAC reached −147.4 ± 1.6 after only 20k steps (≈ 56 s; better than −200 "
         "after ≈ 4.4k steps); PPO needed ≈ 502k steps for −222 ± 22 (≈ 29 s) — off-policy reuse is sample-efficient, "
         "on-policy is simple and cheap per step. Torque = 2·tanh(mean) (say how the network's output becomes a torque). "
         "Sim-to-real: domain randomization (Tobin 2017) and OpenAI's Rubik's-cube hand (notes §19 numbers). Close Module 5: "
         "next, the policy is a language model."),

    # ------------------------------------------------------------------ Module 6 · Teaching Language Models with Reward
    20: ("model-is-the-policy", "TheModelIsThePolicy", ["ag30", "ai30", "rl16"], "§20 (all), §21 (pipeline)",
         "A language model is a policy: the state is the prompt plus the tokens so far, each token is an action (CORAL "
         "token_row), a completion is an episode, the reward arrives once at the end and flows back to every token. Policy "
         "gradient on tokens: the log-probability of the whole answer is the sum over its tokens. Pretraining → supervised "
         "fine-tuning → RL (the pipeline). The KL leash: reward − β·KL(π ‖ π_ref) keeps the model near where it started "
         "(kl_k3 per token). " + RUN + "Use one real question from the run (lm_rl.json examples — e.g. test-1028, Tom's bedroom "
         "set) to show prompt → tokens → boxed answer → checker → reward 1 or 0; introduce the run as the module's running "
         "example (its results come in Episodes 24 and 26 — don't spend them here). Link Episode 16 (REINFORCE) and 'Agents, "
         "Animated' Episode 30."),
    21: ("human-preferences", "LearningFromHumanPreferences", ["ai30", "in13", "rl20"], "§21 (all)",
         "When there is no checker, people compare two answers. Bradley–Terry: P(A preferred) = σ(r_A − r_B) "
         "(bradley_terry_prob; a curve). A reward model is trained on preference pairs, then PPO with a KL penalty optimises "
         "the policy against it — four models in memory (policy, reference, reward model, value network). Christiano et al. "
         "2017 → Stiennon et al. 2020 → InstructGPT (Ouyang et al. 2022): 12,725 / 33,207 / 31,144 prompts for SFT / RM / PPO, "
         "≈ 40 labelers, a 6B reward model, β = 0.02, the RL stage ≈ 60 vs GPT-3's 3,640 petaflop/s-days (≈ 1.65%), and the "
         "headline: the 1.3B model preferred over the 175B GPT-3 (exact wording from notes §21). Costs and failure modes in one "
         "line each (labels are expensive; the reward model can be fooled → Episode 25). Our own run (" + L + ") used a checker "
         "instead of a reward model — say so when contrasting (Episode 24). Link 'AI, Explained' Episode 30."),
    22: ("dpo", "DirectPreferenceOptimization", ["ai30", "in13", "rl21"], "§22 (all)",
         "Derive DPO in four steps (tex, one per line): the KL-regularised objective → its closed-form optimal policy "
         "π* ∝ π_ref·exp(r/β) → solve for r (the implicit reward β·log π/π_ref + const) → plug into Bradley–Terry: the "
         "constant cancels, leaving a loss on log-probability ratios of the chosen and rejected answers. Use the REAL pair from "
         "our run (" + L + ", the DPO extra): the trucks question, chosen 'x + 2x = 300 → 100' vs rejected '2x = 300 → 150'; "
         "the base model finds the WRONG answer more likely (log-prob ≈ −28.3 vs −41.5); after GRPO (which never saw pairs) the "
         "wrong answer was pushed down more (≈ −9.3 vs −3.7 nats); DPO loss at β = 0.05 / 0.1 / 0.5 ≈ 0.563 / 0.452 / 0.059 vs "
         "ln 2 ≈ 0.693 at the start (dpo_loss / dpo_implicit_reward; assert against lm_rl.json). Honest: a short DPO run on "
         "239 pairs drove the loss to ≈ 0.007 but overfit — held-out greedy accuracy fell to ≈ 43.4% vs ≈ 47.4% for the base "
         "decoded the same way. No reward model, no sampling during training; what it gives up (on-policy exploration). "
         "Variants in one line each with years (IPO, KTO, ORPO, SimPO). Rafailov et al. 2023 results from notes §22."),
    23: ("grpo", "GRPO", ["ag30", "in13", "rl18"], "§23 (all), §18",
         "Group Relative Policy Optimization (Shao et al. 2024, DeepSeekMath — it removes the critic, NOT the reward model; "
         "DeepSeekMath used G = 64, β = 0.04 and took GSM8K 82.9% → 88.2%): sample G answers to the same question, score "
         "them, advantage = (r − group mean) / group std, then a PPO-style clipped update with a KL term to the reference (k3, "
         "kl_k3). " + RUN + "Show ONE real group from our run (lm_rl.json, the early group at step 4: 4 of 8 answers right → "
         "advantages exactly ±1 because our run used the POPULATION std with ε = 1e-4 — compute with grpo_advantages(rewards, "
         "eps=1e-4, ddof=0) and assert; say that TRL and verl use the sample std, which would give ±0.935) — eight answer "
         "cards (truncated text), rewards, advantages (PINK), the update pushing the right ones up. All-equal groups give zero "
         "advantage: the fraction of such groups early vs late (zero_adv_groups_first10 / last10). One update's ratio and clip "
         "statistics from the run (clip fraction ≈ clip_frac_mean). The ≤ 12-line loss excerpt from research/experiments/lm/ "
         "as a code_block. Follow-ups in one line each: Dr. GRPO (length/std biases), DAPO (clip-higher, dynamic sampling, "
         "token-level loss) from notes §23. Link Episode 18 (PPO) and 'Agents, Animated' Episode 30."),
    24: ("rewards-you-can-check", "RewardsYouCanCheck", ["ai43", "in24", "rl23"], "§24 (all), §26",
         "Rewards from checkers: a maths answer matched exactly, unit tests passing (RL with verifiable rewards; the term "
         "from Lambert et al. 2024). DeepSeek-R1-Zero and R1: pure RL on a base model, AIME 2024 pass@1 15.6% → 71.0% in the "
         "Jan 2025 preprint (77.9% in the later Nature / arXiv v2 version — say which you cite, notes Appendix A), majority "
         "vote over 64 → 86.7%, rule-based accuracy + format rewards, the 'aha moment' quote and response length growing "
         "(cited). Our run (" + L + ", 500 held-out problems, 8 samples at T = 0.8): pass@1 38.3% → 46.2% (+7.9 ± 0.9), pass@2 "
         "51.7 → 58.1, pass@4 63.0 → 68.4, pass@8 71.6 → 76.4 (+4.8 ± 1.8) — the gain shrinks as k grows, a small, honest data "
         "point for the pass@k debate (Yue et al. 2025); pass_at_k is the unbiased estimator. BE HONEST: our answers did NOT "
         "get longer (≈ 309 → 305 tokens) — don't claim R1-style length growth from our run. The gain is mostly maths, not "
         "format (train_samples_format_vs_maths). A checker stamping answers GREEN/RED. Greedy accuracy is shaky: the same "
         "weights give 46–48% depending on the generation batch size and 14% of problems flip from batch size alone "
         "(greedy_by_batch_size, heldout_flips) — that's why the headline is pass@1 from samples (link 'Tokens per Second' "
         "Episode 17 on batch-dependent sums). Link 'AI, Explained' Episode 43."),
    25: ("reward-hacking", "RewardHacking", ["ai46", "in24", "rl24"], "§25 (all)",
         "Goodhart's law (Goodhart's own wording; 'When a measure becomes a target, it ceases to be a good measure' is "
         "Strathern's 1997 phrasing — notes §25). Specification gaming: the CoastRunners boat circling for points (scored "
         "'20 percent higher than … human players', OpenAI 2016) and two more from Krakovna's list. Reward-model "
         "overoptimisation (Gao, Schulman & Hilton 2022): proxy reward keeps rising while gold reward rises then falls; their "
         "fitted forms R = d(α − β·d) (best-of-n) and d(α − β·ln d) (RL) with d = √KL — draw the shape (constants from notes "
         "if given, else label illustrative). Length bias and sycophancy in one line each. OUR OWN HACK (" + L + ", the "
         "reward-hacking extra): the reward 'correct + 0.001 per token', same questions and seeds as the honest run, 60 steps — "
         "the proxy reward rose in every block (≈ 0.75 → 0.86) while true accuracy stayed ≈ 5–7 points below the honest run; "
         "answers grew to ≈ 390 tokens, a third of late answers hit the 512-token limit with no answer, and the model padded "
         "by faking carefulness ('re-evaluate / recheck' phrases 4.8% → 14.8% vs 2.3% honest); group normalisation turned the "
         "tiny bonus into a full-strength signal (no ties). The defences: the KL leash, better checkers, held-out evaluation. "
         "Link 'AI, Explained' Episode 46."),
    26: ("grpo-on-this-mac", "GRPOOnThisMac", ["ag30", "in24", "rl23"], "§26 (all), §23, §24",
         "The FINALE, built on our real run. " + RUN + "The loop of Episode 1 again, now with a language model: sample a group → "
         "check → advantages → update. The training reward curve (0.476 → 0.570, first vs last 25 steps; Plot from lm_rl.json "
         "curves), pass@1 38.3% → 46.2% on held-out problems, one question before and after verbatim (test-1028: the base model "
         "takes 10% of $3,000 and answers 300 — wrong; the trained model takes 10% of the remaining $2,000 and answers 200 — "
         "right; token counts) and, honestly, the regression example too. Where the time went (sampling vs log-probs vs update "
         "per step from the run log; links to 'Tokens per Second'). One seed only — say so. Then the course revisited: one card "
         "per module with its key idea and episode numbers (six module colours), the two questions (credit and exploration) "
         "answered six ways. Three-word recap. Override series_outro() with a course-closing card: 'That's Trial and Reward', "
         "the six module names in their colours, Rover, 'Thanks for watching!'."),
}
