"""A tour of the "Trial and Reward" kit: every component on screen once.
VOICE=draft manimgx inspect kit_demo.py   → KitDemo.storyboard*.png"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403


class KitDemo(Episode):
    number = 2

    def lesson(self):
        # 1 · the loop
        loop = agent_world_loop(center=(0, -0.3, 0), width=8.6)
        chips = m.VGroup(reward_chip(1), reward_chip(-1), reward_chip(0), reward_chip(-0.04)).arrange(m.RIGHT, buff=0.3)
        chips.move_to([0, -2.9, 0])
        self.say(
            "[The loop]: [an action goes out], [a state and a reward come back.] [Rewards as chips.]",
            m.AnimationGroup(self.new_header("kit · 1", "The loop, Rover, rewards"), m.FadeIn(loop.agent), m.FadeIn(loop.world)),
            m.GrowArrow(loop.action[0]), m.AnimationGroup(m.FadeIn(loop.action[1]), m.GrowArrow(loop.state[0]), m.FadeIn(loop.state[1]),
                                                           m.GrowArrow(loop.reward[0]), m.FadeIn(loop.reward[1])),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in chips], lag_ratio=0.2),
        )
        self.clear()

        # 2 · bandits
        machines = m.VGroup(*[SlotMachine(str(i + 1), size=0.9) for i in range(5)]).arrange(m.RIGHT, buff=0.75).move_to([-0.2, 1.05, 0])
        est = Bars([0.0] * 5, labels=[f"#{i + 1}" for i in range(5)], top=1.0, width=7.4, height=1.7, center=(-0.2, -2.0, 0))
        win = machines[3].payout(1)
        self.say(
            "[Five machines], [their estimates], [a pull], [and a win.]",
            m.AnimationGroup(self.new_header("kit · 2", "Slot machines and bars"), m.LaggedStart(*[m.FadeIn(x, shift=0.2 * m.UP) for x in machines], lag_ratio=0.15)),
            m.FadeIn(est),
            machines[3].pull(),
            m.AnimationGroup(m.FadeIn(win, scale=0.6), est.set([0.2, 0.45, 0.3, 0.62, 0.5])),
        )
        self.say("[A true value marker.]", m.Create(est.marker(3, 0.6, GOLD)))
        self.clear()

        # 3 · the grid world (Russell & Norvig 4×3, row 0 on top)
        world = ROVER_WORLD
        grid = GridView.from_world(world, cell=1.45, center=(-2.6, -0.3, 0))
        V = value_iteration(world)[0]
        pol = greedy_policy(world, V)
        bot = rover(0.55).next_to(grid, m.LEFT, buff=0.35).align_to(grid, m.DOWN)
        arrows = grid.world_arrows(pol, length=0.42, offset=0.2)
        net = NeuralNet((4, 8, 8, 2), width=3.6, height=2.6, center=(4.1, -0.3, 0))
        self.say(
            "[A grid world], [Rover at the start], [values], [the policy], [and a network.]",
            m.AnimationGroup(self.new_header("kit · 3", "Grid world, values, network"), m.FadeIn(grid)),
            m.FadeIn(bot, shift=0.2 * m.UP),
            grid.world_values(V, offset=0.2),
            m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.05),
            m.AnimationGroup(m.FadeIn(net), net.forward()),
        )
        self.clear()

        # 4 · curves and control
        plot = Plot((0, 1000), (0, 1.6), width=5.6, height=3.0, center=(-3.4, -0.5, 0), x_ticks=[0, 500, 1000],
                    y_ticks=[0, 0.5, 1.0, 1.5], x_label="steps", y_label="average reward")
        xs = np.linspace(0, 1000, 60)
        ys1 = 1.0 * (1 - np.exp(-xs / 120))
        ys2 = 1.4 * (1 - np.exp(-xs / 220))
        band = plot.band(xs, ys2 - 0.12, ys2 + 0.12, VALUE)
        c1, c2 = plot.line(xs, ys1, MUTED), plot.line(xs, ys2, VALUE)
        pole = CartPoleView(center=(3.4, 0.6, 0), units_per_m=0.7, pole_units=1.3)
        traj = [(0.3 * np.sin(t / 15), 0, 0.12 * np.sin(t / 7), 0) for t in range(120)]
        pend = PendulumView(center=(3.4, -2.2, 0), length=0.8)
        self.say(
            "[Learning curves], [a spread], [a pole on a cart], [and a pendulum.]",
            m.AnimationGroup(self.new_header("kit · 4", "Curves and control"), m.FadeIn(plot)),
            m.AnimationGroup(m.Create(c1), m.Create(c2), m.FadeIn(band)),
            m.AnimationGroup(m.FadeIn(pole), pole.replay(traj, 2.0)),
            m.AnimationGroup(m.FadeIn(pend), pend.replay(np.linspace(np.pi, 0.2, 40), 2.0)),
        )
        self.clear()

        # 5 · code, tokens, a formula
        src = ("def q_update(Q, s, a, r, s2, alpha, gamma):\n"
               "    target = r + gamma * max(Q[s2])\n"
               "    Q[s][a] += alpha * (target - Q[s][a])")
        block = code_block(src, 20, title="rl_reference.py").move_to([-2.6, 1.3, 0])
        toks = token_row(["The", " answer", " is", " 42"], size=22).move_to([3.6, 1.3, 0])
        formula = tex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]", 40).move_to([0, -1.2, 0])
        term = terminal(["$ python rl_reference.py", "all checks passed"], width=5.4).move_to([0, -2.8, 0])
        self.say(
            "[Code], [tokens as actions], [a formula], [and a terminal.]",
            m.AnimationGroup(self.new_header("kit · 5", "Code, tokens, formulas"), m.FadeIn(block)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in toks], lag_ratio=0.15),
            m.Write(formula),
            m.FadeIn(term),
        )
        self.clear()
