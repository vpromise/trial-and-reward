"""Trial and Reward · Episode 6 · The Bellman Equation (Module 2 · Planning in a Known World)

A state's value is the reward now plus the value of where you land. Three seeded runs from Rover's start cell give
three different returns; the value is their average. A return splits into the reward now plus the return one cell
later (episode 5's lucky run: 0.80 = −0.04 + 0.84), so a value is the reward now plus the value of where you land,
averaged over the slips. The heart: ONE cell by hand, (3,3) just left of the charger, moving East, drawn as a backup
diagram (cell → move → three landings weighted 0.8 / 0.1 / 0.1): 0.8 × (+1) + 0.1 × 0.918 (bump the wall, stay)
+ 0.1 × 0.660 (slip south) − 0.04 = 0.918. Then Q for all four moves there (N 0.881, E 0.918, S 0.675, W 0.812) and
the max; the optimality and expectation equations with their backup diagrams; policy evaluation of a random Rover;
and the catch: every value is defined by the others (episode 7 solves it).

Sources (facts dated October 2026):
- rl_reference.py: ROVER_WORLD (Russell & Norvig 4x3: 0.8 ahead / 0.1 each side, −0.04 per move, γ = 1),
  value_iteration, q_from_v, backup, greedy_policy, policy_evaluation, play_episode, RN_UTILITIES.
- research/notes.md §6 (Bellman expectation S&B eq. 3.14, optimality eq. 3.19, backup diagrams S&B §3.5 and Fig. 3.4,
  "a functional equation, now often called the Bellman equation ... dynamic programming (Bellman, 1957a)" S&B §1.7;
  the worked cells), §5 (the world; Fig. 17.3 utilities; a random Rover's start cell −1.587), Appendix C (V(1,1) =
  0.705, checked here with every other cell).
- research/experiments/classic.md §4 (value iteration: 22 sweeps, every cell equal to R&N to three decimals).

Illustrative: none of the numbers. The three runs are seeded simulations (play_episode, seeds 11, 0, 9) of Rover's
best moves; their trails are drawn slightly offset so they don't hide each other.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep06"

# ---------------------------------------------------------------------------- the numbers (all computed here)

W = ROVER_WORLD
V, HISTORY = value_iteration(W)
SWEEPS = len(HISTORY) - 1
Q = q_from_v(W, V)
POLICY = greedy_policy(W, V)
CELL, SOUTH, START = (3, 3), (3, 2), W.start            # the cell left of the charger, the one under it, the start
assert SWEEPS == 22 and W.slip == 0.2 and W.step_reward == -0.04 and CHARGER == (4, 3) and PUDDLE == (4, 2)
assert all(round(V[s], 3) == RN_UTILITIES[s] for s in RN_UTILITIES)             # Russell & Norvig Fig. 17.3
assert round(V[CELL], 3) == 0.918 and round(V[SOUTH], 3) == 0.660 and round(V[START], 3) == 0.705

# --- one cell by hand: (3,3) moving East
OUT = W.transitions(CELL, "E")
assert [(p, s2, done) for p, s2, _, done in OUT] == [(0.8, CHARGER, True), (0.1, CELL, False), (0.1, SOUTH, False)]
assert [round(r, 2) for _, _, r, _ in OUT] == [0.96, -0.04, -0.04]               # every move pays −0.04, the charger +1
STEP = W.step_reward
HAND = 0.8 * W.terminals[CHARGER] + 0.1 * V[CELL] + 0.1 * V[SOUTH] + STEP
assert abs(HAND - backup(W, V, CELL, "E")) < 1e-12 and abs(HAND - Q[CELL]["E"]) < 1e-12
assert round(HAND, 3) == 0.918 == round(V[CELL], 3)
TERMS = [0.8 * W.terminals[CHARGER], 0.1 * V[CELL], 0.1 * V[SOUTH], STEP]
assert [round(t, 3) for t in TERMS] == [0.8, 0.092, 0.066, -0.04]
assert round(sum(round(t, 3) for t in TERMS), 3) == 0.918                          # the rounded terms add up too

# --- all four moves there
Q33 = Q[CELL]
assert {a: round(q, 3) for a, q in Q33.items()} == {"N": 0.881, "E": 0.918, "S": 0.675, "W": 0.812}
assert max(W.actions, key=lambda a: Q33[a]) == "E" == POLICY[CELL] and abs(max(Q33.values()) - V[CELL]) < 1e-4
assert min(W.actions, key=lambda a: Q33[a]) == "S" and W.transitions(CELL, "S")[0][:2] == (0.8, SOUTH)
_north = [(p, s2) for p, s2, _, _ in W.transitions(CELL, "N")]
assert _north == [(0.8, CELL), (0.1, (2, 3)), (0.1, CHARGER)]                      # bump and stay; a slip east charges

# --- three seeded runs of Rover's best moves from the start (play_episode)
RUN_SEEDS = (11, 0, 9)
RUNS = [play_episode(W, POLICY, np.random.default_rng(seed)) for seed in RUN_SEEDS]


def ret(ep):
    return sum(t.r for t in ep)


def slips(ep):
    return [(t.s, t.s2) for t in ep if t.s2 != W.move(t.s, t.a)]


assert [len(ep) for ep in RUNS] == [5, 8, 8] and [round(ret(ep), 2) for ep in RUNS] == [0.80, 0.68, -1.32]
assert slips(RUNS[0]) == [] and [s for s, _ in slips(RUNS[1])] == [CELL, CELL]      # "slip twice beside the charger"
assert RUNS[1][-1].s2 == CHARGER and RUNS[2][-1].s2 == PUDDLE
assert (CELL, SOUTH) in slips(RUNS[2]) and slips(RUNS[2])[-1] == (SOUTH, PUDDLE)   # slips south, then slips in

# --- the lucky run split in two: reward now + the return one cell later
LUCKY = RUNS[0]
assert "".join(t.a for t in LUCKY) == "NNEEE" and LUCKY[0].s2 == (1, 2)
G0, R0 = ret(LUCKY), LUCKY[0].r
G1 = ret(LUCKY[1:])
assert round(G0, 2) == 0.80 and round(R0, 2) == -0.04 and round(G1, 2) == 0.84 and abs(G0 - (R0 + G1)) < 1e-12

# --- a fixed policy: Rover moving at random (policy evaluation = solving the expectation equation)
RANDOM = {s: {a: 0.25 for a in W.actions} for s in W.states if not W.is_terminal(s)}
V_RAND = policy_evaluation(W, RANDOM, theta=1e-10)
Q_RAND = q_from_v(W, V_RAND)
assert abs(V_RAND[CELL] - sum(Q_RAND[CELL].values()) / 4) < 1e-8                   # the expectation equation holds
assert round(V_RAND[CELL], 3) == -0.315 and round(V_RAND[START], 3) == -1.587       # notes §5: −1.587 at the start
_ends_in_puddle = GridWorld.from_layout(ROVER_LAYOUT, terminal_rewards={"+": 0.0, "-": 1.0}, step_reward=0.0, slip=0.2)
_moves = GridWorld.from_layout(ROVER_LAYOUT, terminal_rewards={"+": 0.0, "-": 0.0}, step_reward=1.0, slip=0.2)
P_PUDDLE = policy_evaluation(_ends_in_puddle, RANDOM, theta=1e-12)[CELL]
N_MOVES = policy_evaluation(_moves, RANDOM, theta=1e-12)[CELL]
assert round(P_PUDDLE, 2) == 0.39 and round(N_MOVES) == 13                          # "about four in ten", "thirteen"

# --- the catch: every cell checks out, and every value leans on the others
OPEN = [s for s in W.states if not W.is_terminal(s)]
assert len(OPEN) == 9 and all(round(max(Q[s].values()), 3) == round(V[s], 3) for s in OPEN)
assert POLICY[SOUTH] == "N" and W.transitions(SOUTH, "N")[0][:2] == (0.8, CELL)    # 0.660 needs 0.918 back
assert POLICY[START] == "N"
PAIRS = [(a, b) for a in OPEN for b in OPEN if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
assert len(PAIRS) == 9

# ---------------------------------------------------------------------------- look

DICE = recolored("dice", ACTION, CORAL_FILL, prefix=PREFIX)


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def tick(center, size=0.26, color=GOOD, width=5):
    """A check mark drawn with two strokes."""
    c = np.array(center, dtype=float)
    pts = [c + size * np.array(p) for p in ((-0.5, 0.0, 0), (-0.15, -0.38, 0), (0.55, 0.45, 0))]
    return m.VMobject().set_points_as_corners(pts).set_stroke(color, width).set_fill(opacity=0)


def static_values(grid, values, fmt="{:.3f}", size=20, offset=0.0, lo=-1.0, hi=1.0):
    """Tint a grid's open cells by value now (no animation) and return the numbers as a VGroup."""
    nums = m.VGroup()
    for s, v in values.items():
        rc = W.to_screen(s)
        if rc in grid.walls or rc in grid.specials:
            continue
        grid.squares[rc].set_fill(heat(v, lo, hi), 1)
        nums.add(code(minus(fmt.format(v)), size, INK).move_to(grid.at(s) + [0, -offset * grid.size, 0]))
    return nums


def run_animation(grid, bot, ep, offset, step=0.3):
    """Rover follows one recorded run: a glide per move (with a trail segment), a shake when it stays put."""
    seq, trail = [], m.VGroup()
    for t in ep:
        a, b = grid.at(t.s), grid.at(t.s2)
        if t.s2 == t.s:
            seq.append(m.Wiggle(bot, scale_value=1.08, rotation_angle=0.05 * m.TAU, n_wiggles=3, run_time=step))
            continue
        seg = m.Line(a + offset, b + offset).set_stroke(ACTION, 5, opacity=0.85)
        trail.add(seg)
        seq.append(m.AnimationGroup(bot.animate(run_time=step).move_to(b), m.Create(seg, run_time=step)))
    return m.Succession(*seq), trail


def state_node(pos, color=STATE, radius=0.22):
    return m.Circle(radius=radius).set_stroke(color, 3).set_fill(BG, 1).move_to(pos)


def terminal_node(pos, color=REWARD, side=0.38):
    return m.Square(side).set_stroke(color, 3).set_fill(FILLS[color], 1).move_to(pos)


def action_node(pos, radius=0.12):
    return m.Dot(pos, radius=radius, color=ACTION)


def edge(a, b, color, width=3, buff_a=0.0, buff_b=0.0):
    d = (b - a) / np.linalg.norm(b - a)
    return m.Line(a + d * buff_a, b - d * buff_b).set_stroke(color, width)


def mini_tree(center, weights="", arc=False, width=2.8, height=1.75):
    """A small backup diagram: a state, four moves under it, three landings under each move. `weights` labels the
    four move edges with the policy's chances; `arc=True` draws the max arc across them instead."""
    c = np.array(center, dtype=float)
    root_p = c + [0, height / 2, 0]
    xs = np.linspace(-width / 2, width / 2, 4)
    act_p = [c + [x, 0.0, 0] for x in xs]
    root = state_node(root_p, radius=0.18)
    acts = m.VGroup(*[action_node(p, 0.1) for p in act_p])
    up = m.VGroup(*[edge(root_p, p, ACTION, 3, 0.18, 0.1) for p in act_p])
    leaves, down = m.VGroup(), m.VGroup()
    for p in act_p:
        for dx in (-0.24, 0.0, 0.24):
            q = p + [dx, -height / 2, 0]
            leaves.add(state_node(q, radius=0.075).set_stroke(STATE, 2))
            down.add(edge(p, q, EXPLORE, 2, 0.1, 0.075))
    parts = m.VGroup(up, down, root, acts, leaves)
    extra = m.VGroup()
    if arc:
        r = 0.62 * height
        a0 = np.arctan2(act_p[0][1] - root_p[1], act_p[0][0] - root_p[0])
        a1 = np.arctan2(act_p[-1][1] - root_p[1], act_p[-1][0] - root_p[0])
        extra.add(m.Arc(radius=r, start_angle=a0, angle=a1 - a0, arc_center=root_p).set_stroke(ACTION, 4))
        extra.add(code("max", 22, ACTION, weight=m.BOLD).next_to(root, m.RIGHT, buff=0.55).shift(0.05 * m.DOWN))
    if weights:
        extra.add(code(weights, 22, ACTION, weight=m.BOLD).next_to(root, m.RIGHT, buff=0.55).shift(0.05 * m.DOWN))
    return m.VGroup(parts, extra)


# ---------------------------------------------------------------------------- the episode


class Ep06TheBellmanEquation(Episode):
    number = 6

    def lesson(self):
        self.question()
        self.one_step()
        self.slippery()
        self.by_hand()
        self.four_moves()
        self.optimality()
        self.evaluation()
        self.recap()
        self.catch()

    # ------------------------------------------------------------------ 1 · how good is a cell?

    def question(self):
        self.next_section("1 · How good is a cell?", notes=(
            "Rover's world (ROVER_WORLD, Russell & Norvig's 4x3: 0.8 ahead, 0.1 each side, −0.04 per move, γ = 1). "
            f"Three seeded runs of Rover's best moves from the start (play_episode, greedy_policy of value_iteration, seeds "
            f"{RUN_SEEDS}): {len(RUNS[0])} moves, return {ret(RUNS[0]):.2f}; {len(RUNS[1])} moves with two slips at (3,3), "
            f"{ret(RUNS[1]):.2f}; {len(RUNS[2])} moves ending in the puddle, {ret(RUNS[2]):.2f}. The value of the start is "
            f"the expected return, V(1,1) = {V[START]:.4f} (R&N Fig. 17.3: 0.705). Episode 5 built the world; episode 8 "
            "estimates values by averaging runs."))
        grid = GridView.from_world(W, cell=1.35, center=(-2.9, -0.5, 0))
        start = grid.at(START)
        bots = [rover(0.42).move_to(start).set_z_index(3) for _ in RUNS]
        offsets = [np.array([0.0, 0.0, 0]), np.array([0.1, -0.1, 0]), np.array([-0.1, 0.1, 0])]
        runs = [run_animation(grid, bot, ep, off) for bot, ep, off in zip(bots, RUNS, offsets)]
        trail_note = cite("three seeded runs · Rover's best moves").next_to(grid, m.DOWN, buff=0.3)

        x0, x1 = 1.0, 6.35
        heads = m.VGroup(code("run", 22, FAINT).move_to([x0, 1.75, 0], aligned_edge=m.LEFT),
                         code("moves", 22, FAINT).move_to([2.6, 1.75, 0], aligned_edge=m.LEFT),
                         code("return", 22, FAINT).move_to([x1, 1.75, 0], aligned_edge=m.RIGHT))
        rows = m.VGroup()
        for i, ep in enumerate(RUNS):
            y = 1.15 - 0.62 * i
            end = icon("charger" if ep[-1].s2 == CHARGER else "puddle", 0.36).move_to([x0 + 0.15, y, 0])
            row = m.VGroup(code(f"{i + 1}", 24, MUTED).move_to([x0 + 0.7, y, 0]),
                           code(f"{len(ep)}", 24, MUTED).move_to([3.05, y, 0]),
                           code(minus(f"{ret(ep):+.2f}"), 28, VALUE, weight=m.BOLD).move_to([x1, y, 0], aligned_edge=m.RIGHT),
                           end)
            rows.add(row)
        rule = m.Line([x0, -0.75, 0], [x1, -0.75, 0]).set_stroke(EDGE, 2)
        avg = text("its value: the return to expect, on average", 22, MUTED).move_to([0, -1.2, 0]).align_to([x0, 0, 0], m.LEFT)
        stat_v = code(f"V(start) = {V[START]:.3f}", 34, VALUE, weight=m.BOLD).move_to([0, -1.85, 0]).align_to([x0, 0, 0], m.LEFT)
        stat_sub = cite("best moves · every slip weighed by its chance").next_to(stat_v, m.DOWN, buff=0.2).align_to([x0, 0, 0], m.LEFT)

        self.say(
            "[Episode five ended on a question]: [how good is it for Rover to stand on a cell?]",
            m.AnimationGroup(self.new_header("01 · the question", "How good is a cell?"), m.FadeIn(grid)),
            m.AnimationGroup(m.FadeIn(bots[0], shift=0.2 * m.UP), m.FadeIn(heads)),
        )
        self.say(
            "[Here's one run from the start: five moves, straight to the charger], [for a return of zero point eight.]",
            m.AnimationGroup(runs[0][0], m.FadeIn(trail_note)),
            m.FadeIn(rows[0], shift=0.1 * m.LEFT),
        )
        self.say(
            "[In a second run, the wheels slip twice beside the charger]: [eight moves, and zero point six eight.]",
            m.Succession(m.AnimationGroup(m.FadeOut(bots[0]), runs[0][1].animate.set_stroke(opacity=0.25),
                                          m.FadeIn(bots[1]), run_time=0.4), runs[1][0]),
            m.FadeIn(rows[1], shift=0.1 * m.LEFT),
        )
        self.say(
            "[In a third, Rover slips south beside the puddle, then slips right in]: [minus one point three two.]",
            m.Succession(m.AnimationGroup(m.FadeOut(bots[1]), runs[1][1].animate.set_stroke(opacity=0.25),
                                          m.FadeIn(bots[2]), run_time=0.4), runs[2][0]),
            m.FadeIn(rows[2], shift=0.1 * m.LEFT),
        )
        self.say(
            "[A cell's value is the return Rover can expect from it, on average]: [from the start, zero point seven zero five.]",
            m.AnimationGroup(m.Create(rule), m.FadeIn(avg, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(stat_v, scale=0.9), m.FadeIn(stat_sub)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · one step, then the rest

    def one_step(self):
        self.next_section("2 · Reward now, value next", notes=(
            f"The lucky run (seed {RUN_SEEDS[0]}): moves {''.join(t.a for t in LUCKY)}, −0.04 per move and +1 on entering the "
            f"charger: G = {G0:.2f}. Splitting off the first reward: G = r + G′ with G′ = {G1:.2f}, the return of the same "
            "run from (1,2) (S&B eq. 3.9: G_t = R_{t+1} + γ G_{t+1}, γ = 1 here). Averaging over the slips turns returns "
            "into values: V(s) = E[r + γ V(s′)] (notes §6). Episode 8 averages played runs (Monte Carlo)."))
        y_arrow, y_chip = 2.05, 1.35
        xs = [-5.75 + 1.5 * i for i in range(6)]
        moves = m.VGroup()
        for x, t in zip(xs, LUCKY):
            d = np.array([*GridView.DIRS[t.a], 0], dtype=float) * 0.24
            moves.add(m.Arrow([x, y_arrow, 0] - d, [x, y_arrow, 0] + d, buff=0, color=ACTION, stroke_width=5,
                              max_tip_length_to_length_ratio=0.45))
        goal = icon("charger", 0.46).move_to([xs[5], y_arrow, 0])
        chips = m.VGroup(*[reward_chip(-0.04, 24).move_to([x, y_chip, 0]) for x in xs[:5]],
                         reward_chip(1, 24).move_to([xs[5], y_chip, 0]))
        total = code(f"= {G0:.2f}", 32, VALUE, weight=m.BOLD).move_to([0, y_chip, 0]).align_to([xs[5] + 0.75, 0, 0], m.LEFT)
        total_lab = cite("the return").move_to([0, y_arrow, 0]).align_to(total, m.LEFT)
        ep8 = pill("episode 8 · average many runs", VALUE, BLUE_FILL, 20).move_to([0, 0.2, 0]).align_to([6.45, 0, 0], m.RIGHT)

        now_lab = text("reward now", 22, REWARD, weight=m.BOLD)
        ring = m.SurroundingRectangle(chips[0], buff=0.1, corner_radius=0.12).set_stroke(REWARD, 3)
        brace = m.Brace(chips[1:], m.DOWN, buff=0.18, color=VALUE)
        rest_lab = text(f"{G1:.2f} · the return of the same run, one cell later", 22, VALUE).next_to(brace, m.DOWN, buff=0.15)
        now_lab.move_to([chips[0].get_x(), rest_lab.get_y(), 0])

        f_g = m.MathTex("G", "=", "r", "+", "G'", font_size=64, color=INK)
        f_g[0].set_color(VALUE)
        f_g[2].set_color(REWARD)
        f_g[4].set_color(VALUE)
        f_g.move_to([-3.6, -1.0, 0])
        f_num = code(minus(f"{G0:.2f} = {R0:.2f} + {G1:.2f}"), 30, INK,
                     t2c={r"^0\.80": VALUE, r"-0\.04|−0\.04": REWARD, r"0\.84$": VALUE})
        f_num.move_to([2.2, -1.0, 0])
        f_v = m.MathTex(r"V(s)", "=", r"\mathbb{E}\big[", "r", "+", r"V(s')", r"\big]", font_size=58, color=INK)
        f_v[0].set_color(VALUE)
        f_v[3].set_color(REWARD)
        f_v[5].set_color(VALUE)
        f_v.move_to([-3.6, -2.45, 0])
        avg_note = cite("averaged over every slip").next_to(f_v, m.RIGHT, buff=0.5)
        slogan = text("value here = reward now + value of where you land", 28, INK,
                      t2c={"value here": VALUE, "reward now": REWARD, "value of where you land": VALUE})
        slogan.move_to([0, -3.3, 0])

        self.say(
            "[Averaging many runs works: that's episode eight.] [But look inside the lucky run's return]: "
            "[minus zero point zero four a move], [and plus one at the charger.]",
            m.AnimationGroup(self.new_header("02 · one step", "Reward now, value next"), m.FadeIn(ep8, shift=0.1 * m.DOWN)),
            m.LaggedStart(*[m.GrowArrow(a) for a in moves], lag_ratio=0.15),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in chips[:5]], lag_ratio=0.15),
            m.AnimationGroup(m.FadeIn(goal), m.FadeIn(chips[5], shift=0.1 * m.DOWN), m.FadeIn(total_lab),
                             m.FadeIn(total, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Split off the first reward.] [What's left, zero point eight four], [is the return of the same run, started one cell later.]",
            m.AnimationGroup(m.Create(ring), m.FadeIn(now_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(rest_lab, shift=0.1 * m.UP)),
            m.Circumscribe(m.VGroup(moves[1:], chips[1:]), color=VALUE, buff=0.1),
        )
        self.say(
            "[So a return is the reward now], [plus the return from where you land.]",
            m.FadeIn(f_g[:3], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(f_g[3:], shift=0.1 * m.UP), m.FadeIn(f_num, shift=0.1 * m.UP)),
        )
        self.say(
            "[Average over every slip, and returns become values]: "
            "[a cell's value is the reward now, plus the value of where you land.]",
            m.AnimationGroup(m.FadeIn(f_v, shift=0.1 * m.UP), m.FadeIn(avg_note)),
            m.Write(slogan, run_time=2.0),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · slippery wheels: the backup diagram

    def slippery(self):
        self.next_section("3 · A backup diagram", notes=(
            f"world.transitions((3,3), 'E') = {[(p, s2, round(r, 2), d) for p, s2, r, d in OUT]}: 0.8 into the charger "
            "(−0.04 + 1, the episode ends), 0.1 slip north into the wall (stay, −0.04), 0.1 slip south to (3,2) (−0.04). "
            "Backup diagram as in S&B §3.5: open circle = state, solid dot = state–action pair, values flow back up from "
            "the landings to the root. Arrow widths ∝ probability; CORAL = the intended move, VIOLET = slips."))
        grid = GridView.from_world(W, cell=1.45, center=(-3.55, -0.5, 0))
        self.grid = grid
        c = grid.at(CELL)
        bot = rover(0.4).move_to(c + [-0.3, 0.0, 0]).set_z_index(3)
        self.bot = bot
        east = m.Arrow(c + [0.02, 0.1, 0], c + [1.0, 0.1, 0], buff=0, color=ACTION, stroke_width=9,
                       max_tip_length_to_length_ratio=0.32)
        north = m.Arrow(c + [0.3, 0.24, 0], c + [0.3, 0.66, 0], buff=0, color=EXPLORE, stroke_width=4,
                        max_tip_length_to_length_ratio=0.4)
        wall = m.Line(c + [0.1, 0.705, 0], c + [0.5, 0.705, 0]).set_stroke(RED, 6)
        south = m.Arrow(c + [0.47, 0.02, 0], c + [0.47, -1.02, 0], buff=0, color=EXPLORE, stroke_width=4,
                        max_tip_length_to_length_ratio=0.2)
        self.slip_arrows = m.VGroup(east, north, wall, south)

        # the backup diagram, on the right
        root_p, act_p = np.array([3.3, 2.1, 0]), np.array([3.3, 1.2, 0])
        leaf_p = [np.array([x, 0.3, 0]) for x in (1.2, 3.3, 5.4)]
        root = state_node(root_p, radius=0.24)
        root_lab = code("(3,3)", 22, STATE).next_to(root, m.LEFT, buff=0.2)
        act = action_node(act_p, 0.14)
        act_lab = code("E", 24, ACTION, weight=m.BOLD).move_to(act_p + [0.32, 0.24, 0])
        stem = edge(root_p, act_p, ACTION, 4, 0.24, 0.14)
        leaves = m.VGroup(terminal_node(leaf_p[0]), state_node(leaf_p[1]), state_node(leaf_p[2]))
        widths = [7, 3, 3]
        branches = m.VGroup(*[edge(act_p, q, EXPLORE, wd, 0.14, 0.22) for q, wd in zip(leaf_p, widths)])
        probs = m.VGroup(
            code("0.8", 22, EXPLORE, weight=m.BOLD).move_to((act_p + leaf_p[0]) / 2 + [-0.35, 0.18, 0]),
            code("0.1", 22, EXPLORE, weight=m.BOLD).move_to((act_p + leaf_p[1]) / 2 + [0.36, -0.05, 0]),
            code("0.1", 22, EXPLORE, weight=m.BOLD).move_to((act_p + leaf_p[2]) / 2 + [0.35, 0.18, 0]),
        )
        names = m.VGroup(code("charger", 20, REWARD), code("(3,3)", 20, STATE), code("(3,2)", 20, STATE))
        hows = m.VGroup(text("straight on", 20, MUTED), text("bump, stay", 20, MUTED), text("slip south", 20, MUTED))
        for n, h, q in zip(names, hows, leaf_p):
            n.move_to(q + [0, -0.45, 0])
            h.move_to(q + [0, -0.8, 0])
        self.tree = m.VGroup(stem, branches, root, root_lab, act, act_lab, leaves, probs, names, hows)
        self.tree_parts = dict(root=root, act=act, leaves=leaves, probs=probs, branches=branches, names=names)
        tree_cite = cite("backup diagram · Sutton & Barto, 2018, §3.5").move_to([3.3, 2.75, 0]).align_to([6.45, 0, 0], m.RIGHT)
        self.tree_cite = tree_cite

        flashes_up = [m.ShowPassingFlash(m.Line(q, act_p).set_stroke(VALUE, 8), time_width=0.8) for q in leaf_p]

        self.say(
            "[Rover stands just left of the charger], [and tries to move east.]",
            m.AnimationGroup(self.new_header("03 · slippery wheels", "A backup diagram"), m.FadeIn(grid),
                             m.FadeIn(bot, shift=0.15 * m.UP)),
            m.GrowArrow(east),
        )
        self.say(
            "[The wheels slip, so the move can end three ways]: [straight on, eight times in ten], "
            "[or sideways, one time in ten each, and north is a wall.]",
            m.Wiggle(bot, n_wiggles=4, run_time=0.9),
            m.Circumscribe(east, color=ACTION, buff=0.06),
            m.AnimationGroup(m.GrowArrow(north), m.Create(wall), m.GrowArrow(south)),
        )
        self.say(
            "[Draw it as a tree: the cell on top], [the move below it], [and every place Rover might land.]",
            m.AnimationGroup(m.FadeIn(root, scale=0.6), m.FadeIn(root_lab)),
            m.AnimationGroup(m.Create(stem), m.FadeIn(act, scale=0.5), m.FadeIn(act_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(b) for b in branches], lag_ratio=0.2),
                             m.LaggedStart(*[m.FadeIn(lf, scale=0.6) for lf in leaves], lag_ratio=0.2),
                             m.LaggedStart(*[m.FadeIn(p) for p in probs], lag_ratio=0.2),
                             m.FadeIn(names), m.FadeIn(hows)),
        )
        self.say(
            "[Values flow up this tree, from where Rover lands to where it stands]: [it's called a backup diagram.]",
            m.Succession(m.AnimationGroup(*flashes_up, run_time=0.9),
                         m.ShowPassingFlash(m.Line(act_p, root_p).set_stroke(VALUE, 8), time_width=0.8, run_time=0.6)),
            m.AnimationGroup(m.FadeIn(tree_cite), m.Circumscribe(root, color=VALUE, buff=0.08)),
        )
        self.wait(0.2)

    # ------------------------------------------------------------------ 4 · one cell, by hand

    def by_hand(self):
        self.next_section("4 · One cell, by hand", notes=(
            f"Values: value_iteration(ROVER_WORLD) ({SWEEPS} sweeps, θ = 1e-4; every cell equals R&N Fig. 17.3 to three "
            f"decimals). Q((3,3), E) = backup(world, V, (3,3), 'E') = 0.8 × (+1) + 0.1 × V(3,3) + 0.1 × V(3,2) − 0.04 = "
            f"0.8 + {TERMS[1]:.4f} + {TERMS[2]:.4f} − 0.04 = {HAND:.4f} → 0.918 = V(3,3) (asserted). The −0.04 is paid on "
            "every outcome, so it comes out of the sum once. Notes §6 worked example."))
        grid, tp = self.grid, self.tree_parts
        nums = grid.world_values(V, fmt="{:.3f}", offset=0.28)
        rc = W.to_screen(CELL)
        x0, ys = 1.45, [-0.95, -1.42, -1.89, -2.36]
        rows_src = ["0.8 × (+1)    0.800", "0.1 × 0.918   0.092", "0.1 × 0.660   0.066", "every move   −0.040"]
        rules = {r"^0\.[18]": EXPLORE, r"\(\+1\)": REWARD, r"(?<=× )0\.\d{3}": VALUE, r"−0\.040": REWARD,
                 r"every move": MUTED}
        rows = m.VGroup(*[code(s, 24, INK, t2c=rules).move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT)
                          for s, y in zip(rows_src, ys)])
        line = m.Line([x0, -2.66, 0], [x0 + rows[0].width + 0.1, -2.66, 0]).set_stroke(MUTED, 2)
        total = code("0.918", 30, VALUE, weight=m.BOLD).move_to([0, -3.05, 0]).align_to(rows[0], m.RIGHT)
        plus = code("+", 24, MUTED).next_to(line, m.LEFT, buff=0.15).shift(0.2 * m.UP)
        check = tick(total.get_right() + [0.45, 0.02, 0], 0.32)
        same = text("= the cell's value", 22, GOOD).next_to(total, m.LEFT, buff=0.35)
        ring = m.SurroundingRectangle(grid.squares[rc], buff=0.04, corner_radius=0.06).set_stroke(GOOD, 4)

        def pick(row, key):
            return glyphs(row, key)

        def rest(row, key):
            taken = {id(g) for g in glyphs(row, key)}
            return m.VGroup(*[g for g in row if id(g) not in taken])

        charger_lab = grid.marks[[i for i, (pos) in enumerate(grid.specials) if pos == W.to_screen(CHARGER)][0] + len(grid.walls)]

        self.say(
            "[Here are the values this world settles on]: [episode seven computes them. Let's check one.]",
            m.AnimationGroup(self.new_header("04 · by hand", "One cell, by hand"), m.FadeOut(self.bot), nums),
            m.Circumscribe(grid.squares[rc], color=VALUE, buff=0.04),
        )
        self.say(
            "[Eight times in ten, Rover reaches the charger]: [plus one, and the episode is over.]",
            m.AnimationGroup(m.Circumscribe(tp["probs"][0], color=EXPLORE, buff=0.08),
                             m.Circumscribe(self.slip_arrows[0], color=ACTION, buff=0.06)),
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.15 * m.UP),
                             m.Circumscribe(charger_lab, color=REWARD, buff=0.06)),
        )
        self.say(
            "[One time in ten, it slips north, bumps the wall], [and stays put: worth zero point nine one eight again.]",
            m.Circumscribe(tp["probs"][1], color=EXPLORE, buff=0.08),
            m.AnimationGroup(m.FadeIn(pick(rows[1], "0.918"), target_position=grid.numbers[rc]),
                             m.FadeIn(rest(rows[1], "0.918"), shift=0.15 * m.UP)),
        )
        rs = W.to_screen(SOUTH)
        self.say(
            "[And one time in ten, it slips south], [beside the puddle: zero point six six zero.]",
            m.Circumscribe(tp["probs"][2], color=EXPLORE, buff=0.08),
            m.AnimationGroup(m.FadeIn(pick(rows[2], "0.660"), target_position=grid.numbers[rs]),
                             m.FadeIn(rest(rows[2], "0.660"), shift=0.15 * m.UP)),
        )
        self.say(
            "[Whatever happens, the move itself costs zero point zero four.]",
            m.FadeIn(rows[3], shift=0.15 * m.UP),
        )
        self.say(
            "[Now add them up]: [zero point nine one eight], [exactly the value we started with.]",
            m.AnimationGroup(m.Create(line), m.FadeIn(plus)),
            m.FadeIn(total, shift=0.15 * m.UP),
            m.AnimationGroup(m.Create(check), m.FadeIn(same, shift=0.1 * m.RIGHT), m.Create(ring)),
        )
        self.say(
            "[That's all the Bellman equation says]: [every value agrees with the values next door.]",
            m.Circumscribe(m.VGroup(rows, total), color=VALUE, buff=0.12),
            m.LaggedStart(*[m.Indicate(grid.numbers[W.to_screen(s)], color=VALUE, scale_factor=1.15)
                            for s in (CHARGER, CELL, SOUTH) if W.to_screen(s) in grid.numbers], lag_ratio=0.3),
        )
        self.sum_parts = m.VGroup(rows, line, total, plus, check, same, ring)
        self.wait(0.2)

    # ------------------------------------------------------------------ 5 · four moves: Q

    def four_moves(self):
        self.next_section("5 · Q: the value of a move", notes=(
            f"q_from_v(world, V)[(3,3)] = {{{', '.join(f'{a}: {q:.4f}' for a, q in Q33.items())}}}. South lands on (3,2) "
            "8 times in 10; North bumps the wall 8 times in 10 (stays at 0.918) and slips east into the charger 1 in 10; "
            "the max is East, so V(3,3) = max_a Q((3,3), a) = 0.918 (Bellman optimality). Q(s, a): the value of taking a "
            "in s, then playing well (S&B eq. 3.17)."))
        grid = self.grid
        rc = W.to_screen(CELL)
        q_head = m.VGroup(m.MathTex("Q(s, a)", font_size=48, color=VALUE),
                          text("move a in cell s, then play well", 22, MUTED))
        q_head.arrange(m.DOWN, buff=0.18).move_to([3.35, 1.95, 0])
        acts = list(W.actions)
        bars = Bars([Q33[a] for a in acts], labels=acts, top=1.0, bottom=0, width=4.6, height=2.1,
                    center=(3.35, -0.2, 0), color=VALUE, gap=0.45, fmt="{:.3f}", size=22)
        for lab in bars.labels:
            lab.set_color(ACTION)
        vmax = m.MathTex(r"V(s)", "=", r"\max_{a}", r"Q(s, a)", font_size=46, color=INK)
        vmax[0].set_color(VALUE)
        vmax[2].set_color(ACTION)
        vmax[3].set_color(VALUE)
        vmax.move_to([3.35, -2.75, 0])
        best = grid.world_arrows({CELL: "E"}, length=0.42, offset=0.1)
        i_n, i_e, i_s = acts.index("N"), acts.index("E"), acts.index("S")

        def bar(i):
            return m.VGroup(bars.bars[i], bars.numbers[i])

        self.say(
            "[That sum scored one move in one cell]: [it's called Q of s and a], [the value of making that move, then playing well.]",
            m.AnimationGroup(self.new_header("05 · four moves", "Q: the value of a move"),
                             m.FadeOut(m.VGroup(self.tree, self.tree_cite, self.sum_parts, self.slip_arrows))),
            m.FadeIn(q_head[0], shift=0.1 * m.UP),
            m.FadeIn(q_head[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Now score all four moves from the same cell, each from its own three landings.]",
            m.AnimationGroup(m.FadeIn(bars.zero), m.FadeIn(bars.labels),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars.bars], lag_ratio=0.25),
                             m.LaggedStart(*[m.FadeIn(n) for n in bars.numbers], lag_ratio=0.25), run_time=2.0),
        )
        self.say(
            "[South is the worst, zero point six seven five]: [eight times in ten, it lands beside the puddle.]",
            m.Circumscribe(bar(i_s), color=ACTION, buff=0.08),
            m.Circumscribe(grid.squares[W.to_screen(SOUTH)], color=EXPLORE, buff=0.04),
        )
        self.say(
            "[Even north, straight into the wall, scores zero point eight eight one]: "
            "[the bump keeps Rover here, and a slip east still reaches the charger.]",
            m.Circumscribe(bar(i_n), color=ACTION, buff=0.08),
            m.AnimationGroup(m.Circumscribe(grid.squares[rc], color=EXPLORE, buff=0.04),
                             m.Circumscribe(grid.squares[W.to_screen(CHARGER)], color=REWARD, buff=0.04)),
        )
        self.say(
            "[East is best, at zero point nine one eight], [and that's the cell's value]: [the value of its best move.]",
            m.AnimationGroup(m.Circumscribe(bar(i_e), color=ACTION, buff=0.08), m.GrowArrow(best[0]),
                             bars.bars[i_e].animate.set_stroke(ACTION, 3)),
            m.Circumscribe(grid.numbers[rc], color=VALUE, buff=0.06),
            m.FadeIn(vmax, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the optimality equation

    def optimality(self):
        self.next_section("6 · Bellman optimality", notes=(
            "Bellman optimality equation, S&B 2018 eq. 3.19: V*(s) = max_a Σ_{s′,r} p(s′, r | s, a)[r + γ V*(s′)]; "
            "γ = 1 in Rover's world. Its backup diagram (S&B Fig. 3.4) has an arc at the action level: max, not average. "
            "S&B §1.7: the optimal return function defines 'a functional equation, now often called the Bellman equation', "
            "and the methods solving it 'came to be known as dynamic programming (Bellman, 1957a)': Richard Bellman, "
            "Dynamic Programming, Princeton University Press, 1957."))
        eq = m.MathTex(r"V^{*}(s)", "=", r"\max_{a}", r"\sum_{s'}", r"p(s' \mid s, a)", r"\big[", "r", "+", r"\gamma",
                       r"V^{*}(s')", r"\big]", font_size=56, color=INK)
        for i, col in ((0, VALUE), (2, ACTION), (4, EXPLORE), (6, REWARD), (8, VALUE), (9, VALUE)):
            eq[i].set_color(col)
        eq.move_to([0, 1.55, 0])

        def item(sym, words, color):
            return m.VGroup(m.MathTex(sym, font_size=36, color=color), text(words, 24, INK)).arrange(m.RIGHT, buff=0.25)

        legend = m.VGroup(item(r"\max_{a}", "the best move", ACTION),
                          item(r"p(s' \mid s, a)", "the chance of each landing", EXPLORE),
                          item("r", "the reward now", REWARD),
                          item(r"\gamma V^{*}(s')", "the value of where you land", VALUE))
        lx = [-5.6, 0.6]
        for k, it in enumerate(legend):
            it.move_to([0, 0.25 - 0.72 * (k // 2), 0]).align_to([lx[k % 2], 0, 0], m.LEFT)
        gamma_note = cite("γ = 1 in Rover's world").next_to(legend[3], m.DOWN, buff=0.12).align_to(legend[3][1], m.LEFT)
        targets = [eq[2], eq[4], eq[6], eq[8:10]]
        tree = mini_tree((-3.6, -2.35, 0), arc=True)
        tree_lab = cite("optimality: max over the moves").next_to(tree, m.DOWN, buff=0.15)
        name = title_text("Richard Bellman", 34, INK).move_to([3.4, -1.75, 0])
        book = cite("Bellman, 1957 · Dynamic Programming").next_to(name, m.DOWN, buff=0.2)
        sb = cite("Sutton & Barto, 2018, eq. 3.19").next_to(book, m.DOWN, buff=0.12)

        self.say(
            "[Write that down for every cell, and you have the Bellman optimality equation.]",
            m.AnimationGroup(self.new_header("06 · the equation", "Bellman optimality"),
                             m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in eq], lag_ratio=0.08)),
        )
        self.say(
            "[The best move], [the chance of each landing], [the reward now], "
            "[and the value of where you land, times gamma, which is one here.]",
            *[m.AnimationGroup(m.FadeIn(it, shift=0.1 * m.UP), m.Circumscribe(t, color=it[0].get_color(), buff=0.07))
              for it, t in zip(legend[:3], targets[:3])],
            m.AnimationGroup(m.FadeIn(legend[3], shift=0.1 * m.UP), m.FadeIn(gamma_note),
                             m.Circumscribe(targets[3], color=VALUE, buff=0.07)),
        )
        self.say(
            "[Its backup diagram has an arc across the moves]: [take the best one, not the average.]",
            m.AnimationGroup(m.FadeIn(tree[0]), m.FadeIn(tree_lab)),
            m.AnimationGroup(m.Create(tree[1][0]), m.FadeIn(tree[1][1])),
        )
        self.say(
            "[It's named after Richard Bellman], [whose book, Dynamic Programming, came out in nineteen fifty-seven.]",
            m.FadeIn(name, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(book, shift=0.1 * m.UP), m.FadeIn(sb)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · policy evaluation

    def evaluation(self):
        self.next_section("7 · Policy evaluation", notes=(
            "Bellman expectation equation, S&B 2018 eq. 3.14: V^π(s) = Σ_a π(a|s) Σ p(s′, r|s, a)[r + γ V^π(s′)]. "
            "policy_evaluation(ROVER_WORLD, uniform random policy, θ = 1e-10): "
            f"V(3,3) = {V_RAND[CELL]:.4f} (and V(1,1) = {V_RAND[START]:.3f}, notes §5); check: V(3,3) equals the average of "
            f"its four Q's under that V. From (3,3) the random Rover ends in the puddle with probability {P_PUDDLE:.3f} after "
            f"{N_MOVES:.2f} moves on average (policy evaluation of two helper worlds)."))
        eq = m.MathTex(r"V^{\pi}(s)", "=", r"\sum_{a} \pi(a \mid s)", r"\sum_{s'}", r"p(s' \mid s, a)", r"\big[", "r", "+",
                       r"\gamma", r"V^{\pi}(s')", r"\big]", font_size=50, color=INK)
        for i, col in ((0, VALUE), (2, ACTION), (4, EXPLORE), (6, REWARD), (8, VALUE), (9, VALUE)):
            eq[i].set_color(col)
        eq.move_to([0, 1.6, 0])
        avg_lab = text("average over the policy's moves", 22, ACTION).next_to(eq[2], m.DOWN, buff=0.3)
        sb = cite("Sutton & Barto, 2018, eq. 3.14").move_to([0, 0.6, 0]).align_to([6.45, 0, 0], m.RIGHT)
        tree = mini_tree((-4.75, -1.75, 0), weights="1/4 each", width=2.6, height=1.65)
        tree_lab = cite("expectation: average").next_to(tree, m.DOWN, buff=0.15)

        grid = GridView.from_world(W, cell=0.95, center=(0.1, -1.75, 0))
        rand_vals = grid.world_values(V_RAND, fmt="{:.2f}", size=20)
        rc = W.to_screen(CELL)
        x_r = 2.75
        bot = rover(0.42).move_to([x_r + 0.3, -0.55, 0])
        dice = icon(DICE, 0.42).next_to(bot, m.RIGHT, buff=0.25)
        who = text("random Rover", 22, MUTED).next_to(dice, m.RIGHT, buff=0.22)
        stat_r = code(minus(f"V(3,3) = {V_RAND[CELL]:.3f}"), 28, VALUE, weight=m.BOLD).move_to([0, -1.35, 0]).align_to([x_r, 0, 0], m.LEFT)
        best_r = code(f"best moves: {V[CELL]:.3f}", 22, MUTED).next_to(stat_r, m.DOWN, buff=0.2).align_to([x_r, 0, 0], m.LEFT)
        wander = text(f"about {N_MOVES:.0f} moves on average", 22, MUTED).next_to(best_r, m.DOWN, buff=0.35).align_to([x_r, 0, 0], m.LEFT)
        puddle = text(f"ends in the puddle: {P_PUDDLE:.0%}", 22, RED).next_to(wander, m.DOWN, buff=0.12).align_to([x_r, 0, 0], m.LEFT)
        pe = pill("policy evaluation", ACTION, CORAL_FILL, 20).next_to(grid, m.UP, buff=0.2)

        self.say(
            "[Now take a Rover that follows one fixed rule instead], [say, picking every move at random.]",
            m.AnimationGroup(self.new_header("07 · a fixed policy", "Policy evaluation"), m.FadeIn(bot, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(dice, scale=0.6), m.FadeIn(who)),
        )
        self.say(
            "[Swap the max for an average over its moves]: [that's the Bellman expectation equation.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in eq], lag_ratio=0.06),
                             m.FadeIn(avg_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(tree), m.FadeIn(tree_lab), m.FadeIn(sb), m.FadeIn(grid)),
        )
        self.say(
            "[Solving it for one policy is called policy evaluation]: [here are the random Rover's values.]",
            m.FadeIn(pe, shift=0.1 * m.DOWN),
            rand_vals,
        )
        self.say(
            "[Even beside the charger, it's worth minus zero point three one five]: "
            "[it wanders for about thirteen moves, and four times in ten it ends in the puddle.]",
            m.AnimationGroup(m.Circumscribe(grid.squares[rc], color=VALUE, buff=0.04), m.FadeIn(stat_r, shift=0.1 * m.UP),
                             m.FadeIn(best_r)),
            m.AnimationGroup(m.FadeIn(wander, shift=0.1 * m.UP), m.FadeIn(puddle, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Now (the reward now: −0.04 a move, +1 on entering the charger), next (the value of where you land, weighted "
            "by the slip chances 0.8 / 0.1 / 0.1), best (the max over the four moves). Cell (3,3), East: 0.800 + 0.092 + "
            "0.066 − 0.040 = 0.918 = V(3,3)."))
        halo = glow(STATE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Now", 70, REWARD), title_text("Next", 70, VALUE), title_text("Best", 70, ACTION))
        words.arrange(m.RIGHT, buff=1.9).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("reward now", 24, MUTED), text("value of where you land", 24, MUTED),
                        text("best move", 24, MUTED))
        for s, w in zip(subs, words):
            s.next_to(w, m.DOWN, buff=0.4)
        base = max(s.get_center()[1] for s in subs)
        for s in subs:
            s.set_y(base)
        proof = code("(3,3), east: 0.800 + 0.092 + 0.066 − 0.040 = 0.918", 24, INK,
                     t2c={r"0\.918$": VALUE, r"−0\.040": REWARD, r"east": ACTION})
        proof.move_to([0, -1.35, 0])
        mascot = rover(0.7).move_to([0, -2.75, 0])

        self.say(
            "[To sum up]: [the reward now], [plus the value of where you land], [for the best move.]",
            m.AnimationGroup(self.new_header("08 · recap", "Now, next, best"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Next to the charger, that gave zero point nine one eight], [by heading east.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · the catch

    def catch(self):
        self.next_section("9 · Defined by each other", notes=(
            "Every one of the 9 non-terminal cells satisfies V(s) = max_a Q(s, a) to three decimals (asserted), e.g. the "
            "start (1,1): 0.705 by heading North (notes Appendix C). But (3,3)'s value uses (3,2)'s, and (3,2)'s best move "
            "(North) lands on (3,3) eight times in ten: the values are defined by each other (9 equations, 9 unknowns, "
            "non-linear because of the max). Double arrows: the 9 pairs of neighbouring open cells. Episode 7 (value "
            "iteration) solves it by repeating the backup from zeros until nothing changes."))
        grid = GridView.from_world(W, cell=1.35, center=(-2.75, -0.45, 0))
        nums = static_values(grid, V, size=20)
        ticks = m.VGroup(*[tick(grid.at(s) + [0.43, 0.42, 0], 0.2, width=4) for s in OPEN])
        x0 = 1.0
        checked = m.VGroup(text(f"{len(OPEN)} of {len(OPEN)} cells check out", 28, GOOD, weight=m.BOLD),
                           text("each value = the backup of its best move", 22, MUTED))
        checked.arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT).move_to([0, 1.3, 0]).align_to([x0, 0, 0], m.LEFT)

        c, s_ = grid.at(CELL), grid.at(SOUTH)
        down = m.ArcBetweenPoints(c + [0.5, -0.22, 0], s_ + [0.5, 0.22, 0], angle=-0.9).set_stroke(VALUE, 4)
        up = m.ArcBetweenPoints(s_ + [-0.5, 0.22, 0], c + [-0.5, -0.22, 0], angle=-0.9).set_stroke(VALUE, 4)
        for arc_ in (down, up):
            arc_.add_tip(tip_length=0.16)
            arc_.set_color(VALUE)
        uses = m.VGroup(code("0.918 uses 0.660", 26, VALUE), code("0.660 uses 0.918", 26, VALUE))
        uses.arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([0, -0.05, 0]).align_to([x0, 0, 0], m.LEFT)

        web = m.VGroup()
        for a, b in PAIRS:
            pa, pb = grid.at(a), grid.at(b)
            d = (pb - pa) / np.linalg.norm(pb - pa)
            mid = (pa + pb) / 2
            half = 0.27 if abs(d[0]) > 0.5 else 0.3
            shift = np.array([0, 0.34, 0]) if abs(d[0]) > 0.5 else np.zeros(3)
            web.add(m.DoubleArrow(mid - d * half + shift, mid + d * half + shift, buff=0, color=VALUE, stroke_width=3,
                                  tip_length=0.12, max_tip_length_to_length_ratio=0.3))
        loop = m.ArcBetweenPoints(c + [-0.2, 0.36, 0], c + [0.2, 0.36, 0], angle=-4.0).set_stroke(EXPLORE, 3)
        loop.add_tip(tip_length=0.13)
        loop.set_color(EXPLORE)
        unknowns = {s: code("?", 30, MUTED, weight=m.BOLD).move_to(grid.at(s)) for s in OPEN}
        nine = m.VGroup(text("9 values, 9 equations", 28, INK, weight=m.BOLD),
                        text("each one defined by the others", 22, MUTED))
        nine.arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT).move_to([0, -1.45, 0]).align_to([x0, 0, 0], m.LEFT)
        bot = rover(0.6).move_to([2.1, -2.85, 0])
        qmark = title_text("?", 64, EXPLORE).next_to(bot, m.RIGHT, buff=0.3)

        self.say(
            "[Back to the best values.] [Every cell passes the same check]: [its value is the backup of its best move.]",
            m.AnimationGroup(self.new_header("09 · the catch", "Defined by each other"), m.FadeIn(grid), m.FadeIn(nums)),
            m.LaggedStart(*[m.Create(t) for t in ticks], lag_ratio=0.15),
            m.FadeIn(checked, shift=0.1 * m.UP),
        )
        self.say(
            "[But look at what the check used]: [zero point nine one eight needs zero point six six zero], "
            "[and zero point six six zero needs zero point nine one eight.]",
            m.FadeOut(ticks),
            m.AnimationGroup(m.Create(up), m.FadeIn(uses[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(down), m.FadeIn(uses[1], shift=0.1 * m.UP)),
        )
        self.say(
            "[In fact, every value leans on its neighbours], [and a bump makes it lean on itself.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(up, down)), m.LaggedStart(*[m.GrowFromCenter(a) for a in web], lag_ratio=0.12)),
            m.Create(loop),
        )
        self.say(
            "[Wipe the numbers away], [and you're left with nine unknowns, each defined by the others.]",
            m.AnimationGroup(*[m.Transform(n, unknowns[s]) for n, s in zip(nums, [s for s in V if s in OPEN])],
                             *[grid.squares[W.to_screen(s)].animate.set_fill(CARD, 1) for s in OPEN]),
            m.FadeIn(nine, shift=0.1 * m.UP),
        )
        self.say(
            "[So how do you solve for all nine at once?]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(qmark, scale=0.6)),
        )
        self.wait(0.4)
        self.clear()
