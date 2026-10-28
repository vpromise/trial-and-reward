"""Trial and Reward · Episode 1 · The Reward Loop (Module 1 · Trial and Reward)

The series opener. Rover, the course's learner, meets its first slot machine: nobody tells it what to do, and a
reward only arrives after it acts. The episode builds the agent–environment loop (state → action → reward + next
state, turning), names its seven words (agent, environment, state, action, reward, policy, return), sets reward
apart from a supervised label (evaluative, not instructive; Rover must explore), then lets a reward arrive late in
Rover's grid world and travel back along the trail (credit assignment), and names the course's two questions:
which action was good, and what haven't I tried? A dated strip runs TD-Gammon → DQN → AlphaGo → InstructGPT →
reasoning models. The payoff is the course map: one loop, six worlds. It ends on Episode 2's question: which of
five machines should Rover pull?

Every number on screen is computed below from rl_reference.py and asserted, or cited with its source.

Sources (research/notes.md §1, Appendix C; facts dated October 2026):
- Sutton & Barto 2018 §1.1 (definition; "trial-and-error search and delayed reward"; evaluative vs instructive
  feedback, §2 intro), §3.1 (S_0, A_0, R_1, … notation; the agent is "the learner and decision maker", the
  environment "everything outside the agent").
- Timeline (notes §1): TD-Gammon (Tesauro 1992, Machine Learning 8; CACM 1995: self-play backgammon); DQN (Mnih
  et al., Nature 518, 26 Feb 2015: 49 Atari games from pixels and the game score, one recipe); AlphaGo beat Lee
  Sedol 4–1 in Seoul, 9–15 Mar 2016 (Silver et al., Nature 529, 2016; DeepMind AlphaGo page); InstructGPT (Ouyang
  et al., arXiv 2203.02155, 4 Mar 2022: RLHF); ChatGPT 30 Nov 2022 ("trained … using RLHF", OpenAI blog);
  OpenAI o1 12 Sep 2024 ("trained with reinforcement learning to perform complex reasoning"); DeepSeek-R1 20–22
  Jan 2025 (rule-based accuracy and format rewards).
- 'AI, Explained' Episode 8 (Four Ways to Learn) is linked on screen.

Computed here (rl_reference): Rover's first pulls of machine 1 (arm 0, ROVER_ARMS[0] = 0.25), seed 11: +1, then 0;
machine 2 (arm 1) next: 0. The grid episode: play_episode(ROVER_WORLD, RN_POLICY, seed 11) goes
(1,1) → (1,2) → (1,3) → (2,3) → (3,3) → (4,3) in five moves with no slip; rewards −0.04 × 4 and +0.96
(+1 charger − 0.04 step); returns with γ = 1 (discounted_returns): 0.80, 0.84, 0.88, 0.92, 0.96.
Illustrative (said "for example" or plainly schematic): the spam e-mail and its label; the generic loop's mini world
and the state marker's path; the cart-and-pole wobble (a simple push-toward-the-lean rule run through
cartpole_step, not a trained policy); the tokens on the language-model card.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- Rover's first pulls (bandit)

BANDIT = BernoulliBandit(ROVER_ARMS)
_rng = np.random.default_rng(11)
HOOK_PULLS = [BANDIT.pull(0, _rng) for _ in range(2)]        # machine 1, twice
LEVER_TWO = BANDIT.pull(1, _rng)                            # then machine 2, once
assert HOOK_PULLS == [1.0, 0.0]                             # "plus one" … "zero"
assert LEVER_TWO == 0.0                                     # "hears only: zero"
assert len(ROVER_ARMS) == 5 and ROVER_BEST_ARM == 3          # "five machines"

# ---------------------------------------------------------------------------- the late reward (Rover's world)

EPISODE = play_episode(ROVER_WORLD, RN_POLICY, np.random.default_rng(11))
TRAIL = [t.s for t in EPISODE] + [EPISODE[-1].s2]
MOVES_TAKEN = [t.a for t in EPISODE]
REWARDS = [t.r for t in EPISODE]
RETURNS = discounted_returns(REWARDS, gamma=ROVER_GAMMA)
assert TRAIL == [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3)] and len(EPISODE) == 5   # "five moves"
assert MOVES_TAKEN == ["N", "N", "E", "E", "E"]                                          # "up and along"
assert ROVER_WORLD.step_reward == -0.04                                                   # "minus zero point zero four"
assert ROVER_WORLD.terminals[CHARGER] == 1.0 and ROVER_WORLD.terminals[PUDDLE] == -1.0     # "plus one", "minus one"
assert np.allclose(REWARDS, [-0.04] * 4 + [0.96]) and ROVER_GAMMA == 1.0
assert [f"{g:.2f}" for g in RETURNS] == ["0.80", "0.84", "0.88", "0.92", "0.96"]          # "zero point eight"
VISITED = set(TRAIL)
UNVISITED = [s for s in ROVER_WORLD.states if s not in VISITED and not ROVER_WORLD.is_terminal(s)]
assert sorted(UNVISITED) == [(2, 1), (3, 1), (3, 2), (4, 1)]

# ---------------------------------------------------------------------------- the six worlds

assert CLIFF_WORLD.cliff_reward == -100.0 and CLIFF_WORLD.step_reward == -1.0              # "minus one hundred"
assert (CLIFF_WORLD.width, CLIFF_WORLD.height) == (12, 4)
assert cartpole_step(np.zeros(4), 1)[1] == 1.0                                            # "+1 every step upright"
assert TicTacToe.winner(MIDGAME) is None
assert EPISODES[2] == ("Explore or Exploit", 1)


def wobble(steps=90, seed=3):
    """A cart and pole kept up by a simple rule (push toward the lean), for the card's art: not a trained policy."""
    s = cartpole_reset(np.random.default_rng(seed))
    s[2] = 0.06
    rows = [s]
    for _ in range(steps):
        a = 1 if s[2] + 0.5 * s[3] > 0 else 0
        s, _, done = cartpole_step(s, a)
        rows.append(s)
        if done:
            break
    return np.array(rows)


WOBBLE = wobble()
assert len(WOBBLE) == 91                                   # the rule keeps it up for the whole clip

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
DOC = recolored("doc", MUTED, CARD_HI, prefix="ep01")
USER = recolored("user", MUTED, CARD_HI, prefix="ep01")
MAIL = recolored("mail", MUTED, CARD_HI, prefix="ep01")
DICE = recolored("dice", BLUE, BLUE_FILL, prefix="ep01")
PAD = recolored("gamepad", VIOLET, VIOLET_FILL, prefix="ep01")
PEOPLE = recolored("user", PINK, PINK_FILL, prefix="ep01")
MODEL = recolored("llm", PINK, PINK_FILL, prefix="ep01")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words, s):
    """The baseline of a Text made from s: the median bottom of its flat-bottomed letters."""
    chars = [c for c in s if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, s, y):
    """Put the baseline of Text `words` (made from s) at height y."""
    return words.shift((y - baseline(words, s)) * m.UP)


def sym(tex_str, color, width=0.82, height=0.64):
    """One step of a trajectory (S_0, A_0, R_1 …) as a small chip in its colour."""
    box = card(width, height, color, FILLS.get(color, CARD), radius=0.12, stroke=2.2)
    return m.VGroup(box, m.MathTex(tex_str, font_size=34, color=INK).move_to(box))


def crossed(name, label):
    pic = icon(name, 0.75)
    slash = m.Line(pic.get_corner(m.UL) + [-0.05, 0.05, 0], pic.get_corner(m.DR) + [0.05, -0.05, 0]).set_stroke(RED, 6)
    words = text(label, 22, MUTED).next_to(pic, m.DOWN, buff=0.18)
    return m.VGroup(pic, slash, words)


def pull(machine, run_time=0.6):
    """SlotMachine.pull() rotates about the pivot where the machine was BUILT; this one uses where it is now."""
    return m.Rotate(machine.lever, angle=-1.1, about_point=machine.lever[0].get_start(), rate_func=m.there_and_back,
                    run_time=run_time)


def badge(n, color):
    disc = m.Circle(radius=0.21).set_stroke(color, 2).set_fill(color, 1)
    return m.VGroup(disc, code(str(n), 22, BG, weight=m.BOLD).move_to(disc))


def loop_glyph(color=MUTED):
    arc = m.Arc(radius=0.16, start_angle=np.pi / 2 + 0.5, angle=2 * np.pi - 1.0).set_stroke(color, 3)
    arc.add_tip(tip_length=0.11)
    return arc


def go_board(color):
    size, n = 0.5, 4
    lines = m.VGroup()
    for i in range(n):
        f = -size / 2 + size * i / (n - 1)
        lines.add(m.Line([-size / 2, f, 0], [size / 2, f, 0]), m.Line([f, -size / 2, 0], [f, size / 2, 0]))
    lines.set_stroke(color, 2)
    step = size / (n - 1)
    stones = m.VGroup(
        m.Dot([-step / 2 * 1, step / 2, 0], radius=0.075, color=INK),
        m.Dot([step / 2, -step / 2, 0], radius=0.075, color=INK),
        m.Dot([step / 2, step / 2, 0], radius=0.075).set_fill(BG, 1).set_stroke(INK, 2),
    )
    return m.VGroup(lines, stones)


# ---------------------------------------------------------------------------- the episode


class Ep01TheRewardLoop(Episode):
    number = 1

    def lesson(self):
        self.hook()
        self.the_loop()
        self.seven_words()
        self.no_labels()
        self.credit()
        self.history()
        self.course_map()
        self.next_question()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Rover meets its first slot machine (machine 1 of the five, ROVER_ARMS[0] = {ROVER_ARMS[0]}). Nobody tells it what to do; it pulls and gets a reward only after acting. The two pulls are BernoulliBandit(ROVER_ARMS).pull(0, rng) with rng seed 11: {HOOK_PULLS}. Definition on screen: Sutton & Barto 2018, §1.1.")
        floor = m.Line([-6.2, -2.0, 0], [6.2, -2.0, 0]).set_stroke(EDGE, 3)
        machine = SlotMachine("1", size=1.5)
        machine.move_to([-1.6, 0, 0]).align_to([0, -2.0, 0], m.DOWN)
        bot = rover(1.25)
        bot.move_to([4.4, 0, 0]).align_to([0, -2.0, 0], m.DOWN)
        ask = title_text("?", 76, VIOLET).move_to(bot.get_top() + [0, 0.65, 0])
        manual = crossed(DOC, "no manual").move_to([0.75, 1.75, 0])
        teacher = crossed(USER, "no teacher").move_to([2.55, 1.75, 0])
        spot = bot.copy().move_to([1.0, bot.get_y(), 0])
        name = title_text("Rover", 54, CORAL)
        role = text("learns only from reward", 26, MUTED)
        tag = m.VGroup(name, role).arrange(m.DOWN, buff=0.18).move_to([0.2, -0.65, 0])

        strip_lab = code("rewards", 22, MUTED).move_to([-2.0, -2.75, 0])
        slots = [np.array([strip_lab.get_right()[0] + 0.85 + 1.2 * i, -2.75, 0]) for i in range(2)]
        coins = [reward_chip(r, 24).move_to(p) for r, p in zip(HOOK_PULLS, slots)]
        shown = [machine.payout(r, 24) for r in HOOK_PULLS]

        self.say(
            "[Meet Rover]: [a small robot that learns only from reward.]",
            m.AnimationGroup(self.new_header("00 · hook", "Rover meets a slot machine"), m.Create(floor),
                             m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(name, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(role, shift=0.1 * m.UP),
                             m.Flash(bot[3][1].get_center(), color=GOLD, flash_radius=0.3, line_length=0.2)),
        )
        self.say(
            "[Today it meets its first slot machine.] [No manual, no teacher]: [nobody tells Rover what to do.]",
            m.AnimationGroup(m.FadeOut(tag, shift=0.2 * m.UP), m.FadeIn(machine, shift=0.2 * m.UP)),
            m.LaggedStart(m.FadeIn(manual, shift=0.15 * m.UP), m.FadeIn(teacher, shift=0.15 * m.UP), lag_ratio=0.3),
            m.FadeIn(ask, scale=0.6),
        )
        self.say(
            "[So it tries something]: [it pulls the lever], [and a coin drops out: plus one.]",
            m.AnimationGroup(m.FadeOut(manual), m.FadeOut(teacher), m.FadeOut(ask), bot.animate.move_to(spot), run_time=1.1),
            pull(machine, 0.8),
            m.Succession(
                m.FadeIn(shown[0], scale=0.6, run_time=0.4),
                m.AnimationGroup(m.FadeIn(strip_lab), m.FadeIn(coins[0], target_position=shown[0]), run_time=0.7),
                m.AnimationGroup(m.FadeOut(shown[0], run_time=0.4),
                                 m.Flash(spot[3][1].get_center(), color=GOLD, flash_radius=0.3, line_length=0.2)),
            ),
        )
        self.say(
            "[It pulls again]: [nothing this time, zero.] [The reward only comes after it acts.]",
            pull(machine, 0.8),
            m.Succession(m.FadeIn(shown[1], scale=0.6, run_time=0.4),
                         m.AnimationGroup(m.FadeIn(coins[1], target_position=shown[1]), m.FadeOut(shown[1]), run_time=0.7)),
            m.Circumscribe(m.VGroup(strip_lab, *coins), color=GOLD, buff=0.12),
        )
        self.wait(0.2)
        halo = glow(GOLD, radius=4.0).move_to(0.4 * m.UP)
        big = title_text("Learning what to do", 64, INK).move_to([0, 1.25, 0])
        small = title_text("from the consequences of doing it", 46, GOLD).next_to(big, m.DOWN, buff=0.3)
        quote = m.VGroup(
            text("“Reinforcement learning is learning what to do, how to map situations", 22, MUTED),
            text("to actions, so as to maximize a numerical reward signal.”", 22, MUTED),
        ).arrange(m.DOWN, buff=0.14).next_to(small, m.DOWN, buff=0.7)
        source = cite("Sutton & Barto, 2018, §1.1").next_to(quote, m.DOWN, buff=0.25)
        scene_bits = m.VGroup(floor, machine, bot, strip_lab, *coins)
        self.say(
            "[That's reinforcement learning]: [learning what to do], [from the consequences of doing it.]",
            m.AnimationGroup(m.FadeOut(scene_bits, shift=0.3 * m.DOWN), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(big, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(small, shift=0.2 * m.UP), m.FadeIn(quote), m.FadeIn(source), lag_ratio=0.3),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ the loop

    def the_loop(self):
        self.next_section("1 · The loop", notes="The agent–environment loop (Sutton & Barto 2018, §3.1): in state S_t the agent takes A_t, then receives R_{t+1} and lands in S_{t+1}; the trajectory S_0, A_0, R_1, S_1, A_1, R_2, … (eq. 3.1). The mini world inside the right card and the TEAL marker's path are schematic.")
        loop = agent_world_loop(center=(0, 0.85, 0), width=9.0, size=1.1)
        mini = loop.world[1]
        cells = [mini[2][0], mini[1][0], mini[0][0], mini[0][1]]
        marker = m.Square(mini[0][0].width).set_stroke(TEAL, 3).set_fill(TEAL, 0.55).move_to(cells[0]).set_z_index(2)
        steps = [("S_0", STATE), ("A_0", ACTION), ("R_1", REWARD), ("S_1", STATE), ("A_1", ACTION), ("R_2", REWARD),
                 ("S_2", STATE), ("A_2", ACTION), ("R_3", REWARD), ("S_3", STATE)]
        chips = m.VGroup(*[sym(s, c) for s, c in steps]).arrange(m.RIGHT, buff=0.14)
        more = m.VGroup(*[m.Dot(radius=0.05, color=MUTED) for _ in range(3)]).arrange(m.RIGHT, buff=0.12)
        more.next_to(chips, m.RIGHT, buff=0.25)
        row = m.VGroup(chips, more).move_to([0, -1.75, 0])
        lab = code("trajectory", 22, MUTED).next_to(row, m.UP, buff=0.3).align_to(row, m.LEFT)
        src = cite("Sutton & Barto, 2018, §3.1").next_to(row, m.DOWN, buff=0.3).align_to(row, m.RIGHT)
        arrow_of = lambda link: link[0]
        enter = lambda k, link: m.FadeIn(chips[k], target_position=arrow_of(link), scale=0.6)

        self.say(
            "[Every problem in this course has the same shape]: [an agent], [and a world.]",
            self.new_header("01 · the loop", "Agent, world, and a loop"),
            m.FadeIn(loop.agent, shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(loop.world, shift=0.2 * m.LEFT), m.FadeIn(marker)),
        )
        self.say(
            "[The world is in some state], [and Rover picks an action.]",
            m.AnimationGroup(m.GrowArrow(loop.state[0]), m.FadeIn(loop.state[1]), m.FadeIn(lab), enter(0, loop.state),
                             m.Indicate(marker, color=TEAL, scale_factor=1.3)),
            m.AnimationGroup(m.GrowArrow(loop.action[0]), m.FadeIn(loop.action[1]), enter(1, loop.action)),
        )
        self.say(
            "[The world answers with a reward], [and a new state.]",
            m.AnimationGroup(m.GrowArrow(loop.reward[0]), m.FadeIn(loop.reward[1]), enter(2, loop.reward)),
            m.AnimationGroup(pulse(loop.state[0], STATE), marker.animate.move_to(cells[1]), enter(3, loop.state)),
        )

        def turn(k, cell):
            return m.Succession(
                m.AnimationGroup(pulse(loop.action[0], ACTION), enter(k, loop.action), run_time=0.7),
                m.AnimationGroup(pulse(loop.reward[0], REWARD), pulse(loop.state[0], STATE), enter(k + 1, loop.reward),
                                 enter(k + 2, loop.state), marker.animate.move_to(cell), run_time=0.8),
            )

        self.say(
            "[Then Rover acts again], [and the loop keeps turning], [step after step.]",
            m.AnimationGroup(pulse(loop.action[0], ACTION), enter(4, loop.action)),
            m.Succession(
                m.AnimationGroup(pulse(loop.reward[0], REWARD), pulse(loop.state[0], STATE), enter(5, loop.reward),
                                 enter(6, loop.state), marker.animate.move_to(cells[2]), run_time=0.8),
                turn(7, cells[3]),
            ),
            m.AnimationGroup(m.FadeIn(more, shift=0.1 * m.RIGHT), m.FadeIn(src)),
        )
        self.wait(0.5)
        self.diagram = m.VGroup(loop, marker)
        self.loop = loop
        self.leftover = m.VGroup(chips, more, lab, src)

    # ------------------------------------------------------------------ seven words

    def seven_words(self):
        self.next_section("2 · Seven words", notes="One line each, with the loop as the picture. Agent: 'the learner and decision maker'; environment: 'everything outside the agent' (Sutton & Barto §3.1). Policy π(a | s): the probability of action a in state s. Return G_t = R_{t+1} + R_{t+2} + … (S&B eq. 3.8 with γ = 1; discounting is Episode 5).")
        loop, diagram = self.loop, self.diagram
        terms = [("agent", ACTION, "the learner that decides"), ("environment", STATE, "everything outside the agent"),
                 ("state", STATE, "what it can see right now"), ("action", ACTION, "a choice it makes"),
                 ("reward", REWARD, "one number: how good it was"), ("policy", POLICY, "its rule for choosing actions"),
                 ("return", VALUE, "all the rewards still to come")]
        rows = []
        for i, (term, color, gloss) in enumerate(terms):
            y = 2.15 - 0.75 * i
            t = code(term, 24, color, weight=m.BOLD).move_to([0, y, 0]).align_to([0.35, 0, 0], m.LEFT)
            g = text(gloss, 22, INK).move_to([0, y, 0]).align_to([2.55, 0, 0], m.LEFT)
            sit(g, gloss, baseline(t, term))
            rows.append(m.VGroup(t, g))
        small = agent_world_loop(center=(-3.3, 0.85, 0), width=6.5, size=0.82)
        small_marker = m.Square(small.world[1][0][1].width).set_stroke(TEAL, 3).set_fill(TEAL, 0.55)
        small_marker.move_to(small.world[1][0][1]).set_z_index(2)
        target = m.VGroup(small, small_marker)
        agent_box = small.agent[0]
        policy = pill("policy · π(a | s)", POLICY, CORAL_FILL, 22, mono=True)
        policy.next_to(agent_box, m.DOWN, buff=0.45).align_to(agent_box, m.LEFT)
        ret = m.MathTex("G_t", "=", "R_{t+1}", "+", "R_{t+2}", "+", "R_{t+3}", "+", r"\cdots", font_size=42, color=INK)
        ret[0].set_color(VALUE)
        for k in (2, 4, 6):
            ret[k].set_color(REWARD)
        ret.move_to([-3.3, -2.25, 0])

        self.say(
            "[The agent is the learner and decision maker]: [here, Rover.]",
            m.AnimationGroup(self.new_header("02 · seven words", "The loop in seven words"), m.FadeOut(self.leftover),
                             m.Transform(diagram, target), m.FadeIn(rows[0], shift=0.1 * m.LEFT)),
            m.Circumscribe(loop.agent[0], color=ACTION, buff=0.06),
        )
        self.say(
            "[The environment is everything outside the agent]: [the world it acts in.]",
            m.FadeIn(rows[1], shift=0.1 * m.LEFT),
            m.Circumscribe(loop.world[0], color=STATE, buff=0.06),
        )
        self.say(
            "[A state is what the agent can see of the world, right now.]",
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.1 * m.LEFT), pulse(loop.state[0], STATE),
                             m.Circumscribe(loop.state[1], color=STATE, buff=0.06)),
        )
        self.say(
            "[An action is a choice]: [which lever to pull, or which way to drive.]",
            m.FadeIn(rows[3], shift=0.1 * m.LEFT),
            m.AnimationGroup(pulse(loop.action[0], ACTION), m.Circumscribe(loop.action[1], color=ACTION, buff=0.06)),
        )
        self.say(
            "[A reward is a single number]: [how good that one step was.]",
            m.FadeIn(rows[4], shift=0.1 * m.LEFT),
            m.AnimationGroup(pulse(loop.reward[0], REWARD), m.Circumscribe(loop.reward[1], color=REWARD, buff=0.06)),
        )
        self.say(
            "[A policy is the agent's rule for choosing actions]: [it's what Rover learns.]",
            m.AnimationGroup(m.FadeIn(rows[5], shift=0.1 * m.LEFT), m.FadeIn(policy, shift=0.15 * m.UP)),
            m.Circumscribe(policy, color=POLICY, buff=0.08),
        )
        self.say(
            "[And the return is the sum of all the rewards still to come]: [what Rover is really trying to maximize.]",
            m.AnimationGroup(m.FadeIn(rows[6], shift=0.1 * m.LEFT), m.FadeIn(ret, shift=0.15 * m.UP)),
            m.Circumscribe(ret, color=VALUE, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ rewards are not labels

    def no_labels(self):
        self.next_section("3 · No labels", notes=f"Supervised vs reinforcement learning (Sutton & Barto 2018 §1.1 and the chapter 2 introduction): a label says what the right output was (instructive feedback); a reward only says how good the action actually taken was (evaluative feedback), so the agent must explore; and its data come from its own choices, not a fixed dataset (notes §1 table). The spam e-mail is illustrative. Rover's pull of machine 2 (arm 1, seed-11 stream after the hook) pays {LEVER_TWO:g}. Linked: 'AI, Explained' Episode 8, Four Ways to Learn.")
        y0 = -0.2
        boxes = [card(6.1, 4.7, EDGE, CARD, radius=0.22, stroke=2).move_to([x, y0, 0]) for x in (-3.3, 3.3)]
        top = boxes[0].get_top()[1]

        def heading(box, title, sub):
            t = title_text(title, 32).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
            t.set_y(top - 0.45)
            s = text(sub, 22, MUTED).next_to(t, m.DOWN, buff=0.12).align_to(t, m.LEFT)
            return m.VGroup(t, s)

        left_head = heading(boxes[0], "Supervised learning", "learns from labels")
        right_head = heading(boxes[1], "Reinforcement learning", "learns from rewards")
        mail = icon(MAIL, 0.85).move_to([-5.15, 0.05, 0])
        to_label = arrow([-4.45, 0.05, 0], [-3.05, 0.05, 0], MUTED)
        label = pill("spam", GREEN, GREEN_FILL, 28).move_to([-2.0, 0.05, 0])
        label_cap = code("label", 22, GREEN).next_to(label, m.UP, buff=0.15)
        left_note = text("instructive: says what was right", 22, INK).move_to([-3.3, -1.6, 0])
        left_data = text("data: a fixed set of examples", 22, MUTED).move_to([-3.3, -2.05, 0])

        machines = m.VGroup(*[SlotMachine(str(i + 1), 0.5) for i in range(5)]).arrange(m.RIGHT, buff=0.24)
        machines.move_to([3.3, 0.3, 0])
        two = machines[1]
        bot = rover(0.55).move_to([two.cabinet.get_x(), -0.95, 0])
        heard = reward_chip(LEVER_TWO, 26).next_to(bot, m.RIGHT, buff=0.45)
        in_window = two.window
        unknown = m.VGroup(*[code("?", 22, VIOLET, weight=m.BOLD).move_to(mc.window) for k, mc in enumerate(machines) if k != 1])
        right_note = text("evaluative: says how good it was", 22, INK).move_to([3.3, -1.6, 0])
        right_data = text("data: made by its own choices", 22, MUTED).move_to([3.3, -2.05, 0])
        explore = pill("explore", VIOLET, VIOLET_FILL, 24).move_to([5.2, bot.get_y(), 0])
        others = [mc for k, mc in enumerate(machines) if k != 1]
        link = cite("The four ways to learn: AI, Explained · Episode 8").move_to([0, -3.15, 0])

        self.say(
            "[How is this different from supervised learning?] [There, every example comes with a label]: [the right answer.]",
            m.AnimationGroup(self.new_header("03 · no labels", "Rewards are not labels"), m.FadeIn(boxes[0]), m.FadeIn(left_head),
                             m.FadeIn(link)),
            m.AnimationGroup(m.FadeIn(mail, shift=0.15 * m.RIGHT), m.GrowArrow(to_label)),
            m.AnimationGroup(m.FadeIn(label, scale=0.7), m.FadeIn(label_cap), m.FadeIn(left_note, shift=0.1 * m.UP)),
        )
        self.say(
            "[In reinforcement learning, Rover pulls lever two], [and hears only: zero.]",
            m.Succession(m.AnimationGroup(m.FadeIn(boxes[1]), m.FadeIn(right_head), m.FadeIn(machines), m.FadeIn(bot, shift=0.15 * m.UP)),
                         pull(two, 0.7)),
            m.FadeIn(heard, target_position=in_window, scale=0.5),
        )
        self.say(
            "[The reward says how good that choice was], [never which lever would have been best.]",
            m.FadeIn(right_note, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(q, scale=0.5) for q in unknown], lag_ratio=0.2),
        )
        self.say(
            "[To find out, Rover has to try the others]: [it has to explore.]",
            m.LaggedStart(*[m.Circumscribe(mc.cabinet, color=EXPLORE, buff=0.05) for mc in others], lag_ratio=0.2),
            m.FadeIn(explore, shift=0.1 * m.LEFT),
        )
        self.say(
            "[And there's no fixed dataset]: [what Rover gets to see depends on what it chooses to do.]",
            m.FadeIn(left_data, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(right_data, shift=0.1 * m.UP), m.Circumscribe(right_data, color=EXPLORE, buff=0.08), lag_ratio=0.6),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ a reward that arrives late

    def credit(self):
        self.next_section("4 · Credit", notes=f"Rover's world (Russell & Norvig 4x3): charger +1, puddle −1, rock, −0.04 per move, 80% intended move, γ = 1. One episode, play_episode(ROVER_WORLD, RN_POLICY, seed 11): {TRAIL}, moves {MOVES_TAKEN}, no slip; rewards {[round(r, 2) for r in REWARDS]} (the last = +1 charger − 0.04 step). Returns with γ = 1 (discounted_returns): {[round(g, 2) for g in RETURNS]}: each move is credited with every reward after it. Credit assignment (Minsky 1961, via Sutton & Barto §1.7) and exploration are the course's two questions (notes §1). Unvisited cells: {sorted(UNVISITED)}.")
        grid = GridView.from_world(ROVER_WORLD, cell=1.25, center=(-3.55, 0.15, 0))
        lift = np.array([0, 0.13, 0])
        bot = rover(0.42).move_to(grid.at(TRAIL[0]) + lift).set_z_index(4)
        segs = [m.Line(grid.at(a), grid.at(b)).set_stroke(ACTION, 6).set_z_index(3) for a, b in zip(TRAIL, TRAIL[1:])]

        cx = {"move": 1.05, "reward": 2.95, "return": 4.95}
        head_y, row_y = 1.75, [1.1 - 0.62 * i for i in range(5)]
        heads = {k: code(k, 22, c, weight=m.BOLD).move_to([cx[k], head_y, 0])
                 for k, c in (("move", ACTION), ("reward", REWARD), ("return", VALUE))}
        rule = m.Line([0.3, head_y - 0.3, 0], [5.9, head_y - 0.3, 0]).set_stroke(EDGE, 2)
        move_cells = [code(f"{i + 1}  {ARROWS[a]}", 24, ACTION).move_to([cx["move"], y, 0]) for i, (a, y) in enumerate(zip(MOVES_TAKEN, row_y))]
        for c in move_cells:
            c[0].set_color(MUTED)
        reward_cells = [reward_chip(r, 22).move_to([cx["reward"], y, 0]) for r, y in zip(REWARDS, row_y)]
        return_cells = [code(f"{g:.2f}", 24, VALUE, weight=m.BOLD).move_to([cx["return"], y, 0]) for g, y in zip(RETURNS, row_y)]
        note = cite("move 5: +1 for the charger, −0.04 for the move").move_to([3.1, row_y[-1] - 0.6, 0])
        spark = m.Dot(grid.at(TRAIL[-1]), radius=0.13, color=GOLD).set_z_index(5)
        tints = [grid.squares[ROVER_WORLD.to_screen(s)] for s in TRAIL[:-1]]

        def move(k, rt=0.55):
            return m.AnimationGroup(bot.animate.move_to(grid.at(TRAIL[k + 1]) + lift), m.Create(segs[k]),
                                    m.FadeIn(move_cells[k], shift=0.1 * m.LEFT), run_time=rt)

        def back(k, rt=0.5):
            return m.AnimationGroup(spark.animate.move_to(grid.at(TRAIL[k])), tints[k].animate.set_fill(heat(RETURNS[k]), 1),
                                    m.FadeIn(return_cells[k], shift=0.1 * m.UP), run_time=rt)

        q1 = m.VGroup(title_text("Credit:", 32, SIGNAL), text("which action was good?", 28, INK))
        q2 = m.VGroup(title_text("Exploration:", 32, EXPLORE), text("what haven't I tried?", 28, INK))
        for q in (q1, q2):
            q.arrange(m.RIGHT, buff=0.2)
            sit(q[1], q[1].text, baseline(q[0], q[0].text))
        q1.move_to([-3.55, -2.85, 0])
        q2.move_to([3.1, -2.85, 0])
        holes = m.VGroup()
        for s in UNVISITED:
            sq = m.DashedVMobject(m.Square(1.0), num_dashes=24).set_stroke(EXPLORE, 3).move_to(grid.at(s))
            holes.add(m.VGroup(sq, title_text("?", 34, EXPLORE).move_to(grid.at(s))))
        holes.set_z_index(3)
        charger_sq = grid.squares[ROVER_WORLD.to_screen(CHARGER)]
        puddle_sq = grid.squares[ROVER_WORLD.to_screen(PUDDLE)]

        self.say(
            "[Rewards can also arrive late.] [Take Rover's grid world]: [a charger worth plus one], [and a puddle worth minus one.]",
            self.new_header("04 · credit", "A reward that arrives late"),
            m.AnimationGroup(m.FadeIn(grid), m.FadeIn(bot, shift=0.15 * m.UP)),
            m.Circumscribe(charger_sq, color=GOLD, buff=0.05),
            m.Circumscribe(puddle_sq, color=RED, buff=0.05),
        )
        self.say(
            "[Say Rover drives up and along], [and every move costs a little: minus zero point zero four.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(heads["move"], heads["reward"])),
                             m.Create(rule), m.Succession(*[move(k) for k in range(4)])),
            m.LaggedStart(*[m.FadeIn(c, scale=0.7) for c in reward_cells[:4]], lag_ratio=0.25),
        )
        self.say(
            "[The fifth move reaches the charger]: [plus one, at last.]",
            move(4, 0.7),
            m.AnimationGroup(m.FadeIn(reward_cells[4], scale=0.6), m.FadeIn(note),
                             m.Flash(grid.at(CHARGER), color=GOLD, flash_radius=0.55, line_length=0.25)),
        )
        self.say(
            "[But which moves earned it?] [The reward travels back along the trail], [so even the first move gets credit: zero point eight.]",
            m.AnimationGroup(m.FadeIn(heads["return"]), m.FadeIn(spark, scale=0.5)),
            m.Succession(*[back(k) for k in range(4, -1, -1)]),
            m.Circumscribe(return_cells[0], color=VALUE, buff=0.1),
        )
        self.say(
            "[Sharing out that credit is the first question every episode answers]: [which action was good?]",
            m.FadeOut(spark),
            m.FadeIn(q1, shift=0.15 * m.UP),
        )
        self.say(
            "[The second is exploration]: [what haven't I tried yet?]",
            m.LaggedStart(*[m.FadeIn(h, scale=0.8) for h in holes], lag_ratio=0.2),
            m.FadeIn(q2, shift=0.15 * m.UP),
        )
        self.wait(0.7)
        self.clear()

    # ------------------------------------------------------------------ history

    def history(self):
        self.next_section("5 · History", notes="Dated strip (notes §1, Appendix C): TD-Gammon, Tesauro 1992 (Machine Learning 8) and 1995 (CACM 38: near the level of the strongest grandmasters, learned by self-play); DQN, Mnih et al., Nature 518, 26 Feb 2015 (49 Atari games, inputs: pixels and the game score, one algorithm, architecture and hyperparameter set, a network trained per game); AlphaGo beat Lee Sedol 4–1, Seoul, 9–15 Mar 2016 (Silver et al., Nature 529, 28 Jan 2016; DeepMind AlphaGo page); InstructGPT, Ouyang et al., arXiv 2203.02155, 4 Mar 2022 (RLHF), ChatGPT 30 Nov 2022 (OpenAI: trained with RLHF); OpenAI o1, 12 Sep 2024 ('trained with reinforcement learning to perform complex reasoning'); DeepSeek-R1, 20–22 Jan 2025 (rule-based accuracy and format rewards). Coda: the 2024 ACM A.M. Turing Award to Andrew Barto and Richard Sutton, 'for developing the conceptual and algorithmic foundations of reinforcement learning' (ACM press release, March 2025; the exact day, 5 March, is unverified per notes Appendix B). Fact date: October 2026. Episode pills point to where the course covers each.")
        axis_y = -0.45
        axis = m.Line([-6.4, axis_y, 0], [6.4, axis_y, 0]).set_stroke(EDGE, 4)
        xs = [-5.2, -2.6, 0.0, 2.6, 5.2]
        eras = [
            ("1992", "TD-Gammon", "taught itself backgammon,\nplaying against itself", "Tesauro, 1992 · 1995", "Ep 9", BLUE, icon(DICE, 0.5)),
            ("2015", "DQN", "49 Atari games, learned\nfrom pixels and the score", "Mnih et al., 2015", "Ep 13", VIOLET, icon(PAD, 0.36)),
            ("2016", "AlphaGo", "beat Lee Sedol at Go, 4–1", "Silver et al., 2016", "Ep 15", VIOLET, go_board(VIOLET)),
            ("2022", "InstructGPT", "learned from human feedback;\nChatGPT followed in November", "Ouyang et al., 2022", "Ep 21", PINK, icon(PEOPLE, 0.5)),
            ("2024–25", "Reasoning models", "o1, DeepSeek-R1: rewarded\nfor answers that check out", "OpenAI, 2024; DeepSeek-AI, 2025", "Ep 24", PINK, icon(MODEL, 0.5)),
        ]
        items = []
        for i, ((year, name, fact, src, ep, color, art), x) in enumerate(zip(eras, xs)):
            above = i % 2 == 0
            node = m.Circle(radius=0.42).set_stroke(color, 3).set_fill(FILLS[color], 1).move_to([x, axis_y, 0])
            art.move_to(node)
            y_words = code(year, 36, color, weight=m.BOLD)
            n_words = text(name, 26, INK, weight=m.BOLD)
            tag = pill(ep, color, FILLS[color], 22, mono=True)
            head = m.VGroup(y_words, tag).arrange(m.RIGHT, buff=0.25)
            tag.set_y(y_words.get_y())
            body = text(fact, 22, INK)
            block = m.VGroup(head, n_words, body, cite(src)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.13)
            block.move_to([x, 0, 0])
            if block.get_left()[0] < -6.55:
                block.shift((-6.55 - block.get_left()[0]) * m.RIGHT)
            if block.get_right()[0] > 6.55:
                block.shift((6.55 - block.get_right()[0]) * m.RIGHT)
            if above:
                block.align_to([0, axis_y + 0.7, 0], m.DOWN)
                stem = m.Line([x, axis_y + 0.42, 0], [x, block.get_bottom()[1] - 0.12, 0])
            else:
                block.align_to([0, axis_y - 0.7, 0], m.UP)
                stem = m.Line([x, axis_y - 0.42, 0], [x, block.get_top()[1] + 0.12, 0])
            stem.set_stroke(color, 2)
            items.append((m.VGroup(node, art), stem, block))

        def show(k):
            node, stem, block = items[k]
            return m.AnimationGroup(m.GrowFromCenter(node), m.Create(stem), m.FadeIn(block, shift=0.15 * (m.UP if k % 2 == 0 else m.DOWN)),
                                    lag_ratio=0.25)

        self.say(
            "[The idea has a long history.] [In 1992, TD-Gammon taught itself backgammon, by playing against itself.]",
            m.AnimationGroup(self.new_header("05 · history", "From backgammon to reasoning"), m.Create(axis)),
            show(0),
        )
        self.say("[In 2015, the DQN learned to play forty-nine Atari games], [from the pixels and the score.]",
                 show(1), m.Circumscribe(items[1][2][2], color=VIOLET, buff=0.08))
        self.say("[In 2016, AlphaGo beat one of the world's best Go players], [four games to one.]",
                 show(2), m.Circumscribe(items[2][2][2], color=VIOLET, buff=0.08))
        self.say("[In 2022, InstructGPT learned from people ranking its answers], [and ChatGPT followed that November.]",
                 show(3), m.Circumscribe(items[3][2][2], color=PINK, buff=0.08))
        self.say("[And since 2024, reasoning models learn to think before they answer], [rewarded when the answer checks out.]",
                 show(4), m.Circumscribe(items[4][2][2], color=PINK, buff=0.08))
        strip = m.VGroup(axis, *[m.VGroup(*it) for it in items])
        cup = icon("trophy", 1.3)
        who = text("Andrew Barto and Richard Sutton", 32, INK, weight=m.BOLD)
        prize = code("2024 ACM A.M. Turing Award", 22, GOLD, weight=m.BOLD)
        why = m.VGroup(text("“for developing the conceptual and algorithmic", 22, MUTED),
                       text("foundations of reinforcement learning”", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        src = cite("ACM, March 2025")
        words = m.VGroup(prize, who, why, src).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        inside = m.VGroup(cup, words).arrange(m.RIGHT, buff=0.55).move_to([0, -0.4, 0])
        shell = card(inside.width + 1.3, inside.height + 0.8, GOLD, CARD, radius=0.22, stroke=2.5).move_to(inside)
        self.say(
            "[In 2025, Andrew Barto and Richard Sutton won the Turing Award], [for laying the foundations of this field.]",
            m.AnimationGroup(m.FadeOut(strip, shift=0.2 * m.DOWN), m.FadeIn(shell, scale=0.95), m.FadeIn(cup, shift=0.15 * m.UP),
                             m.FadeIn(prize), m.FadeIn(who, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(why, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ the course map

    def course_map(self):
        self.next_section("6 · The course", notes="One loop, six worlds, one per module, in the module colours (COURSE.md): Rover's five slot machines (Module 1, Episodes 1–4); Rover's grid world (Module 2, 5–7); the cliff walk, 4 × 12, −1 per step, −100 off the edge (Module 3, Episode 10); a board game (tic-tac-toe searched live, AlphaGo's family; Module 4, Episode 15); a cart and pole (CartPole: +1 for every step upright; DQN in Module 4, policy gradients and PPO in Module 5); a small language model answering grade-school maths, its tokens the actions, a checker's score at the end (Module 6). States, actions and rewards per world: notes §1, worked example 3. The cart's wobble uses a simple push-toward-the-lean rule (illustrative); the language-model tokens are illustrative.")
        W, H = 4.2, 2.85
        centres = [(-4.45, 1.07), (0.0, 1.07), (4.45, 1.07), (-4.45, -1.98), (0.0, -1.98), (4.45, -1.98)]
        accents = [MODULES[k][1:] for k in range(1, 7)]
        boxes = [card(W, H, EDGE, CARD, radius=0.2, stroke=2).move_to([x, y, 0]) for x, y in centres]

        # the art for each world
        slots = m.VGroup(*[SlotMachine(str(i + 1), 0.34) for i in range(5)]).arrange(m.RIGHT, buff=0.12)
        g = cell_grid(3, 4, size=0.27, gap=0.04, color=EDGE, fill=BG)
        g[0][3].set_fill(GOLD_FILL, 1).set_stroke(GOLD, 2)
        g[1][3].set_fill(RED_FILL, 1).set_stroke(RED, 2)
        g[1][1].set_fill("#2A2238", 1)
        route = m.VMobject().set_points_as_corners([g[2][0].get_center(), g[0][0].get_center(), g[0][3].get_center()])
        route.set_stroke(ACTION, 4)
        dot = m.Dot(g[2][0].get_center(), radius=0.07, color=ACTION)
        grid_art = m.VGroup(g, route, dot)
        cliff = cell_grid(4, 12, size=0.15, gap=0.03, color=EDGE, fill=BG)
        for c in range(1, 11):
            cliff[3][c].set_fill(RED_FILL, 1).set_stroke(RED, 1.5)
        cliff[3][0].set_stroke(TEAL, 2)
        cliff[3][11].set_stroke(INK, 2)
        cliff_art = cliff
        cell = 0.26
        board = m.VGroup()
        for i in (1, 2):
            board.add(m.Line([-1.5 * cell, (1.5 - i) * cell, 0], [1.5 * cell, (1.5 - i) * cell, 0]),
                      m.Line([(i - 1.5) * cell, -1.5 * cell, 0], [(i - 1.5) * cell, 1.5 * cell, 0]))
        board.set_stroke(MUTED, 2.5)
        marks = m.VGroup()
        for k, ch in enumerate(MIDGAME):
            r, c = divmod(k, 3)
            p = np.array([(c - 1) * cell, (1 - r) * cell, 0])
            if ch == "X":
                d = 0.08
                marks.add(m.VGroup(m.Line(p + [-d, -d, 0], p + [d, d, 0]), m.Line(p + [-d, d, 0], p + [d, -d, 0])).set_stroke(INK, 3))
            elif ch == "O":
                marks.add(m.Circle(radius=0.085).move_to(p).set_stroke(VIOLET, 3))
        board_art = m.VGroup(board, marks)
        cart_box = boxes[4]
        cart = CartPoleView(center=(cart_box.get_x(), cart_box.get_top()[1] - 1.0, 0), units_per_m=0.36, pole_units=0.5)
        cart.set_state(WOBBLE[0][0], WOBBLE[0][2])   # built in place: set_state works in absolute coordinates
        toks = token_row(["answer", " 12"], ACTION, size=22, gap=0.07)
        prize = reward_chip(1, 22)
        lm_art = m.VGroup(toks, prize).arrange(m.RIGHT, buff=0.18)
        arts = [slots, grid_art, cliff_art, board_art, cart, lm_art]

        names = ["Slot machines", "Grid world", "Cliff walk", "Board game", "Cart and pole", "Language model"]
        triples = [
            ("none at all", "which lever to pull", "a coin, or nothing"),
            ("which cell Rover is in", "a move: N, E, S or W", "+1, −1, and −0.04 a move"),
            ("where Rover stands", "a step in any direction", "−1 a step, −100 off the edge"),
            ("the board", "where to play", "win, draw or loss, at the end"),
            ("four numbers", "push left or right", "+1 for every step upright"),
            ("the text so far", "the next token", "a check of the final answer"),
        ]
        cards = []
        for k, (box, (accent, fill), art, name, triple) in enumerate(zip(boxes, accents, arts, names, triples)):
            top, left = box.get_top()[1], box.get_left()[0]
            if art is not cart:
                if art.height > 0.82:
                    art.scale(0.82 / art.height)
                if art.width > W - 1.1:
                    art.scale((W - 1.1) / art.width)
                art.move_to([box.get_x() - 0.25, top - 0.62, 0])
            b = badge(k + 1, accent)
            t = title_text(name, 27)
            title_row = m.VGroup(b, t).arrange(m.RIGHT, buff=0.18)
            title_row.align_to([left + 0.25, 0, 0], m.LEFT).set_y(top - 1.38)
            rows = m.VGroup()
            for j, (value, color) in enumerate(zip(triple, (STATE, ACTION, REWARD))):
                y = top - 1.86 - 0.33 * j
                d = m.Dot(radius=0.07, color=color).move_to([left + 0.36, y, 0])
                v = text(value, 20, INK).move_to([0, y, 0]).align_to([left + 0.58, 0, 0], m.LEFT)
                rows.add(m.VGroup(d, v))
            glyph = loop_glyph(accent).move_to([box.get_right()[0] - 0.33, top - 0.33, 0])
            box.set_stroke(accent, 2.5)
            cards.append(dict(box=box, art=art, title=title_row, rows=rows, glyph=glyph, accent=accent))

        legend = m.VGroup()
        for word, color in (("state", STATE), ("action", ACTION), ("reward", REWARD)):
            legend.add(m.VGroup(m.Dot(radius=0.07, color=color), code(word, 20, color)).arrange(m.RIGHT, buff=0.12))
        legend.arrange(m.RIGHT, buff=0.35).move_to([0, 3.19, 0]).align_to([6.6, 0, 0], m.RIGHT)
        questions = m.VGroup(pill("credit", SIGNAL, PINK_FILL, 22), pill("exploration", EXPLORE, VIOLET_FILL, 22))
        questions.arrange(m.RIGHT, buff=0.25).move_to(legend).align_to(legend, m.RIGHT)

        def whole(k, *skip):
            c = cards[k]
            parts = [c[p] for p in ("art", "title", "rows", "glyph") if p not in skip]
            return m.AnimationGroup(*[m.FadeIn(p, shift=0.1 * m.UP) for p in parts], lag_ratio=0.1)

        self.say(
            "[So here's the plan for this course]: [one loop, six worlds.]",
            m.AnimationGroup(self.new_header("06 · the course", "One loop, six worlds"), m.FadeIn(legend)),
            m.LaggedStart(*[m.FadeIn(b, scale=0.92) for b in boxes], lag_ratio=0.12),
        )
        self.say(
            "[First, Rover's five slot machines]: [no state at all], [just which lever to pull.]",
            m.AnimationGroup(m.FadeIn(cards[0]["art"], shift=0.1 * m.UP), m.FadeIn(cards[0]["title"]), m.FadeIn(cards[0]["glyph"])),
            m.FadeIn(cards[0]["rows"][0], shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(cards[0]["rows"][1:], shift=0.1 * m.LEFT), pull(slots[3], 0.7)),
        )
        self.say(
            "[Then a grid with a charger and a puddle], [where Rover learns to plan a route.]",
            whole(1),
            m.Circumscribe(route, color=ACTION, buff=0.05),
        )
        self.say(
            "[A cliff walk], [where one careless step costs minus one hundred.]",
            whole(2),
            m.Circumscribe(m.VGroup(*cliff[3][1:11]), color=RED, buff=0.04),
        )
        self.say(
            "[A board game, played against itself], [and a cart that must keep a pole upright.]",
            whole(3),
            m.Succession(whole(4), cart.replay(WOBBLE, 2.2)),
        )
        self.say(
            "[And finally a language model]: [every token is an action], [and the final answer gets checked.]",
            m.AnimationGroup(m.FadeIn(cards[5]["title"]), m.FadeIn(cards[5]["glyph"]), m.FadeIn(cards[5]["rows"][0], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.RIGHT) for t in toks], lag_ratio=0.25),
                             m.FadeIn(cards[5]["rows"][1], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(prize, scale=0.6), m.FadeIn(cards[5]["rows"][2], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Six worlds], [one loop turning in every one], [and the same two questions each time.]",
            m.LaggedStart(*[m.Circumscribe(c["box"], color=c["accent"], buff=0.04) for c in cards], lag_ratio=0.1),
            m.AnimationGroup(*[m.Rotate(c["glyph"], angle=-2 * np.pi, run_time=1.4) for c in cards]),
            m.AnimationGroup(m.FadeOut(legend, shift=0.15 * m.UP), m.FadeIn(questions, shift=0.15 * m.UP)),
        )
        self.wait(1.0)
        self.clear()

    # ------------------------------------------------------------------ the next question

    def next_question(self):
        self.next_section("7 · Next", notes=f"Rover's five machines (ROVER_ARMS = {ROVER_ARMS}): each pays one coin with its own probability, hidden from Rover. Which to pull, and how to balance trying against cashing in, is Episode 2 (Explore or Exploit).")
        machines = m.VGroup(*[SlotMachine(str(i + 1), 1.0) for i in range(5)]).arrange(m.RIGHT, buff=0.5)
        machines.move_to([0, 0.75, 0])
        first = machines[0].copy().move_to([0, 0.75, 0])
        hidden = m.VGroup(*[code("p = ?", 22, EXPLORE, weight=m.BOLD).move_to(mc.window) for mc in machines])
        bot = rover(0.8).move_to([0, -1.1, 0])
        ask = title_text("Which machine should Rover pull?", 52, INK).move_to([0, -2.75, 0])

        self.say(
            "[Back to Rover's first slot machine.] [Now there are five], [and each pays out with its own hidden chance.]",
            m.AnimationGroup(self.new_header("07 · next", "Which machine?"), m.FadeIn(first, shift=0.2 * m.UP),
                             m.FadeIn(bot, shift=0.2 * m.UP)),
            m.AnimationGroup(first.animate.move_to(machines[0]), m.LaggedStart(*[m.FadeIn(mc, shift=0.2 * m.LEFT) for mc in machines[1:]],
                                                                               lag_ratio=0.15)),
            m.LaggedStart(*[m.FadeIn(h, scale=0.6) for h in hidden], lag_ratio=0.12),
        )
        self.say(
            "[So which machine should Rover pull?]",
            m.AnimationGroup(m.FadeIn(ask, shift=0.2 * m.UP), bot.animate(rate_func=m.there_and_back).shift(0.15 * m.UP)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Act (the policy picks an action), reward (the world scores it), learn (do more of what paid). Reinforcement learning is learning what to do from the consequences of doing it.")
        halo = glow(GOLD, radius=4.0)
        words = m.VGroup(*[title_text(w, 88, c) for w, c in (("Act", ACTION), ("Reward", REWARD), ("Learn", SIGNAL))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("try an action", "the world scores it", "do more of what paid")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.0
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Learning what to do, from the consequences of doing it.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)
        mascot = rover(0.45).next_to(tagline, m.DOWN, buff=0.45)

        self.say(
            "[To sum up]: [act], [reward], [learn.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Learning what to do], [from the consequences of doing it.]",
            m.Write(tagline),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline, mascot)], run_time=0.9)
