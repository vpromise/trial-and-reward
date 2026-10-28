"""The series kit for "Trial and Reward" (reinforcement learning, from slot machines to reasoning models):
design system, RL components, voice, and the Episode base class with the series intro and next-episode outro.
Episodes import it:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kit import *  # noqa: F403
    from rl_reference import *  # noqa: F403  (the course's algorithms, numbers and measured runs)

    class Ep02ExploreOrExploit(Episode):
        number = 2
        def lesson(self): ...

Run from an episode folder: `VOICE=draft manimgx inspect scene.py` (silent draft) or, with the narrator
serving from the series folder, `manimgx render scene.py`. A tour of every component: `kit_demo.py`.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import manimgx as m
from manimgx.audio import Word

SERIES = Path(__file__).resolve().parent
ASSETS = SERIES / "assets"
TITLE_WIDTH = 12.5  # the widest an intro/outro card title may be (the frame is about 14.2 wide)

# ---------------------------------------------------------------------------- engine fixes
# (carried over from "AI, Explained" and "Agents, Animated": each fixes a ManimGX timing fault, see the
# tutorial-video skill's reference/manimgx.md, gotchas 19–29)


def _from_copy(mobject, target_mobject, **kwargs):
    """TransformFromCopy shows its finished target from the start of the line; a FadeIn from the source waits."""
    return m.FadeIn(target_mobject, target_position=mobject, **kwargs)


def _fade_transform(mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs):
    """FadeTransform(Pieces) show the target early too; a cross-fade where each side glides waits for its cue."""
    return m.AnimationGroup(
        m.FadeOut(mobject, target_position=target_mobject),
        m.FadeIn(target_mobject, target_position=mobject),
        **kwargs,
    )


m.TransformFromCopy = _from_copy
m.FadeTransform = _fade_transform
m.FadeTransformPieces = _fade_transform


# A line that introduces a group and also moves or highlights a PART of it would show that part early:
# marking those animations as introducers keeps play() from adding them before their cue.
def _leaves(animation):
    subs = getattr(animation, "animations", None)
    if subs:
        for sub in subs:
            yield from _leaves(sub)
    else:
        yield animation


def _cue_safe(scene, animations):
    on_screen = {id(x) for x in scene.get_mobject_family_members()}
    roots, family = set(), set()
    for animation in animations:
        for leaf in _leaves(animation):
            if leaf.is_introducer() and getattr(leaf, "mobject", None) is not None:
                roots.add(id(leaf.mobject))
                family |= {id(x) for x in leaf.mobject.get_family()}
    for animation in animations:
        for leaf in _leaves(animation):
            mob = getattr(leaf, "mobject", None)
            if mob is None or leaf.is_introducer() or id(mob) in roots or id(mob) in on_screen:
                continue
            if id(mob) in family:
                leaf.introducer = True
    return animations


# Compiled under ManimGX's package path so play labels still point at the episode's own scene.py line.
_SAY = """
def say(self, script, *animations, voice=None):
    return _scene_say(self, script, *_cue_safe(self, animations), voice=voice)
"""
_say_namespace = {"_scene_say": m.Scene.say, "_cue_safe": _cue_safe}
exec(compile(_SAY, str(Path(m.__file__).parent / "_series_kit_say.py"), "exec"), _say_namespace)


# ---------------------------------------------------------------------------- design system
# A deep plum stage. Colours carry meaning across the whole course — learn them once:
#   reward GOLD · values (V, Q, returns, estimates) BLUE · policy and actions CORAL · states and the world TEAL
#   exploration and uncertainty VIOLET · the learning signal (TD error, advantage, gradient) PINK
#   good GREEN · bad RED · structure EDGE/MUTED

BG = "#120E1A"
CARD = "#1C1726"
CARD_HI = "#241E31"
EDGE = "#3A3150"
INK = "#F0ECF7"
MUTED = "#A79DB8"
FAINT = "#5F5574"
GOLD, GOLD_FILL = "#FFC93C", "#3B2C0C"
BLUE, BLUE_FILL = "#62A8FF", "#142849"
CORAL, CORAL_FILL = "#FF8A4C", "#3D1F10"
TEAL, TEAL_FILL = "#34D6BD", "#0F322D"
VIOLET, VIOLET_FILL = "#B48CFF", "#2A1C47"
PINK, PINK_FILL = "#F472D0", "#3D1535"
GREEN, GREEN_FILL = "#66E39A", "#133322"
RED, RED_FILL = "#FF4D5E", "#3E141B"
GOOD, GOOD_FILL = GREEN, GREEN_FILL
BAD, BAD_FILL = RED, RED_FILL
FILLS = {GOLD: GOLD_FILL, BLUE: BLUE_FILL, CORAL: CORAL_FILL, TEAL: TEAL_FILL, VIOLET: VIOLET_FILL, PINK: PINK_FILL,
         GREEN: GREEN_FILL, RED: RED_FILL, MUTED: CARD, EDGE: CARD, INK: CARD}

# what each colour means (use these names in scenes so the meaning stays fixed)
REWARD, VALUE, ACTION, STATE, EXPLORE, SIGNAL = GOLD, BLUE, CORAL, TEAL, VIOLET, PINK
POLICY, WORLD = ACTION, STATE

ROLES = {"system": (MUTED, CARD), "user": (TEAL, TEAL_FILL), "assistant": (CORAL, CORAL_FILL)}

HEAD = "Gill Sans"            # titles, headers, recap words (bold)
NUM_FONT = "Avenir Next"      # display text with digits: Gill Sans draws the digit one like a capital I
SANS = "Avenir Next"            # everything you read
MONO = "Menlo"                  # code, numbers in tables, units, labels: no ligatures, so "->", "!=" stay as typed

m.config.background_color = BG

MASCOT = "Rover"  # the course's learner (drawn by rover())

# the course: module → (name, accent, fill); episode → (title, module). See COURSE.md.
MODULES = {
    1: ("TRIAL AND REWARD", GOLD, GOLD_FILL),
    2: ("PLANNING IN A KNOWN WORLD", TEAL, TEAL_FILL),
    3: ("LEARNING FROM EXPERIENCE", BLUE, BLUE_FILL),
    4: ("DEEP VALUE LEARNING", VIOLET, VIOLET_FILL),
    5: ("LEARNING THE POLICY", CORAL, CORAL_FILL),
    6: ("TEACHING LANGUAGE MODELS WITH REWARD", PINK, PINK_FILL),
}
EPISODES = {
    1: ("The Reward Loop", 1), 2: ("Explore or Exploit", 1), 3: ("Optimism and Luck", 1), 4: ("Bandits with Context", 1),
    5: ("States, Actions, Returns", 2), 6: ("The Bellman Equation", 2), 7: ("Value Iteration", 2),
    8: ("Learning by Playing", 3), 9: ("Learning from a Guess", 3), 10: ("Q-Learning and SARSA", 3),
    11: ("When Rewards Are Rare", 3),
    12: ("From Tables to Networks", 4), 13: ("Deep Q-Networks", 4), 14: ("Fixing Overconfidence", 4),
    15: ("Search and Self-Play", 4),
    16: ("Policy Gradients", 5), 17: ("Baselines and Critics", 5), 18: ("Proximal Policy Optimization", 5),
    19: ("Continuous Control", 5),
    20: ("The Model Is the Policy", 6), 21: ("Learning from Human Preferences", 6),
    22: ("Direct Preference Optimization", 6), 23: ("GRPO", 6), 24: ("Rewards You Can Check", 6),
    25: ("Reward Hacking", 6), 26: ("GRPO on This Mac", 6),
}
LEVELS = MODULES  # older kits' name

_SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def spoken_title(title):
    """A title as the narrator should read it: no symbols (the lexicon splits CamelCase names)."""
    return title.rstrip("?.").replace(" & ", " and ").replace(":", ",").replace(" vs ", " versus ")


def say_number(n):
    """A number under a hundred as words, for the narrator."""
    return _SMALL[n] if n < 20 else _TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + _SMALL[n % 10])


# ---------------------------------------------------------------------------- the voice

NARRATOR = os.environ.get("NARRATOR_URL", "http://127.0.0.1:8765/speak")
VOICE_REF = ""  # a clean 10–20 s WAV of a voice to copy; "" = the built-in voice


def narrator(text, voice_ref=VOICE_REF, exaggeration=0.5, cfg_weight=0.3, temperature=0.8):
    """Chatterbox, run locally by the tutorial-video skill's tts_server.py, timed by Whisper.
    Start it from the series folder: ~/.claude/skills/tutorial-video/scripts/narrator.sh"""
    settings = dict(voice_ref=voice_ref, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    request = urllib.request.Request(
        NARRATOR, json.dumps({"text": text, **settings}).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as reply:
            said = json.load(reply)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"no narrator at {NARRATOR}: start ~/.claude/skills/tutorial-video/scripts/narrator.sh"
            " in the series folder (or render with VOICE=draft)"
        ) from error
    return m.Speech(base64.b64decode(said["audio"]), text=text, words=[Word(*w) for w in said["words"]])


def draft_voice(text, seconds_per_word=0.3):
    """A silent stand-in at about the real voice's pace: for layout and timing passes."""
    words = re.findall(r"\S+", text)
    rate, lead = 24000, 0.15
    timed = [Word(w, lead + i * seconds_per_word, lead + (i + 1) * seconds_per_word) for i, w in enumerate(words)]
    silence = np.zeros(int((2 * lead + seconds_per_word * len(words)) * rate), dtype=np.float32)
    return m.Speech(silence, text=text, words=timed, rate=rate)



# ---------------------------------------------------------------------------- basic components


def text(s, size=32, color=INK, weight=m.NORMAL, **kw):
    """Body text (Avenir Next)."""
    return m.Text(s, font=SANS, font_size=size, color=color, weight=weight, **kw)


def title_text(s, size=40, color=INK, **kw):
    """Display text (Gill Sans Bold): titles, big numbers, recap words. Text containing a digit is set in
    Avenir Next Bold instead, because Gill Sans draws "1" like "I" ("1992" would read "I992")."""
    font = NUM_FONT if re.search(r"\d", s) else HEAD
    return m.Text(s, font=font, font_size=size, color=color, weight=m.BOLD, **kw)


def code(s, size=22, color=INK, t2c=None, **kw):
    """Monospace text (Menlo). `t2c={"regex": colour}` colours matching parts (ManimGX's own t2c
    takes literal strings only, so the regexes are matched here; a key that matches nothing as a regex is
    tried as a literal). The first rule to claim a character wins."""
    words = m.Text(s, font=MONO, font_size=size, color=color, **kw)
    if t2c:
        paint(words, s, t2c)
    return words


def paint(words, s, rules):
    """Colour the glyphs of Text `words` (made from string `s`) whose characters match `rules`
    ({regex: colour}, in priority order). Whitespace draws no glyph, so indices skip it."""
    glyph_of, n = {}, 0
    for i, ch in enumerate(s):
        if not ch.isspace():
            glyph_of[i] = n
            n += 1
    if n != len(words):  # a ligature or emoji changed the glyph count: leave the text plain
        return words
    claimed = set()
    for pattern, colour in rules.items():
        try:
            spans = [mt.span() for mt in re.finditer(pattern, s, re.MULTILINE)]
        except re.error:
            spans = []
        if not spans:
            spans = [mt.span() for mt in re.finditer(re.escape(pattern), s)]
        for a, b in spans:
            for i in range(a, b):
                if i in glyph_of and i not in claimed:
                    claimed.add(i)
                    words[glyph_of[i]].set_color(colour)
    return words


def tex(s, size=40, color=INK, **kw):
    """A formula (LaTeX math)."""
    return m.MathTex(s, font_size=size, color=color, **kw)


def glyphs(words, key):
    """The glyphs that draw the first `key` in a Text (spaces and newlines draw none). Mono text only:
    ligatures (fi, fl) in the sans font merge two letters into one glyph."""
    start = words.text.index(key)
    skip = sum(c.isspace() for c in words.text[:start])
    width = sum(not c.isspace() for c in key)
    return words[start - skip : start - skip + width]


def icon(name, height):
    return m.SVGMobject(str(ASSETS / f"{name}.svg"), height=height)


def card(width, height, color=EDGE, fill=CARD, radius=0.2, stroke=2.5):
    return m.RoundedRectangle(
        corner_radius=radius, width=width, height=height,
        stroke_color=color, stroke_width=stroke, fill_color=fill, fill_opacity=1,
    )


def pill(label, color, fill, size=22, mono=False):
    words = code(label, size - 2, color, weight=m.BOLD) if mono else text(label, size, color, weight=m.BOLD)
    height = words.height + 0.3
    box = card(words.width + 0.55, height, color, fill, radius=height / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def chip(word, color=TEAL, fill=TEAL_FILL, width=1.5, height=0.72, size=28, mono=False):
    """A word in a small rounded box, on the same baseline as its neighbours'."""
    box = card(width, height, color, fill, radius=0.14, stroke=2.5)
    label = (code(word, size - 4, INK) if mono else text(word, size, INK, weight=m.BOLD)).move_to(box)
    chars = word.replace(" ", "")
    if len(label) == len(chars):
        sitting = [g for g, c in zip(label, chars) if c not in "gjpqy,;_()[]{}"] or list(label)
        bottom = min(g.get_bottom()[1] for g in sitting)
    else:  # a ligature (fi, fl) merged two letters into one glyph: most glyphs sit on the baseline
        bottom = float(np.median([g.get_bottom()[1] for g in label]))
    baseline = box.get_center()[1] - 0.449 * size / 48 / 2
    label.shift((baseline - bottom) * m.UP)
    return m.VGroup(box, label)


def header(tag, title, color, fill):
    """The section header: an accent bar, a mono tag, a Gill Sans title (top left)."""
    tag_words = code(tag.upper(), 20, color, weight=m.BOLD)
    title_words = title_text(title, 36)
    bar = m.RoundedRectangle(corner_radius=0.04, width=0.1, height=0.62, stroke_width=0, fill_color=color, fill_opacity=1)
    row = m.VGroup(bar, tag_words, title_words).arrange(m.RIGHT, buff=0.3)
    tag_words.align_to(title_words, m.DOWN)
    return row.to_corner(m.UL, buff=0.5)


def bubble(words, color, fill, tail="left"):
    box = card(words.width + 0.7, words.height + 0.55, color, fill, radius=0.3)
    words.move_to(box)
    side = -1 if tail == "left" else 1
    base = box.get_bottom() + np.array([side * (box.width / 2 - 0.6), 0.15, 0])
    nub = m.Polygon(
        base + [-0.2, 0, 0], base + [0.2, 0, 0], base + [side * 0.5, -0.55, 0],
        stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1,
    )
    return m.VGroup(nub, box, words)


def node(name, label, sub, color, fill, width=2.1, height=2.1, label_size=26):
    box = card(width, height, color, fill)
    body = m.VGroup(icon(name, 0.8), text(label, label_size, INK, weight=m.BOLD), text(sub, 20, MUTED))
    body.arrange(m.DOWN, buff=0.13).move_to(box)
    return m.VGroup(box, body)


def tile(name, title, desc, color, width=6.3, height=2.1):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 30, INK, weight=m.BOLD), text(desc, 22, MUTED))
    words.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    body = m.VGroup(icon(name, 1.05), words).arrange(m.RIGHT, buff=0.45)
    body.move_to(box).align_to(box, m.LEFT).shift(0.45 * m.RIGHT)
    return m.VGroup(box, body)


def arrow(start, end, color=MUTED, buff=0.1):
    return m.Arrow(start, end, buff=buff, color=color, stroke_width=5, max_tip_length_to_length_ratio=0.3)


def pulse(a, color, width=10):
    return m.ShowPassingFlash(m.Line(a.get_start(), a.get_end()).set_stroke(color, width=width), time_width=0.7)


def on_baseline(words):
    for w in words[1:]:
        w.shift((words[0][0].get_bottom()[1] - w[0].get_bottom()[1]) * m.UP)
    return words


def glow(color, radius=3.0, layers=28):
    return m.VGroup(
        *[m.Circle(radius=radius * (1 - i / layers), stroke_width=0).set_fill(color, opacity=0.009) for i in range(layers)]
    )


def backdrop():
    image = m.ImageMobject(str(ASSETS / "backdrop.png"))
    image.height = 8
    return image


def axes(x_range, y_range, x_length, y_length, color=FAINT):
    """Quiet axes for the dark stage: thin, no numbers (label them with text()). Note: a range is cut to
    whole steps, so pick a step that divides it (gotcha 27)."""
    return m.Axes(
        x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
        axis_config={"color": color, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18},
    )


def recolored(name, color, fill, prefix):
    """An icon in another accent, written as assets/<prefix>-<name>-<hex>.svg (pass your episode's prefix,
    e.g. "ep07", so parallel episodes never write the same file). Returns the new name."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() in (BG.upper(), CARD.upper()):
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"{prefix}-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


# ---------------------------------------------------------------------------- code and data cards

_PY_KEYWORDS = r"\b(def|while|for|in|if|elif|else|return|import|from|as|with|try|except|raise|class|break|continue|not|and|or|is|None|True|False|lambda|yield|await|async)\b"


def code_block(src, size=20, width=None, title=None, accent=EDGE):
    """Python (or pseudo-code) in a dark card, lightly highlighted: keywords VIOLET, strings GREEN, comments
    FAINT, numbers GOLD, called names TEAL. Keep lines short (about 44 characters at size 20 fill 7 units)."""
    t2c = {
        r"#[^\n]*": FAINT,
        r"\"[^\"\n]*\"|'[^'\n]*'": GREEN,
        _PY_KEYWORDS: VIOLET,
        r"\b\d+(\.\d+)?(e\d+)?\b": GOLD,
        r"\b[a-z_][a-z0-9_]*(?=\()": TEAL,
    }
    body = code(src, size, INK, t2c=t2c)
    w = max(width or 0, body.width + 0.7)
    top, parts = 0.0, []
    if title:
        cap = code(title, 20, MUTED)
        top = cap.height + 0.25
        parts.append(cap)
    box = card(w, body.height + 0.6 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    if title:
        body.shift(-top / 2 * m.UP)
        parts[0].align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
    return m.VGroup(box, body, *parts)


def terminal(lines, width=6.5, size=20, title="terminal"):
    """A terminal window. Lines starting with "$ " are commands (the prompt in GREEN); lines starting
    with "! " are errors (RED); others are output (MUTED)."""
    rows = []
    for line in lines:
        if line.startswith("$ "):
            rows.append(code(line, size, INK, t2c={r"^\$": GREEN}))
        elif line.startswith("! "):
            rows.append(code(line[2:], size, RED))
        else:
            rows.append(code(line or " ", size, MUTED))
    body = m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    bar_h = 0.42
    w = max(width, body.width + 0.7)
    box = card(w, body.height + 0.55 + bar_h, EDGE, "#0C0912", radius=0.16, stroke=2)
    bar = m.Line(box.get_corner(m.UL) + [0.0, -bar_h, 0], box.get_corner(m.UR) + [0.0, -bar_h, 0]).set_stroke(EDGE, 2)
    dots = m.VGroup(*[m.Dot(radius=0.06, color=c) for c in (RED, GOLD, GREEN)]).arrange(m.RIGHT, buff=0.12)
    dots.move_to(box.get_corner(m.UL) + [0.4, -bar_h / 2, 0])
    cap = code(title, 20, FAINT).move_to(box.get_top() + [0, -bar_h / 2, 0])
    body.next_to(bar, m.DOWN, buff=0.28).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, bar, dots, cap, body)


def json_card(obj, size=19, accent=TEAL, fill=CARD, width=None, title=None):
    """A JSON object as a card: keys TEAL, strings GREEN, numbers GOLD, true/false/null VIOLET."""
    src = json.dumps(obj, indent=2, ensure_ascii=False)
    t2c = {
        r"\"[^\"\n]*\"(?=\s*:)": TEAL,
        r":\s*\"[^\"\n]*\"": GREEN,
        r"\b-?\d+(\.\d+)?\b": GOLD,
        r"\b(true|false|null)\b": VIOLET,
    }
    body = code(src, size, INK, t2c=t2c)
    top, cap = 0.0, None
    if title:
        cap = code(title, 20, accent, weight=m.BOLD)
        top = cap.height + 0.22
    w = max(width or 0, body.width + 0.7, (cap.width + 0.7) if cap else 0)
    box = card(w, body.height + 0.55 + top, accent, fill, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    group = m.VGroup(box, body)
    if cap:
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
        group.add(cap)
    return group


def message(role, body, width=5.2, size=20, mono=False):
    """One chat message: a role label in its colour (system MUTED, user TEAL, assistant = the policy CORAL)."""
    color, fill = ROLES[role]
    label = code(role, 22, color, weight=m.BOLD)
    words = code(body, size - 2, INK) if mono else text(body, size, INK)
    inner = m.VGroup(label, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    if words.width > width - 0.6:
        words.scale((width - 0.6) / words.width)
        inner.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def loop_ring(labels, radius=1.6, colors=None, size=24, center=(0, 0, 0)):
    """A cycle of steps around a circle with arrows between them. Returns VGroup(nodes, arcs)."""
    n = len(labels)
    colors = colors or [GOLD, TEAL, CORAL, PINK, VIOLET][:n]
    center = np.array(center, dtype=float)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    nodes = m.VGroup()
    for label, color, a in zip(labels, colors, angles):
        words = text(label, size, INK, weight=m.BOLD)
        c = chip(label, color, FILLS.get(color, CARD), width=max(1.4, words.width + 0.5), height=0.66, size=size)
        nodes.add(c.move_to(center + radius * np.array([np.cos(a), np.sin(a), 0])))

    def clearance(chip_, a):
        half_w, half_h = chip_.width / 2 + 0.14, chip_.height / 2 + 0.14
        reach = half_w * abs(np.sin(a)) + half_h * abs(np.cos(a))
        return min(reach / radius, 0.45 * 2 * np.pi / n)

    arcs = m.VGroup()
    for i in range(n):
        a0, a1 = angles[i], angles[(i + 1) % n]
        if a1 > a0:
            a1 -= 2 * np.pi
        start, end = a0 - clearance(nodes[i], a0), a1 + clearance(nodes[(i + 1) % n], a1)
        arc = m.Arc(radius=radius, start_angle=start, angle=end - start, arc_center=center)
        arcs.add(arc.set_stroke(MUTED, 4).add_tip(tip_length=0.2))
    return m.VGroup(nodes, arcs)

# ---------------------------------------------------------------------------- small data components


def token_row(tokens, color=ACTION, size=20, gap=0.08, max_width=12.0, fill=None):
    """Tokens as small mono chips in a row (wrapping at max_width). Returns a VGroup of chips."""
    ref = code("Mx", size)  # every chip has the same height and baseline, whatever its letters
    chips = []
    for tok in tokens:
        shown = tok if tok.strip() else "␣"
        words = code(shown, size, INK if tok.strip() else MUTED)
        box = card(max(words.width + 0.26, 0.44), ref.height + 0.3, color, fill or FILLS.get(color, CARD),
                   radius=0.08, stroke=1.8)
        words.move_to(box)
        sitting = [g for g, ch in zip(words, shown.replace(" ", "")) if ch not in "gjpqy,;_()[]{}"] or list(words)
        if len(words) == len(shown.replace(" ", "")):
            words.shift((box.get_y() - ref.height / 2 - min(g.get_bottom()[1] for g in sitting)) * m.UP)
        chips.append(m.VGroup(box, words))
    rows, row, width = [], [], 0.0
    for c in chips:
        if row and width + c.width > max_width:
            rows.append(m.VGroup(*row).arrange(m.RIGHT, buff=gap))
            row, width = [], 0.0
        row.append(c)
        width += c.width + gap
    if row:
        rows.append(m.VGroup(*row).arrange(m.RIGHT, buff=gap))
    m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=gap)
    return m.VGroup(*chips)


def stat(value, unit="", label="", color=INK, size=72):
    """A big number (DIN) with a small unit and a MUTED label under it."""
    number = title_text(value, size, color)
    parts = [number]
    if unit:
        u = code(unit, max(20, size // 3), MUTED)
        u.next_to(number, m.RIGHT, buff=0.15).align_to(number, m.DOWN)
        parts.append(u)
    group = m.VGroup(*parts)
    if label:
        group.add(text(label, 22, MUTED).next_to(group, m.DOWN, buff=0.18).align_to(number, m.LEFT))
    return group


def cell_grid(rows, cols, size=0.34, gap=0.06, color=EDGE, fill=CARD):
    """A grid of small squares (KV blocks, batch slots × steps …). grid[r][c] is one cell."""
    grid = m.VGroup(*[
        m.VGroup(*[m.Square(size).set_stroke(color, 1.5).set_fill(fill, 1) for _ in range(cols)]).arrange(m.RIGHT, buff=gap)
        for _ in range(rows)
    ]).arrange(m.DOWN, buff=gap)
    return grid


class Meter(m.VGroup):
    """A horizontal gauge (memory used, a budget …). `fill_to(f)` animates it; it turns RED above 90%."""

    def __init__(self, width=4.0, fraction=0.0, color=TEAL, height=0.24):
        super().__init__()
        self.track_width, self.bar_height, self.color = width, height, color
        self.track = m.RoundedRectangle(corner_radius=height / 2, width=width, height=height,
                                        stroke_color=EDGE, stroke_width=2, fill_color=CARD, fill_opacity=1)
        self.bar = self._bar(fraction)
        self.add(self.track, self.bar)

    def _bar(self, fraction):
        f = float(np.clip(fraction, 0.004, 1.0))
        bar = m.RoundedRectangle(corner_radius=self.bar_height / 2, width=self.track_width * f, height=self.bar_height,
                                 stroke_width=0, fill_color=RED if fraction > 0.9 else self.color, fill_opacity=1)
        return bar.align_to(self.track, m.LEFT).set_y(self.track.get_y())

    def fill_to(self, fraction, **kwargs):
        return m.Transform(self.bar, self._bar(fraction).align_to(self.track, m.LEFT), **kwargs)




# ---------------------------------------------------------------------------- reinforcement-learning components


def rover(size=1.0, color=CORAL, fill=CORAL_FILL):
    """Rover, the course's learner: a small robot with a dark visor and two eyes, an antenna with a GOLD tip
    (it wants reward) and two wheels. Parts: [0] wheels, [1] body, [2] visor with eyes, [3] antenna."""
    s = size
    body = m.RoundedRectangle(corner_radius=0.28 * s, width=1.2 * s, height=0.95 * s,
                              stroke_color=color, stroke_width=4 * s, fill_color=fill, fill_opacity=1)
    visor = m.RoundedRectangle(corner_radius=0.16 * s, width=0.86 * s, height=0.42 * s, stroke_width=0,
                               fill_color=BG, fill_opacity=1).move_to(body).shift(0.08 * s * m.UP)
    eyes = m.VGroup(m.Dot(radius=0.075 * s, color=INK), m.Dot(radius=0.075 * s, color=INK)).arrange(m.RIGHT, buff=0.24 * s)
    eyes.move_to(visor)
    top = body.get_top()
    antenna = m.VGroup(m.Line(top, top + [0, 0.28 * s, 0]).set_stroke(color, 4 * s),
                       m.Dot(top + [0, 0.33 * s, 0], radius=0.08 * s, color=GOLD))
    wheels = m.VGroup(*[
        m.Circle(radius=0.17 * s).set_stroke(color, 4 * s).set_fill(BG, 1).move_to(body.get_bottom() + [side * 0.36 * s, 0, 0])
        for side in (-1, 1)
    ])
    return m.VGroup(wheels, body, m.VGroup(visor, eyes), antenna)


def reward_chip(value, size=24):
    """A reward as a small pill: '+1' GOLD, '−1' RED, '0' MUTED (digits as given, minus sign typeset)."""
    v = float(value)
    label = (f"{v:+g}" if v else "0").replace("-", "−")
    color, fill = (GOLD, GOLD_FILL) if v > 0 else (RED, RED_FILL) if v < 0 else (MUTED, CARD)
    return pill(label, color, fill, size, mono=True)


def mix(a, b, t):
    """The hex colour a fraction t of the way from hex colour a to b."""
    ca, cb = (np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) for h in (a, b))
    r, g, bl = (ca + (cb - ca) * float(np.clip(t, 0, 1))).round().astype(int)
    return f"#{r:02X}{g:02X}{bl:02X}"


def heat(value, lo=-1.0, hi=1.0, strength=0.75):
    """The fill colour for a value on the stage: toward BLUE for high, toward RED for negative, CARD near zero."""
    if value >= 0:
        return mix(CARD, BLUE, strength * (0.0 if hi <= 0 else value / hi))
    return mix(CARD, RED, strength * (0.0 if lo >= 0 else value / lo))


class SlotMachine(m.VGroup):
    """One arm of a bandit: a cabinet, a window and a lever. `.pull()` swings the lever down and back (about the
    lever's own foot, so it works after the machine is moved or arranged);
    `.payout(r)` is a reward chip sitting in the window (fade it in, then out); `.window`, `.label`, `.lever`."""

    def __init__(self, label="1", size=1.0, color=GOLD, fill=CARD):
        super().__init__()
        s = size
        self.cabinet = card(1.3 * s, 1.9 * s, color, fill, radius=0.18 * s, stroke=3)
        self.window = card(0.92 * s, 0.62 * s, EDGE, BG, radius=0.08 * s, stroke=2).move_to(self.cabinet).shift(0.28 * s * m.UP)
        self.label = code(label, max(20, int(26 * s)), MUTED).next_to(self.window, m.DOWN, buff=0.24 * s)
        pivot = self.cabinet.get_right() + np.array([0.02 * s, -0.05 * s, 0])
        stick = m.Line(pivot, pivot + np.array([0.26 * s, 0.62 * s, 0])).set_stroke(MUTED, 5 * s)
        knob = m.Dot(stick.get_end(), radius=0.11 * s, color=CORAL)
        self.lever = m.VGroup(stick, knob)
        self.add(self.cabinet, self.window, self.label, self.lever)

    def pull(self, run_time=0.6):
        return m.Rotate(self.lever, angle=-1.1, about_point=self.lever[0].get_start(), rate_func=m.there_and_back,
                        run_time=run_time)

    def payout(self, value, size=22):
        chip_ = reward_chip(value, size)
        if chip_.width > self.window.width - 0.1:
            chip_.scale((self.window.width - 0.1) / chip_.width)
        return chip_.move_to(self.window)


class Bars(m.VGroup):
    """Vertical bars on a baseline: action-value estimates (VALUE), probabilities (ACTION), counts, rewards.
    Data range [bottom, top] maps to the height; negative values hang below the zero line. `.set(values)`
    animates heights and numbers; `.level(v)` is the scene y of data value v; `.marker(i, v, color)` a short
    dashed tick across bar i at v (e.g. the true value); `.bars[i]`, `.labels[i]`, `.numbers[i]`."""

    def __init__(self, values, labels=None, top=1.0, bottom=0.0, width=6.0, height=3.0, color=VALUE, colors=None,
                 gap=0.3, show_values=True, fmt="{:.2f}", size=20, center=(0, 0, 0)):
        super().__init__()
        n = len(values)
        self.top_v, self.bottom_v, self.h, self.fmt, self.size = top, bottom, height, fmt, size
        self.colors = list(colors) if colors else [color] * n
        self.bar_w = (width - gap * (n - 1)) / n
        c = np.array(center, dtype=float)
        self.y0 = c[1] - height / 2
        self.xs = [c[0] - width / 2 + self.bar_w / 2 + i * (self.bar_w + gap) for i in range(n)]
        self.zero = m.Line([c[0] - width / 2 - 0.15, self.level(0), 0], [c[0] + width / 2 + 0.15, self.level(0), 0]).set_stroke(FAINT, 2)
        self.bars = m.VGroup(*[self._rect(i, v) for i, v in enumerate(values)])
        self.labels = m.VGroup(*[code(str(lab), size, MUTED).move_to([x, self.y0 - 0.32, 0])
                                 for x, lab in zip(self.xs, labels or [""] * n)])
        self.numbers = m.VGroup(*[self._number(i, v) for i, v in enumerate(values)]) if show_values else m.VGroup()
        self.add(self.zero, self.bars, self.labels, self.numbers)

    def level(self, v):
        return self.y0 + self.h * (v - self.bottom_v) / (self.top_v - self.bottom_v)

    def _rect(self, i, v):
        y_zero, y_v = self.level(0), self.level(float(np.clip(v, self.bottom_v, self.top_v)))
        hgt = max(abs(y_v - y_zero), 0.015)
        col = self.colors[i]
        rect = m.Rectangle(width=self.bar_w, height=hgt).set_stroke(col, 2).set_fill(FILLS.get(col, CARD), 1)
        return rect.move_to([self.xs[i], (y_zero + y_v) / 2 if abs(y_v - y_zero) > 0.015 else y_zero, 0])

    def _number(self, i, v):
        words = code(self.fmt.format(v).replace("-", "−"), self.size, self.colors[i])
        y = max(self.level(max(v, 0)), self.level(0)) + 0.24
        return words.move_to([self.xs[i], y, 0])

    def set(self, values, run_time=0.8):
        anims = [m.Transform(self.bars[i], self._rect(i, v)) for i, v in enumerate(values)]
        if len(self.numbers):
            anims += [m.Transform(self.numbers[i], self._number(i, v)) for i, v in enumerate(values)]
        return m.AnimationGroup(*anims, run_time=run_time)

    def marker(self, i, v, color=INK):
        y = self.level(v)
        return m.DashedLine([self.xs[i] - self.bar_w / 2 - 0.08, y, 0], [self.xs[i] + self.bar_w / 2 + 0.08, y, 0],
                            dash_length=0.08).set_stroke(color, 3)


class Plot(m.VGroup):
    """A quiet chart for curves (learning curves, regret, probabilities). Axes with a few labelled ticks;
    `.coords(x, y)` maps data to the scene; `.line(xs, ys, color)` a curve (draw it with Create); `.dot(x, y)`;
    `.hline(y)` / `.vline(x)` dashed guides; `.band(xs, lo, hi, color)` a shaded spread; `.area(xs, ys, color)`
    the region under a curve. `log_x=True` for a log x axis. The whole chart, labels included, is centred on
    `center`; the y label sits above the y axis (horizontal, easy to read)."""

    def __init__(self, x_range, y_range, width=7.0, height=4.0, center=(0, 0, 0), x_ticks=None, y_ticks=None,
                 x_label="", y_label="", log_x=False, x_fmt="{:g}", y_fmt="{:g}", size=20):
        super().__init__()
        (self.x0, self.x1), (self.y0, self.y1) = x_range, y_range
        self.w, self.h, self.log_x = width, height, log_x
        self.origin = np.zeros(3)
        x_axis = m.Arrow(self.origin, self.origin + [width + 0.45, 0, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.035)
        y_axis = m.Arrow(self.origin, self.origin + [0, height + 0.4, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.06)
        self.axes = m.VGroup(x_axis, y_axis)
        self.ticks = m.VGroup()
        for x in x_ticks or []:
            p = self.coords(x, self.y0)
            self.ticks.add(m.Line(p, p + [0, -0.09, 0]).set_stroke(FAINT, 2),
                           code(x_fmt.format(x), size, MUTED).next_to(p, m.DOWN, buff=0.16))
        for y in y_ticks or []:
            p = self.coords(self.x0, y)
            self.ticks.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2),
                           code(y_fmt.format(y).replace("-", "−"), size, MUTED).next_to(p, m.LEFT, buff=0.16))
        self.add(self.axes, self.ticks)
        if x_label:
            self.x_label = text(x_label, size + 2, MUTED)
            self.x_label.next_to(self.ticks if len(self.ticks) else x_axis, m.DOWN, buff=0.16).set_x(width / 2)
            self.add(self.x_label)
        if y_label:
            self.y_label = text(y_label, size + 2, MUTED).next_to(y_axis.get_end(), m.UP, buff=0.12).align_to(y_axis, m.LEFT)
            self.add(self.y_label)
        offset = np.array(center, dtype=float) - self.get_center()
        self.shift(offset)
        self.origin = self.origin + offset

    def coords(self, x, y):
        if self.log_x:
            fx = (np.log10(x) - np.log10(self.x0)) / (np.log10(self.x1) - np.log10(self.x0))
        else:
            fx = (x - self.x0) / (self.x1 - self.x0)
        fy = (y - self.y0) / (self.y1 - self.y0)
        return self.origin + np.array([fx * self.w, fy * self.h, 0])

    def line(self, xs, ys, color=VALUE, width=4):
        pts = [self.coords(x, float(np.clip(y, self.y0, self.y1))) for x, y in zip(xs, ys)]
        return m.VMobject().set_points_as_corners(pts).set_stroke(color, width).set_fill(opacity=0)

    def dot(self, x, y, color=INK, radius=0.08):
        return m.Dot(self.coords(x, y), radius=radius, color=color).set_z_index(3)

    def hline(self, y, color=MUTED, width=2):
        return m.DashedLine(self.coords(self.x0, y), self.coords(self.x1, y), dash_length=0.1).set_stroke(color, width)

    def vline(self, x, color=MUTED, width=2):
        return m.DashedLine(self.coords(x, self.y0), self.coords(x, self.y1), dash_length=0.1).set_stroke(color, width)

    def band(self, xs, lo, hi, color=VALUE, opacity=0.22):
        pts = [self.coords(x, float(np.clip(y, self.y0, self.y1))) for x, y in zip(xs, hi)]
        pts += [self.coords(x, float(np.clip(y, self.y0, self.y1))) for x, y in zip(xs[::-1], lo[::-1])]
        return m.Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)

    def area(self, xs, ys, color=VALUE, opacity=0.25):
        return self.band(xs, [self.y0] * len(xs), ys, color, opacity)


class GridView(m.VGroup):
    """A grid world as square cells, (row, col) with row 0 at the TOP. `walls` are rocks; `specials` maps
    (row, col) → (icon name, label, colour), e.g. {(0, 3): ("charger", "+1", GOLD), (1, 3): ("puddle", "−1", RED)}.
    `.cell(r, c)` is a cell's centre; `.squares[(r, c)]` the drawn square; `.values(V, …)` the animation that tints
    every open cell by its value and writes the number (call it again to update); `.arrows(policy, …)` a VGroup of
    CORAL arrows (policy maps (r, c) → 'N', 'E', 'S' or 'W'); `.path(cells, color)` a polyline through centres."""

    DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

    def __init__(self, rows, cols, cell=1.3, center=(0, 0, 0), walls=(), specials=None, gap=0.06, icon_scale=0.5):
        super().__init__()
        self.rows, self.cols, self.size = rows, cols, cell
        c = np.array(center, dtype=float)
        self.top_left = c + np.array([-cols * cell / 2, rows * cell / 2, 0])
        self.walls, self.specials = set(walls), dict(specials or {})
        self.squares, self.numbers = {}, {}
        tiles = m.VGroup()
        for r in range(rows):
            for col in range(cols):
                sq = m.Square(cell - gap).move_to(self.cell(r, col))
                if (r, col) in self.walls:
                    sq.set_stroke(EDGE, 2).set_fill("#2A2238", 1)
                else:
                    sq.set_stroke(EDGE, 2).set_fill(CARD, 1)
                self.squares[(r, col)] = sq
                tiles.add(sq)
        marks = m.VGroup()
        for (r, col) in self.walls:
            marks.add(icon("rock", cell * icon_scale).move_to(self.cell(r, col)))
        for (r, col), (name, label, color) in self.specials.items():
            self.squares[(r, col)].set_stroke(color, 3).set_fill(FILLS.get(color, CARD), 1)
            mark = m.VGroup(icon(name, cell * icon_scale).move_to(self.cell(r, col))) if name else m.VGroup()
            if label:
                mark.add(code(label.replace("-", "−"), 20, color, weight=m.BOLD)
                         .move_to(self.cell(r, col) + np.array([0, -0.33 * cell, 0])))
                if name:
                    mark[0].shift(0.1 * cell * m.UP)
            marks.add(mark)
        self.tiles, self.marks = tiles, marks
        self.add(tiles, marks)

    def cell(self, r, c):
        return self.top_left + np.array([(c + 0.5) * self.size, -(r + 0.5) * self.size, 0])

    @classmethod
    def from_world(cls, world, cell=1.3, center=(0, 0, 0), goal=("charger", GOLD), bad=("puddle", RED), **kwargs):
        """A GridView of an rl_reference world, whose cells are (col, row) with row 1 at the BOTTOM: walls become
        rocks, a terminal with reward ≥ 0 the charger (labelled with its reward if non-zero), a negative one the
        puddle, cliff cells plain RED squares. Then use the world-coordinate helpers `.at(s)`, `.world_values(V)`,
        `.world_arrows(policy)` and `.world_path(states)`, which take rl_reference's states directly."""
        to = world.to_screen
        specials = {}
        for s, r in world.terminals.items():
            name, color = goal if r >= 0 else bad
            specials[to(s)] = (name, f"{r:+g}" if r else "", color)
        for s in getattr(world, "cliffs", ()):
            specials[to(s)] = (None, "", RED)
        view = cls(world.height, world.width, cell=cell, center=center, walls=[to(s) for s in world.walls],
                   specials=specials, **kwargs)
        view.world = world
        return view

    def at(self, s):
        """The centre of world state s = (col, row), row 1 at the bottom (needs from_world)."""
        return self.cell(*self.world.to_screen(s))

    def world_values(self, V, **kwargs):
        return self.values({self.world.to_screen(s): v for s, v in V.items()}, **kwargs)

    def world_arrows(self, policy, **kwargs):
        return self.arrows({self.world.to_screen(s): a for s, a in policy.items()}, **kwargs)

    def world_path(self, states, **kwargs):
        return self.path([self.world.to_screen(s) for s in states], **kwargs)

    def open_cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.walls and (r, c) not in self.specials]

    def values(self, V, fmt="{:.2f}", lo=-1.0, hi=1.0, size=20, offset=0.0, run_time=0.8, color=INK):
        """V maps (r, c) → value for open cells. Tints squares (heat) and writes numbers `offset` cells below centre."""
        anims = []
        for (r, c), v in V.items():
            if (r, c) in self.walls or (r, c) in self.specials:
                continue
            anims.append(self.squares[(r, c)].animate.set_fill(heat(v, lo, hi), 1))
            words = code(fmt.format(v).replace("-", "−"), size, color).move_to(self.cell(r, c) + [0, -offset * self.size, 0])
            if (r, c) in self.numbers:
                anims.append(m.Transform(self.numbers[(r, c)], words))
            else:
                self.numbers[(r, c)] = words
                anims.append(m.FadeIn(words))
        return m.AnimationGroup(*anims, run_time=run_time)

    def arrows(self, policy, color=ACTION, length=0.5, offset=0.0, width=5):
        group = m.VGroup()
        for (r, c), d in policy.items():
            if d not in self.DIRS or (r, c) in self.walls:
                continue
            v = np.array([*self.DIRS[d], 0], dtype=float) * length * self.size / 2
            p = self.cell(r, c) + [0, offset * self.size, 0]
            group.add(m.Arrow(p - v, p + v, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.4))
        return group

    def path(self, cells, color=ACTION, width=6):
        return m.VMobject().set_points_as_corners([self.cell(r, c) for r, c in cells]).set_stroke(color, width)


class NeuralNet(m.VGroup):
    """A small network diagram: columns of nodes joined by thin lines. `layers` = nodes per column (a column with
    more than `max_nodes` shows dots for the hidden ones). `.columns[i]`, `.edges[i]` (between i and i+1);
    `.forward(color)` a flash sweeping left to right."""

    def __init__(self, layers=(4, 8, 8, 2), width=4.5, height=3.0, center=(0, 0, 0), color=VIOLET, radius=0.11, max_nodes=8):
        super().__init__()
        c = np.array(center, dtype=float)
        n = len(layers)
        self.columns, self.edges = m.VGroup(), m.VGroup()
        for i, k in enumerate(layers):
            shown = min(k, max_nodes)
            x = c[0] - width / 2 + width * i / max(n - 1, 1)
            ys = np.linspace(c[1] + height / 2, c[1] - height / 2, shown) if shown > 1 else [c[1]]
            col = m.VGroup(*[m.Circle(radius=radius).set_stroke(color, 2.5).set_fill(BG, 1).move_to([x, y, 0]) for y in ys])
            self.columns.add(col)
        for a, b in zip(self.columns[:-1], self.columns[1:]):
            self.edges.add(m.VGroup(*[m.Line(p.get_center(), q.get_center(), buff=radius).set_stroke(EDGE, 1.5)
                                      for p in a for q in b]))
        self.add(self.edges, self.columns)

    def forward(self, color=VIOLET, run_time=1.2):
        flashes = [m.ShowPassingFlash(e.copy().set_stroke(color, 3), time_width=0.6) for e in self.edges]
        return m.LaggedStart(*flashes, lag_ratio=0.7, run_time=run_time)


class CartPoleView(m.VGroup):
    """CartPole as gymnasium draws it: a cart on a track (±2.4 m are the edges) and a hinged pole (1 m long,
    angle 0 = upright, positive leans right). `.set_state(x, theta)` moves it in place; `.replay(states, seconds)`
    is an animation that walks through a trajectory of (x, x_dot, theta, theta_dot) rows. Both read the track's
    current position and length, so the view may be moved or scaled after it is built."""

    def __init__(self, center=(0, 0, 0), units_per_m=1.25, pole_units=1.7, color=STATE, pole_color=ACTION):
        super().__init__()
        self.c, self.k, self.pole_len = np.array(center, dtype=float), units_per_m, pole_units
        half = 2.4 * units_per_m
        self.track = m.Line(self.c + [-half - 0.4, 0, 0], self.c + [half + 0.4, 0, 0]).set_stroke(FAINT, 3)
        self.track_len0 = 2 * half + 0.8
        self.edges = m.VGroup(*[m.Line(self.c + [s * half, -0.15, 0], self.c + [s * half, 0.15, 0]).set_stroke(RED, 3) for s in (-1, 1)])
        self.cart = card(0.9, 0.42, color, FILLS.get(color, CARD), radius=0.08, stroke=3)
        self.pole = m.Line(np.zeros(3), [0, pole_units, 0]).set_stroke(pole_color, 9)
        self.hinge = m.Dot(radius=0.07, color=INK)
        self.add(self.track, self.edges, self.cart, self.pole, self.hinge)
        self.set_state(0.0, 0.0)

    def set_state(self, x, theta):
        s = self.track.get_length() / self.track_len0  # the view may have been moved or scaled since it was built
        base = self.track.get_center() + np.array([x * self.k * s, 0.26 * s, 0])
        self.cart.move_to(base)
        top = base + self.pole_len * s * np.array([np.sin(theta), np.cos(theta), 0])
        self.pole.put_start_and_end_on(base, top)
        self.hinge.move_to(base)
        return self

    def replay(self, states, seconds):
        states = np.asarray(states, dtype=float)

        def step(_, alpha):
            i = alpha * (len(states) - 1)
            j = int(np.floor(i))
            k = min(j + 1, len(states) - 1)
            f = i - j
            x = (1 - f) * states[j][0] + f * states[k][0]
            th = (1 - f) * states[j][2] + f * states[k][2]
            self.set_state(x, th)

        return m.UpdateFromAlphaFunc(self, step, run_time=seconds, rate_func=m.linear)


class PendulumView(m.VGroup):
    """The pendulum of Pendulum-v1: a rod from a fixed pivot, angle 0 = upright, positive = counter-clockwise.
    `.set_angle(theta)`; `.replay(thetas, seconds)` walks through a list of angles (both read the pivot's current
    place and the ring's current size, so the view may be moved or scaled)."""

    def __init__(self, center=(0, 0, 0), length=1.6, color=ACTION):
        super().__init__()
        self.c, self.length = np.array(center, dtype=float), length
        self.ring = m.Circle(radius=length).move_to(self.c).set_stroke(EDGE, 2).set_fill(opacity=0)
        self.rod = m.Line(self.c, self.c + [0, length, 0]).set_stroke(color, 10)
        self.bob = m.Dot(radius=0.16, color=color)
        self.pivot = m.Dot(self.c, radius=0.08, color=INK)
        self.add(self.ring, self.rod, self.bob, self.pivot)
        self.set_angle(np.pi)

    def set_angle(self, theta):
        c, length = self.pivot.get_center(), self.ring.width / 2
        tip = c + length * np.array([-np.sin(theta), np.cos(theta), 0])
        self.rod.put_start_and_end_on(c, tip)
        self.bob.move_to(tip)
        return self

    def replay(self, thetas, seconds):
        thetas = np.unwrap(np.asarray(thetas, dtype=float))

        def step(_, alpha):
            i = alpha * (len(thetas) - 1)
            j = int(np.floor(i))
            k = min(j + 1, len(thetas) - 1)
            self.set_angle((1 - (i - j)) * thetas[j] + (i - j) * thetas[k])

        return m.UpdateFromAlphaFunc(self, step, run_time=seconds, rate_func=m.linear)


def agent_world_loop(center=(0, 0, 0), width=8.0, size=1.0, labels=("action", "state", "reward")):
    """The reinforcement-learning loop: Rover (the agent, left) and the world (right). One CORAL arrow carries the
    action over the top; a TEAL arrow (the next state) and a GOLD arrow (the reward) come back underneath.
    Returns a VGroup with attributes .agent, .world, .action, .state, .reward (each arrow is VGroup(arrow, label))."""
    c = np.array(center, dtype=float)
    s = size
    agent_box = card(2.4 * s, 2.3 * s, ACTION, CARD)
    agent = m.VGroup(agent_box, rover(0.95 * s).move_to(agent_box).shift(0.18 * s * m.UP),
                     text("agent", 22, MUTED).move_to(agent_box.get_bottom() + [0, 0.32 * s, 0]))
    world_box = card(2.4 * s, 2.3 * s, STATE, CARD)
    mini = cell_grid(3, 3, size=0.36 * s, gap=0.06 * s, color=EDGE, fill=BG).move_to(world_box).shift(0.18 * s * m.UP)
    mini[0][2].set_fill(GOLD_FILL, 1).set_stroke(GOLD, 2)
    world = m.VGroup(world_box, mini, text("world", 22, MUTED).move_to(world_box.get_bottom() + [0, 0.32 * s, 0]))
    agent.move_to(c + [-width / 2 + 1.2 * s, 0, 0])
    world.move_to(c + [width / 2 - 1.2 * s, 0, 0])
    a0, a1 = agent_box.get_right(), world_box.get_left()

    def link(y, start, end, color, label, above):
        arr = m.Arrow(start + [0, y, 0], end + [0, y, 0], buff=0.12, color=color, stroke_width=6,
                      max_tip_length_to_length_ratio=0.12)
        words = code(label, 22, color, weight=m.BOLD).next_to(arr, m.UP if above else m.DOWN, buff=0.1)
        return m.VGroup(arr, words)

    action = link(0.55 * s, a0, a1, ACTION, labels[0], True)
    state = link(-0.25 * s, a1, a0, STATE, labels[1], True)
    reward = link(-0.7 * s, a1, a0, REWARD, labels[2], False)
    loop = m.VGroup(agent, world, action, state, reward)
    loop.agent, loop.world, loop.action, loop.state, loop.reward = agent, world, action, state, reward
    return loop


# ---------------------------------------------------------------------------- the episode


class Episode(m.Scene):
    """An episode of the course: its intro card, its lesson (`lesson()`), the next-episode card."""

    number = 0
    voice = draft_voice if os.environ.get("VOICE") == "draft" else narrator
    say = _say_namespace["say"]

    def construct(self) -> None:
        self.head = None
        self.add(backdrop().set_z_index(-100))
        self.series_intro()
        self.lesson()
        self.series_outro()

    def lesson(self) -> None:
        raise NotImplementedError

    @property
    def module(self):
        return MODULES[EPISODES[self.number][1]]

    level = module

    def new_header(self, tag, title, color=None, fill=None):
        """The animation that swaps the header: pass it to say() with a short [span]."""
        _, accent, tint = self.module
        new = header(tag, title, color or accent, fill or tint)
        parts = [m.FadeIn(new, shift=0.25 * m.UP)]
        if self.head:
            parts.insert(0, m.FadeOut(self.head, shift=0.25 * m.UP))
        self.head = new
        return m.AnimationGroup(*parts, run_time=0.9)

    def clear(self, *keep, run_time=0.8):
        gone = [mob for mob in self.mobjects[1:] if mob not in keep and mob is not self.head]
        if gone:
            self.play(*[m.FadeOut(mob) for mob in gone], run_time=run_time)

    def drop_header(self):
        old, self.head = self.head, None
        return m.FadeOut(old, shift=0.25 * m.UP) if old else m.Wait(0.01)

    def _fit(self, heading):
        if heading.width > TITLE_WIDTH:
            heading.scale(TITLE_WIDTH / heading.width)
        return heading

    def series_intro(self):
        title, module = EPISODES[self.number]
        name, accent, tint = MODULES[module]
        self.next_section(f"Episode {self.number} · {title}", notes="Series card.")
        halo = glow(accent, radius=4.2).move_to(0.1 * m.DOWN)
        prompt = code(f"$ rover train --episode {self.number:02d}", 22, MUTED, t2c={r"^\$": GREEN})
        brand = title_text("Trial and Reward", 44, INK, gradient=(GOLD, CORAL))
        badge = pill(f"MODULE {module} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 76 if len(title) < 22 else 60))
        top = m.VGroup(prompt, brand).arrange(m.DOWN, buff=0.22)
        group = m.VGroup(top, badge, heading).arrange(m.DOWN, buff=0.45).move_to(0.25 * m.UP)
        half = min(heading.width, 6) / 2
        rule = m.Line([-half, 0, 0], [half, 0, 0]).set_stroke(accent, width=5).next_to(heading, m.DOWN, buff=0.35)
        mascot = rover(0.5).next_to(rule, m.DOWN, buff=0.45)

        self.wait(0.3)
        self.say(
            f"[Welcome to Trial and Reward.] [Episode {say_number(self.number)}]: [{spoken_title(title)}.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.AddTextLetterByLetter(prompt), m.FadeIn(brand, shift=0.2 * m.DOWN), run_time=1.2),
            m.AnimationGroup(m.FadeIn(badge, shift=0.2 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Write(heading), m.GrowFromCenter(rule)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group, rule, mascot)], run_time=0.8)

    def series_outro(self):
        upcoming = self.number + 1
        self.next_section("Next episode", notes="Next-episode card.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        if upcoming not in EPISODES:
            self.say("Thanks for watching!")
            return
        title, module = EPISODES[upcoming]
        name, accent, tint = MODULES[module]
        halo = glow(accent, radius=4.0)
        kicker = code("next episode", 24, MUTED)
        number = pill(f"EP {upcoming:02d} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 64 if len(title) < 26 else 50))
        group = m.VGroup(kicker, number, heading).arrange(m.DOWN, buff=0.4).move_to(0.2 * m.UP)
        self.say(
            f"[Next time, in episode {say_number(upcoming)}]: [{spoken_title(title)}.] Thanks for watching!",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(kicker), m.FadeIn(number, shift=0.2 * m.UP)),
            m.Write(heading),
        )
        self.wait(1.0)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
