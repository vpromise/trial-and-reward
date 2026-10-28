"""Write one subagent brief per episode (briefs/epNN.md) for "Trial and Reward", from the rows of COURSE.md,
the per-episode notes in SPEC (briefs/spec.py), briefs/pronunciation.txt and the series notes below.
Run: python3 briefs/make_briefs.py [N ...]"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE.parent
VIDEOS = SERIES_DIR.parent
sys.path.insert(0, str(HERE))
from spec import SPEC  # noqa: E402

NARRATOR_PYTHON = "~/rag-video/tts/.venv/bin/python"
MODULE_NAMES = {1: "TRIAL AND REWARD", 2: "PLANNING IN A KNOWN WORLD", 3: "LEARNING FROM EXPERIENCE", 4: "DEEP VALUE LEARNING",
                5: "LEARNING THE POLICY", 6: "TEACHING LANGUAGE MODELS WITH REWARD"}


def _scene(series, n):
    hit = sorted(VIDEOS.glob(f"{series}/ep{n:02d}-*/scene.py"))
    return str(hit[0]) if hit else None


AI = {8: "the four ways to learn, a robot's rewards", 30: "RLHF & DPO: preference pairs, the DPO loss",
      43: "reasoning models, verifiable rewards, majority voting", 46: "alignment, reward hacking, a model gaming its reward"}
AG = {30: "training agents with RL: sixteen rollouts in sandboxes, group-relative advantages, a toy update"}
IN = {2: "a measured number always paired with its ceiling; the roofline", 10: "a slot grid stepping through time",
      13: "formulas, an accept/reject strip, a measured comparison", 24: "plots and a step-by-step fixed point",
      25: "a six-panel dashboard driven by a simulation"}


def references(keys):
    out = []
    for key in keys:
        kind, n = key[:2], int(key[2:])
        if kind == "ag":
            path, what, label = _scene("agent-series", n), AG.get(n, ""), "'Agents, Animated' (sister course, OLD palette)"
        elif kind == "ai":
            path, what, label = _scene("ai-series", n), AI.get(n, ""), "'AI, Explained' (sister course, OLD palette)"
        elif kind == "in":
            path, what, label = _scene("inference-series", n), IN.get(n, ""), "'Tokens per Second' (the latest sister course: match its polish, NOT its palette)"
        else:
            path, what = _scene("rl-series", n), "THIS course, THIS palette — match it"
            label = f"this course's Episode {n}"
            if not path:
                out.append(f"{label} (not made yet — skip)")
                continue
        out.append(f"{path} — {label}: {what}")
    return "; ".join(out)


TEMPLATE = """You are producing ONE narrated animated tutorial for the course "Trial and Reward" — reinforcement learning, from slot machines to reasoning models (Mac, Apple Silicon). Work fully autonomously and finish with a short report.

YOUR EPISODE: Episode {N} · "{TITLE}" (Module {M} · {MODULE_NAME}). Folder: {SERIES_DIR}/ep{NN}-{slug}/ (create it). Scene class: `Ep{NN}{ClassName}(Episode)` with `number = {N}`.
Goal (what the viewer will understand): {GOAL}
Key concepts: {CONCEPTS}
Mechanism to animate: {MECHANISM}
Running example: {EXAMPLE}
EPISODE NOTES: {NOTES}

READ FIRST, in this order:
1. ~/.claude/skills/tutorial-video/SKILL.md, then reference/narration.md, reference/design.md, reference/manimgx.md (gotchas 19–31 matter), reference/series.md.
2. {SERIES_DIR}/COURSE.md (the whole course: what comes before and after yours — link to other episodes by number, never repeat them at length).
3. {SERIES_DIR}/kit.py — the visual system (deep plum stage; Gill Sans titles, Avenir Next text, Menlo code) and the RL components: rover() (the course's learner), reward_chip, SlotMachine (pull, payout), Bars (estimates, probabilities; set, marker), Plot (curves, bands, areas, log x), GridView (cells, values, arrows, path; GridView.from_world(world) with .at/.world_values/.world_arrows/.world_path for rl_reference worlds), NeuralNet (forward flash), CartPoleView / PendulumView (set_state / replay a trajectory), agent_world_loop, heat/mix, plus code_block/terminal/json_card/message/token_row/stat/cell_grid/Meter/tex; header via self.new_header(tag, title). Look at its storyboards ({SERIES_DIR}/KitDemo.storyboard*.png, already current — do not re-run kit_demo.py, it writes outside your folder).
4. {SERIES_DIR}/rl_reference.py — the canonical algorithms and numbers (numpy, seeded, self-checked: bandits, GridWorld/ROVER_WORLD/value_iteration/policy_iteration, MC/TD/Q-learning/SARSA/CLIFF_WORLD/SPARSE_MAZE, Baird, maximization bias, CartPole/Pendulum ports, load_policy, TicTacToe/mcts, softmax/gae/ppo_clip_objective/bradley_terry_prob/dpo_loss/grpo_advantages/kl_k3/pass_at_k, measured()). Read its docstring and the functions you use. Import it in your scene (`from rl_reference import *` after the sys.path line) and COMPUTE every number you show from it (fast, seeded calls are fine at render time) or load it with measured(...); any code you show is an excerpt (≤ 10 lines) in its style and names.
5. {SERIES_DIR}/research/notes.md — sections: {NOTES_SECTIONS}; plus its Appendix A (where sources disagree — use the version it recommends) and Appendix C (your episode's must-get-right numbers). And the runs on THIS Mac (Apple M4 Max, 64 GB): {SERIES_DIR}/research/experiments/classic.md, deep_rl.md, lm_rl.md (load numbers with measured(...) — never re-run the experiments yourself). Open the original sources the notes cite when you need more detail (WebFetch).
6. Style references (read their scene.py and look at a few storyboard PNGs; sister courses share the engine and pacing but use OTHER palettes — copy structure and care, NOT colours or fonts): {REFERENCE_EPISODES}.

Match the series: one running example carried through the episode (Rover unless a better example teaches better — then say so), a header (mono tag + Gill Sans title via self.new_header) per section, sections via self.next_section(...), ~38–46 narrated self.say(...) lines, 3.5–4.5 minutes, a three-word recap near the end. Show mechanisms MOVING: Rover pulling levers, estimates growing, values spreading across the grid, errors sparking between neighbours, curves being drawn, poles balancing. Every number on screen is computed in the scene (from rl_reference / measured()) and asserted, or cited with its source; invented data is said "for example" and recorded as illustrative in the section notes.

HARD RULES (other agents work in the same series folder at the same time):
- Write only inside your episode folder. Do NOT edit kit.py, rl_reference.py, COURSE.md, lexicon.json, research/, other episodes, or existing files in {SERIES_DIR}/assets/. You MAY add NEW icon SVGs to assets/ named `ep{NN}-*.svg` (flat 2-tone, this palette — see kit.py), and recoloured copies via kit's recolored(name, color, fill, prefix="ep{NN}").
- Do not run training, benchmarks or models (no new experiments, no downloads): the runs are done and shared. Calling rl_reference functions in the scene is fine.
- The narrator is ALREADY RUNNING at http://127.0.0.1:8765/speak, shared by several agents (requests queue). NEVER start, stop or restart it. If a request fails, wait a minute and retry. Its log is {SERIES_DIR}/narrator.log (shared, appended).
- Python with Whisper: {NARRATOR_PYTHON}. ManimGX CLI: manimgx. Keep each response and tool output compact (pipe long logs through tail/grep): long turns time out. In the shell, `ls` is aliased to eza, which can hang in a background shell: use `/bin/ls`.

PROCESS:
1. Write the script + scene. 2. Draft pass: `cd <folder> && VOICE=draft manimgx inspect scene.py` until exit 0 and "layout: no problems in N samples"; READ EVERY storyboard PNG and fix what you see (overlaps, cramped or empty frames, text over lines, misaligned baselines, faint or half-drawn objects, anything closer than 0.5 units to the frame edge). 3. Voice pass: `manimgx inspect scene.py`. For a line below 100% after 4 takes, transcribe the clip (librosa 16 kHz → mlx_whisper, path_or_hf_repo="mlx-community/whisper-large-v3-turbo") to decide: checker false alarm (keep) or real misread (reword, `python3 ~/.claude/skills/tutorial-video/scripts/drop_lines.py voice '<regex>'`, re-run inspect). 4. `python3 ~/.claude/skills/tutorial-video/scripts/drop_lines.py voice . --voice-id draft_voice`. 5. `manimgx render scene.py`, then `manimgx present scene.py --no-open`. 6. From INSIDE your folder: `{NARRATOR_PYTHON} ~/.claude/skills/tutorial-video/scripts/verify_audio.py <Scene>.mp4 --terms AI,<every ALL-CAPS acronym you say>`; and a sync check of 2 cues with `manimgx inspect scene.py -t a,b -o /tmp/ep{NN}_sync.png` (Read it).

FINAL REPORT (short): paths of MP4/SRT/HTML/scene.py; duration; layout result; verify_audio summary; lines reworded or below 100% and why; new icons/helpers; facts and numbers you checked (with sources) and anything you are unsure about.
"""

COMMON = """
SERIES NOTES (apply to every episode):
- Visual language (keep it consistent — viewers learn it once): the colours carry meaning across the course — reward GOLD · values (V, Q, returns, estimates) BLUE · policy, actions and the language model's tokens CORAL · states and the world TEAL · exploration and uncertainty VIOLET · the learning signal (TD error, advantage, gradient, update) PINK · good GREEN · bad RED · structure EDGE/MUTED. kit.py names them REWARD, VALUE, ACTION (= POLICY), STATE (= WORLD), EXPLORE, SIGNAL. Headers use your module's accent (kit picks it). Display text (titles, big numbers, recap words) in title_text (Gill Sans Bold; text containing a digit is set in Avenir Next Bold automatically, because Gill Sans draws \"1\" like \"I\"); reading text in text() (Avenir Next); code, units, tables of numbers, commands and exact strings in code() / code_block / json_card / terminal (Menlo).
- The running example: Rover, a small robot that learns only from reward (rover(), CORAL, a GOLD antenna tip). Rover's five slot machines are ROVER_ARMS (machine 4 is the best, 0.60 — in code arms count from 0, so it is arm 3). Rover's world is the Russell & Norvig 4×3 grid (ROVER_WORLD: charger +1, puddle −1, a rock, −0.04 per step, 80% intended move); the cliff walk is CLIFF_WORLD; the sparse maze SPARSE_MAZE. Rover appears small, as a presence, not in every frame. Module 6 replaces Rover with a small language model (the policy is the model; its tokens are CORAL actions).
- Facts: prefer research/notes.md (each fact is sourced and dated) and primary sources (papers, Sutton & Barto 2nd ed.). Show "Source, Year" in small MUTED text (font_size ≥ 20) near a claim that needs one, e.g. "Sutton & Barto, 2018, Fig. 6.4", "Mnih et al., 2015", "Shao et al., 2024". Fast-moving facts (model releases, leaderboards) are dated (October 2026) in the section notes.
- Measured runs: load them with measured(...) and say "trained on this Mac" (or "measured here") on screen, with the number of seeds/runs and the spread when the notes give it (e.g. "5 seeds · mean ± std"). Never round a measured number into a stronger claim; when a result is surprising (UCB1 trailing ε-greedy on Rover's arms at 1,000 pulls, CartPole being easy), say so honestly — that is part of the lesson.
- No real logos, screenshots or photos; draw games, robots and models as simple cards/icons (assets: charger, puddle, rock, trophy, dice, gamepad, robot-arm, llm, user, …). Product and paper names are fine as text.
- Numbers as words in narration ("zero point six", "eighty percent", "a thousand pulls"); digits are fine on screen. Avoid spoken symbols and single letters where possible (say "epsilon", "gamma", "the value of state s" — or rephrase).
{PRONUNCIATION}
- ManimGX: kit.py already fixes TransformFromCopy/FadeTransform early-show and parts animated after their group's intro in the same line. Still: never Indicate/Wiggle on a LATER cue an object whose intro may still be running (use Circumscribe); probe any doubtful line with inspect -t just before/after its cue. Within one line, fade out the SAME group object you faded in (gotcha 24). Chained .animate.shift(v) in one Succession land wrong: use absolute move_to (gotcha 25). Never name a scene method or attribute voice, film, say or clock (gotchas 26, 31). Axes ranges are cut to whole steps (gotcha 27) — prefer kit's Plot. Store your own geometry under your own names, never in .width/.height (gotcha 30).
- Text: Text() strips leading spaces of a single line (indent by position; multi-line strings keep indentation). Gill Sans and Avenir merge "fi"/"fl" into one glyph: glyph-indexing helpers (glyphs(), paint()) are for mono text only. code() / code_block / json_card / terminal colour code by regex already.
- Margins: keep every object ≥ 0.5 units inside the frame edges (the layout check only catches cut-offs, not text flush against the edge), and make sure cards placed late never cover earlier arrows or labels still on screen.
"""


def course_rows():
    rows = {}
    for line in (SERIES_DIR / "COURSE.md").read_text().splitlines():
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) == 7 and cells[0].isdigit():
            rows[int(cells[0])] = dict(title=cells[1].strip("*"), goal=cells[2], concepts=cells[3], mechanism=cells[4],
                                       example=cells[5])
    return rows


def module_of(n):
    src = (SERIES_DIR / "kit.py").read_text()
    return {int(a): int(c) for a, b, c in re.findall(r"(\d+): \(\"(.+?)\", (\d)\)", src)}[n]


def brief(n, rows):
    slug, cls, refs, sections, notes = SPEC[n]
    row, module = rows[n], module_of(n)
    pron = (HERE / "pronunciation.txt").read_text().strip() if (HERE / "pronunciation.txt").exists() else ""
    fill = {
        "{N}": str(n), "{NN}": f"{n:02d}", "{TITLE}": row["title"], "{M}": str(module),
        "{MODULE_NAME}": MODULE_NAMES[module], "{SERIES_DIR}": str(SERIES_DIR), "{slug}": slug,
        "{ClassName}": cls, "{GOAL}": row["goal"], "{CONCEPTS}": row["concepts"],
        "{MECHANISM}": row["mechanism"], "{EXAMPLE}": row["example"], "{NOTES}": notes,
        "{NOTES_SECTIONS}": sections, "{REFERENCE_EPISODES}": references(refs), "{NARRATOR_PYTHON}": NARRATOR_PYTHON,
    }
    text = TEMPLATE
    for key, value in fill.items():
        text = text.replace(key, value)
    leftover = re.findall(r"\{(?:[A-Z_]+|slug|ClassName)\}", text)
    assert not leftover, (n, leftover)
    return text.rstrip() + "\n" + COMMON.replace("{PRONUNCIATION}", pron)


def main():
    rows = course_rows()
    for n in [int(a) for a in sys.argv[1:]] or sorted(SPEC):
        path = HERE / f"ep{n:02d}.md"
        path.write_text(brief(n, rows))
        print(path.name, len(path.read_text().split()), "words")


if __name__ == "__main__":
    main()
