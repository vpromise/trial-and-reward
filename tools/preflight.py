"""Pronunciation pre-flight for "Trial and Reward": speak one sentence per risky term through the running
narrator (which reads lexicon.json), then transcribe each clip with Whisper large and small, and report
which terms both listeners heard. Run with the narrator env, from the series folder:
    ~/rag-video/tts/.venv/bin/python tools/preflight.py [out_dir]"""

import base64
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import librosa
import mlx_whisper
import soundfile

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "research/preflight")
OUT.mkdir(parents=True, exist_ok=True)

# (term as written in the script, sentence, regex a correct transcript must match)
TESTS = [
    ("RL", "This course is about RL, learning from reward.", r"\brl\b|r l\b|\bour l\b"),
    ("SARSA", "SARSA learns the value of the action it actually takes.", r"sarsa|sarza|salsa"),
    ("Q-learning", "Q-learning learns the value of the best next action.", r"q[- ]?learning|cue learning"),
    ("GRPO", "GRPO compares a group of answers to each other.", r"\bgrpo\b|g ?r ?p ?o"),
    ("PPO", "PPO clips each update so the policy cannot move too far.", r"\bppo\b|p ?p ?o"),
    ("DQN", "DQN learned to play Atari games from pixels.", r"\bdqn\b|d ?q ?n"),
    ("DPO", "DPO skips the reward model entirely.", r"\bdpo\b|d ?p ?o"),
    ("RLHF", "RLHF trains a model on human preferences.", r"\brlhf\b|r ?l ?h ?f"),
    ("MDP", "An MDP describes the states, actions and rewards.", r"\bmdp\b|m ?d ?p"),
    ("TD", "TD learning updates a guess from the next guess.", r"\btd\b|t ?d\b"),
    ("UCB", "UCB adds a bonus for uncertainty.", r"\bucb\b|u ?c ?b"),
    ("LinUCB", "LinUCB fits one line per arm.", r"lin ?u ?c ?b|linucb"),
    ("KL", "A KL penalty keeps the model close to the reference.", r"\bkl\b|k ?l\b"),
    ("GAE", "GAE blends short and long returns.", r"\bgae\b|g ?a ?e"),
    ("MCTS", "MCTS grows a search tree one simulation at a time.", r"\bmcts\b|m ?c ?t ?s"),
    ("TRPO", "TRPO keeps each step inside a trust region.", r"\btrpo\b|t ?r ?p ?o"),
    ("SAC", "SAC rewards the policy for staying random.", r"\bsac\b|s ?a ?c\b|\bsack\b"),
    ("REINFORCE", "REINFORCE is the simplest policy gradient.", r"reinforce"),
    ("CartPole", "CartPole asks the agent to balance a pole.", r"cart ?pole"),
    ("AlphaGo", "AlphaGo beat Lee Sedol four games to one.", r"alpha ?go"),
    ("AlphaZero", "AlphaZero learned chess from self-play alone.", r"alpha ?zero"),
    ("MuZero", "MuZero learned its own model of the rules.", r"mu ?zero|mew ?zero|muzero"),
    ("Atari", "Atari games were the first big test.", r"atari"),
    ("epsilon", "With epsilon-greedy, Rover explores ten percent of the time.", r"epsilon"),
    ("Thompson", "Thompson sampling draws a guess from each belief.", r"thompson"),
    ("Bellman", "The Bellman equation links neighbouring values.", r"bellman"),
    ("Rover", "Rover tries the fourth machine again.", r"\brover\b"),
    ("DeepSeekMath", "DeepSeekMath introduced GRPO.", r"deep ?seek ?math"),
    ("InstructGPT", "InstructGPT was preferred to a model a hundred times larger.", r"instruct ?g ?p ?t"),
    ("GSM8K", "GSM8K is a set of grade-school maths problems.", r"gsm ?8 ?k|g ?s ?m ?eight ?k|gsm8k"),
    ("Qwen", "Qwen is a family of open models.", r"qwen|quen|kwen|cuen"),
    ("Goodhart", "Goodhart's law warns about measures that become targets.", r"goodhart"),
    ("Montezuma", "Montezuma's Revenge hides its rewards behind many rooms.", r"montezuma"),
    ("Bradley-Terry", "The Bradley-Terry model turns scores into preferences.", r"bradley[- ]?terry"),
    ("gamma", "A discount gamma of zero point nine shrinks future rewards.", r"gamma"),
]

LISTENERS = {"large": "mlx-community/whisper-large-v3-turbo", "small": "mlx-community/whisper-small-mlx"}


def speak(sentence):
    settings = dict(voice_ref="", exaggeration=0.5, cfg_weight=0.3, temperature=0.8)  # the same voice as kit.narrator
    req = urllib.request.Request("http://127.0.0.1:8765/speak", json.dumps({"text": sentence, **settings}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as reply:
        return base64.b64decode(json.load(reply)["audio"])


rows = []
for i, (term, sentence, pattern) in enumerate(TESTS):
    wav = OUT / f"{i:02d}-{re.sub(r'[^A-Za-z0-9]+', '_', term)}.wav"
    if not wav.exists():
        wav.write_bytes(speak(sentence))
    audio, _ = librosa.load(io.BytesIO(wav.read_bytes()), sr=16000)
    heard = {name: mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language="en")["text"].strip()
             for name, repo in LISTENERS.items()}
    ok = {name: bool(re.search(pattern, t.lower())) for name, t in heard.items()}
    rows.append(dict(term=term, sentence=sentence, heard=heard, ok=ok))
    mark = "ok  " if all(ok.values()) else ("half" if any(ok.values()) else "FAIL")
    print(f"{mark} {term:15} large: {heard['large'][:70]!r} | small: {heard['small'][:70]!r}", flush=True)

(OUT / "preflight.json").write_text(json.dumps(rows, indent=1))
print(f"{sum(all(r['ok'].values()) for r in rows)}/{len(rows)} terms heard by both listeners")
