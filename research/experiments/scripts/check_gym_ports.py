"""Check rl_reference's CartPole-v1 and Pendulum-v1 ports against gymnasium, step for step.
Run in the throwaway venv that has gymnasium (from the series folder):
    research/experiments/.venv-classic/bin/python research/experiments/scripts/check_gym_ports.py
Writes research/experiments/scripts/out/gym_ports.json (merged into classic.json by build_classic.py)."""

import json
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

SERIES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERIES))
import rl_reference as rl  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
t0 = time.time()


def check_cartpole(episodes=300):
    env = gym.make("CartPole-v1")
    worst, steps_total, lengths = 0.0, 0, []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        state = rl.cartpole_reset(np.random.default_rng(seed))
        assert np.array_equal(obs, rl.cartpole_obs(state))
        worst = max(worst, np.abs(env.unwrapped.state - state).max())
        policy_rng = np.random.default_rng(10_000 + seed)
        # a mix of behaviours: random pushes, and a crude balancing rule (so some episodes reach the 500 limit)
        balance = seed % 2 == 1
        for t in range(1, 10_000):
            action = int(state[2] + 0.5 * state[3] > 0) if balance else int(policy_rng.integers(2))
            obs, reward, terminated, truncated, _ = env.step(action)
            state, my_reward, my_terminated = rl.cartpole_step(state, action)
            worst = max(worst, np.abs(env.unwrapped.state - state).max())
            assert np.array_equal(obs, rl.cartpole_obs(state)), (seed, t)
            assert reward == my_reward and terminated == my_terminated, (seed, t)
            assert truncated == (t >= rl.CARTPOLE["max_steps"] and not terminated), (seed, t)
            if terminated or truncated:
                break
        steps_total += t
        lengths.append(t)
    return dict(episodes=episodes, steps=steps_total, max_abs_state_diff=float(worst),
                episodes_truncated_at_500=int(sum(n == 500 for n in lengths)))


def check_pendulum(episodes=200):
    env = gym.make("Pendulum-v1")
    worst_state, worst_reward, steps_total = 0.0, 0.0, 0
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        state = rl.pendulum_reset(np.random.default_rng(seed))
        assert np.array_equal(obs, rl.pendulum_obs(state))
        policy_rng = np.random.default_rng(20_000 + seed)
        as_float32 = seed % 2 == 0                       # half the episodes pass float32 arrays, as agents do
        for t in range(1, 1000):
            u = policy_rng.uniform(-3, 3)                # beyond +-2 too, to exercise the clip
            action = np.array([u], dtype=np.float32 if as_float32 else np.float64)
            obs, reward, terminated, truncated, _ = env.step(action)
            state, my_reward = rl.pendulum_step(state, action)
            worst_state = max(worst_state, np.abs(env.unwrapped.state - state).max())
            worst_reward = max(worst_reward, abs(reward - my_reward))
            assert np.array_equal(obs, rl.pendulum_obs(state)), (seed, t)
            assert not terminated and truncated == (t >= rl.PENDULUM["max_steps"])
            if truncated:
                break
        steps_total += t
    return dict(episodes=episodes, steps=steps_total, max_abs_state_diff=float(worst_state),
                max_abs_reward_diff=float(worst_reward))


def reference_trajectories():
    """The short trajectories the self-check in rl_reference.py compares against (recorded from gymnasium)."""
    env = gym.make("CartPole-v1").unwrapped
    env.reset(seed=0)
    start = env.state.copy()
    for t in range(1, 1000):
        _, _, terminated, _, _ = env.step((t - 1) % 2)
        if terminated:
            break
    cart = dict(reset_seed0=start.tolist(), alternate_steps=t, alternate_final=env.state.tolist())
    env = gym.make("Pendulum-v1").unwrapped
    env.reset(seed=0)
    total = 0.0
    for t in range(200):
        _, r, _, _, _ = env.step(np.array([2.0 * math.sin(t / 10)]))
        total += r
    pend = dict(sine_return=float(total), sine_final=env.state.tolist())
    return cart, pend


cartpole, pendulum = check_cartpole(), check_pendulum()
cart_ref, pend_ref = reference_trajectories()
result = dict(gymnasium=gym.__version__, numpy=np.__version__, python=sys.version.split()[0],
              cartpole=cartpole, pendulum=pendulum, cartpole_reference=cart_ref, pendulum_reference=pend_ref,
              wall_s=round(time.time() - t0, 1))
(OUT / "gym_ports.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
