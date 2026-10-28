#!/bin/zsh
# Run grpo.py with the given arguments; on a crash (e.g. GPU out-of-memory while the machine is shared),
# wait and resume from the last checkpoint (every 10 steps). Gives up after 20 attempts.
cd "$(dirname "$0")"
for i in $(seq 1 20); do
  ../.venv-lm/bin/python -I grpo.py "$@" && exit 0
  echo "[run_resume] attempt $i failed, resuming in 30 s" >&2
  sleep 30
done
exit 1
