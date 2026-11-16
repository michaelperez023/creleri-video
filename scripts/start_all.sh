#!/usr/bin/env bash
# Start every CReLeRI service in its own tmux window.
#
#   scripts/start_all.sh [session-name]
#
# Settings come from the repo-root .env (copy .env.example). GPU assignment can be
# overridden with SEG_GPU, OBJDET_GPU and GROUNDING_GPUS, and extra Qwen flags passed
# with QWEN_ARGS; the defaults match the README layout for a 4-GPU machine. Attach with
# `tmux attach -t creleri`, stop everything with `tmux kill-session -t creleri`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${1:-creleri}"
ENV_FILE="$ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

SEG_GPU="${SEG_GPU:-1}"
OBJDET_GPU="${OBJDET_GPU:-0}"
GROUNDING_GPUS="${GROUNDING_GPUS:-0 1 2 3}"
QWEN_ARGS="${QWEN_ARGS:-}"
WEBUI_PORT="${WEBUI_PORT:-5004}"
REDIS_PORT="${REDIS_PORT:-17253}"

for cmd in tmux conda; do
  command -v "$cmd" >/dev/null || { echo "error: $cmd is not installed" >&2; exit 1; }
done
REDIS_SERVER="$ROOT/redis/redis-stable/src/redis-server"
if [[ ! -x "$REDIS_SERVER" ]]; then
  REDIS_SERVER="$(command -v redis-server || true)"
fi
if [[ -z "$REDIS_SERVER" ]]; then
  echo "error: redis-server not found; see 'Install' in README.md" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "error: tmux session '$SESSION' is already running (tmux attach -t $SESSION)" >&2
  exit 1
fi

# Every window loads .env itself: an already-running tmux server would not pass on
# this script's environment.
LOAD_ENV="set -a; [ -f $(printf '%q' "$ENV_FILE") ] && . $(printf '%q' "$ENV_FILE"); set +a"

window() {  # window <name> <directory> <command>
  tmux new-window -t "$SESSION" -n "$1" -c "$ROOT/$2" "$LOAD_ENV; $3; exec \$SHELL"
}

mkdir -p "$ROOT/redis"  # Redis writes its dump.rdb here
tmux new-session -d -s "$SESSION" -n redis -c "$ROOT/redis" \
  "$(printf '%q' "$REDIS_SERVER") --port $REDIS_PORT; exec \$SHELL"

window seg-api adatad_temp_seg \
  "conda run --no-capture-output -n opentad4 python api.py -p 17249"
window seg-worker adatad_temp_seg \
  "CUDA_VISIBLE_DEVICES=$SEG_GPU conda run --no-capture-output -n opentad4 python celery_worker.py -d 0"
window objdet dfine_obj_det/tools/inference \
  "CUDA_VISIBLE_DEVICES=$OBJDET_GPU conda run --no-capture-output -n dfine2 python object_proposer_api.py"
window qwen vlm_action_pipeline/demo_pipeline \
  "PYTHONNOUSERSITE=1 conda run --no-capture-output -n qwen3 python qwen2p5_api.py $QWEN_ARGS"
window llama vlm_action_pipeline/demo_pipeline \
  "conda run --no-capture-output -n ufllama python llama_3p3_api.py"
window grounding vlm_action_pipeline/demo_pipeline \
  "conda run --no-capture-output -n demo_pipeline python grounding_api.py -d $GROUNDING_GPUS"
window webui web_ui \
  "source webui/bin/activate && python app.py $WEBUI_PORT"

echo "CReLeRI is starting in tmux session '$SESSION' (tmux attach -t $SESSION)."
echo "Web UI: http://localhost:$WEBUI_PORT"
