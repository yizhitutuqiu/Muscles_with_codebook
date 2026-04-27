#!/bin/bash
set -e

# Config
BATCH_CONFIG="custom/stage2/configs/batch_train.stage2.yaml"
PYTHON_RUNNER="python custom/stage2/train/batch_train_stage2.py"

SESSION_NAME=""
ATTACH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)
            SESSION_NAME="$2"
            shift 2
            ;;
        --no-attach)
            ATTACH="0"
            shift
            ;;
        --config)
            BATCH_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Please install tmux, or run train_stage2_pose2emg.py manually."
    exit 1
fi

mkdir -p custom/stage2/logs

read_tmux_defaults() {
    python - "$1" <<'PY'
import sys
import yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
tmux_cfg = cfg.get("tmux", {}) or {}
session_name = tmux_cfg.get("session_name", "batch_stage2_train")
attach = "1" if bool(tmux_cfg.get("attach", True)) else "0"
fail_if_exists = "1" if bool(tmux_cfg.get("fail_if_exists", True)) else "0"
print(session_name)
print(attach)
print(fail_if_exists)
PY
}

tmux_defaults=$(read_tmux_defaults "$BATCH_CONFIG")
tmux_session_default=$(echo "$tmux_defaults" | sed -n '1p')
tmux_attach_default=$(echo "$tmux_defaults" | sed -n '2p')

# Precedence: CLI args > env > yaml defaults
if [ -z "$SESSION_NAME" ]; then
    SESSION_NAME="${TMUX_SESSION_NAME:-$tmux_session_default}"
fi
if [ -z "$ATTACH" ]; then
    ATTACH="${TMUX_ATTACH:-$tmux_attach_default}"
fi

# Get all cases
cases=$($PYTHON_RUNNER --config "$BATCH_CONFIG" --list_cases)
if [ -z "$cases" ]; then
    echo "No cases found in $BATCH_CONFIG"
    exit 1
fi

# Convert to array
case_array=()
while IFS= read -r line; do
    case_array+=("$line")
done <<< "$cases"

total_cases=${#case_array[@]}
echo "Found $total_cases cases to run."

# Get available GPUs from config, fallback to CUDA_VISIBLE_DEVICES
read_gpus() {
    python - "$1" <<'PY'
import sys
import os
import yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
gpus = cfg.get("gpus", [])
if len(gpus) > 0:
    print(",".join(map(str, gpus)))
else:
    print(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
PY
}

gpu_str=$(read_gpus "$BATCH_CONFIG")
IFS=',' read -ra gpu_list <<< "$gpu_str"

num_gpus=${#gpu_list[@]}
echo "Available GPUs for this batch: ${gpu_list[*]} ($num_gpus GPUs)"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME (killing it and all its processes)"
    tmux kill-session -t "$SESSION_NAME"
fi

echo "Creating tmux session: $SESSION_NAME"

# Create session with a monitor window
tmux new-session -d -s "$SESSION_NAME" -n "monitor" "echo 'Session: $SESSION_NAME'; echo 'BATCH_CONFIG: $BATCH_CONFIG'; echo 'GPUs: ${gpu_list[*]}'; echo 'Cases: $total_cases'; echo ''; echo 'Attach: tmux attach -t $SESSION_NAME'; bash"

# Assign cases to GPUs in round-robin; each GPU window runs its assigned cases sequentially
declare -a gpu_cases
for ((i=0; i<num_gpus; i++)); do
    gpu_cases[$i]=""
done

for ((i=0; i<total_cases; i++)); do
    gi=$((i % num_gpus))
    if [ -z "${gpu_cases[$gi]}" ]; then
        gpu_cases[$gi]="${case_array[$i]}"
    else
        gpu_cases[$gi]="${gpu_cases[$gi]} ${case_array[$i]}"
    fi
done

for ((gi=0; gi<num_gpus; gi++)); do
    gpu="${gpu_list[$gi]}"
    win_name="gpu_${gpu}"
    assigned="${gpu_cases[$gi]}"
    if [ -z "$assigned" ]; then
        continue
    fi

    run_script=$(cat <<EOF
set -e
if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate gvhmr
else
  echo "conda not found in PATH" >&2
  exit 1
fi
echo "[GPU $gpu] cases: ${assigned}"
export CUDA_VISIBLE_DEVICES=$gpu
for c in ${assigned}; do
  echo ""
  echo "[GPU $gpu] start: \$c"
  tmux pipe-pane -t "$SESSION_NAME:$win_name" "cat > custom/stage2/logs/batch_stage2_\${c}.log"
  # IMPORTANT: Do not pass --device "cuda:0" to let train_stage2_pose2emg.py respect CUDA_VISIBLE_DEVICES
  $PYTHON_RUNNER --config "$BATCH_CONFIG" --case "\$c"
  echo "[GPU $gpu] done: \$c"
done
echo "[GPU $gpu] all done"
tmux kill-window -t "$SESSION_NAME:$win_name"
EOF
)
    tmux new-window -t "$SESSION_NAME" -n "$win_name" "bash -lc $(printf %q "$run_script")"
done

# Summary window waits until only monitor+summary remain, then generates summary.csv
summary_script=$(cat <<EOF
set -e
if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate gvhmr
else
  echo "conda not found in PATH" >&2
  exit 1
fi
echo "[Summary] waiting for training windows to finish..."
while true; do
  w=\$(tmux list-windows -t "$SESSION_NAME" -F '#{window_name}' | grep -E '^gpu_' | wc -l)
  if [ "\$w" -eq 0 ]; then
    break
  fi
  sleep 5
done
echo "[Summary] generating summary.csv"
$PYTHON_RUNNER --config "$BATCH_CONFIG" --summary
echo "[Summary] done"
bash
EOF
)
tmux new-window -t "$SESSION_NAME" -n "summary" "bash -lc $(printf %q "$summary_script")"

echo "tmux session ready: $SESSION_NAME"
echo "Monitor: tmux attach -t $SESSION_NAME"

if [ "$ATTACH" == "1" ]; then
    tmux attach -t "$SESSION_NAME"
fi