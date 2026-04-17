#!/usr/bin/env bash
set -euo pipefail

OOD_CONFIG="${OOD_CONFIG:-/data/litengmo/HSMR/mia_custom/custom/configs/ood/ood.yaml}"
OOD_PARALLEL="${OOD_PARALLEL:-0}"
CUDA_VISIBLE="${CUDA_VISIBLE:-${cuda_visible:-}}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE}"

if [[ "$OOD_PARALLEL" == "1" && -n "$GPU_LIST" ]]; then
  IFS=',' read -ra GPUS <<< "$GPU_LIST"
  if [[ "${#GPUS[@]}" -lt 1 ]]; then
    echo "Invalid GPU list: $GPU_LIST"
    exit 2
  fi

  mapfile -t CASES < <(python /data/litengmo/HSMR/mia_custom/custom/stage2/train/ood_batch_train.py --ood_config "$OOD_CONFIG" --list_cases true)
  if [[ "${#CASES[@]}" -eq 0 ]]; then
    echo "No OOD cases found."
    exit 3
  fi

  running=0
  idx=0
  for line in "${CASES[@]}"; do
    proto="$(echo "$line" | cut -f1)"
    target="$(echo "$line" | cut -f2)"
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    idx=$((idx + 1))

    while [[ "$running" -ge "${#GPUS[@]}" ]]; do
      wait -n
      running=$((running - 1))
    done

    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python /data/litengmo/HSMR/mia_custom/custom/stage2/train/ood_batch_train.py \
        --ood_config "$OOD_CONFIG" \
        --protocol "$proto" \
        --only_target "$target"
    ) &
    running=$((running + 1))
  done
  wait
else
  python /data/litengmo/HSMR/mia_custom/custom/stage2/train/ood_batch_train.py \
    --ood_config "$OOD_CONFIG"
fi
