#!/usr/bin/env bash
# H6 Batch Ablation Studies

source /data/litengmo/anaconda3/etc/profile.d/conda.sh
conda activate gvhmr

cd /data/litengmo/HSMR/mia_custom

# Call the original bash script to start tmux sessions
bash custom/stage2/train/batch_train_stage2.sh \
    --config custom/stage2/configs/batch_train.stage2_h6.yaml
