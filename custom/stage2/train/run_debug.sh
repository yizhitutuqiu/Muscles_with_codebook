#!/bin/bash
source /data/litengmo/anaconda3/etc/profile.d/conda.sh
conda activate gvhmr
export CUDA_VISIBLE_DEVICES=6
python custom/stage2/train/train_stage2_pose2emg.py --config custom/stage2/configs/temp_run_stage2_exp_1_pure_continuous_lightweight.yaml
